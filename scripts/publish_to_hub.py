"""Publish the final model and the dataset to the Hugging Face Hub, then check what arrived.

Two repositories, and nothing else: the model served, the supervised and the preference steps
merged into it, and the dataset it was trained and evaluated on. Neither adapter nor
the merged supervised model goes to the Hub: the pipeline rebuilds them, and the model card says how.

Requires a write token in ``HF_TOKEN``; the target account is set through ``HF_NAMESPACE``::

    $env:HF_TOKEN = "hf_..."
    uv run python scripts/publish_to_hub.py --what dataset
    uv run python scripts/publish_to_hub.py --what model
    uv run python scripts/publish_to_hub.py --what cards --tag model-v2.0.0
    uv run python scripts/publish_to_hub.py --what verify --tag model-v2.0.0

The cards are the files ``scripts/build_hub_cards.py`` renders, sent byte for byte: what the Hub
shows is what the repository versions. ``--tag`` places a tag on the repositories touched, and
refuses one the model card does not pin: an inference server asks for ``revision=model-v2.0.0`` and must
get the weights the card describes, where ``main`` moves at every publication.

``--what verify`` reads back what the Hub holds at the tag: each card equal to its file, each
weight file with the digest of the weights evaluated, each subset of the dataset loadable.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path

from clinical_triage.config import MODEL, PATHS
from clinical_triage.data.dataset_io import DATASET_FILES
from clinical_triage.training.export import export_problems, weight_digests
from clinical_triage.utils import get_logger

logger = get_logger("publish")

MODEL_CARD = PATHS.root / "infra" / "hub" / "model-card.md"
DATASET_CARD = PATHS.root / "infra" / "hub" / "dataset-card.md"
EVALUATION = PATHS.reports / "evaluation_results.json"

# What goes up from the merged model's folder: the weights, the configuration and the tokenizer.
# Anything else — a checkpoint, a card written by a library — stays on the machine.
MODEL_FILES = ["*.safetensors", "*.json", "*.txt", "*.jinja"]

# The subsets the dataset card declares, as `datasets` loads them.
SUBSETS = {"sft": ("train", "validation", "test"), "dpo": ("train",), "clinical_eval": ("test",)}


def _token() -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN missing. Create a write token at "
            "https://huggingface.co/settings/tokens then export it."
        )
    return token


def evaluated_weights(results: Path = EVALUATION) -> dict[str, str]:
    """The digests of the weight files the committed evaluation measured."""
    if not results.exists():
        raise SystemExit(
            f"Evaluation not found ({results}): nothing says which weights to publish."
        )
    digests = json.loads(results.read_text(encoding="utf-8")).get("evaluated_weights")
    if not digests:
        raise SystemExit(
            "The evaluation records no weight digest: run `scripts/run_evaluation.py`."
        )
    return digests


def publishable_model(folder: Path = PATHS.dpo_merged, results: Path = EVALUATION) -> None:
    """Refuse an export that is wrong, or that is not the one the evaluation measured."""
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")
    problems = export_problems(folder)
    if problems:
        raise SystemExit("Export refused: " + "; ".join(problems))
    if weight_digests(folder) != evaluated_weights(results):
        raise SystemExit(
            "These weights are not the ones evaluated: their digests differ from those recorded "
            "in the evaluation. Evaluate them, or publish the weights that were."
        )


def pinned_revision() -> str | None:
    """The revision the model card's serving command pins."""
    for line in MODEL_CARD.read_text(encoding="utf-8").splitlines():
        if "--revision" in line:
            return line.split("--revision", 1)[1].split()[0]
    return None


def complete_dataset(folder: Path = PATHS.data_processed) -> None:
    """Refuse an incomplete set rather than publish part of it as a success.

    The four training files are not versioned: they are rebuilt by ``scripts/build_dataset.py``.
    A publication launched from a fresh clone would otherwise upload two files out of six.
    """
    missing = [name for name in DATASET_FILES if not (folder / name).exists()]
    if missing:
        raise SystemExit(
            "Incomplete dataset, publication refused. Missing: "
            + ", ".join(missing)
            + ".\nRun `python scripts/build_dataset.py` on the machine that publishes."
        )


def publish_dataset(api) -> None:
    complete_dataset()
    # `private=False` is explicit on purpose: publishing data cannot be undone with a click, and
    # that choice is not left to a library default.
    api.create_repo(MODEL.hub_dataset_id, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(PATHS.data_processed),
        repo_id=MODEL.hub_dataset_id,
        repo_type="dataset",
        allow_patterns=list(DATASET_FILES),
    )
    publish_card(api, DATASET_CARD, MODEL.hub_dataset_id, "dataset")


def publish_model(api) -> None:
    publishable_model()
    api.create_repo(MODEL.hub_model_id, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(PATHS.dpo_merged),
        repo_id=MODEL.hub_model_id,
        repo_type="model",
        allow_patterns=MODEL_FILES,
    )
    publish_card(api, MODEL_CARD, MODEL.hub_model_id, "model")


def publish_card(api, card: Path, repository: str, repo_type: str) -> None:
    """Send a rendered card as the repository's README, unchanged."""
    if not card.exists():
        raise SystemExit(f"Card not found: {card}. Run `python scripts/build_hub_cards.py`.")
    api.upload_file(
        path_or_fileobj=card.read_bytes(),
        path_in_repo="README.md",
        repo_id=repository,
        repo_type=repo_type,
    )
    logger.info("Card of %s published.", repository)


def place_tag(api, repository: str, repo_type: str, name: str) -> None:
    """Place a tag, replacing the previous one of the same name.

    ``create_tag`` can ignore an existing tag but not move it onto new content: a publication
    redone after a correction removes it first.
    """
    from huggingface_hub.errors import RevisionNotFoundError

    with contextlib.suppress(RevisionNotFoundError):
        api.delete_tag(repository, tag=name, repo_type=repo_type)
    api.create_tag(repository, tag=name, repo_type=repo_type)
    logger.info("Tag %s placed on %s", name, repository)


def verify(api, tag: str) -> list[str]:
    """What differs between the Hub at ``tag`` and what the repository says it should hold."""
    from huggingface_hub import hf_hub_download

    problems = []
    for card, repository, repo_type in (
        (MODEL_CARD, MODEL.hub_model_id, "model"),
        (DATASET_CARD, MODEL.hub_dataset_id, "dataset"),
    ):
        published = Path(
            hf_hub_download(repository, "README.md", repo_type=repo_type, revision=tag)
        ).read_bytes()
        if published != card.read_bytes():
            problems.append(f"{repository}: the card differs from {card.name}")

    stored = {
        entry.path: entry.lfs.sha256
        for entry in api.list_repo_tree(MODEL.hub_model_id, revision=tag, expand=True)
        if getattr(entry, "lfs", None)
    }
    for name, digest in evaluated_weights().items():
        if stored.get(name) != digest:
            problems.append(f"{MODEL.hub_model_id}: {name} is not the file evaluated")

    from datasets import load_dataset

    for subset, splits in SUBSETS.items():
        try:
            loaded = load_dataset(MODEL.hub_dataset_id, subset, revision=tag)
        except Exception as error:  # noqa: BLE001 - any failure to load is the finding itself
            problems.append(f"{MODEL.hub_dataset_id}: subset {subset} does not load ({error})")
            continue
        if tuple(loaded) != splits:
            problems.append(f"{MODEL.hub_dataset_id}: subset {subset} has splits {tuple(loaded)}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--what", choices=["dataset", "model", "cards", "verify"], required=True)
    parser.add_argument("--tag", help="the version to tag, or to verify: model-v2.0.0")
    args = parser.parse_args()
    if args.tag and pinned_revision() != args.tag:
        raise SystemExit(
            f"The model card pins {pinned_revision()}, not {args.tag}: the tag would point "
            "readers to weights the card does not describe."
        )
    if args.what == "verify" and not args.tag:
        parser.error("--what verify needs the --tag to read back")

    from huggingface_hub import HfApi

    api = HfApi(token=_token())
    if args.what == "verify":
        problems = verify(api, args.tag)
        for problem in problems:
            logger.error(problem)
        if not problems:
            logger.info("The Hub holds, at %s, what the repository describes.", args.tag)
        return 1 if problems else 0

    touched = []
    if args.what in ("model", "cards"):
        if args.what == "model":
            publish_model(api)
        else:
            publish_card(api, MODEL_CARD, MODEL.hub_model_id, "model")
        touched.append((MODEL.hub_model_id, "model"))
    if args.what in ("dataset", "cards"):
        if args.what == "dataset":
            publish_dataset(api)
        else:
            publish_card(api, DATASET_CARD, MODEL.hub_dataset_id, "dataset")
        touched.append((MODEL.hub_dataset_id, "dataset"))
    if args.tag:
        for repository, repo_type in touched:
            place_tag(api, repository, repo_type, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
