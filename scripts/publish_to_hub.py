"""Publish the dataset and the models to the Hugging Face Hub.

The git repository already carries the dataset and its card; the Hub adds online browsing, a
data preview and the download of weights too large for a git repository.

Requires a write token in ``HF_TOKEN``. The target account is set through ``HF_NAMESPACE``,
without touching the code::

    $env:HF_TOKEN = "hf_..."
    uv run python scripts/publish_to_hub.py --what dataset
    uv run python scripts/publish_to_hub.py --what final-model

``--tag`` additionally places a tag on the repositories touched. That is what makes a model
version citable: an inference server can then ask for ``revision=model-v1.0.0`` and always get
the same weights, where ``main`` moves at every publication. Continuous deployment uses it to
pin the version it puts online::

    uv run python scripts/publish_to_hub.py --what cards --tag model-v1.0.0

Model folders contain intermediate checkpoints, useful locally but worthless on the Hub: they
are excluded from publication.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path

from clinical_triage.config import MODEL, PATHS
from clinical_triage.data.dataset_io import DATASET_FILES
from clinical_triage.utils import get_logger

logger = get_logger("publish")

# Files excluded from every publication: intermediate checkpoints and optimiser states, which
# weigh a lot and only serve to resume an interrupted training run.
EXCLUSIONS = [
    "checkpoint-*/*",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    # The training library drops its own card into the adapter folder, and writes the **local
    # path** of the base model into it. The Hub refuses the whole upload for that single line,
    # and the project's card arrives right after anyway, under the same name.
    "README.md",
]

# The four model repositories, with the local folder that feeds each.
MODELS = {
    "sft-adapter": "sft_adapter",
    "merged-sft-model": "sft_merged",
    "dpo-adapter": "dpo_adapter",
    "final-model": "dpo_merged",
}

# What ``--what all`` publishes: the four models and the dataset.
#
# The merged supervised model is part of it, although it weighs four gigabytes and nobody serves
# it. There is a reason, and it is not negotiable: **the DPO adapter is trained on top of it**,
# and its configuration designates it as the base model. Not publishing it would make the DPO
# adapter an unloadable artefact — ``PeftModel.from_pretrained`` would look for a repository that
# does not exist. Four gigabytes of storage are better than a public artefact nobody can open.
EVERYTHING = [
    "dataset",
    "sft-adapter",
    "merged-sft-model",
    "dpo-adapter",
    "final-model",
]


def _token() -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN missing. Create a write token at "
            "https://huggingface.co/settings/tokens then export it."
        )
    return token


def _model_repository(what: str) -> str:
    identifiers = {
        "sft-adapter": MODEL.hub_sft_model_id,
        "merged-sft-model": MODEL.hub_sft_merged_model_id,
        "dpo-adapter": MODEL.hub_dpo_model_id,
        "final-model": MODEL.hub_merged_model_id,
    }
    return identifiers[what]


def _model_card(what: str) -> Path:
    return PATHS.root / "infra" / "model-cards" / f"{what}.md"


def publish_dataset(api) -> str:
    """Publish the JSONL files, the data card and the metadata."""
    # The four training sets are not versioned: they are rebuilt by ``scripts/build_dataset.py``.
    # Without this refusal, a publication launched from a freshly cloned repository would upload
    # two files out of six and announce it as a success.
    missing = [n for n in DATASET_FILES if not (PATHS.data_processed / n).exists()]
    if missing:
        raise SystemExit(
            "Incomplete dataset, publication refused. Missing: "
            + ", ".join(missing)
            + ".\nRun `python scripts/build_dataset.py` on the machine that publishes."
        )
    repository = MODEL.hub_dataset_id
    # ``private=False`` is explicit on purpose: the set is meant to be public, and publishing
    # data cannot be undone with a click. That choice is not left to a library default.
    api.create_repo(repository, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(PATHS.data_processed),
        repo_id=repository,
        repo_type="dataset",
        allow_patterns=["*.jsonl", "*.json"],
    )
    api.upload_file(
        path_or_fileobj=str(PATHS.data / "README.md"),
        path_in_repo="README.md",
        repo_id=repository,
        repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{repository}"


def publish_model(api, what: str, tag: str | None = None) -> str:
    """Publish a LoRA adapter or the final merged model, with its card."""
    folder = getattr(PATHS, MODELS[what])
    repository = _model_repository(what)
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")
    api.create_repo(repository, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repository,
        repo_type="model",
        ignore_patterns=EXCLUSIONS,
    )
    publish_card(api, what, tag)
    return f"https://huggingface.co/{repository}"


# Which evaluated model matches which card. The supervised adapter and the merged supervised
# model carry the same training: they share their figures, as they share their weights to the
# rounding.
CARD_EVALUATION = {
    "sft-adapter": "sft",
    "merged-sft-model": "sft",
    "dpo-adapter": "dpo",
    "final-model": "dpo-merged",
}


def _evaluation_table(what: str) -> str:
    """Render the model's figures, as they will be read on the Hub.

    A model card with no measurable performance is not a model card: it is the first page anyone
    opening the repository sees, and a pointer to a document hosted elsewhere does not do.
    """
    results = PATHS.reports / "evaluation_results.json"
    if not results.exists():
        raise SystemExit(
            f"Evaluation not found ({results}). Run `python scripts/run_evaluation.py` before "
            "publishing: a model card with no figures has no purpose."
        )
    block = json.loads(results.read_text(encoding="utf-8"))["clinical_set"]
    name = CARD_EVALUATION[what]
    measures = block["models"].get(name)
    if measures is None:
        raise SystemExit(f"Model `{name}` was not evaluated: card `{what}` cannot be published.")

    low, high = measures["accuracy_ci95"]

    def percent(value: float) -> str:
        return f"{value * 100:.1f}%".replace(".0%", "%")

    return "\n".join(
        [
            f"| Measure | Value on {block['n']} cases |",
            "|---|---|",
            f"| Triage level accuracy | {measures['accuracy']:.3f} [{low:.2f} - {high:.2f}] |",
            f"| **Undertriage of urgent cases** | **{percent(measures['undertriage'])}** |",
            f"| Overtriage, all cases | {percent(measures['overtriage'])} |",
            (
                "| Answers the information system can parse | "
                f"{percent(measures['format_compliance'])} |"
            ),
        ]
    )


def publish_card(api, what: str, tag: str | None = None) -> str:
    """Update a model's card alone, without touching the weights.

    That is what continuous integration needs: the cards are versioned in the git repository and
    evolve with the documentation, whereas the weights are uploaded from the training machine.
    The figures are injected here from the results file: the published card therefore carries the
    measurements of the version it describes, with nothing copied by hand.

    The serving command it gives pins the tag placed by the same run. Hard-coded, it would
    designate the first publication after every new one: the card would describe one set of
    weights and have another served.
    """
    repository = _model_repository(what)
    card = _model_card(what)
    if not card.exists():
        raise SystemExit(f"Card not found: {card}")
    text = card.read_text(encoding="utf-8").replace("{{EVALUATION}}", _evaluation_table(what))
    text = text.replace("{{REVISION}}", tag or "main")
    api.create_repo(repository, repo_type="model", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repository,
        repo_type="model",
    )
    return f"https://huggingface.co/{repository}"


def place_tag(api, repository: str, repo_type: str, name: str) -> None:
    """Place a tag on a Hub repository, replacing the previous one.

    Re-tagging is useful when a publication is redone after a correction: ``create_tag`` can
    ignore an existing tag, but not move it onto the new content. So it is removed first, if it
    is there.
    """
    from huggingface_hub.errors import RevisionNotFoundError

    with contextlib.suppress(RevisionNotFoundError):
        api.delete_tag(repository, tag=name, repo_type=repo_type)
    api.create_tag(repository, tag=name, repo_type=repo_type)
    logger.info("Tag %s placed on %s", name, repository)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--what",
        choices=["dataset", *MODELS, "cards", "all"],
        required=True,
        help="dataset, one specific model, `cards` for the cards alone, or `all`",
    )
    parser.add_argument(
        "--tag",
        help="tag name to place on the repositories touched, for example model-v1.0.0",
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi(token=_token())

    if args.what == "all":
        requested = list(EVERYTHING)
    elif args.what == "cards":
        # Only the models actually published: refreshing the card of a repository that was never
        # created would create it, empty, with a README for all content.
        requested = [name for name in EVERYTHING if name != "dataset"]
    else:
        requested = [args.what]

    cards_only = args.what == "cards"
    touched: list[tuple[str, str]] = []
    for request in requested:
        if request == "dataset":
            address = publish_dataset(api)
            repository, repo_type = MODEL.hub_dataset_id, "dataset"
        elif cards_only:
            address = publish_card(api, request, args.tag)
            repository, repo_type = _model_repository(request), "model"
        else:
            address = publish_model(api, request, args.tag)
            repository, repo_type = _model_repository(request), "model"
        touched.append((repository, repo_type))
        logger.info("Published: %s", address)

    # With ``--what cards`` the dataset is not republished — the continuous-integration runner
    # does not have its files — but its repository must receive the tag like the others: the tag
    # is what makes the version citable, and a revision where the dataset is untagged is
    # incomplete.
    if cards_only and api.repo_exists(MODEL.hub_dataset_id, repo_type="dataset"):
        touched.append((MODEL.hub_dataset_id, "dataset"))

    if args.tag:
        for repository, repo_type in touched:
            place_tag(api, repository, repo_type, args.tag)


if __name__ == "__main__":
    main()
