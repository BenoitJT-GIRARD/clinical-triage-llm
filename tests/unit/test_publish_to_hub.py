"""What the publication sends to the Hugging Face Hub, and what it refuses to send.

The Hub is replaced by a recorder: these tests read what the script would upload, and touch no
network.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys

import pytest

from clinical_triage.config import MODEL, PATHS
from clinical_triage.data import dataset_io


def _publication_script():
    """Load ``scripts/publish_to_hub.py``, which is not importable as a module."""
    path = PATHS.root / "scripts" / "publish_to_hub.py"
    spec = importlib.util.spec_from_file_location("publish_to_hub_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["publish_to_hub_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def publication():
    return _publication_script()


class RecordingHub:
    """Stands for ``HfApi``: records every call instead of sending anything."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args, kwargs))


def _export(folder, weights=b"weights", tied=False):
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"tie_word_embeddings": tied}), encoding="utf-8")
    (folder / "model.safetensors").write_bytes(weights)
    index = {"weight_map": {"lm_head.weight": "model.safetensors"}}
    (folder / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return folder


def _evaluation(path, weights=b"weights"):
    digests = {"model.safetensors": hashlib.sha256(weights).hexdigest()}
    path.write_text(json.dumps({"evaluated_weights": digests}), encoding="utf-8")
    return path


def test_the_weights_evaluated_are_publishable(publication, tmp_path):
    publication.publishable_model(_export(tmp_path / "merged"), _evaluation(tmp_path / "e.json"))


def test_weights_other_than_the_ones_evaluated_are_refused(publication, tmp_path):
    """A later export, even a correct one, is not what the published figures describe."""
    folder = _export(tmp_path / "merged", weights=b"retrained")
    with pytest.raises(SystemExit, match="not the ones evaluated"):
        publication.publishable_model(folder, _evaluation(tmp_path / "e.json"))


def test_an_export_that_lost_its_output_head_is_refused(publication, tmp_path):
    folder = _export(tmp_path / "merged", tied=True)
    with pytest.raises(SystemExit, match="Export refused"):
        publication.publishable_model(folder, _evaluation(tmp_path / "e.json"))


def test_an_evaluation_without_digests_says_which_step_to_run(publication, tmp_path):
    results = tmp_path / "e.json"
    results.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="run_evaluation"):
        publication.evaluated_weights(results)


def test_an_incomplete_dataset_does_not_go_to_the_hub(publication, tmp_path):
    """The refusal names what is missing, and only that."""
    (tmp_path / "clinical_eval.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as refusal:
        publication.complete_dataset(tmp_path)
    assert "sft_train.jsonl" in str(refusal.value)
    assert "clinical_eval.jsonl" not in str(refusal.value)


def test_the_six_expected_files_are_the_ones_the_preparation_produces(publication):
    """Written out here: the four training files are not versioned, so the disk cannot say."""
    assert publication.DATASET_FILES is dataset_io.DATASET_FILES
    assert set(publication.DATASET_FILES) == {
        "sft_train.jsonl",
        "sft_validation.jsonl",
        "sft_test.jsonl",
        "dpo_train.jsonl",
        "clinical_eval.jsonl",
        "metadata.json",
    }


def test_a_card_goes_up_byte_for_byte(publication, tmp_path):
    card = tmp_path / "card.md"
    card.write_bytes("---\nlicense: apache-2.0\n---\n\n# Titre accentué\n".encode())
    hub = RecordingHub()
    publication.publish_card(hub, card, "account/model", "model")
    [(name, _, arguments)] = hub.calls
    assert name == "upload_file"
    assert arguments["path_or_fileobj"] == card.read_bytes()
    assert arguments["path_in_repo"] == "README.md"


def test_a_tag_the_model_card_does_not_pin_is_refused(publication, tmp_path, monkeypatch):
    """The tag would send readers to weights the card does not describe."""
    card = tmp_path / "model-card.md"
    card.write_text("vllm serve account/model --revision model-v2.0.0\n", encoding="utf-8")
    monkeypatch.setattr(publication, "MODEL_CARD", card)
    monkeypatch.setattr(sys, "argv", ["publish", "--what", "cards", "--tag", "model-v9.9.9"])
    with pytest.raises(SystemExit, match="pins model-v2.0.0"):
        publication.main()


def test_the_cards_go_to_the_two_repositories_and_both_are_tagged(
    publication, tmp_path, monkeypatch
):
    card = tmp_path / "model-card.md"
    card.write_text("vllm serve account/model --revision model-v2.0.0\n", encoding="utf-8")
    monkeypatch.setattr(publication, "MODEL_CARD", card)
    monkeypatch.setattr(publication, "DATASET_CARD", card)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    hub = RecordingHub()
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token: hub)
    monkeypatch.setattr(
        publication,
        "place_tag",
        lambda api, repo, kind, tag: hub.calls.append(("tag", (repo, kind, tag), {})),
    )
    monkeypatch.setattr(sys, "argv", ["publish", "--what", "cards", "--tag", "model-v2.0.0"])

    assert publication.main() == 0
    uploads = {arguments["repo_id"] for name, _, arguments in hub.calls if name == "upload_file"}
    tags = {args for name, args, _ in hub.calls if name == "tag"}
    assert uploads == {MODEL.hub_model_id, MODEL.hub_dataset_id}
    assert tags == {
        (MODEL.hub_model_id, "model", "model-v2.0.0"),
        (MODEL.hub_dataset_id, "dataset", "model-v2.0.0"),
    }


def test_the_model_and_the_dataset_repositories_are_distinct():
    """Two artefacts sharing a repository would overwrite each other."""
    assert MODEL.hub_model_id != MODEL.hub_dataset_id


def test_the_subsets_verified_are_the_ones_the_dataset_card_declares(publication):
    """The verification loads what the card announces, not a list of its own."""
    import re

    import yaml

    template = (PATHS.root / "infra" / "hub" / "dataset-card.in.md").read_text(encoding="utf-8")
    header = yaml.safe_load(re.sub(r"\{\{[^}]+\}\}", "0", template.split("---\n")[1]))
    declared = {
        config["config_name"]: tuple(item["split"] for item in config["data_files"])
        for config in header["configs"]
    }
    assert declared == publication.SUBSETS
    files = {item["path"] for config in header["configs"] for item in config["data_files"]}
    assert files == {name for name in publication.DATASET_FILES if name.endswith(".jsonl")}
