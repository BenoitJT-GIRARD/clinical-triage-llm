"""Consistency of what is published on the Hugging Face Hub.

A LoRA adapter contains no model: it describes a correction to apply to specific weights, which
it designates by their repository. Publishing the adapter without those weights produces an
artefact nobody can open — and the mistake only shows at the first download, that is to say too
late.

These tests read the publication configuration and the model cards: they touch no network.
"""

from __future__ import annotations

import importlib.util
import json
import re
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


def _declared_base(card) -> str | None:
    """Read ``base_model`` from the YAML header of a model card."""
    found = re.search(r"^base_model:\s*(\S+)\s*$", card.read_text(encoding="utf-8"), re.MULTILINE)
    return found.group(1) if found else None


def test_every_published_model_has_its_card(publication):
    missing = [
        what
        for what in publication.EVERYTHING
        if what != "dataset" and not publication._model_card(what).exists()
    ]
    assert missing == []


def test_the_base_model_of_every_adapter_is_published(publication):
    """Pins the rule that every adapter's declared base is itself published."""
    published = {
        publication._model_repository(what) for what in publication.EVERYTHING if what != "dataset"
    }
    published.add(MODEL.base_model)  # the base model comes from its own publisher

    for what in publication.EVERYTHING:
        if not what.endswith("-adapter"):
            continue
        base = _declared_base(publication._model_card(what))
        assert base is not None, f"{what}: the card does not declare its base model"
        assert base in published, f"{what}: its base model {base} is not published"


def test_the_dpo_adapter_on_disk_designates_the_public_repository(publication):
    """The local path written by the library would make the weights unloadable elsewhere."""
    configuration = PATHS.dpo_adapter / "adapter_config.json"
    if not configuration.exists():
        pytest.skip("DPO adapter absent: run `python scripts/train_dpo.py` to produce it.")
    content = json.loads(configuration.read_text(encoding="utf-8"))
    assert content["base_model_name_or_path"] == MODEL.hub_sft_merged_model_id


def test_the_repository_identifiers_are_all_distinct():
    """Two artefacts sharing a repository would overwrite each other."""
    identifiers = [
        MODEL.hub_dataset_id,
        MODEL.hub_sft_model_id,
        MODEL.hub_sft_merged_model_id,
        MODEL.hub_dpo_model_id,
        MODEL.hub_merged_model_id,
    ]
    assert len(set(identifiers)) == len(identifiers)


def test_an_incomplete_dataset_does_not_go_to_the_hub(publication, tmp_path, monkeypatch):
    """Pins the refusal: an incomplete set stops the publication instead of truncating it.

    The refusal names what is missing, and names only that — a file already on disk must not
    appear in the message, or the operator looks for a problem that is not there.
    """
    import dataclasses

    (tmp_path / "clinical_eval.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        publication, "PATHS", dataclasses.replace(publication.PATHS, data_processed=tmp_path)
    )

    with pytest.raises(SystemExit) as refusal:
        publication.publish_dataset(api=None)
    assert "sft_train.jsonl" in str(refusal.value)
    assert "clinical_eval.jsonl" not in str(refusal.value)


def test_the_six_expected_files_are_the_ones_the_preparation_produces(publication):
    """The list must not drift from what ``scripts/build_dataset.py`` actually writes.

    It is written once, in ``dataset_io``, and both the preparation script and the publication
    read it from there. The training set not being versioned — four files out of six — comparing
    it with the contents of the disk would pass on the machine that just built the set and fail
    everywhere else: the expected list is therefore written out here.
    """
    assert publication.DATASET_FILES is dataset_io.DATASET_FILES
    assert set(publication.DATASET_FILES) == {
        "sft_train.jsonl",
        "sft_validation.jsonl",
        "sft_test.jsonl",
        "dpo_train.jsonl",
        "clinical_eval.jsonl",
        "metadata.json",
    }


def test_every_card_carries_the_marker_for_its_figures(publication):
    """Pins the marker every card must carry, since the figures are substituted into it."""
    for what in publication.MODELS:
        card = publication._model_card(what)
        assert "{{EVALUATION}}" in card.read_text(encoding="utf-8"), what


def test_every_card_knows_which_evaluated_model_it_describes(publication):
    assert set(publication.CARD_EVALUATION) == set(publication.MODELS)


def test_a_card_without_an_evaluation_cannot_be_published(publication, tmp_path, monkeypatch):
    import dataclasses

    monkeypatch.setattr(
        publication, "PATHS", dataclasses.replace(publication.PATHS, reports=tmp_path)
    )
    with pytest.raises(SystemExit, match="Evaluation not found"):
        publication._evaluation_table("final-model")


def test_the_card_table_carries_the_figures_of_the_model_it_describes(
    publication, tmp_path, monkeypatch
):
    import dataclasses
    import json as json_

    (tmp_path / "evaluation_results.json").write_text(
        json_.dumps(
            {
                "clinical_set": {
                    "n": 60,
                    "models": {
                        "dpo-merged": {
                            "accuracy": 0.917,
                            "accuracy_ci95": [0.82, 0.96],
                            "undertriage": 0.048,
                            "overtriage": 0.117,
                            "format_compliance": 1.0,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        publication, "PATHS", dataclasses.replace(publication.PATHS, reports=tmp_path)
    )
    table = publication._evaluation_table("final-model")
    assert "0.917 [0.82 - 0.96]" in table
    assert "4.8%" in table
    assert "100%" in table
    assert "on 60 cases" in table


def test_no_card_freezes_a_revision_in_its_serving_command(publication):
    """The serving command must pin the published version, not the first one.

    Hard-coded, it went on designating ``model-v1.0.0`` after every new publication: the card
    described one set of weights and had another served.
    """
    for what in publication.MODELS:
        text = publication._model_card(what).read_text(encoding="utf-8")
        assert "--revision model-v" not in text, what
        if "--revision" in text:
            assert "{{REVISION}}" in text, what


def test_the_card_written_by_the_library_is_not_published(publication):
    """It carries the local path of the base model, which the Hub rejects.

    Uploading the whole folder failed on it, before even starting.
    """
    assert "README.md" in publication.EXCLUSIONS

    folder = PATHS.dpo_adapter
    card = folder / "README.md"
    if card.exists():
        assert "base_model: " in card.read_text(encoding="utf-8")
