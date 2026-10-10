"""The rendering of the Hub cards: every figure read from its file, nothing written with a hole.

The renderer runs on a miniature repository: a ``pyproject.toml``, one results file and one
document, enough to check the placeholders, the formats and the links.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys

import pytest

from clinical_triage.config import MODEL, PATHS


def _script():
    path = PATHS.root / "scripts" / "build_hub_cards.py"
    spec = importlib.util.spec_from_file_location("build_hub_cards_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_hub_cards_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cards():
    return _script()


@pytest.fixture
def root(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion = "2.0.0"\n\n[project.urls]\n'
        'Repository = "https://github.com/account/project"\n',
        encoding="utf-8",
    )
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "evaluation_results.json").write_text(
        json.dumps(
            {
                "clinical_set": {
                    "n": 60,
                    "models": {"dpo-merged": {"accuracy": 0.9167, "accuracy_ci95": [0.82, 0.96]}},
                },
                "model_versus_rule": {"accuracy": {"p_mcnemar": 0.0002}},
                "evaluated_weights": {"model.safetensors": "ab" * 32},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_a_figure_is_read_from_its_file_and_written_in_its_format(cards, root):
    text = "{{evaluation:clinical_set.models.dpo-merged.accuracy|f3}} {{evaluation:clinical_set.models.dpo-merged.accuracy_ci95|ci}} on {{evaluation:clinical_set.n}}"
    assert cards.render(text, root) == "0.917 [0.82 – 0.96] on 60"


def test_the_formats_write_what_a_reader_expects(cards, root):
    assert (
        cards.render("{{evaluation:model_versus_rule.accuracy.p_mcnemar|p}}", root) == "p < 0.001"
    )
    assert (
        cards.render("{{evaluation:clinical_set.models.dpo-merged.accuracy|pct}}", root) == "91.7%"
    )
    assert cards.FORMATS["size"](7460) == "1K<n<10K"
    assert cards.FORMATS["minutes"](1830.0) == "30 min"


def test_paths_joined_by_a_plus_add_up(cards, root):
    assert cards.render("{{evaluation:clinical_set.n+clinical_set.n}}", root) == "120"


def test_a_missing_value_stops_the_rendering(cards, root):
    with pytest.raises(cards.MissingValue, match="internal_set"):
        cards.render("{{evaluation:internal_set.n}}", root)


def test_a_missing_results_file_stops_the_rendering(cards, root):
    with pytest.raises(cards.MissingValue, match="seed_study.json"):
        cards.render("{{seeds:spread}}", root)


def test_the_weight_digests_become_table_rows(cards, root):
    assert cards.render("{{evaluation:evaluated_weights|digests}}", root) == (
        f"| `model.safetensors` | `{'ab' * 32}` |"
    )


def test_the_release_values_come_from_the_project(cards, root):
    text = "{{release:model_id}}@{{release:model_tag}} {{release:repository}}/tree/{{release:tag}}"
    assert cards.render(text, root) == (
        f"{MODEL.hub_model_id}@model-v2.0.0 https://github.com/account/project/tree/v2.0.0"
    )


def test_an_included_document_loses_its_header_and_gets_absolute_links(cards, root):
    """A relative link written for GitHub becomes an absolute address on the Hub."""
    (root / "data").mkdir()
    (root / "data" / "README.md").write_text(
        "---\nlicense: x\n---\n\n# Data\n\n"
        "[the generator](../src/generator.py#L10), [the set](processed/metadata.json), "
        "[the Hub](https://huggingface.co), [below](#limits)\n\n"
        "![composition](../reports/figures/composition.png)\n",
        encoding="utf-8",
    )
    text = cards.render("{{include:data/README.md}}", root)
    assert text.startswith("# Data")
    assert "(https://github.com/account/project/blob/v2.0.0/src/generator.py#L10)" in text
    assert "(https://github.com/account/project/blob/v2.0.0/data/processed/metadata.json)" in text
    assert "(https://huggingface.co)" in text
    assert "(#limits)" in text
    assert (
        "(https://raw.githubusercontent.com/account/project/v2.0.0/reports/figures/composition.png)"
        in text
    )


@pytest.mark.parametrize("template", ["model-card.in.md", "dataset-card.in.md"])
def test_every_placeholder_of_the_templates_names_a_known_source_and_format(cards, template):
    text = (PATHS.root / "infra" / "hub" / template).read_text(encoding="utf-8")
    placeholders = cards.PLACEHOLDER.findall(text)
    assert placeholders
    for source, _, form in placeholders:
        assert source in {*cards.SOURCES, "release", "include"}, source
        assert not form or form in cards.FORMATS, form


def test_the_model_card_pins_the_release_tag_in_its_serving_command(cards):
    """The publication refuses a tag the card does not pin: the two must come from one place."""
    text = (PATHS.root / "infra" / "hub" / "model-card.in.md").read_text(encoding="utf-8")
    assert re.search(r"--revision \{\{release:model_tag\}\}", text)
    assert "model-v" not in text
