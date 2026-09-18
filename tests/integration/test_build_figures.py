"""Every published figure can be redrawn from the artefacts this repository carries.

The figures are committed images, and nobody diffs a PNG. What can be checked is that the
script that writes them still runs against the result files as they are committed, that it
writes the ten the repository publishes, and that each one arrives with its entry in the
manifest. Without that, the first evaluation whose output changes shape breaks the drawing
silently, and the figures in the README go on showing the previous run.

The real artefacts are used rather than fixtures: what is under test is the agreement between
the script and the files it reads, and a fixture would agree with itself.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from clinical_triage.config import PATHS
from clinical_triage.reporting.artifacts import read_results

pytestmark = pytest.mark.integration

#: What ``reports/figures/`` publishes, and what the smoke run compares against its committed
#: version. The list is written out rather than read from the folder: a figure that stopped
#: being produced would otherwise stop being expected at the same time.
PUBLISHED = (
    "accuracy_by_case_type.png",
    "confusion_matrices.png",
    "dataset_composition.png",
    "dpo_alignment.png",
    "endpoint_latency.png",
    "hyperparameter_tuning.png",
    "recall_versus_transfer.png",
    "sft_training.png",
    "systems_comparison.png",
    "undertriage_by_case_type.png",
)


@pytest.fixture(scope="module")
def build_figures():
    """Load ``scripts/build_figures.py``, which is not importable as a module."""
    import importlib.util
    import sys

    path = PATHS.root / "scripts" / "build_figures.py"
    spec = importlib.util.spec_from_file_location("build_figures_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_figures_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def drawn(build_figures, tmp_path_factory) -> dict:
    """Run the script once, into a temporary folder, and return its manifest."""
    directory = tmp_path_factory.mktemp("figures")
    original = build_figures.PATHS
    build_figures.PATHS = dataclasses.replace(original, figures=directory)
    try:
        build_figures.main([])
    finally:
        build_figures.PATHS = original
    manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
    return {"directory": directory, "images": manifest["images"]}


def test_the_ten_published_figures_are_all_produced(drawn):
    produced = {path.name for path in drawn["directory"].glob("*.png")}
    assert produced == set(PUBLISHED)


def test_each_figure_records_where_its_numbers_came_from(drawn):
    """A figure that names no source cannot be checked against anything."""
    for name in PUBLISHED:
        entry = drawn["images"][name]
        assert entry["source"] == "scripts/build_figures.py", name
        assert entry["n"], name
        assert entry["sha256"], name


def test_every_figure_labels_both_of_its_axes(drawn):
    """The style refuses an unlabelled axis; this reads back what it accepted."""
    for name in PUBLISHED:
        for axes in drawn["images"][name]["axes"]:
            assert axes["x"].strip(), f"{name}: an axis has no x label"
            assert axes["y"].strip(), f"{name}: an axis has no y label"


def test_the_shipped_model_is_the_most_advanced_one_evaluated(build_figures):
    """The undertriage figure describes one model, and it must be the one that is served."""
    evaluation = read_results(PATHS.reports / "evaluation_results.json")
    assert build_figures._shipped_model(evaluation) == "dpo-merged"


def test_the_two_forms_of_the_aligned_model_are_not_drawn_twice(build_figures):
    """Adapter and merged model share their weights: two columns would be one column twice."""
    kept = build_figures._progression({"base": 1, "sft": 2, "dpo": 3, "dpo-merged": 4})
    assert set(kept) == {"base", "sft", "dpo-merged"}

    # With only one of the two present, nothing is removed.
    assert set(build_figures._progression({"sft": 1, "dpo": 2})) == {"sft", "dpo"}


def test_a_repository_without_results_says_which_script_to_run(
    build_figures, monkeypatch, tmp_path
):
    """A clone carries the artefacts; a fresh pipeline does not yet."""
    monkeypatch.setattr(
        build_figures, "PATHS", dataclasses.replace(build_figures.PATHS, reports=tmp_path)
    )
    with pytest.raises(SystemExit, match="scripts/build_dataset.py"):
        build_figures.main([])
