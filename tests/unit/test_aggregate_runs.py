"""The seed study and the before/after, computed from evaluation results built for the test."""

from __future__ import annotations

import importlib.util
import sys

import pytest

from clinical_triage.config import PATHS

CASES = [
    {"id": "a", "level": "URGENCE_VITALE"},
    {"id": "b", "level": "URGENCE_MODEREE"},
    {"id": "c", "level": "CONSULTATION_DIFFEREE"},
]


@pytest.fixture(scope="module")
def aggregate():
    spec = importlib.util.spec_from_file_location(
        "aggregate_runs_script", PATHS.root / "scripts" / "aggregate_runs.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _results(model: str, accuracy: float, misses: int, errors: list[dict], task: float = 0.9):
    measures = {
        "accuracy": accuracy,
        "format_compliance": 1.0,
        "undertriage_detail": {"misses": misses, "urgent_cases": 2},
    }
    return {
        "clinical_set": {"models": {model: measures}},
        "internal_set": {"models": {model: {"accuracy": 1.0}}},
        "errors": {model: errors},
        "task_preferences": {"correctly_ordered_share": task},
    }


def test_the_seed_study_reports_each_seed_and_the_spread(aggregate):
    runs = {
        42: _results("dpo-merged", 0.70, 1, []),
        43: _results("dpo-merged", 0.60, 2, []),
        44: _results("dpo-merged", 0.65, 1, []),
    }
    study = aggregate.seed_study(runs)
    assert study["shipped_seed"] == 42
    assert set(study["seeds"]) == {"42", "43", "44"}
    spread = study["spread"]["dpo-merged"]["clinical_accuracy"]
    assert (spread["min"], spread["max"], spread["mean"]) == (0.60, 0.70, 0.65)


def test_the_before_after_rebuilds_the_predictions_from_the_errors(aggregate):
    """Case a: the previous release sent it down, this one does not. Case b: the reverse."""
    previous = _results(
        "previous-release",
        0.33,
        1,
        [{"id": "a", "predicted": "URGENCE_MODEREE"}, {"id": "c", "predicted": "URGENCE_MODEREE"}],
    )
    current = _results("dpo-merged", 0.67, 1, [{"id": "b", "predicted": "CONSULTATION_DIFFEREE"}])
    comparison = aggregate.before_after(previous, current, CASES)
    assert comparison["before"]["label"] == "previous-release"
    assert comparison["undertriage"] == {
        "urgent_cases": 2,
        "rescued_by_this_release": 1,
        "lost_by_this_release": 1,
        "p_mcnemar": 1.0,
    }
