"""The release gate, on evaluations built to pass and to fail each criterion."""

from __future__ import annotations

import importlib.util
import sys

import pytest

from clinical_triage.config import PATHS


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location(
        "check_release_script", PATHS.root / "scripts" / "check_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _evaluation(format_compliance: float = 1.0, model_misses: int = 11, rule_misses: int = 22):
    return {
        "clinical_set": {"models": {"dpo-merged": {"format_compliance": format_compliance}}},
        "model_versus_rule": {
            "undertriage": {
                "model_undertriages": model_misses,
                "baseline_undertriages": rule_misses,
            }
        },
    }


def test_a_model_that_meets_both_criteria_is_released(gate):
    assert gate.release_problems(_evaluation()) == []


def test_an_unreadable_answer_refuses_the_release(gate):
    assert gate.release_problems(_evaluation(format_compliance=0.983))


def test_more_urgent_cases_missed_than_the_rule_refuses_the_release(gate):
    problems = gate.release_problems(_evaluation(model_misses=23, rule_misses=22))
    assert "23 urgent cases missed" in problems[0]


def test_an_evaluation_without_the_shipped_model_refuses_the_release(gate):
    assert gate.release_problems({"clinical_set": {"models": {}}})


@pytest.mark.claim
def test_the_committed_evaluation_passes_the_gate(gate):
    """A claim test: the published model must pass the gate the deployment applies."""
    assert gate.main() == 0
