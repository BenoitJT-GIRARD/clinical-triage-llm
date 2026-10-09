"""Which cases the deployment check sends, read from the committed evaluation."""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from clinical_triage.config import PATHS, TRIAGE
from clinical_triage.data.dataset_io import read_jsonl


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location(
        "check_deployment_script", PATHS.root / "scripts" / "check_deployment.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CASES = [
    {"id": "a", "level": "URGENCE_VITALE", "case_type": ""},
    {"id": "b", "level": "URGENCE_VITALE", "case_type": ""},
    {"id": "c", "level": "URGENCE_MODEREE", "case_type": "falsely_alarming"},
    {"id": "d", "level": "URGENCE_MODEREE", "case_type": ""},
    {"id": "e", "level": "CONSULTATION_DIFFEREE", "case_type": ""},
]


def test_one_direct_case_per_level_that_the_model_got_right(check):
    results = {"errors": {"dpo-merged": [{"id": "a"}]}}
    assert [c["id"] for c in check.reference_cases(results, CASES)] == ["b", "d", "e"]


def test_a_level_the_model_never_got_right_stops_the_check(check):
    results = {"errors": {"dpo-merged": [{"id": "e"}]}}
    with pytest.raises(SystemExit, match="CONSULTATION_DIFFEREE"):
        check.reference_cases(results, CASES)


@pytest.mark.claim
def test_the_committed_evaluation_offers_a_case_for_every_level(check):
    results = json.loads((PATHS.reports / "evaluation_results.json").read_text(encoding="utf-8"))
    cases = read_jsonl(PATHS.data_processed / "clinical_eval.jsonl")
    assert [c["level"] for c in check.reference_cases(results, cases)] == list(TRIAGE.levels)
