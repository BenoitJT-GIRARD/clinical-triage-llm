"""The comparisons the published results rest on.

Every headline claim of this project is a comparison: the model against the rule a department
could deploy in an afternoon, the model against an ordinary classifier trained on the same
pairs, the aligned model against the supervised one. All three are settled by a paired test on
the same cases, and all three are computed here, in the evaluation script.

The counting is where a paired test goes wrong in silence — swap ``model_only`` and
``baseline_only`` and the published p-value still looks like a p-value. These tests fix the
counting on cases small enough to check by hand.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from clinical_triage.config import MODEL, PATHS

IMMEDIATE = "URGENCE_VITALE"
URGENT = "URGENCE_MODEREE"
DEFERRED = "CONSULTATION_DIFFEREE"


@pytest.fixture(scope="module")
def evaluation():
    """Load ``scripts/run_evaluation.py``, which is not importable as a module."""
    path = PATHS.root / "scripts" / "run_evaluation.py"
    spec = importlib.util.spec_from_file_location("run_evaluation_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_evaluation_under_test"] = module
    spec.loader.exec_module(module)
    return module


def _cases(*levels: str) -> list[dict]:
    return [
        {"id": f"case_{index}", "level": level, "description": "..."}
        for index, level in enumerate(levels)
    ]


# --- Which weights each evaluated model is made of ---------------------------


def test_the_aligned_adapter_is_evaluated_on_top_of_the_model_it_was_trained_on(evaluation):
    """Applied to the base model, the DPO adapter would compose weights that do not go together.

    It was trained above the merged supervised model, and that is the only base on which its
    measurement means anything.
    """
    base, adapter = evaluation._specification("dpo")
    assert base == str(PATHS.sft_merged)
    assert adapter == str(PATHS.dpo_adapter)


def test_the_supervised_adapter_is_evaluated_on_the_base_model(evaluation):
    base, adapter = evaluation._specification("sft")
    assert base == MODEL.base_model
    assert adapter == str(PATHS.sft_adapter)


def test_a_merged_model_is_evaluated_without_any_adapter(evaluation):
    for name in ("base", "dpo-merged"):
        assert evaluation._specification(name)[1] is None


def test_every_evaluable_model_has_its_specification(evaluation):
    """A typo in ``--models`` must fail at once, not after two hours of evaluation."""
    for name in evaluation.EVALUABLE_MODELS:
        assert evaluation._specification(name)


def test_absent_weights_weigh_nothing_rather_than_failing(evaluation, tmp_path):
    """The comparison of deliveries runs on a machine that does not carry the weights."""
    assert evaluation._weights_on_disk(tmp_path / "nothing-here") == 0.0

    (tmp_path / "model.safetensors").write_bytes(b"0" * 2_097_152)
    (tmp_path / "README.md").write_text("not a weight file", encoding="utf-8")
    assert evaluation._weights_on_disk(tmp_path) == 2.0


# --- Model against baseline, on the same cases -------------------------------


def test_a_paired_comparison_counts_only_the_cases_where_the_two_disagree(evaluation):
    """That is the whole content of a paired test: agreements carry no information.

    Four cases, and one each way: the model saves the second, the baseline saves the third, and
    the other two are agreements. With that many cases nothing is established, which is what the
    p-value says.
    """
    cases = _cases(IMMEDIATE, IMMEDIATE, URGENT, DEFERRED)
    model = [IMMEDIATE, IMMEDIATE, DEFERRED, DEFERRED]
    baseline = [IMMEDIATE, URGENT, URGENT, DEFERRED]

    comparison = evaluation._compare_to_baseline(cases, model, "dpo-merged", baseline, "rule")

    assert comparison["model"] == "dpo-merged"
    assert comparison["baseline"] == "rule"
    assert comparison["accuracy"] == {
        "n": 4,
        "model_only": 1,
        "baseline_only": 1,
        "agreements": 2,
        "p_mcnemar": 1.0,
    }


def test_a_model_that_is_right_everywhere_the_baseline_is_wrong_is_separated(evaluation):
    cases = _cases(*[IMMEDIATE] * 8)
    model = [IMMEDIATE] * 8
    baseline = [DEFERRED] * 8

    comparison = evaluation._compare_to_baseline(cases, model, "dpo-merged", baseline, "rule")

    assert comparison["accuracy"]["model_only"] == 8
    assert comparison["accuracy"]["baseline_only"] == 0
    assert comparison["accuracy"]["p_mcnemar"] < 0.01


def test_no_prediction_produces_no_comparison(evaluation):
    """A model that was not evaluated is not compared to anything."""
    assert evaluation._compare_to_baseline(_cases(IMMEDIATE), [], "sft", [IMMEDIATE], "rule") is None


# --- The failure that costs a patient ----------------------------------------


def test_undertriage_is_counted_on_urgent_cases_alone(evaluation):
    """Overtriaging a cold costs a place in the waiting room; the reverse costs a patient.

    Two of the four cases are urgent. The deferred ones cannot be undertriaged and must not
    dilute the count.
    """
    cases = _cases(IMMEDIATE, URGENT, DEFERRED, DEFERRED)
    model = [IMMEDIATE, DEFERRED, IMMEDIATE, DEFERRED]
    baseline = [DEFERRED, URGENT, DEFERRED, DEFERRED]

    counted = evaluation._compare_undertriage(cases, model, baseline)

    assert counted["urgent_cases"] == 2
    assert counted["model_undertriages"] == 1  # the urgent case sent to deferred
    assert counted["baseline_undertriages"] == 1  # the immediate case sent to deferred
    assert counted["model_only"] == 1  # the model saves the immediate case
    assert counted["baseline_only"] == 1  # the baseline saves the urgent one


def test_an_unparsable_answer_counts_as_an_undertriage(evaluation):
    """An answer the information system cannot read sends nobody anywhere.

    Counted as a non-failure, it would let a model that stops answering look safe.
    """
    cases = _cases(IMMEDIATE, IMMEDIATE)
    counted = evaluation._compare_undertriage(cases, [None, IMMEDIATE], [IMMEDIATE, IMMEDIATE])

    assert counted["model_undertriages"] == 1
    assert counted["baseline_undertriages"] == 0
    assert counted["baseline_only"] == 1


# --- Adapter against merged model --------------------------------------------


def test_the_two_deliveries_are_compared_on_what_separates_them(evaluation):
    """Same weights to the rounding: what differs is operational, and that is what is published."""
    results = {
        "clinical_set": {
            "models": {
                "dpo": {
                    "accuracy": 0.80,
                    "accuracy_ci95": [0.68, 0.88],
                    "latency": {"p50_ms": 910.0},
                },
                "dpo-merged": {
                    "accuracy": 0.80,
                    "accuracy_ci95": [0.68, 0.88],
                    "latency": {"p50_ms": 840.0},
                },
            }
        },
        "_predictions": {
            "dpo": [IMMEDIATE, URGENT, DEFERRED, IMMEDIATE],
            "dpo-merged": [IMMEDIATE, URGENT, DEFERRED, URGENT],
        },
    }

    comparison = evaluation._compare_deliveries(results)

    assert comparison["prediction_agreement"] == 0.75
    assert comparison["adapter"]["median_latency_ms"] == 910.0
    assert comparison["adapter"]["base_model_required"] == PATHS.sft_merged.name
    assert comparison["merged"]["base_model_required"] is None


def test_deliveries_are_not_compared_when_only_one_was_evaluated(evaluation):
    results = {"clinical_set": {"models": {"dpo-merged": {}}}, "_predictions": {}}
    assert evaluation._compare_deliveries(results) == {}


# --- Supervised against aligned, on the same preference pairs ----------------


def test_the_alignment_is_judged_on_the_pairs_where_the_two_models_differ(evaluation):
    results = {
        "external_preferences": {
            "sft": {"correctly_ordered": [True, False, False, True]},
            "dpo": {"correctly_ordered": [True, True, False, False]},
        }
    }

    comparison = evaluation._compare_preferences(results)

    assert comparison["n"] == 4
    assert comparison["aligned_only"] == 1
    assert comparison["supervised_only"] == 1
    assert comparison["agreements"] == 2


def test_two_preference_runs_of_different_lengths_are_not_paired(evaluation):
    """Pairing two lists that do not describe the same pairs would compare nothing."""
    results = {
        "external_preferences": {
            "sft": {"correctly_ordered": [True, False]},
            "dpo": {"correctly_ordered": [True, False, True]},
        }
    }
    assert evaluation._compare_preferences(results) is None


def test_a_run_without_the_external_preference_set_publishes_no_comparison(evaluation):
    assert evaluation._compare_preferences({}) is None
