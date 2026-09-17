"""Tests of the evaluation metrics and of the baselines.

The triage level names are French constants of the model contract: they are values, not prose.
"""

from __future__ import annotations

import pytest

from clinical_triage.evaluation.baselines import (
    always_critical,
    classical_classifier,
    explicit_rule,
    majority_class,
)
from clinical_triage.evaluation.metrics import (
    HORS_FORMAT,
    accuracy,
    clopper_pearson,
    confusion,
    difference_newcombe,
    format_compliance,
    interval_for_proportion,
    latency_summary,
    mcnemar_exact,
    overtriage_rate,
    per_class,
    subgroup_accuracy,
    summarize,
    undertriage_rate,
    wilson_interval,
)

CRITICAL = "URGENCE_VITALE"
URGENT = "URGENCE_MODEREE"
DEFERRED = "CONSULTATION_DIFFEREE"


def test_perfect_accuracy():
    expected = [CRITICAL, DEFERRED]
    assert accuracy(expected, expected) == 1.0


def test_an_off_format_answer_counts_as_wrong():
    assert accuracy([CRITICAL, URGENT], [CRITICAL, None]) == 0.5
    assert format_compliance([CRITICAL, None, URGENT]) == 2 / 3


def test_undertriage_counts_only_the_urgent_cases():
    rate, misses, urgent = undertriage_rate(
        [CRITICAL, CRITICAL, DEFERRED], [DEFERRED, CRITICAL, CRITICAL]
    )
    assert (rate, misses, urgent) == (0.5, 1, 2)


def test_an_off_format_answer_counts_as_an_undertriage():
    """An unreadable answer triggers no care at all."""
    rate, misses, _ = undertriage_rate([CRITICAL], [None])
    assert (rate, misses) == (1.0, 1)


def test_an_overtriage_is_not_an_undertriage():
    assert undertriage_rate([DEFERRED], [CRITICAL])[0] == 0.0
    assert overtriage_rate([DEFERRED], [CRITICAL]) == 1.0


def test_the_wilson_interval():
    low, high = wilson_interval(57, 60)
    assert low < 57 / 60 < high
    assert 0.85 < low < 0.95


def test_the_interval_narrows_as_the_effective_grows():
    small = wilson_interval(9, 10)
    large = wilson_interval(900, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_the_confusion_matrix_isolates_the_off_format_answers():
    """Filing an unreadable answer under a triage class would bend the reading."""
    matrix = confusion([CRITICAL, CRITICAL, DEFERRED], [CRITICAL, None, DEFERRED])
    assert HORS_FORMAT in matrix["columns"]
    assert matrix["matrix"][CRITICAL][HORS_FORMAT] == 1
    assert matrix["matrix"][CRITICAL][CRITICAL] == 1
    assert matrix["matrix"][DEFERRED][DEFERRED] == 1


def test_precision_and_recall_per_level():
    results = per_class([CRITICAL, CRITICAL, URGENT], [CRITICAL, URGENT, URGENT])
    assert results[CRITICAL]["recall"] == 0.5
    assert results[CRITICAL]["precision"] == 1.0
    assert results[URGENT]["n"] == 1


def test_the_full_summary():
    expected = [CRITICAL, URGENT, DEFERRED]
    summary = summarize(
        expected, expected, latencies_ms=[100.0, 120.0, 110.0], clean_stops=[True, True, False]
    )
    for key in (
        "accuracy",
        "accuracy_ci95",
        "undertriage",
        "format_compliance",
        "per_level",
        "confusion",
        "latency",
    ):
        assert key in summary
    assert summary["clean_stops"] == round(2 / 3, 4)
    assert summary["latency"]["p50_ms"] == 110.0


def test_the_accuracy_interval_goes_through_the_single_door():
    """Below six cases no interval is publishable, including in the overall summary.

    The function hard-wired Wilson. On sixty cases the two paths agree, but the same function is
    called on subgroups whose effectives fall to three, and a published interval that stops
    following the written rule is exactly the kind of drift nothing would report.
    """
    assert summarize([CRITICAL] * 3, [CRITICAL] * 3)["accuracy_ci95"] is None
    assert summarize([CRITICAL] * 8, [CRITICAL] * 8)["accuracy_ci95"] == clopper_pearson(8, 8)


def test_latency_on_an_empty_series():
    assert latency_summary([]) == {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}


def test_the_latency_percentiles_are_real_measurements():
    """The 95th percentile drives a go-live criterion: it is not invented.

    On sixty measurements from 1 to 60 ms, the nearest rank designates the 57th value — the
    smallest under which at least 95% of the observations fall.
    """
    summary = latency_summary([float(i) for i in range(1, 61)])
    assert summary["p95_ms"] == 57.0
    assert summary["p50_ms"] == 30.0


def test_the_median_does_not_average_the_central_pair():
    """On an even n, a latency that was observed is published, not their average."""
    assert latency_summary([10.0, 20.0, 30.0, 40.0])["p50_ms"] == 20.0


def test_the_subgroup_breakdown():
    expected = [CRITICAL, CRITICAL, DEFERRED, DEFERRED]
    predicted = [CRITICAL, DEFERRED, DEFERRED, DEFERRED]
    detail = subgroup_accuracy(expected, predicted, ["fr", "fr", "en", "en"])
    assert detail["fr"]["accuracy"] == 0.5
    assert detail["en"]["accuracy"] == 1.0
    assert detail["fr"]["undertriage_detail"]["misses"] == 1


def test_the_unnamed_group_is_given_a_name():
    detail = subgroup_accuracy([CRITICAL], [CRITICAL], [""])
    assert "direct_presentation" in detail


# --- Baselines ---


def test_the_majority_class_baseline():
    predictions = majority_class([DEFERRED, DEFERRED, CRITICAL], 3)
    assert predictions == [DEFERRED] * 3


def test_the_maximum_caution_baseline_never_undertriages():
    expected = [CRITICAL, URGENT, DEFERRED]
    predictions = always_critical(len(expected))
    assert undertriage_rate(expected, predictions)[0] == 0.0
    assert overtriage_rate(expected, predictions) > 0.5


def test_the_explicit_rule_baseline():
    descriptions = ["Douleur thoracique et sueurs.", "Rhume banal depuis deux jours."]
    assert explicit_rule(descriptions) == [CRITICAL, DEFERRED]


def _separable_set(count: int) -> tuple[list[str], list[str]]:
    """Two disjoint vocabularies: a bag of words must succeed there without effort."""
    texts = [f"douleur thoracique sueurs cas {i}" for i in range(count)]
    levels = [CRITICAL] * count
    texts += [f"rhume banal repos cas {i}" for i in range(count)]
    levels += [DEFERRED] * count
    return texts, levels


def test_the_classical_baseline_learns_and_predicts():
    pytest.importorskip("sklearn")
    texts, levels = _separable_set(20)
    predictions, trace = classical_classifier(
        texts, levels, texts, levels, ["douleur thoracique sueurs cas 99"]
    )
    assert predictions == [CRITICAL]
    # The baseline is not retrained on training + validation: it would see more examples than
    # the model it is compared with.
    assert trace["training_examples"] == len(texts)


def test_the_classical_baseline_chooses_its_setting_on_the_validation_split():
    """The evaluation set must never enter the choice of the baseline.

    A baseline tuned on the set that judges it is no longer one. The trace therefore publishes
    the setting kept and the score that kept it, both from the validation split alone.
    """
    pytest.importorskip("sklearn")
    texts, levels = _separable_set(20)
    _, trace = classical_classifier(texts, levels, texts, levels, texts[:1])
    assert trace["configuration"] in trace["candidates"]
    assert trace["candidates"][trace["configuration"]] == max(trace["candidates"].values())
    assert 0.0 <= trace["selection_accuracy"] <= 1.0


# --- Comparing two systems on the same cases ---


def test_without_disagreement_nothing_separates_them():
    """Two systems that never diverge cannot be compared."""
    assert mcnemar_exact(0, 0) == 1.0


@pytest.mark.parametrize(
    ("won", "lost", "expected"),
    [(2, 0, 0.5), (5, 0, 0.0625), (10, 0, 0.001953125), (3, 3, 1.0)],
)
def test_the_mcnemar_test_reads_the_binomial_law(won, lost, expected):
    """On a handful of disagreements, the chi-square approximation makes no sense.

    Each disagreement falls one way or the other on a coin toss under the null hypothesis: the
    binomial law is read directly.
    """
    assert mcnemar_exact(won, lost) == pytest.approx(expected)


def test_the_test_is_two_sided_and_symmetric():
    assert mcnemar_exact(9, 1) == mcnemar_exact(1, 9)


def test_nine_wins_against_one_are_significant_twelve_against_ten_are_not():
    """That is the whole difference between a gap and a demonstrated gap.

    Both pairs of numbers give a difference of proportions of the same order: without a paired
    test, one would conclude in both cases.
    """
    assert mcnemar_exact(9, 1) <= 0.05
    assert mcnemar_exact(12, 10) > 0.05


def test_a_mute_system_gets_the_best_overtriage_score():
    """The three columns of the results table only read together.

    A system that produces nothing usable enters neither failure rate on non-urgent cases: it
    therefore shows 0% overtriage — the best possible score in that column — while being
    unusable. Format compliance, published right next to it, is what keeps that from reading as
    a quality.
    """
    gold = [CRITICAL, CRITICAL, URGENT, URGENT, DEFERRED, DEFERRED]
    mute: list[str | None] = [None] * 6

    assert overtriage_rate(gold, mute) == 0.0
    assert undertriage_rate(gold, mute)[0] == 1.0
    summary = summarize(gold, mute)
    assert summary["accuracy"] == 0.0
    assert summary["format_compliance"] == 0.0


# --- Uncertainty estimators: the right tool for each quantity ---


def test_with_no_observation_the_interval_is_total_ignorance():
    """Zero measurements do not establish a null proportion, they establish nothing.

    The function used to return [0, 0]: a zero-width interval drawn from nothing, that is to
    say absolute certainty — exactly the mistake a confidence interval exists to prevent.
    """
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert clopper_pearson(0, 0) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("successes", "total", "expected"),
    [
        (3, 3, (0.2924, 1.0)),
        (4, 4, (0.3976, 1.0)),
        (2, 40, (0.0061, 0.1692)),
        (0, 40, (0.0, 0.0881)),
        (51, 60, (0.7343, 0.929)),
    ],
)
def test_the_exact_interval_matches_its_reference_values(successes, total, expected):
    assert clopper_pearson(successes, total) == expected


def test_the_exact_interval_agrees_with_the_beta_law():
    """An independent check: the bisection must find the beta quantile again.

    The implementation looks for its bounds on the cumulative binomial rather than adding a
    dependency for two calls. This test verifies that the choice costs no precision, by
    comparing it with the closed form.
    """
    stats = pytest.importorskip("scipy.stats")
    for successes, total in [(3, 3), (1, 7), (17, 40), (19, 20), (32, 32), (10, 10)]:
        low = 0.0 if successes == 0 else stats.beta.ppf(0.025, successes, total - successes + 1)
        high = 1.0 if successes == total else stats.beta.ppf(0.975, successes + 1, total - successes)
        obtained = clopper_pearson(successes, total)
        assert abs(obtained[0] - low) < 1e-4, (successes, total)
        assert abs(obtained[1] - high) < 1e-4, (successes, total)


def test_the_exact_method_is_more_cautious_than_wilson_on_small_effectives():
    """That is why Clopper-Pearson exists on the tiny subgroups."""
    exact_low = clopper_pearson(3, 3)[0]
    wilson_low = wilson_interval(3, 3)[0]
    assert exact_low < wilson_low
    assert round(wilson_low - exact_low, 2) >= 0.14


def test_no_rate_is_published_below_six_observations():
    """On three cases the exact interval covers 71% of the scale.

    A whisker like that reads as "measured with uncertainty" when the correct reading is "not
    measurable". The caller must write the raw fraction.
    """
    assert interval_for_proportion(3, 3) is None
    assert interval_for_proportion(4, 4) is None
    assert interval_for_proportion(5, 5) is None
    assert interval_for_proportion(6, 6) is not None


def test_the_policy_picks_the_exact_method_then_wilson():
    assert interval_for_proportion(19, 20) == clopper_pearson(19, 20)
    assert interval_for_proportion(51, 60) == wilson_interval(51, 60)


def test_a_gap_is_read_on_its_own_interval_and_not_on_the_overlap():
    """The overlap fallacy gives the opposite of the right conclusion here.

    On this project's numbers — 465/500 against 51/60 — the two Wilson intervals overlap widely,
    and yet the interval on the difference excludes zero: the eight-point gap is real.
    """
    internal = wilson_interval(465, 500)
    clinical = wilson_interval(51, 60)
    assert internal[0] <= clinical[1], "the marginal intervals do overlap"

    gap, low, high = difference_newcombe(465, 500, 51, 60)
    assert gap == 0.08
    assert (low, high) == (0.0063, 0.1927)
    assert low > 0, "the interval on the gap excludes zero"


def test_a_null_gap_has_an_interval_that_contains_zero():
    gap, low, high = difference_newcombe(30, 60, 30, 60)
    assert gap == 0.0
    assert low < 0 < high
