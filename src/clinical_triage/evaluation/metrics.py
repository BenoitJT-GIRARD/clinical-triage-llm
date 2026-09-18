"""Triage evaluation metrics.

Four families, in the order that matters to an emergency department:

1. **clinical safety** — the undertriage rate: the share of genuinely urgent cases the model
   files at a lower severity. That is the failure that kills, and it does not show up in
   overall accuracy;
2. **performance** — accuracy and per-level F1, with their confidence intervals. On sixty
   cases a three-point gap is not a gap; publishing accuracy without its interval lets a
   reader believe otherwise;
3. **format** — the share of answers the hospital information system can parse, and the share
   of generations that stop on their own;
4. **latency** — mean, median and 95th percentile.

The confusion matrix carries an extra ``HORS_FORMAT`` column. An unreadable answer is not a
prediction: filing it under a triage class would bend the reading one way or the other.

Level names are French because they are the model's own output contract, written into the
dataset, the prompt and every published artefact. They are values, not prose.
"""

from __future__ import annotations

from math import ceil, comb, sqrt

from clinical_triage.config import TRIAGE

HORS_FORMAT = "HORS_FORMAT"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion, Wilson's method.

    Wilson is preferred over the normal approximation — ±SEM — because it stays correct on
    small effectives and near 0 or 1, which is exactly the regime of a clinical evaluation set
    of a few dozen cases. The normal approximation produces bounds outside [0, 1] there: on 2
    undertriages out of 40 urgent cases it gives [-1.8%, 11.8%], a negative lower bound on a
    rate.

    With no observation the interval is **[0, 1]** and not [0, 0]: zero measurements do not
    establish a null proportion, they establish nothing. A zero-width interval would claim
    absolute certainty drawn from nothing — the very mistake this function exists to avoid.
    """
    if total == 0:
        return (0.0, 1.0)
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z * sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    return (round(max(0.0, centre - half_width), 4), round(min(1.0, centre + half_width), 4))


# The effective below which no rate is publishable.
#
# On three cases the exact interval of a perfect score is [0.29, 1.00]: it covers 71% of the
# scale. A bar topped by a whisker like that reads as "measured with uncertainty" when the
# correct reading is "not measurable". Below this threshold the caller publishes the raw
# fraction instead.
MINIMUM_FOR_A_RATE = 6


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), computed exactly.

    A direct sum of terms: ``n`` never exceeds a few hundred in this project, and an exact sum
    saves having to justify an approximation.
    """
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """**Exact** interval for a proportion, Clopper-Pearson.

    Wilson is the right default: shorter, and with correct coverage on average. Two situations
    ask for more, and that is what this function provides — coverage guaranteed at 95% or
    better for **every** value of p, at the price of a wider interval:

    - **the safety metric.** On undertriage, coverage dropping below 95% for some values of p
      is not acceptable: it is the one measure in this repository whose underestimation is
      paid in patients;
    - **very small effectives.** Wilson is markedly anticonservative at the boundary there: on
      3 successes out of 3 it announces a lower bound of 0.44 where the exact method gives
      0.29. Fifteen points of false precision on the most dangerous subgroup of the evaluation
      set.

    The bounds are found by bisection on the cumulative binomial, which avoids adding a
    dependency on a statistics library for two calls — and reads without knowing the
    incomplete beta function.
    """
    if total == 0:
        return (0.0, 1.0)

    def _bisect(target, increasing: bool) -> float:
        low, high = 0.0, 1.0
        for _ in range(60):
            middle = (low + high) / 2
            if (target(middle) > 0) == increasing:
                high = middle
            else:
                low = middle
        return (low + high) / 2

    # Lower bound: the p at which observing at least `successes` successes has only alpha/2
    # chance left. Upper bound: the p at which observing at most `successes` has only alpha/2.
    low = (
        0.0
        if successes == 0
        else _bisect(lambda p: (1 - _binomial_cdf(successes - 1, total, p)) - alpha / 2, True)
    )
    high = (
        1.0
        if successes == total
        else _bisect(lambda p: _binomial_cdf(successes, total, p) - alpha / 2, False)
    )
    return (round(low, 4), round(high, 4))


def difference_newcombe(
    successes_a: int, total_a: int, successes_b: int, total_b: int, z: float = 1.96
) -> tuple[float, float, float]:
    """Difference between two **independent** proportions, with its interval.

    Returns (difference, lower bound, upper bound). The difference is ``a - b``.

    This function exists because reading two marginal intervals and checking whether they
    overlap is a **fallacy**, and here it gives the opposite of the right answer.
    Non-overlap proves a difference; overlap proves nothing. On this project's numbers —
    465/500 against 51/60 — the two Wilson intervals overlap widely, and yet the interval on
    the difference excludes zero.

    When the difference is the question, the difference is what must be estimated. The method
    is Newcombe's, which composes the Wilson intervals of both proportions: it inherits their
    good behaviour near 0 and 1, where the normal approximation on the difference fails.

    The two samples must be **independent**. Two systems evaluated on the same cases are not:
    they need :func:`mcnemar_exact`.
    """
    proportion_a = successes_a / total_a if total_a else 0.0
    proportion_b = successes_b / total_b if total_b else 0.0
    low_a, high_a = wilson_interval(successes_a, total_a, z)
    low_b, high_b = wilson_interval(successes_b, total_b, z)

    difference = proportion_a - proportion_b
    downward = sqrt((proportion_a - low_a) ** 2 + (high_b - proportion_b) ** 2)
    upward = sqrt((high_a - proportion_a) ** 2 + (proportion_b - low_b) ** 2)
    return (round(difference, 4), round(difference - downward, 4), round(difference + upward, 4))


def interval_for_proportion(successes: int, total: int) -> tuple[float, float] | None:
    """The interval to publish for a proportion, or ``None`` if the effective forbids one.

    One door, so that the choice of estimator is made in the same place everywhere and can be
    re-read:

    - fewer than six observations: **nothing**. The exact interval covers two thirds of the
      scale there; publishing it would give a non-measurement the appearance of a measurement.
      The caller writes the raw fraction;
    - up to thirty: **Clopper-Pearson**, exact, because Wilson is still anticonservative near
      the bounds at those effectives;
    - beyond: **Wilson**, shorter and with correct coverage.
    """
    if total < MINIMUM_FOR_A_RATE:
        return None
    if total <= 30:
        return clopper_pearson(successes, total)
    return wilson_interval(successes, total)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar test on two **paired** measurements.

    Two systems evaluated on the same cases do not compare like two independent proportions:
    what carries the information is the cases where they disagree. ``only_a`` counts those the
    first gets right and the second misses, ``only_b`` the reverse; agreements do not enter the
    computation.

    Under the hypothesis that neither is better, each disagreement falls one way or the other
    on a coin toss. The binomial law is therefore read directly, without approximation: on a
    handful of disagreements, the continuity correction of the chi-square makes no sense.

    Returns 1.0 when the two systems never diverge: there is nothing to separate.
    """
    discordant = only_a + only_b
    if discordant == 0:
        return 1.0
    rarer = min(only_a, only_b)
    tail = sum(comb(discordant, k) for k in range(rarer + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def _as_label(prediction: str | None) -> str:
    """Reduce a prediction to a class, ``HORS_FORMAT`` included."""
    return prediction if prediction in TRIAGE.levels else HORS_FORMAT


def format_compliance(preds: list[str | None]) -> float:
    """Share of answers whose triage level can be extracted."""
    if not preds:
        return 0.0
    return sum(p in TRIAGE.levels for p in preds) / len(preds)


def accuracy(gold: list[str], preds: list[str | None]) -> float:
    """Triage-level accuracy. An off-format answer counts as wrong."""
    if not gold:
        return 0.0
    return sum(g == p for g, p in zip(gold, preds, strict=True)) / len(gold)


def undertriage_rate(gold: list[str], preds: list[str | None]) -> tuple[float, int, int]:
    """Undertriage rate, with its detail (rate, number of misses, cases concerned).

    Among genuinely urgent cases — immediate or urgent — the share the model files at a
    strictly lower severity. An off-format answer counts as an undertriage here: it triggers
    no care at all.

    On a **non-urgent** case, by contrast, an unreadable answer enters neither failure rate:
    undertriage does not look at those cases, and overtriage excludes them too. It then shows
    up only in format compliance and in accuracy — which is why the published table puts those
    columns side by side, an overtriage of 0% otherwise reading as a quality when it describes
    a mute system.
    """
    urgent = [(g, p) for g, p in zip(gold, preds, strict=True) if TRIAGE.severity[g] >= 1]
    if not urgent:
        return (0.0, 0, 0)
    misses = 0
    for expected, predicted in urgent:
        predicted_rank = TRIAGE.severity.get(predicted, -1) if predicted is not None else -1
        if predicted_rank < TRIAGE.severity[expected]:
            misses += 1
    return (misses / len(urgent), misses, len(urgent))


def overtriage_rate(gold: list[str], preds: list[str | None]) -> float:
    """Overtriage rate: cases filed as more severe than they are.

    Overtriage congests the department without endangering the patient: it is the cost
    of caution, and it is measured.

    The denominator is **every case**, not only those that can be overtriaged. That is the
    convention of triage scales, and it is also the only one a head of department can read
    directly: "out of a hundred patients presenting, this many are sent higher than needed".
    An off-format answer does not count here — it is already counted as an undertriage.
    """
    if not gold:
        return 0.0
    misses = 0
    for expected, predicted in zip(gold, preds, strict=True):
        if predicted in TRIAGE.levels and TRIAGE.severity[predicted] > TRIAGE.severity[expected]:
            misses += 1
    return misses / len(gold)


def per_class(gold: list[str], preds: list[str | None]) -> dict[str, dict[str, float]]:
    """Precision, recall, F1 and effective for each triage level.

    One caveat before quoting **precision**: it is a positive predictive value, and it depends
    entirely on the prevalence of the set it is measured on. The clinical evaluation set is
    balanced by design — twenty cases per level — whereas an emergency department sees a very
    different mix. The precision published here describes this evaluation plan, not what the
    triage nurse would observe: transposing it as is would be a reading error. **Recall** does
    not depend on prevalence and does transpose, which is why the published documents show the
    confusion matrix and the undertriage rate; these three numbers stay in
    the results file for analysis.
    """
    labels = [_as_label(p) for p in preds]
    result: dict[str, dict[str, float]] = {}
    for level in TRIAGE.levels:
        true_positives = sum(g == level and p == level for g, p in zip(gold, labels, strict=True))
        predicted = sum(p == level for p in labels)
        actual = sum(g == level for g in gold)
        precision = true_positives / predicted if predicted else 0.0
        recall = true_positives / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[level] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n": actual,
        }
    return result


def confusion(gold: list[str], preds: list[str | None]) -> dict:
    """Confusion matrix: rows = truth, columns = prediction plus off-format.

    Returns three keys: ``rows`` and ``columns`` give the display order, and
    ``matrix[true][predicted]`` gives the count. The headers travel with the numbers because
    the result ends up in JSON and then in a figure: re-reading them from the configuration
    would make an archived artefact depend on a level order that may change.
    """
    columns = [*TRIAGE.levels, HORS_FORMAT]
    matrix = {level: dict.fromkeys(columns, 0) for level in TRIAGE.levels}
    for expected, predicted in zip(gold, preds, strict=True):
        matrix[expected][_as_label(predicted)] += 1
    return {"rows": list(TRIAGE.levels), "columns": columns, "matrix": matrix}


def _percentile(sorted_values: list[float], share: float) -> float:
    """Nearest-rank percentile: the smallest measurement below which at least ``share`` of the
    observations fall.

    This convention never interpolates between two measurements: the published number is
    always a latency that was actually observed. That is what a performance report wants,
    where a value invented between two measured points would be uncheckable.
    """
    rank = ceil(share * len(sorted_values))
    return sorted_values[min(len(sorted_values) - 1, max(0, rank - 1))]


def latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    """Mean, median and 95th percentile of the observed latencies.

    The median is the 50th percentile in the sense above: on an even effective it takes the
    lower of the two central measurements, and not their average, for the same reason — only
    measured values are published.
    """
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(latencies_ms)
    return {
        "mean_ms": round(sum(ordered) / len(ordered), 1),
        "p50_ms": round(_percentile(ordered, 0.50), 1),
        "p95_ms": round(_percentile(ordered, 0.95), 1),
    }


def summarize(
    gold: list[str],
    preds: list[str | None],
    latencies_ms: list[float] | None = None,
    clean_stops: list[bool] | None = None,
) -> dict:
    """Aggregate every metric into one dictionary, intervals included."""
    n = len(gold)
    correct = sum(g == p for g, p in zip(gold, preds, strict=True))
    undertriage, misses, urgent = undertriage_rate(gold, preds)
    summary = {
        "n": n,
        "accuracy": round(accuracy(gold, preds), 4),
        # Through the single door rather than a hard-wired Wilson. On sixty cases the two
        # agree, but this same function is called on subgroups whose effectives fall to three,
        # and a published interval that stops following the written rule is exactly the kind
        # of drift nothing would report.
        "accuracy_ci95": interval_for_proportion(correct, n),
        "undertriage": round(undertriage, 4),
        # Clopper-Pearson and not Wilson: this is the one measure whose underestimation is
        # paid in patients, and the exact method guarantees its coverage for every value of
        # the rate where Wilson holds it only on average. The interval is a little wider;
        # that is the right price.
        "undertriage_ci95": clopper_pearson(misses, urgent),
        "undertriage_detail": {"misses": misses, "urgent_cases": urgent},
        "overtriage": round(overtriage_rate(gold, preds), 4),
        "format_compliance": round(format_compliance(preds), 4),
        "per_level": per_class(gold, preds),
        "confusion": confusion(gold, preds),
    }
    if clean_stops is not None:
        summary["clean_stops"] = round(sum(clean_stops) / max(1, len(clean_stops)), 4)
    if latencies_ms is not None:
        summary["latency"] = latency_summary(latencies_ms)
    return summary


def subgroup_accuracy(
    gold: list[str], preds: list[str | None], groups: list[str]
) -> dict[str, dict]:
    """Accuracy and undertriage broken down by subgroup (language, kind of difficulty...)."""
    result: dict[str, dict] = {}
    for name in sorted(set(groups)):
        indices = [i for i, g in enumerate(groups) if g == name]
        subgroup_gold = [gold[i] for i in indices]
        subgroup_preds = [preds[i] for i in indices]
        correct = sum(g == p for g, p in zip(subgroup_gold, subgroup_preds, strict=True))
        rate, misses, urgent = undertriage_rate(subgroup_gold, subgroup_preds)
        result[name or "direct_presentation"] = {
            "n": len(indices),
            "accuracy": round(accuracy(subgroup_gold, subgroup_preds), 4),
            "accuracy_ci95": interval_for_proportion(correct, len(indices)),
            "undertriage": round(rate, 4),
            # Clopper-Pearson directly, without going through interval_for_proportion:
            # undertriage is the safety metric, and it is by subgroup that its effectives are
            # smallest. The exact interval is published here even below the minimum.
            "undertriage_ci95": clopper_pearson(misses, urgent),
            "undertriage_detail": {"misses": misses, "urgent_cases": urgent},
        }
    return result
