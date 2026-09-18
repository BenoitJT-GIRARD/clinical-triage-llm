"""Tests of the preference measurement on an external set.

No model is loaded: the likelihoods are supplied directly. What is checked is the reading made
of them — the order of the two answers, the margin, the detail by the difficulty the corpus
announces, and the share of answers the model window cut short.
"""

from __future__ import annotations

import pytest

from clinical_triage.evaluation.preference import PreferenceScore, summarize


def test_a_correctly_ordered_pair():
    score = PreferenceScore(chosen=-1.20, rejected=-1.80, label_type="hard")
    assert score.correctly_ordered
    assert score.margin == pytest.approx(0.6)


def test_a_wrongly_ordered_pair():
    score = PreferenceScore(chosen=-2.0, rejected=-1.5, label_type="easy")
    assert not score.correctly_ordered
    assert score.margin < 0


def test_the_summary_counts_the_correctly_ordered_pairs():
    scores = [
        PreferenceScore(-1.0, -1.5, "hard"),
        PreferenceScore(-1.0, -1.2, "hard"),
        PreferenceScore(-2.0, -1.0, "easy"),
        PreferenceScore(-1.0, -1.1, "easy"),
    ]
    summary = summarize(scores)
    assert summary["n"] == 4
    assert summary["correctly_ordered_share"] == 0.75


def test_the_summary_breaks_down_by_difficulty():
    scores = [
        PreferenceScore(-1.0, -1.5, "hard"),
        PreferenceScore(-2.0, -1.0, "hard"),
        PreferenceScore(-1.0, -1.5, "easy"),
        PreferenceScore(-1.0, -1.5, "easy"),
    ]
    detail = summarize(scores)["per_difficulty"]
    assert detail["hard"]["correctly_ordered_share"] == 0.5
    assert detail["easy"]["correctly_ordered_share"] == 1.0


def test_the_summary_publishes_the_share_of_truncated_answers():
    """The essays regularly exceed the window, and the comparison then covers their beginning.

    That share qualifies a result that sits below chance, so it is published next to it rather
    than left in a log line.
    """
    scores = [
        PreferenceScore(-1.0, -1.5, "hard", truncated_sides=2),
        PreferenceScore(-1.0, -1.2, "hard", truncated_sides=1),
        PreferenceScore(-1.0, -1.2, "easy"),
        PreferenceScore(-1.0, -1.2, "easy"),
    ]
    summary = summarize(scores)
    assert summary["truncated_answers"] == 3
    assert summary["truncated_share"] == pytest.approx(3 / 8)


def test_the_summary_on_an_empty_set():
    summary = summarize([])
    assert summary["n"] == 0
    assert summary["correctly_ordered_share"] == 0.0
