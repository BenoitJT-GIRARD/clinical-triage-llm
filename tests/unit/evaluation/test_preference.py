"""Tests of the alignment measurement by DPO's implicit reward.

No model is loaded: what the alignment changed to each answer's likelihood is supplied
directly. What is checked is the reading made of it — which answer the alignment favoured, the
detail per group, and the two shares that travel with the result.
"""

from __future__ import annotations

from clinical_triage.evaluation.preference import AlignmentEffect, summarize


def _effect(preferred: float, rejected: float, group: str = "hard", longer: bool = True):
    return AlignmentEffect(preferred, rejected, group, longer)


def test_the_alignment_favoured_the_preferred_answer():
    """Both answers lost likelihood, the rejected one more: the pair counts."""
    assert _effect(preferred=-1.0, rejected=-3.0).correctly_ordered


def test_the_alignment_favoured_the_rejected_answer():
    assert not _effect(preferred=2.0, rejected=2.5).correctly_ordered


def test_the_summary_counts_the_correctly_ordered_pairs_per_group():
    summary = summarize(
        [
            _effect(1.0, 0.0, "length"),
            _effect(0.0, 1.0, "length"),
            _effect(1.0, 0.0, "human"),
            _effect(1.0, 0.0, "human"),
        ]
    )
    assert summary["n"] == 4
    assert summary["correctly_ordered_share"] == 0.75
    assert summary["per_group"]["length"] == {"n": 2, "correctly_ordered_share": 0.5}
    assert summary["per_group"]["human"] == {"n": 2, "correctly_ordered_share": 1.0}


def test_the_summary_counts_the_answers_the_alignment_lowered():
    """Three of the four answers lost likelihood with the alignment."""
    assert summarize([_effect(-1.0, -2.0), _effect(0.5, -0.5)])["lowered_answers_share"] == 0.75


def test_the_summary_says_when_the_longer_answer_wins():
    """The longer answer wins when the alignment is right and it is the preferred one, or when the
    alignment is wrong and it is the rejected one."""
    effects = [
        _effect(1.0, 0.0, longer=True),  # right, preferred answer longer
        _effect(0.0, 1.0, longer=False),  # wrong, rejected answer longer
        _effect(0.0, 1.0, longer=True),  # wrong, preferred answer longer
        _effect(1.0, 0.0, longer=False),  # right, preferred answer shorter
    ]
    assert summarize(effects)["longer_favoured_share"] == 0.5


def test_the_summary_of_an_empty_set():
    summary = summarize([])
    assert summary["n"] == 0
    assert summary["correctly_ordered_share"] == 0.0
    assert summary["per_group"] == {}


def test_the_length_bias_is_counted_outside_the_length_matched_group():
    from clinical_triage.data.corpus_sources import PreferencePair
    from clinical_triage.evaluation.preference import preferred_longer_share

    pairs = [
        PreferencePair("q", "a long preferred answer", "short", "easy"),
        PreferencePair("q", "short", "a long rejected answer", "hard"),
        PreferencePair("q", "a long preferred answer", "short", "human"),
        PreferencePair("q", "a long preferred answer", "short", "length"),
    ]
    assert preferred_longer_share(pairs) == {"n": 3, "preferred_is_longer": 2, "share": 0.6667}
