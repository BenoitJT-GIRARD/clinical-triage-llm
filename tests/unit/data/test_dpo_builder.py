"""Tests of the preference pairs used for the alignment.

What is checked here is not cosmetic. The model will learn the difference it finds between the
preferred answer and the rejected one, not the one that was intended. If that difference is
systematically length, then length is what it learns — and it stops knowing how to stop.
"""

from __future__ import annotations

import random
from collections import Counter

from clinical_triage.config import TRIAGE
from clinical_triage.data.case_generator import generate_cases
from clinical_triage.data.dataset_io import dpo_record
from clinical_triage.data.dpo_builder import build_preference_pairs, length_balance
from clinical_triage.data.sft_builder import assemble
from clinical_triage.prompts import extract_level


def _built_pairs(count: int = 200):
    """Build a preference set from generated vignettes."""
    generator = random.Random(12)
    vignettes = []
    for level in TRIAGE.levels:
        vignettes += generate_cases(count, level, "fr", generator)
    examples = assemble(vignettes, [], set(), generator)
    return examples, build_preference_pairs(examples, count, generator)


def test_rejected_answers_are_never_an_escalation():
    """Escalating is the right reaction under doubt: it cannot be a counter-example."""
    _, pairs = _built_pairs()
    for pair in pairs:
        rejected_level = extract_level(pair.rejected)
        if rejected_level is None:
            continue
        assert TRIAGE.severity[rejected_level] <= TRIAGE.severity[pair.level]


def test_the_pairs_carry_no_length_bias():
    _, pairs = _built_pairs()
    balance = length_balance(pairs)
    assert 0.35 <= balance["chosen_longer_share"] <= 0.65
    ratio = balance["rejected_mean_chars"] / balance["chosen_mean_chars"]
    assert 0.85 <= ratio <= 1.15


def test_rejected_answers_keep_the_expected_structure():
    """A malformed rejected answer would teach the format instead of the substance."""
    _, pairs = _built_pairs()
    structured = [p for p in pairs if p.strategy != "answered_in_english"]
    for pair in structured:
        assert pair.rejected.startswith("Niveau de priorité :")
        assert "Justification :" in pair.rejected
        assert "Recommandation :" in pair.rejected


def test_the_pairs_cover_the_four_defects():
    _, pairs = _built_pairs()
    assert len({p.strategy for p in pairs}) == 4


def test_serious_cases_are_over_represented():
    _, pairs = _built_pairs()
    distribution = Counter(p.level for p in pairs)
    assert distribution["URGENCE_VITALE"] > distribution["CONSULTATION_DIFFEREE"]


def test_one_case_does_not_give_two_pairs():
    _, pairs = _built_pairs()
    turns = [p.user_turn for p in pairs]
    assert len(turns) == len(set(turns))


def test_a_preference_record_exposes_the_three_expected_columns():
    _, pairs = _built_pairs(30)
    record = dpo_record(pairs[0])
    assert set(record) >= {"prompt", "chosen", "rejected", "strategy"}
    assert record["prompt"].endswith("<|im_start|>assistant\n")
