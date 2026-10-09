"""The task-format preference set: what it is built from, and that length cannot decide it."""

from __future__ import annotations

from collections import Counter

from clinical_triage.data.clinical_eval_set import eval_cases
from clinical_triage.evaluation.task_preferences import task_preference_pairs
from clinical_triage.prompts import extract_level


def test_every_urgent_case_gives_an_undertriage_pair_and_every_case_an_asserted_diagnosis():
    cases = eval_cases()
    pairs = task_preference_pairs()
    urgent = sum(case.level != "CONSULTATION_DIFFEREE" for case in cases)
    assert Counter(pair.label_type for pair in pairs) == {
        "undertriage": urgent,
        "asserted_diagnosis": len(cases),
    }


def test_the_rejected_undertriage_answer_sits_below_the_reference_level():
    order = ("CONSULTATION_DIFFEREE", "URGENCE_MODEREE", "URGENCE_VITALE")
    for pair in task_preference_pairs():
        if pair.label_type == "undertriage":
            assert order.index(extract_level(pair.rejected)) < order.index(
                extract_level(pair.chosen)
            )


def test_the_prompt_ends_on_the_opening_of_the_assistant_turn():
    """The answers are read straight after it: no separator may stand between them."""
    assert all(pair.prompt.endswith("<|im_start|>assistant\n") for pair in task_preference_pairs())


def test_the_set_is_the_same_at_every_build():
    assert task_preference_pairs() == task_preference_pairs()
