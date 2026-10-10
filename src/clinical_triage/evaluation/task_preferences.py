"""A preference set in the exact format of the answer, on narratives the model never trained on.

The alignment pairs are built on the training narratives, which the generator wrote from
templates. Whether the preference they teach holds elsewhere is measured here, on the sixty
hand-written cases of the clinical set, which nothing in training resembles.

Each case gives one preferred answer — its reference level, the clinical note written beside
it, the course of action standard for that level — and up to two rejected answers, built by the
same strategies as the training pairs:

- **undertriage**, for the forty urgent cases: one level too low, with a minimising
  justification;
- **asserted diagnosis**, for all sixty: the right level, a diagnosis stated as certain.

So the pairs share the training pairs' construction and only their narratives differ: what is
measured is the transfer from generated narratives to written ones, nothing more. The cases and
their notes were written by the author of the rest of the project; the result is read with that
in mind.

Its prompt stops where the assistant's turn opens, so each answer is read right after it, with no
separator.
"""

from __future__ import annotations

import random

from clinical_triage.config import SEED
from clinical_triage.data.clinical_eval_set import EvalCase, eval_cases
from clinical_triage.data.corpus_sources import PreferencePair
from clinical_triage.data.dpo_builder import BUILDERS
from clinical_triage.data.sft_builder import TriageExample
from clinical_triage.data.triage_rules import recommendation_for
from clinical_triage.prompts import build_messages, build_target_response, format_chatml

#: The strategies measured, and the levels each applies to.
MEASURED_STRATEGIES = {
    "undertriage": ("URGENCE_VITALE", "URGENCE_MODEREE"),
    "asserted_diagnosis": ("URGENCE_VITALE", "URGENCE_MODEREE", "CONSULTATION_DIFFEREE"),
}


def _example(case: EvalCase) -> TriageExample:
    return TriageExample(
        user_turn=case.user_turn,
        assistant_turn=build_target_response(case.level, case.note, recommendation_for(case.level)),
        level=case.level,
        lang=case.lang,
        source="clinical_eval",
        confidence="high",
        symptoms=(),
        medical_history=(),
        vitals="",
        presentation_id=case.id,
    )


def task_preference_pairs(cases: list[EvalCase] | None = None) -> list[PreferencePair]:
    """The pairs of every case, ``label_type`` naming the strategy of the rejected answer.

    Among the candidates a strategy offers, the rejected answer is the one closest in length to
    the preferred answer: the comparison must not turn on length.
    """
    rng = random.Random(SEED)
    pairs = []
    for case in cases if cases is not None else eval_cases():
        example = _example(case)
        prompt = format_chatml(build_messages(case.user_turn), add_generation_prompt=True)
        for strategy, levels in MEASURED_STRATEGIES.items():
            if case.level not in levels:
                continue
            candidates = [
                c for c in BUILDERS[strategy](example, rng) if c != example.assistant_turn
            ]
            rejected = min(candidates, key=lambda c: abs(len(c) - len(example.assistant_turn)))
            pairs.append(PreferencePair(prompt, example.assistant_turn, rejected, strategy))
    return pairs
