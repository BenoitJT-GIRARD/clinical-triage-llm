"""Assembling the supervised training set.

Two contributions meet here under a single format, the triage example:

- the **generated vignettes** built from the presentation catalogue, which carry a ``high``
  confidence label and cover the three levels in both languages;
- the **cases extracted from the public corpora**, labelled by the rule with ``medium``
  confidence, which bring authentic clinical turns of phrase and vocabulary that templates do
  not produce.

Every example carries its metadata: symptoms, medical history, vital signs, source and
confidence level.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass

from clinical_triage.data.case_generator import ClinicalCase
from clinical_triage.data.corpus_cases import CorpusCase
from clinical_triage.prompts import build_target_response


def case_fingerprint(text: str) -> str:
    """Normalised form of a patient turn, to compare near-identical cases.

    A character-exact comparison lets near-duplicates through, and that is exactly what a public
    corpus produces: "A 6 year old child" and "A 6-year-old child" are the same case, extracted
    twice. Split either side of the train/test boundary, that duplicate would make the test set
    flattering with no check seeing it. So texts are compared lower-cased, without accents or
    punctuation.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", without_accents)).strip()


@dataclass(frozen=True)
class TriageExample:
    """One triage training example, metadata included."""

    user_turn: str
    assistant_turn: str
    level: str
    lang: str
    source: str
    confidence: str
    symptoms: tuple[str, ...]
    medical_history: tuple[str, ...]
    vitals: str
    presentation_id: str


def from_clinical_case(case: ClinicalCase) -> TriageExample:
    """Convert a generated vignette into a training example."""
    return TriageExample(
        user_turn=case.user_turn,
        assistant_turn=build_target_response(case.level, case.justification, case.recommendation),
        level=case.level,
        lang=case.lang,
        source=case.source,
        confidence=case.confidence,
        symptoms=case.symptoms,
        medical_history=case.medical_history,
        vitals=case.vitals.render(case.lang) if case.vitals else "",
        presentation_id=case.presentation_id,
    )


def from_corpus_case(case: CorpusCase) -> TriageExample:
    """Convert a case extracted from a public corpus into a training example."""
    return TriageExample(
        user_turn=case.user_turn,
        assistant_turn=build_target_response(case.level, case.justification, case.recommendation),
        level=case.level,
        lang=case.lang,
        source=case.source,
        confidence=case.confidence,
        symptoms=case.symptoms,
        medical_history=(),
        vitals="",
        presentation_id="",
    )


def assemble(
    generated: list[ClinicalCase],
    from_corpora: list[CorpusCase],
    excluded_user_turns: set[str],
    rng: random.Random,
) -> list[TriageExample]:
    """Join both contributions, drop duplicates and the turns reserved for evaluation.

    Two precautions, learnt on this corpus:

    - deduplication runs **after** every text transformation. Deduplicating earlier lets through
      the duplicates that anonymisation then creates, by making two hitherto distinct statements
      identical;
    - it works on the **normalised form** of the patient turn, not on the exact text. Public
      corpora deliver the same case under neighbouring spellings — "A 6 year old child" and
      "A 6-year-old child" — and a duplicate split either side of the boundary makes the test
      set flattering.
    """
    examples = [from_clinical_case(c) for c in generated]
    examples += [from_corpus_case(c) for c in from_corpora]

    unique: list[TriageExample] = []
    seen = {case_fingerprint(turn) for turn in excluded_user_turns}
    for example in examples:
        fingerprint = case_fingerprint(example.user_turn)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(example)

    rng.shuffle(unique)
    return unique
