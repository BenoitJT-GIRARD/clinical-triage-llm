"""Extracting the triage cases that the public corpora can actually supply.

Public medical corpora are not triage sets. Taking them as they are — wrapping "Levamisole is
used as all except -" inside "A patient presents with:" then labelling it because the word
"infection" appears — produces absurd training targets. This module does the sorting.

Two transformations only, both verifiable by reading:

1. **exam clinical vignettes** (MedMCQA, FrenchMedMCQA). Some questions open on a genuine
   patient presentation: "A 60-year-old chronic smoker presents with painless gross hematuria of
   1 day duration." That presentation is kept and the exam question that follows it is removed.
2. **symptom descriptions** (MedQuAD). Entries of type ``symptoms`` describe the signs of a
   condition; they become a patient's complaint.

The label then comes from the triage rule, with a **``medium`` confidence level** that
distinguishes it from the catalogue vignettes. One safety rule frames that label: **a case is
kept only when the rule explicitly identifies a sign**. The absence of a detected sign does not
prove the absence of severity — "suspected pneumoperitoneum" contains no alert keyword and
remains a surgical emergency. Manufacturing a "deferred consultation" label out of the rule's
silence would be dangerous; the non-urgent cases therefore come from the catalogue, where they
are described and validated one by one.

The patterns are French and English clinical vocabulary: they are the data being matched.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from clinical_triage.data.case_generator import USER_TEMPLATES
from clinical_triage.data.corpus_sources import CorpusEntry
from clinical_triage.data.triage_rules import (
    DEFERRED,
    MODERATE,
    VITAL,
    classify,
    matched_flags,
    recommendation_for,
)

# A patient presentation is recognised by an age, or by a presenting verb.
#
# The French pattern spots a person followed by their age within forty characters, whatever the
# punctuation between the two. Requiring the form "homme **de** 58 ans" would discard 342
# authentic MediQAl vignettes — the only source corpus describing patients in French, and one
# that writes "Homme, 58 ans, …" or "M. Dupont, 30 ans.". Checked on all three corpora: this
# latitude gains no false positive on MedQuAD or MedMCQA.
#
# The abbreviation "M." carries its own full stop: it cannot be followed by a word boundary,
# hence its separate alternative.
PERSON = (
    r"(?:\b(?:monsieur|madame|mademoiselle|mme|mlle|melle|mr"
    r"|homme|femme|patiente?|enfant|nourrisson|gar[cç]on|fille|b[eé]b[eé]"
    r"|adolescente?|jeune|nouveau-n[eé])\b|\bm\.)"
)
VIGNETTE_MARKERS = re.compile(
    r"\b\d{1,2}[- ]?(?:year|yr)[- ]?old\b"
    r"|\bpresents? with\b"
    r"|\bbrought to the emergency\b"
    r"|" + PERSON + r"[^.]{0,40}?\b\d{1,2}\s+ans?\b",
    re.IGNORECASE,
)

# --- The exam question that follows the vignette, and must be removed ---
#
# One pattern is not enough, because the same word plays two roles. "Which" opens a question put
# to the student, but also serves as a relative pronoun mid-narrative: "She had flu like symptoms
# 20 days ago **which** resolved spontaneously." Relying on interrogative words alone would
# delete that whole sentence — clinical content — and would let through the most common families
# of MedMCQA statements, which use no interrogative word at all: "All of the following ...
# except", "Most appropriate management is".
#
# Hence two lists and a rule of position.

# Turns of phrase that make a sentence an exam statement wherever they appear: none has a
# narrative use.
EXAM_STEM = re.compile(
    r"\ball of the following\b"
    r"|\bwhich of the following\b"
    r"|\bnot true\b|\bis not seen\b|\bare seen in\b|\btrue about\b"
    r"|\b(?:drug|treatment|investigation|management|test|view|method|procedure)"
    r"\s+of\s+choice\b"
    r"|\bmost (?:likely|probable|common|appropriate|useful)\b"
    r"|\bbest (?:initial|next|prognostic|indicator|view|test|investigation)\b"
    r"|\bnext step\b|\bdiagnosis is\b|\bthe toxin is\b"
    r"|\b(?:aiims|jipmer|neet|pgi|comed[ck])\b"
    r"|\bexcept\b\s*[-:.]?\s*$"
    r"|\bparmi les (?:propositions|affirmations|items|réponses)\b"
    r"|\b(?:indiquer|cocher) (?:laquelle|lequel|la|le)\b",
    re.IGNORECASE,
)

# Interrogative words. They count only by their position: at the head of a sentence, or in a
# sentence whose punctuation marks it as a question. Elsewhere they are relatives — "la douleur
# **qui** irradie", "**quel** traitement il prend" — and deleting their sentence throws away
# clinical content.
INTERROGATIVE = re.compile(
    r"\b(?:which|what|how|why|when|whom|whose"
    r"|quel|quelle|quels|quelles|laquelle|lequel|lesquels|lesquelles|combien)\b",
    re.IGNORECASE,
)


# A stem the statement left open, waiting for the option the student picks: it ends on a copula,
# a preposition or a bare noun of the answer sheet — "Best treatment is.", "Poisoning is due
# to.", "... appropriate treatment." The full stop that `_vignette_narrative` may have appended
# is ignored. A narrative sentence ends on what was observed: "... associated with nausea."
OPEN_STEM = re.compile(
    r"(?:\b(?:is|are|was|were|be|includes?|shows?|reveals?|suggests?|due to|caused by"
    r"|associated with|the following|with|by|to|of)"
    r"|\b(?:appropriate|best|preferred|initial|definitive|next) (?:treatment|management|drug"
    r"|diagnosis|investigation|step)"
    # A sentence that ends on "the following deformity", "the above ECG": it points at an
    # image the statement showed, and the reader of a description will never see it.
    r"|\bthe (?:following|above|below)(?:\s+\w+){1,2})\s*[-–:]?\s*[.!]?\s*$",
    re.IGNORECASE,
)

#: A blank to fill in belongs to an answer sheet, never to a narrative.
BLANK = re.compile(r"_{2,}")


def is_an_exam_question(sentence: str) -> bool:
    """Say whether a sentence is the question put to the student rather than narrative."""
    text = sentence.strip()
    if EXAM_STEM.search(text) or OPEN_STEM.search(text) or BLANK.search(text):
        return True
    # A patient narrative states; it does not ask. "Diagnosis?", "ECG shows?" carry no
    # interrogative word, and the question mark alone says what they are.
    if text.endswith("?"):
        return True
    match = INTERROGATIVE.search(text)
    if match is None:
        return False
    # A narrative sentence ends with a full stop. An exam stem does not: "Most appropriate
    # management is", "... which measure is largest".
    if not text.endswith((".", "!")):
        return True
    # Punctuated as a sentence, only one opening on the interrogative word is a question.
    return match.start() == 0


# MedQuAD documentation boilerplate, with no clinical content.
MEDQUAD_BOILERPLATE = (
    "human phenotype ontology",
    "medlineplus medical dictionary",
    "the following list of signs and symptoms",
    "you can use the",
    "visiting the following link",
)

# Acceptable lengths for a patient description.
#
# The upper bound is not a round number. It is the largest value for which **no** kept case
# exceeds the service's description budget: 324 tokens, that is the 768 window minus the system
# prompt and the room reserved for the answer. At the density measured on the corpora, 800
# characters fit in 100% of cases, 900 in 99.3%, 1,200 in 92.6%. Training beyond that would
# teach the model on narratives the service would truncate. Lowering the bound to 600 would
# discard 635 authentic MediQAl vignettes with nothing requiring it.
# ``tests/unit/data/test_corpus_cases.py`` redoes this computation and fails if the model window
# or the generation budget changes.
MIN_LENGTH = 60
MAX_LENGTH = 800


@dataclass(frozen=True)
class CorpusCase:
    """A triage case derived from a public corpus, with its weak label."""

    description: str
    user_turn: str
    level: str
    lang: str
    source: str
    topic: str
    symptoms: tuple[str, ...]
    justification: str
    recommendation: str
    confidence: str


def _sentences(text: str) -> list[str]:
    """Split a text into sentences, on strong punctuation."""
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _vignette_narrative(question: str) -> str | None:
    """The patient narrative, exam question removed, with no length check.

    Separated from :func:`extract_vignette` so that :func:`funnel` can tell what is not a patient
    presentation from what is one but falls outside the length bounds. The two losses do not mean
    the same thing.
    """
    if not VIGNETTE_MARKERS.search(question):
        return None
    narrative = [s for s in _sentences(question) if not is_an_exam_question(s)]
    text = " ".join(narrative).strip(" -–:;,")
    if not text or not VIGNETTE_MARKERS.search(text):
        return None
    # The existing strong punctuation is enough: adding a full stop behind a question mark would
    # produce "?." at the end of the description.
    return text if text.endswith((".", "!", "?")) else text + "."


def extract_vignette(question: str) -> str | None:
    """Isolate the patient presentation inside a clinical exam question.

    The sentences that are the question put to the student are removed and only the narrative is
    kept. If what is left no longer looks like a patient presentation, or falls outside the
    length bounds, the entry is dropped.
    """
    text = _vignette_narrative(question)
    if text is None or not MIN_LENGTH <= len(text) <= MAX_LENGTH:
        return None
    return text


def extract_symptom_description(entry: CorpusEntry) -> str | None:
    """Turn a MedQuAD symptom sheet into a patient's complaint."""
    if entry.topic != "symptoms":
        return None
    answer = entry.answer
    lowered = answer.lower()
    if any(boilerplate in lowered for boilerplate in MEDQUAD_BOILERPLATE):
        return None
    # The answer often repeats the question before answering: that is removed.
    sentences = [s for s in _sentences(answer) if not s.lower().startswith("what are")]
    # Whole sentences are stacked as long as they fit inside the bound: a training target must be
    # a complete sentence, whereas cutting the string at `[:MAX_LENGTH]` would slice mid-word, on
    # "Symptoms in Toddle" or "a transient is".
    #
    # The preamble counts towards the bound: it is what is delivered to the model, and leaving it
    # out of the computation would deliver descriptions longer than MAX_LENGTH, so that the
    # published bound would not describe the set.
    condition = entry.text.replace("What are the symptoms of", "").strip(" ?")
    preamble = f"Patient reporting the following complaints, possibly related to {condition}: "
    budget = MAX_LENGTH - len(preamble)
    if budget < MIN_LENGTH:
        return None

    text = ""
    for sentence in sentences:
        candidate = f"{text} {sentence}" if text else sentence
        if len(candidate) > budget:
            break
        text = candidate
    text = text.strip()
    if len(text) < MIN_LENGTH:
        return None
    return preamble + text


def _justification(level: str, signs: tuple[str, ...]) -> str:
    """Write a justification anchored in the signs actually spotted."""
    listed = ", ".join(signs)
    if level == VITAL:
        return (
            f"Signe(s) de détresse vitale identifié(s) dans la description : {listed}. "
            "Toute suspicion de détresse vitale impose une prise en charge immédiate."
        )
    return (
        f"Signe(s) d'alerte identifié(s) dans la description : {listed}. "
        "Une évaluation médicale rapprochée est justifiée, sans critère de détresse vitale immédiat."
    )


def build_corpus_case(
    description: str, entry: CorpusEntry, rng: random.Random
) -> CorpusCase | None:
    """Label a description with the rule, and keep it only when a sign is found."""

    level = classify(description)
    if level == DEFERRED:
        # No sign spotted: a "non-urgent" label is not manufactured out of the rule's silence.
        # The case is dropped.
        return None
    signs = tuple(matched_flags(description, VITAL if level == VITAL else MODERATE))
    if not signs:
        return None
    return CorpusCase(
        description=description,
        user_turn=rng.choice(USER_TEMPLATES[entry.lang]).format(description=description),
        level=level,
        lang=entry.lang,
        source=entry.source,
        topic=entry.topic,
        symptoms=signs,
        justification=_justification(level, signs),
        recommendation=recommendation_for(level),
        confidence="medium",
    )


def extract_cases(entries: list[CorpusEntry], rng: random.Random) -> list[CorpusCase]:
    """Walk the entries of a corpus and return the usable triage cases."""
    cases: list[CorpusCase] = []
    seen_descriptions: set[str] = set()
    for entry in entries:
        description = extract_vignette(entry.text) or extract_symptom_description(entry)
        if description is None or description in seen_descriptions:
            continue
        case = build_corpus_case(description, entry, rng)
        if case is None:
            continue
        seen_descriptions.add(description)
        cases.append(case)
    return cases


def funnel(entries: list[CorpusEntry]) -> dict[str, int]:
    """Count the entries lost at each stage of the extraction.

    A corpus's overall yield — so many entries read, so many cases kept — does not say *where*
    the entries are lost, and therefore does not say whether what discards them belongs to the
    corpus or to the filter. This breakdown does, and it is what the dataset card publishes.

    Two stages do not have the same status. "Not a patient presentation" and "too long or too
    short" are decisions of **form**, revisable. "The rule identifies no sign" is a decision of
    **safety**: labelling "non-urgent" from the silence of a keyword rule is refused, and that
    share is not recovered by widening a pattern.
    """
    count = {
        "entries": len(entries),
        "no_patient_presentation": 0,
        "outside_length_bounds": 0,
        "no_identified_sign": 0,
        "duplicates": 0,
        "kept": 0,
    }
    seen: set[str] = set()
    rng = random.Random(0)
    for entry in entries:
        # `extract_cases` writes `extract_vignette(...) or extract_symptom_description(...)`: an
        # out-of-bounds vignette does not condemn the entry, the symptom sheet can still rescue
        # it. The funnel must follow the same path, failing which it would charge to "out of
        # bounds" entries that the extraction keeps.
        narrative = _vignette_narrative(entry.text)
        out_of_bounds = narrative is not None and not MIN_LENGTH <= len(narrative) <= MAX_LENGTH
        description = None if out_of_bounds else narrative
        if description is None:
            description = extract_symptom_description(entry)
        if description is None:
            count["outside_length_bounds" if out_of_bounds else "no_patient_presentation"] += 1
            continue
        if description in seen:
            count["duplicates"] += 1
            continue
        # `extract_cases` records a description only once the case is kept: a description
        # rejected by the rule and met twice counts twice as "no sign", never as a duplicate.
        if build_corpus_case(description, entry, rng) is None:
            count["no_identified_sign"] += 1
            continue
        seen.add(description)
        count["kept"] += 1
    return count
