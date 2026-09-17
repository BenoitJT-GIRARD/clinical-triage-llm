"""Tests of the extraction filters applied to the public corpora.

They lock down defects that really did truncate the delivered dataset: a marker blind to
MediQAl's syntax, a truncation mid-word, and a length bound set by eye rather than on the
service's budget. The last ones verify that the funnel published in the dataset card really
describes what the extraction does, and not what one believes it does.
"""

from __future__ import annotations

import random

import pytest

from clinical_triage.config import MODEL, SERVING
from clinical_triage.data.corpus_cases import (
    MAX_LENGTH,
    MIN_LENGTH,
    extract_cases,
    extract_symptom_description,
    extract_vignette,
    funnel,
    is_an_exam_question,
)
from clinical_triage.data.corpus_sources import CorpusEntry

# --- The marker must recognise the syntax MediQAl actually uses ---


@pytest.mark.parametrize(
    "opening",
    [
        "Homme, 58 ans, majoration de dyspnée chez un BPCO connu depuis vingt ans.",
        "M. Dupont, 30 ans. Agitation à la sortie d'une boîte de nuit, amené par les pompiers.",
        "Mme Barbie, 72 ans, percutée par un bus alors qu'elle circulait à vélo sans casque.",
        "Mme N, âgée de 35 ans, présente en quelques jours une chute de la paupière gauche.",
        "Un enfant de 2 ans est adressé pour des otites moyennes aiguës récidivantes.",
        "Une patiente de 25 ans consulte pour une pâleur et une asthénie d'installation récente.",
    ],
)
def test_the_french_presentation_forms_are_recognised(opening: str):
    """MediQAl writes "Homme, 58 ans", not "homme de 58 ans".

    The pattern required the preposition. 342 authentic French vignettes were therefore not even
    examined, in the only source corpus that describes patients.
    """
    assert extract_vignette(opening) is not None


def test_a_statement_with_no_patient_is_still_discarded():
    """Widening the marker must not open the door to exam questions."""
    assert extract_vignette("Parmi les propositions suivantes, laquelle est exacte ?") is None
    assert extract_vignette("Levamisole is used as all except -") is None


# --- A training target ends on a whole sentence ---


def test_a_symptom_sheet_that_is_too_long_is_cut_at_a_sentence():
    """The truncation used to happen at ``[:MAX_LENGTH]``, hence mid-word.

    120 of the 172 MedQuAD cases delivered ended on "Symptoms in Toddle" or "a transient is":
    text cut in the middle of a word, learnt as a target.
    """
    sentence = "The patient reports abdominal cramping and persistent nausea after meals. "
    entry = CorpusEntry(
        text="What are the symptoms of Gastroparesis ?",
        answer=sentence * 30,
        lang="en",
        source="medquad",
        topic="symptoms",
    )
    description = extract_symptom_description(entry)
    assert description is not None
    assert len(description) <= MAX_LENGTH + len("Patient reporting the following complaints, ")
    # The text kept ends on strong punctuation, never on a cut word.
    assert description.rstrip().endswith(".")


def test_a_symptom_sheet_that_is_too_short_is_discarded():
    entry = CorpusEntry(
        text="What are the symptoms of X ?",
        answer="Rare.",
        lang="en",
        source="medquad",
        topic="symptoms",
    )
    assert extract_symptom_description(entry) is None


# --- The length bound is the service's, not a round number ---


def test_the_length_bound_fits_inside_the_service_budget():
    """No case kept may exceed what the service accepts.

    The service leaves the description what the model window leaves it once the system prompt
    and the room reserved for the answer are removed. Training beyond that would teach the model
    on narratives it will never see whole in production.

    This test fails if the model window, the generation budget or the bound change without the
    computation being redone.
    """
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL.base_model)
    except Exception as exc:  # noqa: BLE001 - offline, the test has nothing to say
        pytest.skip(f"tokenizer unavailable: {exc}")

    from clinical_triage.prompts import description_budget

    budget = description_budget(tokenizer, MODEL.max_seq_length, SERVING.max_new_tokens)
    # Density measured on the 546 descriptions kept from MediQAl and MedQuAD: median 0.287
    # tokens per character, 95th percentile 0.375. The longest description of the delivered set
    # occupies 312 tokens. The build-time check (``scripts/build_dataset.py``) verifies the real
    # case; this one keeps the setting coherent offline.
    tokens_at_p95 = MAX_LENGTH * 0.375
    assert tokens_at_p95 <= budget, (
        f"MAX_LENGTH={MAX_LENGTH} produces about {tokens_at_p95:.0f} tokens at the 95th "
        f"percentile, for a service budget of {budget}."
    )
    assert MIN_LENGTH < MAX_LENGTH


# --- The funnel must describe exactly what the extraction does ---


def _entry(text: str) -> CorpusEntry:
    return CorpusEntry(text=text, answer="", lang="en", source="medmcqa", topic="")


def _control_corpus() -> list[CorpusEntry]:
    """A miniature corpus exercising each of the four losses, plus one case kept."""
    return [
        # Neither an age nor a presenting verb: this is not a patient.
        _entry("Which vitamin is supplied from only animal source:"),
        # A patient presentation, but well beyond the upper bound.
        _entry("A 55-year-old man presents with chest pain. " + "Il décrit la douleur. " * 60),
        # A patient presentation with no sign the rule can spot.
        _entry("A 40-year-old man presents for a routine administrative certificate request."),
        # A keepable case, present twice: the second is a duplicate.
        _entry("A 55-year-old man presents with chest pain radiating to the left arm."),
        _entry("A 55-year-old man presents with chest pain radiating to the left arm."),
    ]


def test_the_funnel_columns_add_up_to_the_entries():
    """That is the property the dataset card publishes: nothing is lost outside the columns."""
    entries = _control_corpus()
    count = funnel(entries)
    total = (
        count["no_patient_presentation"]
        + count["outside_length_bounds"]
        + count["no_identified_sign"]
        + count["duplicates"]
        + count["kept"]
    )
    assert total == count["entries"] == len(entries)


def test_the_funnel_finds_the_same_count_as_the_extraction():
    """Two paths, one result: otherwise the published table describes something else."""
    entries = _control_corpus()
    assert funnel(entries)["kept"] == len(extract_cases(entries, random.Random(42)))


def test_each_loss_is_charged_to_the_right_column():
    count = funnel(_control_corpus())
    assert count["no_patient_presentation"] == 1
    assert count["outside_length_bounds"] == 1
    assert count["no_identified_sign"] == 1
    assert count["duplicates"] == 1
    assert count["kept"] == 1


# --- Exam question against relative pronoun ---


@pytest.mark.parametrize(
    "statement",
    [
        "All of the following are surgical options for morbid obesity except -",
        "Which of the following is the investigation of choice?",
        "The most common cause of renal scaring in a 3 year old child is -",
        "Most appropriate management is",
        "Investigation of choice to know the cause of hematuria",
        "BEST prognostic factor for head injury is",
        "Parmi les affirmations suivantes, une seule est fausse",
        "What is the next step in management?",
    ],
)
def test_an_exam_statement_is_recognised(statement: str):
    assert is_an_exam_question(statement)


@pytest.mark.parametrize(
    "narrative",
    [
        "She had flu like symptoms 20 days ago which resolved spontaneously.",
        "There is circumoral cyanosis, which is not alleviated by nasal oxygen.",
        "A 50-year-old lady presented with a lump in the left breast, which developed suddenly.",
        "Il décrit une douleur qui irradie vers la mâchoire.",
        "Le patient ne sait plus quel traitement il prend.",
    ],
)
def test_a_narrative_sentence_is_not_taken_for_a_question(narrative: str):
    """The previous pattern deleted the whole sentence on a relative "which".

    That was clinical content lost: "flu like symptoms 20 days ago" disappeared from the
    narrative because the sentence contained a relative pronoun.
    """
    assert not is_an_exam_question(narrative)


def test_the_exam_question_is_removed_but_the_narrative_stays():
    question = (
        "A 60 yr old chronic smoker presents with painless gross hematuria of 1 day duration, "
        "which started this morning. All of the following are possible causes except -"
    )
    vignette = extract_vignette(question)
    assert vignette is not None
    assert "All of the following" not in vignette
    # The narrative sentence carries a relative "which": it must survive.
    assert "which started this morning" in vignette
