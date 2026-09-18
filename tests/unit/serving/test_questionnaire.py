"""Tests of the adaptive questionnaire.

The complaints, the questions and the answers are French: they are what the service exchanges
with the staff, and the triage rule reads them back.
"""

from __future__ import annotations

import pytest

from clinical_triage.config import TRIAGE
from clinical_triage.data.triage_rules import DEFERRED, classify, explain
from clinical_triage.serving.questionnaire import (
    ALARMING_ANSWER,
    GENERAL_QUESTIONS,
    QUESTIONS_BY_THEME,
    RED_FLAG_QUESTIONS,
    STATEMENTS,
    _binary_answer,
    compile_symptoms,
    detect_theme,
    has_red_flag,
    next_question,
    plan,
)


@pytest.mark.parametrize(
    ("complaint", "theme"),
    [
        ("douleur dans la poitrine", "douleur_thoracique"),
        ("essoufflement depuis hier", "respiratoire"),
        ("mal de tête violent", "neurologique"),
        ("idées suicidaires", "psychiatrique"),
        ("chute de vélo", "traumatologie"),
        ("mal au ventre", "digestif"),
        ("fièvre depuis deux jours", "fievre"),
        ("démarche administrative", "general"),
    ],
)
def test_the_complaint_determines_the_theme(complaint, theme):
    assert detect_theme(complaint) == theme


def test_the_questions_depend_on_the_complaint():
    """That is what makes the questionnaire adaptive rather than fixed."""
    chest = [identifier for identifier, _ in plan("douleur dans la poitrine")]
    trauma = [identifier for identifier, _ in plan("entorse de la cheville")]
    assert chest != trauma
    assert "irradiation" in chest
    assert "appui" in trauma


def test_the_red_flag_questions_come_first_whatever_the_complaint():
    for complaint in ("mal de gorge", "douleur dans la poitrine", "chute de vélo"):
        first = [identifier for identifier, _ in plan(complaint)][: len(RED_FLAG_QUESTIONS)]
        assert first == [identifier for identifier, _ in RED_FLAG_QUESTIONS]


def test_a_red_flag_stops_the_collection_immediately():
    step = next_question("douleur thoracique intense", {})
    assert step.finished is True
    assert step.identifier is None


def test_a_benign_complaint_triggers_questions():
    step = next_question("mal de gorge léger", {})
    assert step.finished is False
    assert step.identifier and step.text


def test_the_collection_ends_when_everything_has_been_answered():
    answers = {identifier: "non" for identifier, _ in plan("fatigue")}
    assert next_question("fatigue", answers).finished is True


def test_a_red_flag_appearing_mid_collection_stops_the_questionnaire():
    step = next_question("fatigue", {"conscience": "non, le patient est inconscient"})
    assert step.finished is True


def test_the_summary_keeps_the_meaning_of_each_answer():
    """Keeping only the answers would produce "non. non. oui", with no referent."""
    summary = compile_symptoms("toux", {"respiration": "non", "temperature": "38,5 depuis hier"})
    assert "toux" in summary
    assert "Pas de difficulté à respirer" in summary
    assert "38,5 depuis hier" in summary


def test_a_negative_answer_is_written_as_a_negation():
    """A trailing negation would be read back as the symptom itself.

    That is the trap this module already avoids for the early stop, and which reappeared here:
    the compiled description is read back by the triage rule in the service, and "Difficulté à
    respirer : non" reads there as a breathing difficulty.
    """
    summary = compile_symptoms("toux", {"respiration": "non"})
    assert "Difficulté à respirer :" not in summary
    assert summary.endswith("Pas de difficulté à respirer ni d'essoufflement au repos.")


@pytest.mark.parametrize(
    "complaint",
    [
        "gene dans la poitrine",
        "petite toux",
        "mal de tete leger",
        "ventre un peu gonfle",
        "petite coupure au doigt",
        "fievre a 38",
        "un peu d angoisse",
        "fatigue generale",
    ],
)
def test_reassuring_answers_never_worsen_the_rule_verdict(complaint):
    """A cold with everything denied must not come out as life-threatening.

    One case walks the eight themes: it was the text of the questions, copied into the
    description, that tipped the rule over.
    """
    # "non" is not reassuring everywhere: to "is the patient conscious?" it is the alarming
    # answer. So the opposite of what `ALARMING_ANSWER` declares is answered for each question.
    reassuring = {
        identifier: "oui" if ALARMING_ANSWER.get(identifier) == "non" else "non"
        for identifier, _ in plan(complaint)
    }
    description = compile_symptoms(complaint, reassuring)
    assert TRIAGE.severity[classify(description)] <= TRIAGE.severity[classify(complaint)]


def test_every_question_knows_how_to_report_its_answer():
    """A question with no wording would fall back on its raw identifier."""
    asked = {identifier for identifier, _ in RED_FLAG_QUESTIONS}
    for questions in QUESTIONS_BY_THEME.values():
        asked.update(identifier for identifier, _ in questions)
    asked.update(identifier for identifier, _ in GENERAL_QUESTIONS)
    assert asked <= set(STATEMENTS)


def test_empty_answers_are_ignored():
    summary = compile_symptoms("toux", {"respiration": "   ", "saignement": "non"})
    assert "respirer" not in summary
    assert "saignement" in summary.lower()


# --- Reading a yes / no answer ---


@pytest.mark.parametrize(
    "answer",
    [
        "Notre fille saigne du nez en abondance depuis 20 minutes",
        "Nous ne savons pas",
        "Normalement non",
        "Nouvelle crise depuis ce matin",
    ],
)
def test_a_word_starting_with_no_is_not_a_negation(answer):
    """The match is on a whole word: "non" must not be found inside "Notre".

    The comparison used to be on a prefix: "Notre fille saigne du nez" was read as a "non", and
    the summary handed to the model wrote "Aucun saignement actif" — the exact opposite of what
    the relative had just declared.
    """
    assert _binary_answer(answer) is None


@pytest.mark.parametrize(
    ("expected", "answer"),
    [
        ("oui", "oui"),
        ("oui", "Oui"),
        ("non", "non"),
        ("non", "  NON  "),
        ("non", "Aucun"),
        ("non", "no"),
        ("non", "rien"),
        (None, "oui, beaucoup de sang"),
        (None, "peut-etre"),
    ],
)
def test_only_an_answer_reduced_to_one_word_is_binary(expected, answer):
    """As soon as there is anything else, the text is carried through: that is safer."""
    assert _binary_answer(answer) == expected


def test_a_free_text_answer_reaches_the_summary_intact():
    summary = compile_symptoms(
        "Chute de velo", {"saignement": "Notre fille saigne du nez en abondance depuis 20 minutes"}
    )
    assert "saigne du nez en abondance" in summary
    assert "Aucun saignement" not in summary


# --- Red-flag screening ---


@pytest.mark.parametrize(
    ("complaint", "answers"),
    [
        ("mal de tete", {"deficit": "oui"}),
        ("mal de tete", {"cephalee": "oui"}),
        ("angoisse", {"intention": "oui"}),
        ("toux", {"parole": "non"}),
        ("fievre", {"nuque": "non"}),
        ("mal de tete", {"deficit": "oui, hemiplegie droite depuis 30 minutes"}),
        ("chute", {"saignement": "Notre fille saigne beaucoup du nez"}),
    ],
)
def test_an_alarming_answer_stops_the_collection(complaint, answers):
    """Five questions out of eight triggered nothing, for want of being listed.

    And the triage rule was applied to non-binary answers only: everything following a "oui"
    was never read.
    """
    assert has_red_flag(complaint, answers)
    assert next_question(complaint, answers).finished


@pytest.mark.parametrize(
    ("complaint", "answers"),
    [
        ("rhume", {"conscience": "oui", "respiration": "non", "saignement": "non"}),
        ("entorse", {"appui": "non", "deformation": "non"}),
        ("fievre", {"nuque": "oui", "frissons": "non"}),
        ("angoisse", {"intention": "non", "moyens": "non"}),
        ("ventre", {"vomissements": "oui"}),
    ],
)
def test_a_reassuring_collection_carries_on(complaint, answers):
    assert not has_red_flag(complaint, answers)


def test_no_free_text_label_triggers_the_rule_on_its_own():
    """A label is our vocabulary, not the patient's.

    The compiled summary is read back by the triage rule, and its opinion is shown next to the
    model's. "Idées suicidaires : je ne sais pas" made "idees suicidaires" appear among the
    reasons shown to the nurse, for a patient who had expressed nothing at all: a justification
    manufactured by the wording of the questionnaire itself.
    """
    for identifier, (label, _, _) in STATEMENTS.items():
        sentence = f"{label} : je ne sais pas."
        assert classify(sentence) == DEFERRED, (identifier, label, explain(sentence))


def test_a_genuinely_alarming_answer_is_still_detected():
    """The canonical sentences, on the other hand, report what the patient said."""
    for identifier in ("intention", "frissons", "respiration"):
        _, affirmative, _ = STATEMENTS[identifier]
        assert classify(affirmative) != DEFERRED, identifier


def test_a_free_text_answer_to_a_non_binary_question_does_not_alarm():
    """Not every question expects a yes or a no.

    The mechanism of a trauma is told in full sentences. Compared without a guard, it was
    ``None`` on both sides — non-binary answer and question with no alarming answer — and the
    questionnaire stopped on a life-threatening distress nobody had described.
    """
    complaint = "entorse de la cheville apres une chute"
    answers = {
        "conscience": "oui",
        "respiration": "non",
        "saignement": "non",
        "mecanisme": "chute de sa hauteur dans l escalier",
    }

    assert has_red_flag(complaint, answers) is False
    assert next_question(complaint, answers).finished is False


def test_a_free_text_answer_describing_a_serious_sign_still_alarms():
    """The guard must not make the questionnaire deaf to content."""
    complaint = "chute a domicile"
    answers = {"mecanisme": "chute de quatre metres, le patient est inconscient"}

    assert has_red_flag(complaint, answers) is True
