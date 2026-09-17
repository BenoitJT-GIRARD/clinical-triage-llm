"""Tests of the explicit triage rule.

They cover the three defects that would make the rule unusable as a baseline: failing to
recognise inflected forms, ignoring negations, and not reading the vital signs written into the
narrative.

The cases are French and English clinical text: they are the data the rule matches against.
"""

from __future__ import annotations

import pytest

from clinical_triage.data.triage_rules import (
    DEFERRED,
    MODERATE,
    RED_FLAGS,
    VITAL,
    WARNING_FLAGS,
    classify,
    explain,
    matched_flags,
)
from clinical_triage.data.vital_signs import VitalSigns


@pytest.mark.parametrize(
    "text",
    [
        "Le patient présente une douleur thoracique.",
        "Le patient a des convulsions répétées.",
        "Patient avec douleurs thoraciques intenses.",
        "Hémorragies digestives abondantes.",
        "Patiente retrouvée inconsciente au sol.",
        "Chest pain radiating to the left arm.",
        "Seizures started ten minutes ago.",
    ],
)
def test_red_flags_are_recognised_in_plural_and_feminine_forms(text):
    assert classify(text) == VITAL


def test_a_red_flag_wins_over_a_warning_flag():
    assert classify("Fièvre élevée puis convulsion.") == VITAL


@pytest.mark.parametrize(
    "text",
    [
        "Chute de sa hauteur, sans perte de connaissance, examen normal.",
        "Le patient n'a pas de douleur thoracique et ne présente aucun saignement abondant.",
        "No chest pain, no difficulty breathing, no loss of consciousness.",
    ],
)
def test_negated_signs_trigger_nothing(text):
    assert classify(text) == DEFERRED


def test_a_warning_flag_is_recognised():
    assert classify("Vomissements répétés depuis ce matin, déshydratation.") == MODERATE


def test_no_sign_at_all():
    assert classify("Petit rhume et fatigue légère depuis deux jours.") == DEFERRED


def test_the_detected_signs_are_returned():
    signs = matched_flags("chest pain and stroke symptoms", VITAL)
    assert "chest pain" in signs
    assert "stroke" in signs


def test_negated_signs_are_not_returned():
    assert matched_flags("pas de douleur thoracique", VITAL) == []


def test_collapsed_vital_signs_classify_as_life_threatening():
    vitals = VitalSigns(spo2=88, systolic_bp=86, heart_rate=118, conscious=True)
    assert classify("Patient fatigué depuis ce matin.", vitals, age=55) == VITAL


def test_the_vital_signs_are_read_out_of_the_text():
    """Without that reading, the rule would be compared to the model on unequal terms."""
    text = "Homme de 29 ans venu pour une grosse fatigue. FC 38, TA 82/48, SpO2 97 %."
    assert classify(text) == VITAL


def test_the_paediatric_thresholds_are_applied():
    """140 beats per minute is normal at two years old, critical at forty."""
    vitals = VitalSigns(heart_rate=140)
    assert classify("Enfant enrhumé.", vitals, age=2) == DEFERRED
    assert classify("Adulte enrhumé.", vitals, age=40) == VITAL


def test_the_decision_is_justified():
    reasons = explain("Douleur thoracique avec sueurs.")
    assert any("douleur thoracique" in reason for reason in reasons)


def test_the_reasons_quote_the_abnormal_vital_signs():
    reasons = explain("Patient calme.", VitalSigns(spo2=85), age=40)
    assert any("saturation" in reason for reason in reasons)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Canonical French clinical vocabulary: these are the words a triage nurse writes, and
        # the French-language corpora use them.
        ("Dyspnée au repos depuis ce matin.", "URGENCE_VITALE"),
        ("Marbrures des genoux, extrémités froides.", "URGENCE_VITALE"),
        ("Raideur de nuque fébrile.", "URGENCE_VITALE"),
        ("Défense abdominale à la palpation.", "URGENCE_VITALE"),
        ("Dyspnée à l'effort depuis une semaine.", "URGENCE_MODEREE"),
        ("Syncope brève, récupération complète.", "URGENCE_MODEREE"),
    ],
)
def test_the_lexicon_covers_the_french_clinical_vocabulary(text, expected):
    assert classify(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # "Brûlure" means both a lesion *and* a sensation in French. Only the lesion belongs to
        # triage.
        ("Brûlures remontant derrière le sternum après les repas.", "CONSULTATION_DIFFEREE"),
        ("Brûlure chimique de l'avant-bras.", "URGENCE_MODEREE"),
        ("Brûlure par eau bouillante sur la main.", "URGENCE_MODEREE"),
    ],
)
def test_a_burning_sensation_is_not_a_burn(text, expected):
    assert classify(text) == expected


# --- Scope of a negation ---


@pytest.mark.parametrize(
    "text",
    [
        "Homme de 55 ans, pas de fievre, douleur thoracique constrictive depuis 20 minutes.",
        "Sans perte de connaissance, mais convulsion en cours.",
        "Pas de fievre. Douleur thoracique constrictive.",
        "Apyretique, cependant detresse respiratoire.",
    ],
)
def test_a_negation_does_not_cross_punctuation(text):
    """A reception note enumerates: "pas de fièvre, douleur thoracique".

    The search window reached back over the comma, found "pas de" and cancelled the sign that
    follows — a heart attack filed as a deferred consultation, with nothing shown to explain it.
    """
    assert classify(text) == VITAL


@pytest.mark.parametrize(
    "text",
    [
        "Pas de douleur thoracique.",
        "Aucune convulsion.",
        "Le patient ne presente pas de detresse respiratoire.",
        "Sans perte de connaissance.",
    ],
)
def test_a_negation_in_the_same_clause_still_cancels_the_sign(text):
    assert classify(text) == DEFERRED


# --- Singular and plural forms ---


@pytest.mark.parametrize(
    ("singular", "plural"),
    [
        ("Le patient exprime une idee suicidaire.", "Le patient exprime des idees suicidaires."),
        ("Levre bleue.", "Levres bleues."),
        ("Pause in breathing.", "Pauses in breathing."),
        ("Purple skin blotch on the leg.", "Purple skin blotches on the leg."),
    ],
)
def test_both_numbers_fire_equally(singular, plural):
    """Inflection adds an ending, it cannot remove one.

    A term stored in the plural was therefore recognised in the plural only: "idée suicidaire"
    in the singular escaped the screening.
    """
    assert classify(singular) == VITAL
    assert classify(plural) == VITAL


# --- Lay vocabulary ---


@pytest.mark.parametrize(
    "text",
    [
        "Elle a du mal a respirer au repos.",
        "Il n'arrive pas a respirer.",
        "Notre fille saigne beaucoup du nez.",
        "Le patient a perdu connaissance.",
        "Il ne se reveille pas.",
        "She is struggling to breathe.",
    ],
)
def test_the_words_of_a_relative_fire_too(text):
    """The questionnaire collects sentences as they are said.

    A rule that only knows "détresse respiratoire" does not read "elle a du mal à respirer",
    which is what a relative actually writes.
    """
    assert classify(text) == VITAL


@pytest.mark.parametrize(
    "text",
    [
        "Rhume, pas de mal a respirer.",
        "Petite coupure au doigt, saigne un peu.",
        "Le patient se reveille facilement le matin.",
        "Toux seche depuis deux jours.",
    ],
)
def test_that_vocabulary_does_not_fire_wrongly(text):
    assert classify(text) == DEFERRED


def test_no_term_is_listed_twice():
    """A duplicate has no effect on detection, and signals a list re-read too quickly."""
    for listing in (RED_FLAGS, WARNING_FLAGS):
        duplicates = sorted({t for t in listing if listing.count(t) > 1})
        assert duplicates == []


# --- Over-firing: two common clinical turns of phrase ---


def test_a_failure_to_respond_to_treatment_is_not_life_threatening():
    """"Ne répond pas" used to be searched for without context.

    It is an ordinary turn of phrase in a report — "ne répond pas au traitement antibiotique" —
    and it classified the case as life-threatening. Concrete effect: the questionnaire stopped
    at the complaint, and the patient was not questioned at all.
    """
    text = "Le patient ne répond pas au traitement antibiotique depuis trois jours."
    assert classify(text) == DEFERRED
    assert explain(text) == []


def test_an_unresponsive_patient_is_still_life_threatening():
    assert classify("Patient inconscient, ne répond pas aux stimulations.") == VITAL
    assert classify("Le patient ne réagit plus.") == VITAL


def test_a_subconjunctival_haemorrhage_is_not_life_threatening():
    """Spectacular and benign: the daily false positive of a reception desk."""
    text = "Hémorragie sous-conjonctivale isolée, indolore, vision normale."
    assert classify(text) == DEFERRED
    assert explain(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "Hémorragie digestive avec méléna abondant.",
        "Hémorragie de la délivrance après accouchement.",
        "Elle saigne beaucoup de la cuisse.",
        "Massive hemorrhage from the thigh wound.",
    ],
)
def test_serious_haemorrhages_are_still_detected(text):
    assert classify(text) == VITAL
