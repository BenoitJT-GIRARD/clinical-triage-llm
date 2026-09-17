"""Tests of the vital signs: thresholds, reading them out of a text, and generation.

The readings and the finding labels are French: they are what the service exchanges and what the
rule reads back.
"""

from __future__ import annotations

import random

import pytest

from clinical_triage.data.vital_signs import (
    DEFAULT_AGE,
    VitalSigns,
    critical_findings,
    generate,
    normal_ranges,
    parse,
    parse_age,
    warning_findings,
)


def test_an_empty_reading_triggers_nothing():
    empty = VitalSigns()
    assert empty.is_empty()
    assert critical_findings(empty, 40) == []
    assert warning_findings(empty, 40) == []


def test_the_normal_ranges_depend_on_the_age():
    assert normal_ranges(1)[0] == (100, 160)
    assert normal_ranges(40)[0] == (60, 100)


def test_the_adult_critical_thresholds():
    assert critical_findings(VitalSigns(spo2=88), 40)
    assert critical_findings(VitalSigns(systolic_bp=85), 40)
    assert critical_findings(VitalSigns(temperature=34.5), 40)
    assert critical_findings(VitalSigns(conscious=False), 40)
    assert not critical_findings(VitalSigns(spo2=97, systolic_bp=120, conscious=True), 40)


def test_the_adult_warning_thresholds():
    assert warning_findings(VitalSigns(spo2=93), 40)
    assert warning_findings(VitalSigns(temperature=38.9), 40)
    assert warning_findings(VitalSigns(pain_score=8), 40)
    assert not warning_findings(VitalSigns(spo2=98, temperature=36.8, pain_score=2), 40)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "FC 118/min, TA 92/60 mmHg, SpO2 91 %",
            {"heart_rate": 118, "systolic_bp": 92, "spo2": 91},
        ),
        ("HR 62, BP 128/76, RR 16, SpO2 98%", {"heart_rate": 62, "resp_rate": 16, "spo2": 98}),
        ("Température 38,7 °C, douleur 7/10", {"temperature": 38.7, "pain_score": 7}),
        ("Enfant fébrile à 39.8, vigilance normale", {"temperature": 39.8, "conscious": True}),
    ],
)
def test_reading_the_vital_signs_out_of_a_text(text, expected):
    reading = parse(text)
    for field, value in expected.items():
        assert getattr(reading, field) == value


def test_reading_an_altered_consciousness():
    assert parse("Patient somnolent, difficile à réveiller.").conscious is False


def test_a_text_with_no_vital_sign_gives_an_empty_reading():
    assert parse("Le patient se plaint de fatigue.").is_empty()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Homme de 67 ans", 67),
        ("A 5-year-old boy", 5),
        ("Nourrisson de 6 mois", 0),
        ("An 18-month-old infant", 1),
        ("Un patient sans âge précisé", 40),
    ],
)
def test_reading_the_age(text, expected):
    assert parse_age(text) == expected


def test_the_normal_profile_never_produces_a_vital_sign_in_the_alert_zone():
    """A case labelled "deferred consultation" must not come out with a red flag."""
    generator = random.Random(0)
    for age in (0, 3, 10, 35, 80):
        for _ in range(40):
            reading = generate("normal", age, generator)
            assert critical_findings(reading, age) == []
            assert warning_findings(reading, age) == []


def test_the_critical_profile_produces_at_least_one_abnormality_most_of_the_time():
    generator = random.Random(0)
    abnormal = sum(
        bool(critical_findings(generate("critical", 45, generator), 45)) for _ in range(60)
    )
    assert abnormal >= 40


def test_bilingual_rendering():
    reading = VitalSigns(heart_rate=80, systolic_bp=120, diastolic_bp=75, spo2=98, conscious=True)
    assert "FC 80/min" in reading.render("fr")
    assert "vigilance normale" in reading.render("fr")
    assert "HR 80/min" in reading.render("en")
    assert "alert" in reading.render("en")


def test_absent_measurements_are_not_shown():
    assert "SpO2" not in VitalSigns(heart_rate=80).render("fr")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("T 38,2 °C", 38.2),
        # Without a space before the unit: a common spelling that the word boundary rejected.
        ("temp 39.1C", 39.1),
        ("T = 38,5°C", 38.5),
        # With a cue word, the decimal is optional.
        ("fever 39C", 39.0),
        ("T 39 °C", 39.0),
        # Without a cue word, a bare integer is not a temperature: here they are an age, a
        # respiratory rate and a heart rate.
        ("Patient de 38 ans, FR 40, FC 39.", None),
        ("TA 148/92, FC 102", None),
    ],
)
def test_the_temperature_is_read_in_its_common_spellings(text, expected):
    assert parse(text).temperature == expected


def test_a_text_with_no_age_gives_the_adult_age():
    """The default is deliberate: the adult bands are the most cautious.

    A heart rate of 130 is normal in an infant and alarming in an adult. Absent age
    information, better to alert wrongly.
    """
    assert parse_age("FC 102, TA 148/92, sans autre précision.") == DEFAULT_AGE
    assert parse_age("Enfant de 6 mois.") == 0
    assert parse_age("A 3-year-old boy.") == 3


# --- The patient's age and how long the symptoms have lasted ---


@pytest.mark.parametrize(
    ("expected", "text"),
    [
        (58, "Homme de 58 ans, toux depuis 3 mois. Constantes : FC 128/min, FR 26/min."),
        (32, "Femme de 32 ans, enceinte de 8 mois."),
        (0, "Nourrisson de 8 mois, fievre."),
        (6, "Enfant de 6 ans, otite."),
        (72, "72-year-old woman, chest pain for three weeks."),
        (0, "8-month-old infant, fever."),
    ],
)
def test_the_duration_of_the_symptoms_is_not_the_age_of_the_patient(expected, text):
    """ "Toux depuis 3 mois" turned a 58-year-old man into an infant.

    The patient then moved onto the thresholds of the first paediatric band, where a heart rate
    of 128 is normal: his abnormalities became invisible.
    """
    assert parse_age(text) == expected


def test_a_text_with_no_age_stays_adult():
    assert parse_age("Toux depuis 3 mois, sans autre precision.") == DEFAULT_AGE


# --- Lower bounds, by age band ---


def test_an_infants_bradycardia_is_critical():
    """Its normal rate starts at 100: 52 beats is pre-arrest.

    The lower bounds were frozen on adult values — 40 and 8 — whatever the age, and this infant
    came out as a deferred consultation.
    """
    infant = VitalSigns(heart_rate=52, resp_rate=14, spo2=97)
    assert critical_findings(infant, age=0)


def test_the_adult_lower_thresholds_do_not_change():
    """The derived bounds give back exactly the historical values: 40 and 8."""
    assert critical_findings(VitalSigns(heart_rate=39), age=40)
    assert not critical_findings(VitalSigns(heart_rate=41), age=40)
    assert critical_findings(VitalSigns(resp_rate=7), age=40)
    assert not critical_findings(VitalSigns(resp_rate=9), age=40)


def test_moderate_slowing_is_a_warning_and_not_an_emergency():
    """Acceleration had its intermediate degree, slowing did not."""
    warnings = warning_findings(VitalSigns(heart_rate=52, resp_rate=10), age=40)
    assert any("bradycardie" in w for w in warnings)
    assert any("bradypnée" in w for w in warnings)


# --- Reading consciousness: negation and missing accents ---


@pytest.mark.parametrize(
    "text",
    [
        "Patient sans trouble de la conscience, orienté.",
        "Aucun trouble de la vigilance à l'examen.",
        "Pas de désorientation, pas de somnolence.",
        "No altered consciousness reported.",
    ],
)
def test_a_negated_consciousness_does_not_count_as_altered(text):
    """ "Sans trouble de la conscience" describes a normal patient.

    The reading ignored the negation: these four wordings, ordinary in a reception note, yielded
    ``conscious=False``, that is to say the one criterion that classifies as life-threatening on
    its own.
    """
    assert parse(text).conscious is not False


def test_consciousness_is_read_even_without_accents():
    """A reception note is often written without accents; the patterns did not see it."""
    assert parse("Patient desoriente et obnubile.").conscious is False
    assert parse("Patiente eveille et oriente.").conscious is True


def test_a_negation_in_another_clause_does_not_protect():
    """A negation carries over its own clause only, not the whole sentence."""
    assert parse("Pas de fièvre, mais patient confus.").conscious is False


def test_an_infant_carries_no_self_reported_pain():
    """The visual analogue scale is self-reported: it does not exist before four to six years.

    The generator drew one at every age, and infant bronchiolitis vignettes came out with
    "douleur 9/10" — an impossible measurement, which additionally triggered the "severe pain"
    sign.
    """
    generator = random.Random(11)
    for age in (0, 1, 4):
        for _ in range(20):
            assert generate("critical", age, generator).pain_score is None


def test_a_patient_old_enough_to_self_report_carries_a_pain_score():
    generator = random.Random(11)
    scores = [generate("critical", 40, generator).pain_score for _ in range(20)]
    assert all(score is not None for score in scores)
