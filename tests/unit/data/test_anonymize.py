"""Tests of the GDPR anonymisation and of its independent check.

Two distinct things are required: mask the identifying data, and **check the quality of the
masking**. These tests cover both, and a third requirement the project set itself — not to
destroy the clinical content along the way, without which the dataset is worth nothing.

The Presidio engines load two spaCy models: they are initialised once for the whole module.
"""

from __future__ import annotations

import pytest

from clinical_triage.data.anonymize import (
    ENTITIES_LEFT_OUT,
    RESIDUAL_PII_PATTERNS,
    SCORE_THRESHOLD,
    _engines,
    analyze_and_anonymize,
    audit_corpus,
    residual_pii,
)


@pytest.fixture(scope="module", autouse=True)
def _engines_loaded():
    """Force the engines to load before the first measurement."""
    analyze_and_anonymize("Texte de mise en route.", "fr")


# --- What must be masked ---


@pytest.mark.parametrize(
    ("language", "text", "marker"),
    [
        ("fr", "Mme Martine Dupont tousse depuis trois jours.", "<PERSON>"),
        ("fr", "Joindre au 06 11 22 33 44.", "<FR_TELEPHONE>"),
        # The international form designates the same subscriber as the national one.
        ("fr", "Joindre au +33 6 11 22 33 44.", "<FR_TELEPHONE>"),
        ("fr", "Née le 12/04/1953.", "<DATE_NAISSANCE>"),
        # ISO format, used by the English-language corpora.
        ("en", "Patient born 1953-04-12 with chest pain.", "<DATE_NAISSANCE_ISO>"),
        ("fr", "NIR 1 53 04 75 116 001 23.", "<FR_NIR>"),
        ("fr", "Écrire à jean.martin@chu-exemple.fr.", "<EMAIL_ADDRESS>"),
    ],
)
def test_an_identifying_item_is_masked(language, text, marker):
    masked, count = analyze_and_anonymize(text, language)
    assert marker in masked, masked
    assert count >= 1


def test_several_entities_in_one_text_are_all_masked():
    text = "Mr John Smith, born 1953-04-12, phone +33 6 11 22 33 44, reports chest pain."
    masked, count = analyze_and_anonymize(text, "en")
    assert count >= 3
    assert not residual_pii(masked)


# --- What must on no account be masked ---


@pytest.mark.parametrize(
    "text",
    [
        "Patient de 62 ans, douleur thoracique, FC 102, TA 148/92, SpO2 94 %.",
        "Traitement par inhibiteurs de recapture de la sérotonine depuis deux ans.",
        "Suspicion de syndrome coronarien aigu, appel du 15 (SAMU).",
        "Enfant de 6 mois, fièvre à 39,2 °C, FR 48/min, geignement.",
    ],
)
def test_the_clinical_content_survives_the_masking(text):
    """Masking a word that identifies nobody protects nobody.

    That is the flaw of Presidio's default setting on medical text: the names of molecules and
    of organs are taken there for names of people.
    """
    masked, count = analyze_and_anonymize(text, "fr")
    assert masked == text, f"{count} entity(ies) masked wrongly: {masked}"


# --- The independent check ---


def test_the_check_detects_data_left_in_the_clear():
    """A check unable to report anything checks nothing."""
    text = (
        "Mme Martine Dupont, née le 12/04/1953, tél. 06 11 22 33 44, "
        "jean.martin@chu-exemple.fr, NIR 1 53 04 75 116 001 23, born 1953-04-12."
    )
    found = residual_pii(text)
    expected = {
        "title followed by a name",
        "date of birth",
        "ISO date of birth",
        "French phone number",
        "email address",
        "social security number",
    }
    assert expected <= set(found), f"not detected: {expected - set(found)}"


def test_the_check_reports_nothing_on_a_masked_text():
    text = "Mme Martine Dupont, née le 12/04/1953, tél. 06 11 22 33 44, tousse."
    masked, _ = analyze_and_anonymize(text, "fr")
    assert residual_pii(masked) == {}


def test_the_check_is_independent_of_the_detector():
    """The check's patterns do not reuse the masking ones.

    A check that queried the same detector would confirm its own blind spots. This one is a
    battery of regular expressions written separately, and it is what reported that the
    international phone number escaped the masking.
    """
    assert len(RESIDUAL_PII_PATTERNS) >= 8
    assert "international phone number" in RESIDUAL_PII_PATTERNS


def test_auditing_a_corpus_aggregates_the_occurrences():
    corpus = [
        "Mme Dupont tousse.",
        "Mme Martin tousse aussi.",
        "Patient de 62 ans, FC 102.",
    ]
    assert audit_corpus(corpus)["title followed by a name"] == 2


# --- Why three entities are left out ---


def test_the_entities_left_out_would_destroy_the_clinical_narrative():
    """The ``ENTITIES_LEFT_OUT`` list is a decision, not a superstition.

    This test justifies it on fixed sentences: with those entities active, Presidio would mask
    the French abbreviation for blood pressure, and the onset delay as well as the patient's age
    in English. Three triage criteria out of three.
    """
    analyzer, _ = _engines()

    def detections(text: str, language: str) -> set[str]:
        available = set(analyzer.get_supported_entities(language=language))
        entities = [e for e in ENTITIES_LEFT_OUT if e in available]
        found = analyzer.analyze(
            text=text, language=language, entities=entities, score_threshold=SCORE_THRESHOLD
        )
        return {text[t.start : t.end] for t in found}

    french = "Homme de 62 ans, douleur thoracique depuis trois semaines. TA 148/92, FC 102."
    english = "62-year-old man, chest pain for three weeks. BP 148/92, HR 102."

    assert "TA" in detections(french, "fr")
    assert {"62-year-old", "three weeks"} <= detections(english, "en")


@pytest.mark.parametrize(
    ("language", "text", "survivors"),
    [
        (
            "fr",
            "Homme de 62 ans, douleur thoracique depuis trois semaines. TA 148/92, FC 102.",
            ("trois semaines", "TA 148/92", "62 ans"),
        ),
        (
            "en",
            "62-year-old man, chest pain for three weeks. BP 148/92, HR 102.",
            ("three weeks", "BP 148/92", "62-year-old"),
        ),
    ],
)
def test_the_chosen_setting_leaves_onset_age_and_vitals_intact(language, text, survivors):
    """Those are the three criteria a triage level is decided on."""
    masked, _ = analyze_and_anonymize(text, language)
    for survivor in survivors:
        assert survivor in masked
