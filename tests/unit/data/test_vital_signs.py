"""Tests des constantes vitales : seuils, lecture dans le texte et génération."""

from __future__ import annotations

import random

import pytest

from clinical_triage.data.vital_signs import (
    AGE_PAR_DEFAUT,
    VitalSigns,
    critical_findings,
    generate,
    normal_ranges,
    parse,
    parse_age,
    warning_findings,
)


def test_un_releve_vide_ne_declenche_rien():
    vide = VitalSigns()
    assert vide.is_empty()
    assert critical_findings(vide, 40) == []
    assert warning_findings(vide, 40) == []


def test_les_plages_normales_dependent_de_l_age():
    assert normal_ranges(1)[0] == (100, 160)
    assert normal_ranges(40)[0] == (60, 100)


def test_seuils_critiques_adulte():
    assert critical_findings(VitalSigns(spo2=88), 40)
    assert critical_findings(VitalSigns(systolic_bp=85), 40)
    assert critical_findings(VitalSigns(temperature=34.5), 40)
    assert critical_findings(VitalSigns(conscious=False), 40)
    assert not critical_findings(VitalSigns(spo2=97, systolic_bp=120, conscious=True), 40)


def test_seuils_d_alerte_adulte():
    assert warning_findings(VitalSigns(spo2=93), 40)
    assert warning_findings(VitalSigns(temperature=38.9), 40)
    assert warning_findings(VitalSigns(pain_score=8), 40)
    assert not warning_findings(VitalSigns(spo2=98, temperature=36.8, pain_score=2), 40)


@pytest.mark.parametrize(
    ("texte", "attendu"),
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
def test_lecture_des_constantes_dans_le_texte(texte, attendu):
    releve = parse(texte)
    for champ, valeur in attendu.items():
        assert getattr(releve, champ) == valeur


def test_lecture_de_la_vigilance_alteree():
    assert parse("Patient somnolent, difficile à réveiller.").conscious is False


def test_un_texte_sans_constante_donne_un_releve_vide():
    assert parse("Le patient se plaint de fatigue.").is_empty()


@pytest.mark.parametrize(
    ("texte", "attendu"),
    [
        ("Homme de 67 ans", 67),
        ("A 5-year-old boy", 5),
        ("Nourrisson de 6 mois", 0),
        ("An 18-month-old infant", 1),
        ("Un patient sans âge précisé", 40),
    ],
)
def test_lecture_de_l_age(texte, attendu):
    assert parse_age(texte) == attendu


def test_le_profil_normal_ne_produit_jamais_de_constante_en_alerte():
    """Un cas étiqueté « consultation différée » ne doit pas sortir avec un signe vital."""
    generateur = random.Random(0)
    for age in (0, 3, 10, 35, 80):
        for _ in range(40):
            releve = generate("normal", age, generateur)
            assert critical_findings(releve, age) == []
            assert warning_findings(releve, age) == []


def test_le_profil_critique_produit_au_moins_une_anomalie_le_plus_souvent():
    generateur = random.Random(0)
    anormaux = sum(
        bool(critical_findings(generate("critique", 45, generateur), 45)) for _ in range(60)
    )
    assert anormaux >= 40


def test_mise_en_forme_bilingue():
    releve = VitalSigns(heart_rate=80, systolic_bp=120, diastolic_bp=75, spo2=98, conscious=True)
    assert "FC 80/min" in releve.render("fr")
    assert "vigilance normale" in releve.render("fr")
    assert "HR 80/min" in releve.render("en")
    assert "alert" in releve.render("en")


def test_les_mesures_absentes_ne_sont_pas_affichees():
    assert "SpO2" not in VitalSigns(heart_rate=80).render("fr")


@pytest.mark.parametrize(
    ("texte", "attendue"),
    [
        ("T 38,2 °C", 38.2),
        # Sans espace avant l'unité : écriture courante, que la frontière de mot
        # rejetait.
        ("temp 39.1C", 39.1),
        ("T = 38,5°C", 38.5),
        # Avec mot d'annonce, la décimale est facultative.
        ("fever 39C", 39.0),
        ("T 39 °C", 39.0),
        # Sans mot d'annonce, un entier seul n'est pas une température : ce sont
        # ici un âge, une fréquence respiratoire et une fréquence cardiaque.
        ("Patient de 38 ans, FR 40, FC 39.", None),
        ("TA 148/92, FC 102", None),
    ],
)
def test_la_temperature_est_lue_dans_ses_ecritures_courantes(texte, attendue):
    assert parse(texte).temperature == attendue


def test_un_texte_sans_age_donne_l_age_adulte():
    """Le défaut est délibéré : les bandes adultes sont les plus prudentes.

    Une fréquence cardiaque de 130 est normale chez un nourrisson et alarmante
    chez un adulte. Sans information d'âge, mieux vaut alerter à tort.
    """
    assert parse_age("FC 102, TA 148/92, sans autre précision.") == AGE_PAR_DEFAUT
    assert parse_age("Enfant de 6 mois.") == 0
    assert parse_age("A 3-year-old boy.") == 3


# --- L'âge du patient et l'ancienneté des symptômes ---


@pytest.mark.parametrize(
    ("attendu", "texte"),
    [
        (58, "Homme de 58 ans, toux depuis 3 mois. Constantes : FC 128/min, FR 26/min."),
        (32, "Femme de 32 ans, enceinte de 8 mois."),
        (0, "Nourrisson de 8 mois, fievre."),
        (6, "Enfant de 6 ans, otite."),
        (72, "72-year-old woman, chest pain for three weeks."),
        (0, "8-month-old infant, fever."),
    ],
)
def test_l_anciennete_des_symptomes_n_est_pas_l_age_du_patient(attendu, texte):
    """« Toux depuis 3 mois » faisait un nourrisson d'un homme de 58 ans.

    Le patient passait alors sur les seuils de la première bande pédiatrique,
    où une fréquence cardiaque à 128 est normale : ses anomalies devenaient
    invisibles.
    """
    assert parse_age(texte) == attendu


def test_un_texte_sans_age_reste_adulte():
    assert parse_age("Toux depuis 3 mois, sans autre precision.") == AGE_PAR_DEFAUT


# --- Bornes basses, par bande d'âge ---


def test_la_bradycardie_du_nourrisson_est_critique():
    """Sa fréquence normale commence à 100 : 52 battements, c'est un pré-arrêt.

    Les bornes basses étaient figées sur les valeurs de l'adulte — 40 et 8 —
    quel que soit l'âge, et ce nourrisson ressortait en consultation différée.
    """
    nourrisson = VitalSigns(heart_rate=52, resp_rate=14, spo2=97)
    assert critical_findings(nourrisson, age=0)


def test_les_seuils_bas_de_l_adulte_ne_changent_pas():
    """Les bornes dérivées redonnent exactement les valeurs historiques : 40 et 8."""
    assert critical_findings(VitalSigns(heart_rate=39), age=40)
    assert not critical_findings(VitalSigns(heart_rate=41), age=40)
    assert critical_findings(VitalSigns(resp_rate=7), age=40)
    assert not critical_findings(VitalSigns(resp_rate=9), age=40)


def test_le_ralentissement_modere_est_une_alerte_et_non_une_urgence():
    """L'accélération avait son degré intermédiaire, le ralentissement non."""
    alertes = warning_findings(VitalSigns(heart_rate=52, resp_rate=10), age=40)
    assert any("bradycardie" in a for a in alertes)
    assert any("bradypnée" in a for a in alertes)


# --- Lecture de la vigilance : négation et absence d'accents ---


@pytest.mark.parametrize(
    "texte",
    [
        "Patient sans trouble de la conscience, orienté.",
        "Aucun trouble de la vigilance à l'examen.",
        "Pas de désorientation, pas de somnolence.",
        "No altered consciousness reported.",
    ],
)
def test_une_vigilance_niee_ne_compte_pas_comme_alteree(texte):
    """« Sans trouble de la conscience » décrit un patient normal.

    La lecture ignorait la négation : ces quatre formulations, banales dans une
    note d'accueil, rendaient `conscious=False`, c'est-à-dire le critère qui
    classe à lui seul en urgence vitale.
    """
    assert parse(texte).conscious is not False


def test_la_vigilance_se_lit_meme_sans_accents():
    """Une note d'accueil s'écrit souvent sans accents ; les motifs ne la voyaient pas."""
    assert parse("Patient desoriente et obnubile.").conscious is False
    assert parse("Patiente eveille et oriente.").conscious is True


def test_une_negation_dans_une_autre_proposition_ne_protege_pas():
    """La négation ne porte que sur sa propre proposition, pas sur la phrase entière."""
    assert parse("Pas de fièvre, mais patient confus.").conscious is False


def test_un_nourrisson_ne_porte_pas_de_douleur_auto_evaluee():
    """L'EVA est une autoévaluation : elle n'existe pas avant quatre à six ans.

    Le générateur en tirait une pour tous les âges, et les vignettes de
    bronchiolite du nourrisson sortaient avec « douleur 9/10 » — une mesure
    impossible, qui déclenchait en plus le signe « douleur intense ».
    """
    generateur = random.Random(11)
    for age in (0, 1, 4):
        for _ in range(20):
            assert generate("critique", age, generateur).pain_score is None


def test_un_patient_en_age_de_s_auto_evaluer_porte_une_douleur():
    generateur = random.Random(11)
    scores = [generate("critique", 40, generateur).pain_score for _ in range(20)]
    assert all(score is not None for score in scores)
