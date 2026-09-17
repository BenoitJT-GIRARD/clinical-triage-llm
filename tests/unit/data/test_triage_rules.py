"""Tests de la règle de triage explicite.

Ces tests couvrent les trois défauts qui rendraient la règle inutilisable comme
référence : ne pas reconnaître les formes fléchies, ignorer les négations, et
ne pas lire les constantes écrites dans le récit.
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
    "texte",
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
def test_les_signes_vitaux_sont_reconnus_au_pluriel_et_au_feminin(texte):
    assert classify(texte) == VITAL


def test_un_signe_vital_prime_sur_un_signe_modere():
    assert classify("Fièvre élevée puis convulsion.") == VITAL


@pytest.mark.parametrize(
    "texte",
    [
        "Chute de sa hauteur, sans perte de connaissance, examen normal.",
        "Le patient n'a pas de douleur thoracique et ne présente aucun saignement abondant.",
        "No chest pain, no difficulty breathing, no loss of consciousness.",
    ],
)
def test_les_signes_nies_ne_declenchent_rien(texte):
    assert classify(texte) == DEFERRED


def test_signe_modere_reconnu():
    assert classify("Vomissements répétés depuis ce matin, déshydratation.") == MODERATE


def test_absence_de_signe():
    assert classify("Petit rhume et fatigue légère depuis deux jours.") == DEFERRED


def test_les_signes_detectes_sont_restitues():
    signes = matched_flags("chest pain and stroke symptoms", VITAL)
    assert "chest pain" in signes
    assert "stroke" in signes


def test_les_signes_nies_ne_sont_pas_restitues():
    assert matched_flags("pas de douleur thoracique", VITAL) == []


def test_des_constantes_effondrees_classent_en_urgence_vitale():
    constantes = VitalSigns(spo2=88, systolic_bp=86, heart_rate=118, conscious=True)
    assert classify("Patient fatigué depuis ce matin.", constantes, age=55) == VITAL


def test_les_constantes_sont_lues_dans_le_texte():
    """Sans cette lecture, la règle serait comparée au modèle à armes inégales."""
    texte = "Homme de 29 ans venu pour une grosse fatigue. FC 38, TA 82/48, SpO2 97 %."
    assert classify(texte) == VITAL


def test_les_seuils_pediatriques_sont_appliques():
    """140 battements par minute est normal à deux ans, critique à quarante ans."""
    constantes = VitalSigns(heart_rate=140)
    assert classify("Enfant enrhumé.", constantes, age=2) == DEFERRED
    assert classify("Adulte enrhumé.", constantes, age=40) == VITAL


def test_la_decision_est_justifiee():
    raisons = explain("Douleur thoracique avec sueurs.")
    assert any("douleur thoracique" in raison for raison in raisons)


def test_les_raisons_citent_les_constantes_anormales():
    raisons = explain("Patient calme.", VitalSigns(spo2=85), age=40)
    assert any("saturation" in raison for raison in raisons)


@pytest.mark.parametrize(
    ("texte", "attendu"),
    [
        # Vocabulaire clinique français canonique : ce sont les mots qu'une
        # infirmière d'accueil écrit, et les corpus francophones les emploient.
        ("Dyspnée au repos depuis ce matin.", "URGENCE_VITALE"),
        ("Marbrures des genoux, extrémités froides.", "URGENCE_VITALE"),
        ("Raideur de nuque fébrile.", "URGENCE_VITALE"),
        ("Défense abdominale à la palpation.", "URGENCE_VITALE"),
        ("Dyspnée à l'effort depuis une semaine.", "URGENCE_MODEREE"),
        ("Syncope brève, récupération complète.", "URGENCE_MODEREE"),
    ],
)
def test_le_lexique_couvre_le_vocabulaire_clinique_francais(texte, attendu):
    assert classify(texte) == attendu


@pytest.mark.parametrize(
    ("texte", "attendu"),
    [
        # « Brûlure » désigne en français une lésion *et* une sensation. Seule la
        # lésion relève du triage.
        ("Brûlures remontant derrière le sternum après les repas.", "CONSULTATION_DIFFEREE"),
        ("Brûlure chimique de l'avant-bras.", "URGENCE_MODEREE"),
        ("Brûlure par eau bouillante sur la main.", "URGENCE_MODEREE"),
    ],
)
def test_une_sensation_de_brulure_n_est_pas_une_brulure(texte, attendu):
    assert classify(texte) == attendu


# --- Portée de la négation ---


@pytest.mark.parametrize(
    "texte",
    [
        "Homme de 55 ans, pas de fievre, douleur thoracique constrictive depuis 20 minutes.",
        "Sans perte de connaissance, mais convulsion en cours.",
        "Pas de fievre. Douleur thoracique constrictive.",
        "Apyretique, cependant detresse respiratoire.",
    ],
)
def test_une_negation_ne_franchit_pas_la_ponctuation(texte):
    """Une note d'accueil énumère : « pas de fièvre, douleur thoracique ».

    La fenêtre de recherche remontait par-dessus la virgule, trouvait « pas de »
    et annulait le signe qui suit — un infarctus classé en consultation
    différée, sans rien afficher qui l'explique.
    """
    assert classify(texte) == VITAL


@pytest.mark.parametrize(
    "texte",
    [
        "Pas de douleur thoracique.",
        "Aucune convulsion.",
        "Le patient ne presente pas de detresse respiratoire.",
        "Sans perte de connaissance.",
    ],
)
def test_une_negation_de_la_meme_proposition_annule_toujours_le_signe(texte):
    assert classify(texte) == DEFERRED


# --- Formes singulier et pluriel ---


@pytest.mark.parametrize(
    ("singulier", "pluriel"),
    [
        ("Le patient exprime une idee suicidaire.", "Le patient exprime des idees suicidaires."),
        ("Levre bleue.", "Levres bleues."),
        ("Pause in breathing.", "Pauses in breathing."),
        ("Purple skin blotch on the leg.", "Purple skin blotches on the leg."),
    ],
)
def test_les_deux_nombres_declenchent_autant(singulier, pluriel):
    """La flexion ajoute une terminaison, elle ne sait pas en retirer une.

    Un terme stocké au pluriel n'était donc reconnu qu'au pluriel : « idée
    suicidaire » au singulier échappait au dépistage.
    """
    assert classify(singulier) == VITAL
    assert classify(pluriel) == VITAL


# --- Vocabulaire profane ---


@pytest.mark.parametrize(
    "texte",
    [
        "Elle a du mal a respirer au repos.",
        "Il n'arrive pas a respirer.",
        "Notre fille saigne beaucoup du nez.",
        "Le patient a perdu connaissance.",
        "Il ne se reveille pas.",
        "She is struggling to breathe.",
    ],
)
def test_les_mots_d_un_accompagnant_declenchent_aussi(texte):
    """Le questionnaire recueille les phrases telles qu'elles sont dites.

    Une règle qui ne connaît que « détresse respiratoire » ne lit pas « elle a
    du mal à respirer », qui est ce qu'un accompagnant écrit réellement.
    """
    assert classify(texte) == VITAL


@pytest.mark.parametrize(
    "texte",
    [
        "Rhume, pas de mal a respirer.",
        "Petite coupure au doigt, saigne un peu.",
        "Le patient se reveille facilement le matin.",
        "Toux seche depuis deux jours.",
    ],
)
def test_ce_vocabulaire_ne_declenche_pas_a_tort(texte):
    assert classify(texte) == DEFERRED


def test_aucun_terme_n_est_liste_deux_fois():
    """Un doublon est sans effet sur la détection, et signale une liste relue trop vite."""
    for liste in (RED_FLAGS, WARNING_FLAGS):
        doublons = sorted({t for t in liste if liste.count(t) > 1})
        assert doublons == []


# --- Sur-déclenchements : deux tournures cliniques courantes ---


def test_une_absence_de_reponse_au_traitement_n_est_pas_une_urgence_vitale():
    """« Ne répond pas » était cherché sans contexte.

    C'est une tournure banale d'un compte rendu — « ne répond pas au traitement
    antibiotique » — et elle classait le cas en urgence vitale. Effet concret :
    le questionnaire s'arrêtait dès le motif, et le patient n'était plus
    interrogé du tout.
    """
    texte = "Le patient ne répond pas au traitement antibiotique depuis trois jours."
    assert classify(texte) == DEFERRED
    assert explain(texte) == []


def test_un_patient_areactif_reste_une_urgence_vitale():
    assert classify("Patient inconscient, ne répond pas aux stimulations.") == VITAL
    assert classify("Le patient ne réagit plus.") == VITAL


def test_une_hemorragie_sous_conjonctivale_n_est_pas_une_urgence_vitale():
    """Spectaculaire et bénigne : le faux positif quotidien d'un accueil."""
    texte = "Hémorragie sous-conjonctivale isolée, indolore, vision normale."
    assert classify(texte) == DEFERRED
    assert explain(texte) == []


@pytest.mark.parametrize(
    "texte",
    [
        "Hémorragie digestive avec méléna abondant.",
        "Hémorragie de la délivrance après accouchement.",
        "Elle saigne beaucoup de la cuisse.",
        "Massive hemorrhage from the thigh wound.",
    ],
)
def test_les_hemorragies_graves_restent_detectees(texte):
    assert classify(texte) == VITAL
