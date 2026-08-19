"""Tests de l'anonymisation RGPD et de son contrôle indépendant.

Le cahier des charges demande deux choses distinctes : masquer les données
identifiantes, et **contrôler la qualité du masquage**. Ces tests couvrent les
deux, et une troisième exigence que le projet s'est donnée — ne pas détruire le
contenu clinique au passage, sans quoi le dataset ne vaut plus rien.

Les moteurs Presidio chargent deux modèles spaCy : ils sont initialisés une seule
fois pour tout le module.
"""

from __future__ import annotations

import pytest

from clinical_triage.data.anonymize import (
    ENTITES_ECARTEES,
    RESIDUAL_PII_PATTERNS,
    SCORE_THRESHOLD,
    _engines,
    analyze_and_anonymize,
    audit_corpus,
    residual_pii,
)


@pytest.fixture(scope="module", autouse=True)
def _moteurs_charges():
    """Force le chargement des moteurs avant la première mesure."""
    analyze_and_anonymize("Texte de mise en route.", "fr")


# --- Ce qui doit être masqué ---


@pytest.mark.parametrize(
    ("langue", "texte", "marqueur"),
    [
        ("fr", "Mme Martine Dupont tousse depuis trois jours.", "<PERSON>"),
        ("fr", "Joindre au 06 11 22 33 44.", "<FR_TELEPHONE>"),
        # La forme internationale désigne le même abonné que la forme nationale.
        ("fr", "Joindre au +33 6 11 22 33 44.", "<FR_TELEPHONE>"),
        ("fr", "Née le 12/04/1953.", "<DATE_NAISSANCE>"),
        # Format ISO, employé par les corpus anglophones.
        ("en", "Patient born 1953-04-12 with chest pain.", "<DATE_NAISSANCE_ISO>"),
        ("fr", "NIR 1 53 04 75 116 001 23.", "<FR_NIR>"),
        ("fr", "Écrire à jean.martin@chu-exemple.fr.", "<EMAIL_ADDRESS>"),
    ],
)
def test_une_donnee_identifiante_est_masquee(langue, texte, marqueur):
    masque, nombre = analyze_and_anonymize(texte, langue)
    assert marqueur in masque, masque
    assert nombre >= 1


def test_plusieurs_entites_d_un_meme_texte_sont_toutes_masquees():
    texte = "Mr John Smith, born 1953-04-12, phone +33 6 11 22 33 44, reports chest pain."
    masque, nombre = analyze_and_anonymize(texte, "en")
    assert nombre >= 3
    assert not residual_pii(masque)


# --- Ce qui ne doit surtout pas l'être ---


@pytest.mark.parametrize(
    "texte",
    [
        "Patient de 62 ans, douleur thoracique, FC 102, TA 148/92, SpO2 94 %.",
        "Traitement par inhibiteurs de recapture de la sérotonine depuis deux ans.",
        "Suspicion de syndrome coronarien aigu, appel du 15 (SAMU).",
        "Enfant de 6 mois, fièvre à 39,2 °C, FR 48/min, geignement.",
    ],
)
def test_le_contenu_clinique_survit_au_masquage(texte):
    """Masquer un mot qui n'identifie personne ne protège personne.

    C'est le défaut du réglage par défaut de Presidio sur du texte médical : les
    noms de molécules et d'organes y sont pris pour des noms de personnes.
    """
    masque, nombre = analyze_and_anonymize(texte, "fr")
    assert masque == texte, f"{nombre} entité(s) masquée(s) à tort : {masque}"


# --- Le contrôle indépendant ---


def test_le_controle_detecte_les_donnees_laissees_en_clair():
    """Un contrôle incapable de signaler quoi que ce soit ne contrôle rien."""
    texte = (
        "Mme Martine Dupont, née le 12/04/1953, tél. 06 11 22 33 44, "
        "jean.martin@chu-exemple.fr, NIR 1 53 04 75 116 001 23, born 1953-04-12."
    )
    trouve = residual_pii(texte)
    attendus = {
        "civilité suivie d'un nom",
        "date de naissance",
        "date de naissance ISO",
        "téléphone français",
        "adresse e-mail",
        "numéro de sécurité sociale",
    }
    assert attendus <= set(trouve), f"non détecté : {attendus - set(trouve)}"


def test_le_controle_ne_signale_rien_sur_un_texte_masque():
    texte = "Mme Martine Dupont, née le 12/04/1953, tél. 06 11 22 33 44, tousse."
    masque, _ = analyze_and_anonymize(texte, "fr")
    assert residual_pii(masque) == {}


def test_le_controle_est_independant_du_detecteur():
    """Les motifs du contrôle ne réutilisent pas ceux du masquage.

    Un contrôle qui interrogerait le même détecteur confirmerait ses propres
    angles morts. Celui-ci est une batterie d'expressions régulières écrite à
    part, et c'est elle qui a signalé que le téléphone international échappait au
    masquage.
    """
    assert len(RESIDUAL_PII_PATTERNS) >= 8
    assert "téléphone international" in RESIDUAL_PII_PATTERNS


def test_l_audit_d_un_corpus_agrege_les_occurrences():
    corpus = [
        "Mme Dupont tousse.",
        "Mme Martin tousse aussi.",
        "Patient de 62 ans, FC 102.",
    ]
    assert audit_corpus(corpus)["civilité suivie d'un nom"] == 2


# --- Pourquoi trois entités sont écartées ---


def test_les_entites_ecartees_detruiraient_le_recit_clinique():
    """La liste `ENTITES_ECARTEES` est une décision, pas une superstition.

    Ce test la justifie sur des phrases fixes : avec ces entités actives,
    Presidio masquerait l'abréviation de la tension artérielle en français,
    et le délai d'évolution comme l'âge du patient en anglais. Trois critères
    de triage sur trois.
    """
    analyseur, _ = _engines()

    def detections(texte: str, langue: str) -> set[str]:
        disponibles = set(analyseur.get_supported_entities(language=langue))
        entites = [e for e in ENTITES_ECARTEES if e in disponibles]
        trouvees = analyseur.analyze(
            text=texte, language=langue, entities=entites, score_threshold=SCORE_THRESHOLD
        )
        return {texte[t.start : t.end] for t in trouvees}

    francais = "Homme de 62 ans, douleur thoracique depuis trois semaines. TA 148/92, FC 102."
    anglais = "62-year-old man, chest pain for three weeks. BP 148/92, HR 102."

    assert "TA" in detections(francais, "fr")
    assert {"62-year-old", "three weeks"} <= detections(anglais, "en")


@pytest.mark.parametrize(
    ("langue", "texte", "survivants"),
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
def test_la_configuration_retenue_laisse_intacts_delai_age_et_constantes(langue, texte, survivants):
    """Ce sont les trois critères sur lesquels se décide un niveau de triage."""
    masque, _ = analyze_and_anonymize(texte, langue)
    for survivant in survivants:
        assert survivant in masque
