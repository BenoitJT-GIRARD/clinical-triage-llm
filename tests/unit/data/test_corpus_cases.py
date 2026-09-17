"""Tests des filtres d'extraction appliqués aux corpus publics.

Ils verrouillent des défauts qui ont réellement amputé le dataset livré : un
marqueur aveugle à la syntaxe de MediQAl, une troncature en plein mot, et une
borne de longueur posée au jugé plutôt que sur le budget du service. Les
derniers vérifient que l'entonnoir publié dans la carte du dataset décrit bien
ce que l'extraction fait, et non ce qu'on croit qu'elle fait.
"""

from __future__ import annotations

import random

import pytest

from clinical_triage.config import MODEL, SERVING
from clinical_triage.data.corpus_cases import (
    MAX_LENGTH,
    MIN_LENGTH,
    entonnoir,
    est_une_question_d_examen,
    extract_cases,
    extract_symptom_description,
    extract_vignette,
)
from clinical_triage.data.corpus_sources import CorpusEntry

# --- Le marqueur doit reconnaître la syntaxe réellement employée par MediQAl ---


@pytest.mark.parametrize(
    "ouverture",
    [
        "Homme, 58 ans, majoration de dyspnée chez un BPCO connu depuis vingt ans.",
        "M. Dupont, 30 ans. Agitation à la sortie d'une boîte de nuit, amené par les pompiers.",
        "Mme Barbie, 72 ans, percutée par un bus alors qu'elle circulait à vélo sans casque.",
        "Mme N, âgée de 35 ans, présente en quelques jours une chute de la paupière gauche.",
        "Un enfant de 2 ans est adressé pour des otites moyennes aiguës récidivantes.",
        "Une patiente de 25 ans consulte pour une pâleur et une asthénie d'installation récente.",
    ],
)
def test_les_tournures_francaises_de_presentation_sont_reconnues(ouverture: str):
    """MediQAl écrit « Homme, 58 ans », pas « homme de 58 ans ».

    Le motif exigeait la préposition. 342 vignettes françaises authentiques
    n'étaient donc pas même examinées, sur le seul corpus imposé qui décrive
    des patients.
    """
    assert extract_vignette(ouverture) is not None


def test_un_enonce_sans_patient_reste_ecarte():
    """L'élargissement du marqueur ne doit pas ouvrir la porte aux questions d'examen."""
    assert extract_vignette("Parmi les propositions suivantes, laquelle est exacte ?") is None
    assert extract_vignette("Levamisole is used as all except -") is None


# --- Une cible d'entraînement se termine sur une phrase entière ---


def test_une_fiche_de_symptomes_trop_longue_est_coupee_a_la_phrase():
    """La troncature se faisait à `[:MAX_LENGTH]`, donc en plein mot.

    120 des 172 cas MedQuAD livrés se terminaient sur « Symptoms in Toddle » ou
    « a transient is » : du texte coupé au milieu d'un mot, appris comme cible.
    """
    phrase = "The patient reports abdominal cramping and persistent nausea after meals. "
    entree = CorpusEntry(
        text="What are the symptoms of Gastroparesis ?",
        answer=phrase * 30,
        lang="en",
        source="medquad",
        topic="symptoms",
    )
    description = extract_symptom_description(entree)
    assert description is not None
    assert len(description) <= MAX_LENGTH + len("Patient reporting the following complaints, ")
    # Le texte retenu se termine sur une ponctuation forte, jamais sur un mot coupé.
    assert description.rstrip().endswith(".")


def test_une_fiche_de_symptomes_trop_courte_est_ecartee():
    entree = CorpusEntry(
        text="What are the symptoms of X ?",
        answer="Rare.",
        lang="en",
        source="medquad",
        topic="symptoms",
    )
    assert extract_symptom_description(entree) is None


# --- La borne de longueur est celle du service, pas un chiffre rond ---


def test_la_borne_de_longueur_tient_dans_le_budget_du_service():
    """Aucun cas retenu ne doit dépasser ce que le service accepte.

    Le service laisse à la description ce que la fenêtre du modèle lui laisse une
    fois retirées la consigne système et la place réservée à la réponse.
    S'entraîner au-delà apprendrait le modèle sur des récits qu'il ne verra
    jamais entiers en production.

    Ce test échoue si la fenêtre du modèle, le budget de génération ou la borne
    changent sans qu'on ait refait le calcul.
    """
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL.base_model)
    except Exception as exc:  # noqa: BLE001 - hors ligne, le test n'a rien à dire
        pytest.skip(f"tokenizer indisponible : {exc}")

    from clinical_triage.prompts import description_budget

    budget = description_budget(tokenizer, MODEL.max_seq_length, SERVING.max_new_tokens)
    # Densité mesurée sur les 546 descriptions retenues de MediQAl et MedQuAD :
    # médiane 0,287 jeton par caractère, 95e centile 0,375. La description la plus
    # longue du jeu livré occupe 312 jetons. Le contrôle à la construction
    # (`scripts/build_dataset.py`) vérifie le cas réel ; celui-ci garde la
    # cohérence du réglage hors ligne.
    jetons_au_95e_centile = MAX_LENGTH * 0.375
    assert jetons_au_95e_centile <= budget, (
        f"MAX_LENGTH={MAX_LENGTH} produit environ {jetons_au_95e_centile:.0f} jetons "
        f"au 95e centile, pour un budget de service de {budget}."
    )
    assert MIN_LENGTH < MAX_LENGTH


# --- L'entonnoir doit décrire exactement ce que fait l'extraction ---


def _entree(texte: str) -> CorpusEntry:
    return CorpusEntry(text=texte, answer="", lang="en", source="medmcqa", topic="")


def _corpus_temoin() -> list[CorpusEntry]:
    """Un corpus miniature qui exerce chacune des quatre pertes, et un cas retenu."""
    return [
        # Ni âge, ni verbe de présentation : ce n'est pas un patient.
        _entree("Which vitamin is supplied from only animal source:"),
        # Une présentation de patient, mais bien au-delà de la borne haute.
        _entree("A 55-year-old man presents with chest pain. " + "Il décrit la douleur. " * 60),
        # Une présentation de patient sans aucun signe repérable par la règle.
        _entree("A 40-year-old man presents for a routine administrative certificate request."),
        # Un cas conservable, présent deux fois : la seconde est un doublon.
        _entree("A 55-year-old man presents with chest pain radiating to the left arm."),
        _entree("A 55-year-old man presents with chest pain radiating to the left arm."),
    ]


def test_les_colonnes_de_l_entonnoir_s_additionnent_avec_les_entrees():
    """C'est la propriété que le rapport publie : rien ne se perd hors des colonnes."""
    entrees = _corpus_temoin()
    compte = entonnoir(entrees)
    somme = (
        compte["sans_presentation_de_patient"]
        + compte["hors_bornes_de_longueur"]
        + compte["sans_signe_identifie"]
        + compte["doublons"]
        + compte["retenus"]
    )
    assert somme == compte["entrees"] == len(entrees)


def test_l_entonnoir_retrouve_le_compte_de_l_extraction():
    """Deux parcours, un seul résultat : sinon le tableau publié décrit autre chose."""
    entrees = _corpus_temoin()
    assert entonnoir(entrees)["retenus"] == len(extract_cases(entrees, random.Random(42)))


def test_chaque_perte_est_imputee_a_la_bonne_colonne():
    compte = entonnoir(_corpus_temoin())
    assert compte["sans_presentation_de_patient"] == 1
    assert compte["hors_bornes_de_longueur"] == 1
    assert compte["sans_signe_identifie"] == 1
    assert compte["doublons"] == 1
    assert compte["retenus"] == 1


# --- Question d'examen contre pronom relatif ---


@pytest.mark.parametrize(
    "enonce",
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
def test_un_enonce_d_examen_est_reconnu(enonce: str):
    assert est_une_question_d_examen(enonce)


@pytest.mark.parametrize(
    "recit",
    [
        "She had flu like symptoms 20 days ago which resolved spontaneously.",
        "There is circumoral cyanosis, which is not alleviated by nasal oxygen.",
        "A 50-year-old lady presented with a lump in the left breast, which developed suddenly.",
        "Il décrit une douleur qui irradie vers la mâchoire.",
        "Le patient ne sait plus quel traitement il prend.",
    ],
)
def test_une_phrase_de_recit_n_est_pas_prise_pour_une_question(recit: str):
    """Le motif précédent supprimait la phrase entière sur un « which » relatif.

    C'était du contenu clinique perdu : « symptômes grippaux il y a vingt jours »
    disparaissait du récit parce que la phrase contenait un pronom relatif.
    """
    assert not est_une_question_d_examen(recit)


def test_la_question_d_examen_est_retiree_mais_le_recit_reste():
    question = (
        "A 60 yr old chronic smoker presents with painless gross hematuria of 1 day duration, "
        "which started this morning. All of the following are possible causes except -"
    )
    vignette = extract_vignette(question)
    assert vignette is not None
    assert "All of the following" not in vignette
    # La phrase de récit porte un « which » relatif : elle doit survivre.
    assert "which started this morning" in vignette
