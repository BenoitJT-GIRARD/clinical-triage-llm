"""Extraction des cas de triage réellement exploitables dans les corpus publics.

Les corpus médicaux publics ne sont pas des jeux de triage. Les prendre tels
quels — envelopper « Levamisole is used as all except - » dans « Un patient se
présente avec : » puis l'étiqueter parce que le mot « infection » apparaît —
produit des cibles d'entraînement absurdes. Ce module fait le tri.

Deux transformations seulement, l'une et l'autre vérifiables à la lecture :

1. **Vignettes cliniques d'examen** (MedMCQA, FrenchMedMCQA). Une partie des
   questions commence par une véritable présentation de patient : « A 60-year-old
   chronic smoker presents with painless gross hematuria of 1 day duration. » On
   garde cette présentation et on retire la question d'examen qui la suit.
2. **Descriptions de symptômes** (MedQuAD). Les entrées de type `symptoms`
   décrivent les signes d'une pathologie ; on en fait la plainte d'un patient.

L'étiquette vient ensuite de la règle de triage, avec un **niveau de confiance
`moyenne`** qui la distingue des vignettes du catalogue. Une règle de sécurité
encadre cette étiquette : **on ne conserve un cas que si la règle identifie
explicitement un signe**. L'absence de signe détecté ne prouve pas l'absence de
gravité — « suspected pneumoperitoneum » ne contient aucun mot-clé d'alerte et
reste une urgence chirurgicale. Fabriquer une étiquette « consultation
différée » à partir d'un silence de la règle serait dangereux ; les cas non
urgents viennent donc du catalogue, où ils sont décrits et validés un par un.
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

# Une présentation de patient se reconnaît à l'âge, ou à un verbe de présentation.
#
# Le motif français repère une personne suivie de son âge à moins de quarante
# caractères, quelle que soit la ponctuation entre les deux. Exiger la tournure
# « homme **de** 58 ans » écarterait 342 vignettes authentiques de MediQAl, seul
# corpus imposé qui décrive des patients en français, et qui écrit « Homme,
# 58 ans, … » ou « M. Dupont, 30 ans. ». Vérifié sur les trois corpus : cette
# latitude ne gagne aucun faux positif sur MedQuAD ni MedMCQA.
#
# L'abréviation « M. » porte son propre point : elle ne peut pas être suivie
# d'une frontière de mot, d'où son alternative séparée.
PERSONNE = (
    r"(?:\b(?:monsieur|madame|mademoiselle|mme|mlle|melle|mr"
    r"|homme|femme|patiente?|enfant|nourrisson|gar[cç]on|fille|b[eé]b[eé]"
    r"|adolescente?|jeune|nouveau-n[eé])\b|\bm\.)"
)
VIGNETTE_MARKERS = re.compile(
    r"\b\d{1,2}[- ]?(?:year|yr)[- ]?old\b"
    r"|\bpresents? with\b"
    r"|\bbrought to the emergency\b"
    r"|" + PERSONNE + r"[^.]{0,40}?\b\d{1,2}\s+ans?\b",
    re.IGNORECASE,
)

# --- La question d'examen qui suit la vignette, et qu'il faut retirer ---
#
# Un seul motif ne suffit pas, parce qu'un même mot joue deux rôles. « Which »
# ouvre une question posée à l'étudiant, mais sert aussi de pronom relatif au
# milieu d'un récit : « She had flu like symptoms 20 days ago **which** resolved
# spontaneously. » S'en remettre aux seuls mots interrogatifs supprimerait cette
# phrase entière — donc du contenu clinique — et laisserait passer les familles
# d'énoncés les plus courantes de MedMCQA, qui n'emploient aucun mot
# interrogatif : « All of the following ... except », « Most appropriate
# management is ».
#
# D'où deux listes et une règle de position.

# Tournures qui font d'une phrase un énoncé d'examen où qu'elles apparaissent :
# aucune n'a d'usage narratif.
ENONCE_D_EXAMEN = re.compile(
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

# Mots interrogatifs. Ils ne valent que par leur place : en tête de phrase, ou
# dans une phrase que sa ponctuation désigne comme une question. Ailleurs, ce
# sont des relatifs — « la douleur **qui** irradie », « **quel** traitement il
# prend » — et supprimer leur phrase revient à jeter du contenu clinique.
MOT_INTERROGATIF = re.compile(
    r"\b(?:which|what|how|why|when|whom|whose"
    r"|quel|quelle|quels|quelles|laquelle|lequel|lesquels|lesquelles|combien)\b",
    re.IGNORECASE,
)


def est_une_question_d_examen(phrase: str) -> bool:
    """Dit si une phrase est la question posée à l'étudiant plutôt que du récit."""
    texte = phrase.strip()
    if ENONCE_D_EXAMEN.search(texte):
        return True
    motif = MOT_INTERROGATIF.search(texte)
    if motif is None:
        return False
    # Une phrase de récit se termine par un point. Une amorce d'examen, non :
    # « Most appropriate management is », « ... which measure is largest ».
    if not texte.endswith((".", "!")):
        return True
    # Ponctuée en phrase, seule celle qui s'ouvre sur le mot interrogatif est
    # une question.
    return motif.start() == 0


# Passe-partout documentaire de MedQuAD, sans contenu clinique.
MEDQUAD_BOILERPLATE = (
    "human phenotype ontology",
    "medlineplus medical dictionary",
    "the following list of signs and symptoms",
    "you can use the",
    "visiting the following link",
)

# Longueurs acceptables pour une description de patient.
#
# La borne haute n'est pas un chiffre rond. C'est la plus grande valeur pour
# laquelle **aucun** cas retenu ne dépasse le budget de description du service :
# 324 jetons, soit la fenêtre de 768 moins la consigne système et la place
# réservée à la réponse. À la densité mesurée sur les corpus, 800 caractères y
# tiennent dans 100 % des cas, 900 dans 99,3 %, 1 200 dans 92,6 %. S'entraîner
# au-delà apprendrait le modèle sur des récits que le service tronquerait.
# Descendre la borne à 600 écarterait 635 vignettes MediQAl authentiques sans
# que rien ne l'impose. `tests/test_corpus_cases.py` refait ce calcul et échoue
# si la fenêtre du modèle ou le budget de génération changent.
MIN_LENGTH = 60
MAX_LENGTH = 800


@dataclass(frozen=True)
class CorpusCase:
    """Cas de triage dérivé d'un corpus public, avec son étiquette faible."""

    description: str
    user_turn: str
    level: str
    lang: str
    source: str
    topic: str
    symptomes: tuple[str, ...]
    justification: str
    recommandation: str
    confiance: str


def _sentences(text: str) -> list[str]:
    """Découpe un texte en phrases, sur la ponctuation forte."""
    return [phrase.strip() for phrase in re.split(r"(?<=[.!?])\s+", text) if phrase.strip()]


def _recit_de_vignette(question: str) -> str | None:
    """Le récit du patient, la question d'examen retirée, sans contrôle de longueur.

    Séparée de `extract_vignette` pour que `entonnoir` puisse distinguer ce qui
    n'est pas une présentation de patient de ce qui en est une mais sort des
    bornes de longueur. Les deux pertes n'ont pas la même signification.
    """
    if not VIGNETTE_MARKERS.search(question):
        return None
    recit = [p for p in _sentences(question) if not est_une_question_d_examen(p)]
    texte = " ".join(recit).strip(" -–:;,")
    if not texte or not VIGNETTE_MARKERS.search(texte):
        return None
    # La ponctuation forte existante suffit : ajouter un point derrière un
    # point d'interrogation produirait « ?. » en fin de description.
    return texte if texte.endswith((".", "!", "?")) else texte + "."


def extract_vignette(question: str) -> str | None:
    """Isole la présentation du patient dans une question d'examen clinique.

    On retire les phrases qui sont la question posée à l'étudiant et on ne garde
    que le récit. Si ce qu'il reste ne ressemble plus à une présentation de
    patient, ou s'il sort des bornes de longueur, on renonce.
    """
    texte = _recit_de_vignette(question)
    if texte is None or not MIN_LENGTH <= len(texte) <= MAX_LENGTH:
        return None
    return texte


def extract_symptom_description(entry: CorpusEntry) -> str | None:
    """Transforme une fiche de symptômes MedQuAD en plainte de patient."""
    if entry.topic != "symptoms":
        return None
    reponse = entry.answer
    minuscule = reponse.lower()
    if any(passe_partout in minuscule for passe_partout in MEDQUAD_BOILERPLATE):
        return None
    # La réponse rappelle souvent la question avant de répondre : on la retire.
    phrases = [p for p in _sentences(reponse) if not p.lower().startswith("what are")]
    # On empile des phrases entières tant qu'elles tiennent dans la borne : une
    # cible d'entraînement doit être une phrase entière, alors que couper la
    # chaîne à `[:MAX_LENGTH]` trancherait en plein mot, sur « Symptoms in
    # Toddle » ou « a transient is ».
    #
    # Le préambule compte dans la borne : c'est lui qui est livré au modèle, et
    # le laisser hors du calcul livrerait des descriptions plus longues que
    # MAX_LENGTH, si bien que la borne publiée ne décrirait pas le jeu.
    pathologie = entry.text.replace("What are the symptoms of", "").strip(" ?")
    preambule = f"Patient reporting the following complaints, possibly related to {pathologie}: "
    budget = MAX_LENGTH - len(preambule)
    if budget < MIN_LENGTH:
        return None

    texte = ""
    for phrase in phrases:
        candidat = f"{texte} {phrase}" if texte else phrase
        if len(candidat) > budget:
            break
        texte = candidat
    texte = texte.strip()
    if len(texte) < MIN_LENGTH:
        return None
    return preambule + texte


def _justification(level: str, signes: tuple[str, ...]) -> str:
    """Rédige une justification adossée aux signes effectivement repérés."""
    liste = ", ".join(signes)
    if level == VITAL:
        return (
            f"Signe(s) de détresse vitale identifié(s) dans la description : {liste}. "
            "Toute suspicion de détresse vitale impose une prise en charge immédiate."
        )
    return (
        f"Signe(s) d'alerte identifié(s) dans la description : {liste}. "
        "Une évaluation médicale rapprochée est justifiée, sans critère de détresse vitale immédiat."
    )


def build_corpus_case(
    description: str, entry: CorpusEntry, rng: random.Random
) -> CorpusCase | None:
    """Étiquette une description par la règle, et ne la garde que si un signe est trouvé."""

    level = classify(description)
    if level == DEFERRED:
        # Aucun signe repéré : on ne fabrique pas une étiquette « non urgent »
        # à partir du silence de la règle. Le cas est écarté.
        return None
    signes = tuple(matched_flags(description, VITAL if level == VITAL else MODERATE))
    if not signes:
        return None
    return CorpusCase(
        description=description,
        user_turn=rng.choice(USER_TEMPLATES[entry.lang]).format(description=description),
        level=level,
        lang=entry.lang,
        source=entry.source,
        topic=entry.topic,
        symptomes=signes,
        justification=_justification(level, signes),
        recommandation=recommendation_for(level),
        confiance="moyenne",
    )


def extract_cases(entries: list[CorpusEntry], rng: random.Random) -> list[CorpusCase]:
    """Parcourt les entrées d'un corpus et renvoie les cas de triage exploitables."""
    cases: list[CorpusCase] = []
    descriptions_vues: set[str] = set()
    for entry in entries:
        description = extract_vignette(entry.text) or extract_symptom_description(entry)
        if description is None or description in descriptions_vues:
            continue
        case = build_corpus_case(description, entry, rng)
        if case is None:
            continue
        descriptions_vues.add(description)
        cases.append(case)
    return cases


def entonnoir(entries: list[CorpusEntry]) -> dict[str, int]:
    """Compte les entrées perdues à chaque étape de l'extraction.

    Le rendement global d'un corpus — tant d'entrées lues, tant de cas retenus —
    ne dit pas *où* les entrées sont perdues, et donc ne dit pas si ce qui les
    écarte tient au corpus ou au filtre. Cette ventilation le dit, et c'est elle
    que publie la carte du dataset.

    Deux étapes n'ont pas le même statut. « Pas une présentation de patient » et
    « trop long ou trop court » sont des décisions de **forme**, révisables. « La
    règle n'identifie aucun signe » est une décision de **sécurité** : on refuse
    d'étiqueter « non urgent » à partir du silence d'une règle à mots-clés, et
    cette part-là ne se récupère pas en élargissant un motif.
    """
    compte = {
        "entrees": len(entries),
        "sans_presentation_de_patient": 0,
        "hors_bornes_de_longueur": 0,
        "sans_signe_identifie": 0,
        "doublons": 0,
        "retenus": 0,
    }
    vues: set[str] = set()
    rng = random.Random(0)
    for entry in entries:
        # `extract_cases` écrit `extract_vignette(...) or extract_symptom_description(...)` :
        # une vignette hors bornes ne condamne pas l'entrée, la fiche de symptômes
        # peut encore la rattraper. L'entonnoir doit suivre le même chemin, sans
        # quoi il impute à « hors bornes » des entrées que l'extraction retient.
        recit = _recit_de_vignette(entry.text)
        hors_bornes = recit is not None and not MIN_LENGTH <= len(recit) <= MAX_LENGTH
        description = None if hors_bornes else recit
        if description is None:
            description = extract_symptom_description(entry)
        if description is None:
            compte[
                "hors_bornes_de_longueur" if hors_bornes else "sans_presentation_de_patient"
            ] += 1
            continue
        if description in vues:
            compte["doublons"] += 1
            continue
        # `extract_cases` n'enregistre une description qu'une fois le cas
        # conservé : une description rejetée par la règle et rencontrée deux
        # fois compte deux fois comme « sans signe », jamais comme doublon.
        if build_corpus_case(description, entry, rng) is None:
            compte["sans_signe_identifie"] += 1
            continue
        vues.add(description)
        compte["retenus"] += 1
    return compte
