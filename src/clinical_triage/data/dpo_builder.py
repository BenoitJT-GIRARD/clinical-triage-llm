"""Construction des paires de préférences pour l'alignement DPO.

Le DPO apprend au modèle à préférer une réponse à une autre. Tout dépend donc de
ce que les deux réponses ont de différent : le modèle apprend la différence qu'il
trouve, pas celle qu'on avait en tête. Opposer une bonne réponse de trois lignes
à une réponse rejetée évasive d'une demi-ligne ferait de la longueur la seule
différence systématique du jeu : le modèle apprendrait « plus long vaut mieux »
et cesserait d'émettre son jeton de fin.

Trois règles encadrent donc la construction des paires :

1. **même format, même longueur.** La réponse rejetée respecte la structure en
   trois lignes. Chaque défaut est décliné en plusieurs longueurs, et l'on retient
   celle qui colle au plus près de la réponse préférée : la différence entre les
   deux porte alors sur le fond, et sur rien d'autre.
2. **jamais de surclassement en réponse rejetée.** La consigne système impose de
   surclasser au moindre doute. Opposer une réponse « trop prudente » comme
   mauvais exemple apprendrait exactement l'inverse. Les réponses rejetées sont
   des sous-triages, des recommandations qui retardent la prise en charge, des
   diagnostics affirmés ou des réponses hors contrat de langue.
3. **les cas graves pèsent plus.** Le sous-triage d'une urgence vitale est la
   faute la plus coûteuse : ces cas sont sur-représentés dans le jeu.

Les paires sont construites à partir du **seul jeu d'entraînement** : un prompt
du jeu de test ou du jeu d'évaluation clinique ne doit jamais apparaître ici.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from clinical_triage.config import TRIAGE
from clinical_triage.data.sft_builder import TriageExample
from clinical_triage.data.triage_rules import recommendation_for
from clinical_triage.prompts import build_target_response
from clinical_triage.utils import get_logger

logger = get_logger(__name__)

# Niveau immédiatement inférieur, utilisé pour fabriquer un sous-triage.
NIVEAU_INFERIEUR = {
    "URGENCE_VITALE": "URGENCE_MODEREE",
    "URGENCE_MODEREE": "CONSULTATION_DIFFEREE",
}

# Justifications minimisantes. Les longueurs sont volontairement variées, des
# plus courtes aux plus longues que les vraies justifications : si toutes les
# réponses rejetées étaient plus courtes, le DPO apprendrait la longueur.
JUSTIFICATIONS_MINIMISANTES = (
    "Rien d'inquiétant dans ce qui est décrit.",
    (
        "Les éléments rapportés paraissent rester dans les limites de la normale et ne "
        "réunissent pas, en l'état, de critère imposant une prise en charge rapprochée."
    ),
    (
        "La description ne comporte pas d'élément suffisamment inquiétant pour justifier "
        "une prise en charge accélérée ; l'évolution peut être surveillée à distance."
    ),
    (
        "Le tableau semble relever d'une situation banale et fréquente, du type de celles "
        "que l'on voit quotidiennement à l'accueil sans qu'elles ne débouchent sur quoi que "
        "ce soit de sérieux ; aucun des signes rapportés ne paraît devoir modifier le délai "
        "habituel de prise en charge, ni justifier d'examen complémentaire immédiat."
    ),
)

# Recommandations qui retardent la prise en charge : le niveau annoncé est le bon,
# mais la conduite à tenir contredit l'urgence.
RECOMMANDATIONS_DANGEREUSES = (
    "Faire patienter en salle d'attente.",
    (
        "Faire patienter le patient en salle d'attente et le réévaluer si son état se "
        "dégrade franchement ; une consultation de ville suffira sinon dans la semaine."
    ),
    (
        "Proposer au patient de rentrer chez lui et de revenir demain matin si les "
        "symptômes n'ont pas cédé ; aucune surveillance particulière n'est nécessaire."
    ),
    (
        "Donner un antalgique simple et laisser le patient repartir sans examen "
        "complémentaire ni surveillance ; il consultera son médecin traitant à l'occasion, "
        "en prenant rendez-vous dans les prochaines semaines selon ses disponibilités, et "
        "reviendra de lui-même si quelque chose venait à changer nettement."
    ),
)

# Diagnostics affirmés : l'agent doit fournir une aide à la décision, jamais un
# diagnostic définitif.
DIAGNOSTICS_AFFIRMES = (
    "Le diagnostic est certain.",
    (
        "Il s'agit d'un infarctus du myocarde constitué ; le diagnostic est certain et ne "
        "nécessite aucun examen complémentaire pour être retenu."
    ),
    (
        "Le diagnostic est établi : c'est une infection virale bénigne, sans qu'aucun autre "
        "examen ne soit nécessaire pour l'affirmer avec certitude."
    ),
    (
        "Il s'agit assurément d'une crise d'angoisse, comme le montre l'ensemble du tableau "
        "clinique décrit ici, qui ne laisse place à aucune autre hypothèse raisonnable ; "
        "toute recherche de cause organique est inutile et le patient peut être rassuré "
        "définitivement, sans surveillance particulière ni consultation de contrôle."
    ),
)

# Réponse hors contrat : bon fond, mais rédigée en anglais alors que la consigne
# système impose le français. Deux longueurs, pour la sélection ci-dessous.
REPONSES_EN_ANGLAIS = {
    "URGENCE_VITALE": (
        (
            "Priority level: LIFE-THREATENING EMERGENCY\n"
            "Reasoning: the findings point to an immediately life-threatening condition.\n"
            "Recommendation: immediate assessment and a call to the emergency services."
        ),
        (
            "Priority level: LIFE-THREATENING EMERGENCY\n"
            "Reasoning: the combination of findings reported in this presentation points to a "
            "condition that is immediately life-threatening, and the outcome depends directly on "
            "how quickly the patient is assessed and treated.\n"
            "Recommendation: immediate assessment by the emergency physician in the resuscitation "
            "area, continuous monitoring of vital signs, and a call to the emergency services."
        ),
    ),
    "URGENCE_MODEREE": (
        (
            "Priority level: MODERATE EMERGENCY\n"
            "Reasoning: the findings justify a medical assessment within a few hours.\n"
            "Recommendation: arrange a review within a few hours and watch for deterioration."
        ),
        (
            "Priority level: MODERATE EMERGENCY\n"
            "Reasoning: the findings reported here justify a medical assessment within the next few "
            "hours, but none of them indicates an immediately life-threatening condition that would "
            "require resuscitation.\n"
            "Recommendation: arrange a medical review within a few hours, provide analgesia as "
            "needed, and monitor closely for any deterioration that would change the priority."
        ),
    ),
    "CONSULTATION_DIFFEREE": (
        (
            "Priority level: DEFERRED CONSULTATION\n"
            "Reasoning: no criterion of immediate severity is present in this description.\n"
            "Recommendation: arrange a scheduled appointment and give safety advice."
        ),
        (
            "Priority level: DEFERRED CONSULTATION\n"
            "Reasoning: no criterion of immediate severity is present in this description, and the "
            "situation appears to be one that can safely be managed outside the emergency "
            "department by the general practitioner.\n"
            "Recommendation: arrange a scheduled appointment with the general practitioner, give "
            "clear written safety advice, and explain which warning signs should prompt a return."
        ),
    ),
}


@dataclass(frozen=True)
class PreferencePairRecord:
    """Paire de préférence prête pour l'entraînement DPO."""

    user_turn: str
    chosen: str
    rejected: str
    level: str
    lang: str
    strategie: str
    source: str


# Chaque constructeur renvoie plusieurs candidats de longueurs différentes. Le
# constructeur de paires retient celui dont la longueur est la plus proche de la
# réponse préférée.


def _sous_triage(example: TriageExample, rng: random.Random) -> list[str]:
    """Réponses à un niveau inférieur, avec une justification minimisante.

    Depuis une urgence vitale, on tire au sort entre la sous-évaluation d'un cran
    et celle de deux crans : la seconde est la faute la plus grave, elle doit
    aussi être présente dans le jeu de préférences.
    """
    inferieur = NIVEAU_INFERIEUR.get(example.level)
    if inferieur is None:
        return []
    if example.level == "URGENCE_VITALE" and rng.random() < 0.4:
        inferieur = "CONSULTATION_DIFFEREE"
    return [
        build_target_response(inferieur, justification, recommendation_for(inferieur))
        for justification in JUSTIFICATIONS_MINIMISANTES
    ]


def _recommandation_dangereuse(example: TriageExample, rng: random.Random) -> list[str]:
    """Bon niveau annoncé, mais conduite à tenir qui retarde la prise en charge."""
    justification = example.assistant_turn.split("\n")[1].replace("Justification : ", "")
    return [
        build_target_response(example.level, justification, recommandation)
        for recommandation in RECOMMANDATIONS_DANGEREUSES
    ]


def _diagnostic_affirme(example: TriageExample, rng: random.Random) -> list[str]:
    """Bon niveau, mais diagnostic présenté comme certain."""
    return [
        build_target_response(example.level, diagnostic, recommendation_for(example.level))
        for diagnostic in DIAGNOSTICS_AFFIRMES
    ]


def _reponse_en_anglais(example: TriageExample, rng: random.Random) -> list[str]:
    """Contenu acceptable, mais langue et libellés hors du contrat de sortie."""
    return list(REPONSES_EN_ANGLAIS[example.level])


# Stratégies applicables selon le niveau. Le sous-triage n'existe pas pour le
# niveau le plus bas, et le surclassement n'est jamais une réponse rejetée.
STRATEGIES = {
    "URGENCE_VITALE": (
        "sous_triage",
        "recommandation_dangereuse",
        "diagnostic_affirme",
        "reponse_en_anglais",
    ),
    "URGENCE_MODEREE": (
        "sous_triage",
        "recommandation_dangereuse",
        "diagnostic_affirme",
        "reponse_en_anglais",
    ),
    "CONSULTATION_DIFFEREE": (
        "recommandation_dangereuse",
        "diagnostic_affirme",
        "reponse_en_anglais",
    ),
}

CONSTRUCTEURS = {
    "sous_triage": _sous_triage,
    "recommandation_dangereuse": _recommandation_dangereuse,
    "diagnostic_affirme": _diagnostic_affirme,
    "reponse_en_anglais": _reponse_en_anglais,
}

# Poids de tirage par niveau : le sous-triage d'un cas grave est la faute la plus
# coûteuse, ces cas sont donc sur-représentés.
POIDS_PAR_NIVEAU = {"URGENCE_VITALE": 3, "URGENCE_MODEREE": 2, "CONSULTATION_DIFFEREE": 1}


def build_preference_pairs(
    train_examples: list[TriageExample],
    target_size: int,
    rng: random.Random,
) -> list[PreferencePairRecord]:
    """Construit les paires de préférence de sécurité clinique."""
    ponderes: list[TriageExample] = []
    for example in train_examples:
        ponderes.extend([example] * POIDS_PAR_NIVEAU[example.level])
    rng.shuffle(ponderes)

    paires: list[PreferencePairRecord] = []
    tours_vus: set[str] = set()
    for example in ponderes:
        if len(paires) >= target_size:
            break
        if example.user_turn in tours_vus:
            continue
        strategie = rng.choice(STRATEGIES[example.level])
        candidats = [
            candidat
            for candidat in CONSTRUCTEURS[strategie](example, rng)
            if candidat != example.assistant_turn
        ]
        if not candidats:
            continue
        # On garde les deux candidats les plus proches en longueur, puis on tire
        # entre les deux : retenir systématiquement le plus proche ferait pencher
        # les réponses rejetées toujours du même côté de la réponse préférée.
        candidats.sort(key=lambda c: abs(len(c) - len(example.assistant_turn)))
        rejected = rng.choice(candidats[:2])
        tours_vus.add(example.user_turn)
        paires.append(
            PreferencePairRecord(
                user_turn=example.user_turn,
                chosen=example.assistant_turn,
                rejected=rejected,
                level=example.level,
                lang=example.lang,
                strategie=strategie,
                source="preference_securite",
            )
        )

    repartition = {niveau: sum(1 for p in paires if p.level == niveau) for niveau in TRIAGE.levels}
    logger.info("Paires de préférence : %d (%s)", len(paires), repartition)
    return paires


def length_balance(pairs: list[PreferencePairRecord]) -> dict[str, float]:
    """Mesure l'écart de longueur entre réponses préférées et rejetées.

    Un écart systématique serait le signal que le DPO risque d'apprendre la
    longueur plutôt que le fond : cette mesure est publiée dans les métadonnées
    du dataset pour que le contrôle soit vérifiable par un lecteur.
    """
    if not pairs:
        return {"chosen_moyenne": 0.0, "rejected_moyenne": 0.0, "part_chosen_plus_long": 0.0}
    longueurs_chosen = [len(p.chosen) for p in pairs]
    longueurs_rejected = [len(p.rejected) for p in pairs]
    plus_longues = sum(c > r for c, r in zip(longueurs_chosen, longueurs_rejected, strict=True))
    return {
        "chosen_moyenne": round(sum(longueurs_chosen) / len(pairs), 1),
        "rejected_moyenne": round(sum(longueurs_rejected) / len(pairs), 1),
        "part_chosen_plus_long": round(plus_longues / len(pairs), 3),
    }
