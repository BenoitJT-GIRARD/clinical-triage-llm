"""Building the preference pairs for the DPO alignment.

DPO teaches the model to prefer one answer over another. Everything therefore depends on what
differs between the two: the model learns the difference it finds, not the one that was
intended. Opposing a good three-line answer to an evasive half-line rejected answer would make
length the only systematic difference in the set: the model would learn "longer is better" and
stop emitting its end token.

Three rules frame the construction of the pairs:

1. **same format, same length.** The rejected answer respects the three-line structure. Each
   defect comes in several lengths, and the one closest to the preferred answer is kept: the
   difference between the two then lies in the substance, and in nothing else.
2. **never an escalation as the rejected answer.** The system prompt requires escalating at the
   slightest doubt. Opposing an "over-cautious" answer as a bad example would teach exactly the
   opposite. Rejected answers are undertriages, recommendations that delay care, asserted
   diagnoses, or answers outside the language contract.
3. **serious cases weigh more.** Undertriaging a life-threatening case is the costliest
   failure: those cases are over-represented in the set.

The pairs are built from the **training split only**: a prompt from the test split or from the
clinical evaluation set must never appear here.

The answer texts are French, as the model's output is: they are the data.
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

# The level immediately below, used to manufacture an undertriage.
LOWER_LEVEL = {
    "URGENCE_VITALE": "URGENCE_MODEREE",
    "URGENCE_MODEREE": "CONSULTATION_DIFFEREE",
}

# Minimising justifications. The lengths are deliberately varied, from shorter to longer than
# the real justifications: if every rejected answer were shorter, DPO would learn length.
MINIMISING_JUSTIFICATIONS = (
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

# Recommendations that delay care: the announced level is right, but the course of action
# contradicts the urgency.
UNSAFE_RECOMMENDATIONS = (
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

# Asserted diagnoses: the agent must provide decision support, never a definitive diagnosis.
ASSERTED_DIAGNOSES = (
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

# Answer outside the contract: acceptable substance, but written in English when the system
# prompt requires French. Two lengths, for the selection below.
ENGLISH_ANSWERS = {
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
    """A preference pair ready for DPO training."""

    user_turn: str
    chosen: str
    rejected: str
    level: str
    lang: str
    strategy: str
    source: str


# Each builder returns several candidates of different lengths. The pair builder keeps the one
# whose length is closest to the preferred answer.


def _undertriage(example: TriageExample, rng: random.Random) -> list[str]:
    """Answers one level below, with a minimising justification.

    From a life-threatening case, the draw picks between dropping one step and dropping two: the
    second is the most serious failure, and it must be present in the preference set too.
    """
    lower = LOWER_LEVEL.get(example.level)
    if lower is None:
        return []
    if example.level == "URGENCE_VITALE" and rng.random() < 0.4:
        lower = "CONSULTATION_DIFFEREE"
    return [
        build_target_response(lower, justification, recommendation_for(lower))
        for justification in MINIMISING_JUSTIFICATIONS
    ]


def _unsafe_recommendation(example: TriageExample, rng: random.Random) -> list[str]:
    """Right level announced, but a course of action that delays care."""
    justification = example.assistant_turn.split("\n")[1].replace("Justification : ", "")
    return [
        build_target_response(example.level, justification, recommendation)
        for recommendation in UNSAFE_RECOMMENDATIONS
    ]


def _asserted_diagnosis(example: TriageExample, rng: random.Random) -> list[str]:
    """Right level, but a diagnosis presented as certain."""
    return [
        build_target_response(example.level, diagnosis, recommendation_for(example.level))
        for diagnosis in ASSERTED_DIAGNOSES
    ]


def _answered_in_english(example: TriageExample, rng: random.Random) -> list[str]:
    """Acceptable content, but a language and labels outside the output contract."""
    return list(ENGLISH_ANSWERS[example.level])


# Strategies applicable per level. Undertriage does not exist for the lowest level, and
# escalation is never a rejected answer.
STRATEGIES = {
    "URGENCE_VITALE": (
        "undertriage",
        "unsafe_recommendation",
        "asserted_diagnosis",
        "answered_in_english",
    ),
    "URGENCE_MODEREE": (
        "undertriage",
        "unsafe_recommendation",
        "asserted_diagnosis",
        "answered_in_english",
    ),
    "CONSULTATION_DIFFEREE": (
        "unsafe_recommendation",
        "asserted_diagnosis",
        "answered_in_english",
    ),
}

BUILDERS = {
    "undertriage": _undertriage,
    "unsafe_recommendation": _unsafe_recommendation,
    "asserted_diagnosis": _asserted_diagnosis,
    "answered_in_english": _answered_in_english,
}

# Draw weight per level: undertriaging a serious case is the costliest failure, so those cases
# are over-represented.
WEIGHT_PER_LEVEL = {"URGENCE_VITALE": 3, "URGENCE_MODEREE": 2, "CONSULTATION_DIFFEREE": 1}


def build_preference_pairs(
    train_examples: list[TriageExample],
    target_size: int,
    rng: random.Random,
) -> list[PreferencePairRecord]:
    """Build the clinical-safety preference pairs."""
    weighted: list[TriageExample] = []
    for example in train_examples:
        weighted.extend([example] * WEIGHT_PER_LEVEL[example.level])
    rng.shuffle(weighted)

    pairs: list[PreferencePairRecord] = []
    seen_turns: set[str] = set()
    for example in weighted:
        if len(pairs) >= target_size:
            break
        if example.user_turn in seen_turns:
            continue
        strategy = rng.choice(STRATEGIES[example.level])
        candidates = [
            candidate
            for candidate in BUILDERS[strategy](example, rng)
            if candidate != example.assistant_turn
        ]
        if not candidates:
            continue
        # Keep the two candidates closest in length, then draw between them: always keeping the
        # closest would tilt the rejected answers to the same side of the preferred one.
        candidates.sort(key=lambda c: abs(len(c) - len(example.assistant_turn)))
        rejected = rng.choice(candidates[:2])
        seen_turns.add(example.user_turn)
        pairs.append(
            PreferencePairRecord(
                user_turn=example.user_turn,
                chosen=example.assistant_turn,
                rejected=rejected,
                level=example.level,
                lang=example.lang,
                strategy=strategy,
                source="safety_preference",
            )
        )

    distribution = {level: sum(1 for p in pairs if p.level == level) for level in TRIAGE.levels}
    logger.info("Preference pairs: %d (%s)", len(pairs), distribution)
    return pairs


def length_balance(pairs: list[PreferencePairRecord]) -> dict[str, float]:
    """Measure the length gap between preferred and rejected answers.

    A systematic gap would be the signal that DPO risks learning length rather than substance:
    this measurement is published in the dataset metadata so that the check is verifiable by a
    reader.
    """
    if not pairs:
        return {"chosen_mean_chars": 0.0, "rejected_mean_chars": 0.0, "chosen_longer_share": 0.0}
    chosen_lengths = [len(p.chosen) for p in pairs]
    rejected_lengths = [len(p.rejected) for p in pairs]
    longer = sum(c > r for c, r in zip(chosen_lengths, rejected_lengths, strict=True))
    return {
        "chosen_mean_chars": round(sum(chosen_lengths) / len(pairs), 1),
        "rejected_mean_chars": round(sum(rejected_lengths) / len(pairs), 1),
        "chosen_longer_share": round(longer / len(pairs), 3),
    }
