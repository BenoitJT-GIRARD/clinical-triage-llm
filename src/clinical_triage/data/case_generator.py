"""Generating clinical vignettes from the presentation catalogue.

A vignette is a typical presentation from the catalogue dressed as a patient: an age, a sex, a
medical history, an onset delay, a set of vital signs and a wording. The triage level comes
from the source presentation — never from re-reading the produced text. That is what makes the
evaluation honest: the model cannot settle for relearning a lexical rule, since no lexical rule
was used to label.

Every vignette carries its metadata: symptoms, history, vital signs, source and confidence
level.

The wordings are French and English clinical text: they are the data being generated.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from clinical_triage.data.clinical_catalogue import (
    Presentation,
    altered_consciousness_likely,
    forced_vitals,
    presentations_by_level,
)
from clinical_triage.data.vital_signs import VitalSigns
from clinical_triage.data.vital_signs import generate as generate_vitals

# Presentations specific to women: the sex draw must take them into account.
FEMALE_ONLY_PRESENTATIONS = frozenset(
    {"pre_eclampsie_severe", "hemorragie_du_post_partum", "cystite_simple"}
)

# Wordings of the onset delay, by time scale.
ONSET_PHRASINGS = {
    "minutes": {
        "fr": (
            "depuis dix minutes",
            "depuis une demi-heure",
            "depuis vingt minutes",
            "brutalement il y a un quart d'heure",
        ),
        "en": (
            "for ten minutes",
            "for half an hour",
            "for twenty minutes",
            "suddenly a quarter of an hour ago",
        ),
    },
    "heures": {
        "fr": ("depuis deux heures", "depuis ce matin", "depuis cette nuit", "depuis six heures"),
        "en": ("for two hours", "since this morning", "since last night", "for six hours"),
    },
    "jours": {
        "fr": ("depuis deux jours", "depuis hier", "depuis trois jours", "depuis quatre jours"),
        "en": ("for two days", "since yesterday", "for three days", "for four days"),
    },
    "semaines": {
        "fr": (
            "depuis trois semaines",
            "depuis un mois",
            "depuis plusieurs semaines",
            "depuis six semaines",
        ),
        "en": ("for three weeks", "for a month", "for several weeks", "for six weeks"),
    },
}

# Templates for the patient description. The fields are filled in below.
DESCRIPTION_TEMPLATES = {
    "fr": (
        "{who}, {complaint} {onset}, avec {signs}.{history}{vitals}",
        "Motif d'admission : {complaint} {onset}. Patient : {who}. Signes associés : {signs}.{history}{vitals}",
        "{who} se présente pour {complaint} {onset}. On note {signs}.{history}{vitals}",
    ),
    "en": (
        "{who}, {complaint} {onset}, with {signs}.{history}{vitals}",
        "Presenting complaint: {complaint} {onset}. Patient: {who}. Associated findings: {signs}.{history}{vitals}",
        "{who} presents with {complaint} {onset}. Examination shows {signs}.{history}{vitals}",
    ),
}

# Framing around the description in the user turn.
USER_TEMPLATES = {
    "fr": (
        "Situation clinique à l'accueil des urgences.\n{description}\nQuel est le niveau de priorité de triage ?",
        "Patient à l'accueil des urgences.\n{description}\nÉvalue le degré d'urgence.",
        "{description}\nIndique le niveau de triage et la conduite à tenir.",
    ),
    "en": (
        "Emergency department triage case.\n{description}\nWhat is the triage priority level?",
        "Patient at the emergency desk.\n{description}\nAssess the level of urgency.",
        "{description}\nGive the triage level and the recommended course of action.",
    ),
}


@dataclass(frozen=True)
class ClinicalCase:
    """A complete clinical vignette, ready to become a training pair."""

    presentation_id: str
    level: str
    lang: str
    age: int
    sex: str
    symptoms: tuple[str, ...]
    medical_history: tuple[str, ...]
    vitals: VitalSigns | None
    onset: str
    description: str
    user_turn: str
    justification: str
    recommendation: str
    confidence: str
    source: str


def _who(age: int, sex: str, lang: str) -> str:
    """Describe the patient in a short phrase suited to their age.

    Catalogue ages are whole years: an age of 0 or 1 describes an infant, presented in months as
    a paediatric record would.
    """
    infant_in_months = {0: 6, 1: 18}
    if age in infant_in_months:
        months = infant_in_months[age]
        return f"Nourrisson de {months} mois" if lang == "fr" else f"{months}-month-old infant"
    if age < 15:
        if lang == "fr":
            return f"{'Garçon' if sex == 'M' else 'Fille'} de {age} ans"
        return f"{age}-year-old {'boy' if sex == 'M' else 'girl'}"
    if lang == "fr":
        return f"{'Homme' if sex == 'M' else 'Femme'} de {age} ans"
    return f"{age}-year-old {'man' if sex == 'M' else 'woman'}"


def _join(items: tuple[str, ...], lang: str) -> str:
    """List items with the right coordinating conjunction."""
    if len(items) == 1:
        return items[0]
    conjunction = " et " if lang == "fr" else " and "
    return ", ".join(items[:-1]) + conjunction + items[-1]


def build_case(
    presentation: Presentation,
    lang: str,
    rng: random.Random,
    with_vitals: bool = True,
) -> ClinicalCase:
    """Compose a clinical vignette from a typical presentation."""
    age = rng.randint(*presentation.age_range)
    feminine = presentation.id in FEMALE_ONLY_PRESENTATIONS
    sex = "F" if feminine else rng.choice(("M", "F"))

    signs_pool = presentation.signs_fr if lang == "fr" else presentation.signs_en
    symptoms = tuple(rng.sample(signs_pool, k=rng.randint(2, min(3, len(signs_pool)))))

    history_pool = presentation.history_fr if lang == "fr" else presentation.history_en
    medical_history = tuple(rng.sample(history_pool, k=rng.randint(1, min(2, len(history_pool)))))

    # The vital signs the narrative names are forced: without that, a "fever with chills"
    # vignette comes out afebrile and contradicts itself.
    vitals = (
        generate_vitals(
            presentation.vitals_profile,
            age,
            rng,
            forced_vitals(presentation.id),
            altered_consciousness_likely(presentation.id),
        )
        if with_vitals
        else None
    )

    complaint = presentation.complaint_fr if lang == "fr" else presentation.complaint_en
    onset = rng.choice(ONSET_PHRASINGS[presentation.onset][lang])

    if lang == "fr":
        history_block = f" Antécédents : {_join(medical_history, lang)}."
        vitals_block = f" Constantes : {vitals.render(lang)}." if vitals else ""
    else:
        history_block = f" Past history: {_join(medical_history, lang)}."
        vitals_block = f" Observations: {vitals.render(lang)}." if vitals else ""

    description = rng.choice(DESCRIPTION_TEMPLATES[lang]).format(
        who=_who(age, sex, lang),
        complaint=complaint,
        onset=onset,
        signs=_join(symptoms, lang),
        history=history_block,
        vitals=vitals_block,
    )
    user_turn = rng.choice(USER_TEMPLATES[lang]).format(description=description)

    return ClinicalCase(
        presentation_id=presentation.id,
        level=presentation.level,
        lang=lang,
        age=age,
        sex=sex,
        symptoms=symptoms,
        medical_history=medical_history,
        vitals=vitals,
        onset=presentation.onset,
        description=description,
        user_turn=user_turn,
        justification=presentation.justification,
        recommendation=presentation.recommendation,
        confidence="high",
        source="clinical_vignette",
    )


def generate_cases(
    total: int,
    level: str,
    lang: str,
    rng: random.Random,
    vitals_share: float = 0.75,
    exclude: set[str] | None = None,
) -> list[ClinicalCase]:
    """Produce ``total`` vignettes of a given triage level and language.

    The preparation script calls this function cell by cell of the grid (three levels × two
    languages) after counting what the corpora supplied. That is what allows the dataset to be
    balanced to the unit — target volumes do not always divide by three — whereas the cases
    extracted from the corpora are almost all urgent.

    A share of the vignettes is produced without vital signs: at reception the readings are not
    always available at sorting time, and the model must be able to decide on the clinical
    description alone.

    ``exclude`` holds user turns already produced elsewhere — first of all those of the clinical
    evaluation set, which must reappear nowhere.
    """
    presentations = presentations_by_level(level)
    if not presentations:
        raise ValueError(f"No presentation in the catalogue for level {level!r}")

    cases: list[ClinicalCase] = []
    seen_texts: set[str] = set(exclude or ())
    attempts = 0
    attempt_limit = max(total * 40, 200)

    while len(cases) < total and attempts < attempt_limit:
        attempts += 1
        presentation = rng.choice(presentations)
        case = build_case(presentation, lang, rng, with_vitals=rng.random() < vitals_share)
        if case.user_turn in seen_texts:
            continue
        seen_texts.add(case.user_turn)
        cases.append(case)

    return cases
