"""Génération de vignettes cliniques à partir du catalogue de présentations.

Une vignette, c'est une présentation type du catalogue habillée d'un patient :
un âge, un sexe, des antécédents, un délai d'installation, un relevé de
constantes et une formulation. Le niveau de triage vient de la présentation
d'origine — jamais d'une relecture du texte produit. C'est ce qui rend
l'évaluation honnête : le modèle ne peut pas se contenter de réapprendre une
règle lexicale, puisque aucune règle lexicale n'a servi à étiqueter.

Chaque vignette porte les métadonnées demandées par le brief : symptômes,
antécédents, constantes, source et niveau de confiance.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from clinical_triage.data.clinical_catalogue import (
    Presentation,
    constantes_imposees,
    presentations_by_level,
    vigilance_alteree_probable,
)
from clinical_triage.data.vital_signs import VitalSigns
from clinical_triage.data.vital_signs import generate as generate_vitals

# Présentations propres à la femme : le tirage du sexe doit en tenir compte.
PRESENTATIONS_FEMININES = frozenset(
    {"pre_eclampsie_severe", "hemorragie_du_post_partum", "cystite_simple"}
)

# Formulations du délai d'installation, par échelle de temps.
DELAIS = {
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

# Gabarits de description du patient. Les champs sont remplis plus bas.
DESCRIPTION_TEMPLATES = {
    "fr": (
        "{demographie}, {motif} {delai}, avec {signes}.{antecedents}{constantes}",
        "Motif d'admission : {motif} {delai}. Patient : {demographie}. Signes associés : {signes}.{antecedents}{constantes}",
        "{demographie} se présente pour {motif} {delai}. On note {signes}.{antecedents}{constantes}",
    ),
    "en": (
        "{demographie}, {motif} {delai}, with {signes}.{antecedents}{constantes}",
        "Presenting complaint: {motif} {delai}. Patient: {demographie}. Associated findings: {signes}.{antecedents}{constantes}",
        "{demographie} presents with {motif} {delai}. Examination shows {signes}.{antecedents}{constantes}",
    ),
}

# Consignes encadrant la description dans le tour utilisateur.
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
    """Vignette clinique complète, prête à devenir une paire d'entraînement."""

    presentation_id: str
    level: str
    lang: str
    age: int
    sex: str
    symptomes: tuple[str, ...]
    antecedents: tuple[str, ...]
    constantes: VitalSigns | None
    delai: str
    description: str
    user_turn: str
    justification: str
    recommandation: str
    confiance: str
    source: str


def _demographie(age: int, sex: str, lang: str) -> str:
    """Décrit le patient en une formule courte adaptée à son âge.

    Les âges du catalogue sont en années entières : un âge de 0 ou 1 an décrit un
    nourrisson, que l'on présente en mois comme le fait un dossier de pédiatrie.
    """
    nourrisson_en_mois = {0: 6, 1: 18}
    if age in nourrisson_en_mois:
        mois = nourrisson_en_mois[age]
        return f"Nourrisson de {mois} mois" if lang == "fr" else f"{mois}-month-old infant"
    if age < 15:
        if lang == "fr":
            return f"{'Garçon' if sex == 'M' else 'Fille'} de {age} ans"
        return f"{age}-year-old {'boy' if sex == 'M' else 'girl'}"
    if lang == "fr":
        return f"{'Homme' if sex == 'M' else 'Femme'} de {age} ans"
    return f"{age}-year-old {'man' if sex == 'M' else 'woman'}"


def _join(items: tuple[str, ...], lang: str) -> str:
    """Énumère des éléments avec la bonne conjonction de coordination."""
    if len(items) == 1:
        return items[0]
    conjonction = " et " if lang == "fr" else " and "
    return ", ".join(items[:-1]) + conjonction + items[-1]


def build_case(
    presentation: Presentation,
    lang: str,
    rng: random.Random,
    with_vitals: bool = True,
) -> ClinicalCase:
    """Compose une vignette clinique à partir d'une présentation type."""
    age = rng.randint(*presentation.age_range)
    feminine = presentation.id in PRESENTATIONS_FEMININES
    sex = "F" if feminine else rng.choice(("M", "F"))

    signes_pool = presentation.signes_fr if lang == "fr" else presentation.signes_en
    symptomes = tuple(rng.sample(signes_pool, k=rng.randint(2, min(3, len(signes_pool)))))

    antecedents_pool = presentation.antecedents_fr if lang == "fr" else presentation.antecedents_en
    antecedents = tuple(
        rng.sample(antecedents_pool, k=rng.randint(1, min(2, len(antecedents_pool))))
    )

    # Les constantes que le récit nomme sont imposées : sans cela, une vignette
    # « fièvre avec frissons » sort apyrétique et se contredit elle-même.
    constantes = (
        generate_vitals(
            presentation.vitals_profile,
            age,
            rng,
            constantes_imposees(presentation.id),
            vigilance_alteree_probable(presentation.id),
        )
        if with_vitals
        else None
    )

    motif = presentation.motif_fr if lang == "fr" else presentation.motif_en
    delai = rng.choice(DELAIS[presentation.delai][lang])

    if lang == "fr":
        bloc_antecedents = f" Antécédents : {_join(antecedents, lang)}."
        bloc_constantes = f" Constantes : {constantes.render(lang)}." if constantes else ""
    else:
        bloc_antecedents = f" Past history: {_join(antecedents, lang)}."
        bloc_constantes = f" Observations: {constantes.render(lang)}." if constantes else ""

    description = rng.choice(DESCRIPTION_TEMPLATES[lang]).format(
        demographie=_demographie(age, sex, lang),
        motif=motif,
        delai=delai,
        signes=_join(symptomes, lang),
        antecedents=bloc_antecedents,
        constantes=bloc_constantes,
    )
    user_turn = rng.choice(USER_TEMPLATES[lang]).format(description=description)

    return ClinicalCase(
        presentation_id=presentation.id,
        level=presentation.level,
        lang=lang,
        age=age,
        sex=sex,
        symptomes=symptomes,
        antecedents=antecedents,
        constantes=constantes,
        delai=presentation.delai,
        description=description,
        user_turn=user_turn,
        justification=presentation.justification,
        recommandation=presentation.recommandation,
        confiance="haute",
        source="vignette_clinique",
    )


def generate_cases(
    total: int,
    level: str,
    lang: str,
    rng: random.Random,
    vitals_share: float = 0.75,
    exclude: set[str] | None = None,
) -> list[ClinicalCase]:
    """Produit `total` vignettes d'un niveau de triage et d'une langue donnés.

    Le script de préparation appelle cette fonction case par case de la grille
    (trois niveaux × deux langues) après avoir compté ce que les corpus ont
    fourni. C'est ce qui permet d'équilibrer le dataset à une unité près — les
    volumes cibles ne se divisent pas toujours par trois —, alors
    que les cas extraits des corpus sont, eux, presque tous urgents.

    Une part des vignettes est produite sans constantes vitales : à l'accueil, le
    relevé n'est pas toujours disponible au moment du tri, et le modèle doit
    savoir décider sur la seule description clinique.

    `exclude` contient des tours utilisateur déjà produits ailleurs — en premier
    lieu ceux du jeu d'évaluation clinique, qui ne doivent réapparaître nulle part.
    """
    presentations = presentations_by_level(level)
    if not presentations:
        raise ValueError(f"Aucune présentation au catalogue pour le niveau {level!r}")

    cases: list[ClinicalCase] = []
    textes_vus: set[str] = set(exclude or ())
    tentatives = 0
    limite_tentatives = max(total * 40, 200)

    while len(cases) < total and tentatives < limite_tentatives:
        tentatives += 1
        presentation = rng.choice(presentations)
        case = build_case(presentation, lang, rng, with_vitals=rng.random() < vitals_share)
        if case.user_turn in textes_vus:
            continue
        textes_vus.add(case.user_turn)
        cases.append(case)

    return cases
