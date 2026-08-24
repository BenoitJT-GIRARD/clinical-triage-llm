"""Assemblage du jeu d'entraînement supervisé.

Deux apports se rejoignent ici sous un format unique, l'exemple de triage :

- les **vignettes générées** à partir du catalogue de présentations, qui portent
  une étiquette de confiance `haute` et couvrent les trois niveaux dans les deux
  langues ;
- les **cas extraits des corpus publics**, étiquetés par la règle avec une
  confiance `moyenne`, qui apportent des tournures et un vocabulaire clinique
  authentiques que des gabarits ne produisent pas.

Chaque exemple porte les métadonnées demandées par le cahier des charges :
symptômes, antécédents, constantes, source et niveau de confiance.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass

from clinical_triage.data.case_generator import ClinicalCase
from clinical_triage.data.corpus_cases import CorpusCase
from clinical_triage.prompts import build_target_response


def empreinte_de_cas(texte: str) -> str:
    """Forme normalisée d'un tour patient, pour comparer des cas quasi identiques.

    Une comparaison au caractère près laisse passer les quasi-doublons, et c'est
    exactement ce qu'un corpus public produit : « A 6 year old child » et
    « A 6-year-old child » sont le même cas, extrait deux fois. Réparti de part et
    d'autre du découpage, ce doublon rendrait le jeu de test complaisant sans
    qu'aucun contrôle ne le voie. On compare donc les textes mis en minuscules,
    sans accents ni ponctuation.
    """
    decompose = unicodedata.normalize("NFKD", texte.lower())
    sans_accents = "".join(c for c in decompose if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", sans_accents)).strip()


@dataclass(frozen=True)
class TriageExample:
    """Exemple d'entraînement au triage, métadonnées comprises."""

    user_turn: str
    assistant_turn: str
    level: str
    lang: str
    source: str
    confiance: str
    symptomes: tuple[str, ...]
    antecedents: tuple[str, ...]
    constantes: str
    presentation_id: str


def from_clinical_case(case: ClinicalCase) -> TriageExample:
    """Convertit une vignette générée en exemple d'entraînement."""
    return TriageExample(
        user_turn=case.user_turn,
        assistant_turn=build_target_response(case.level, case.justification, case.recommandation),
        level=case.level,
        lang=case.lang,
        source=case.source,
        confiance=case.confiance,
        symptomes=case.symptomes,
        antecedents=case.antecedents,
        constantes=case.constantes.render(case.lang) if case.constantes else "",
        presentation_id=case.presentation_id,
    )


def from_corpus_case(case: CorpusCase) -> TriageExample:
    """Convertit un cas extrait d'un corpus public en exemple d'entraînement."""
    return TriageExample(
        user_turn=case.user_turn,
        assistant_turn=build_target_response(case.level, case.justification, case.recommandation),
        level=case.level,
        lang=case.lang,
        source=case.source,
        confiance=case.confiance,
        symptomes=case.symptomes,
        antecedents=(),
        constantes="",
        presentation_id="",
    )


def assemble(
    generated: list[ClinicalCase],
    from_corpora: list[CorpusCase],
    excluded_user_turns: set[str],
    rng: random.Random,
) -> list[TriageExample]:
    """Réunit les deux apports, écarte les doublons et les tours réservés à l'évaluation.

    Deux précautions, apprises sur ce corpus :

    - la déduplication s'applique **après** toutes les transformations de texte.
      Dédupliquer plus tôt laisse passer les doublons que crée ensuite
      l'anonymisation, en rendant identiques deux énoncés jusque-là distincts ;
    - elle porte sur la **forme normalisée** du tour patient, pas sur le texte
      exact. Les corpus publics livrent le même cas sous des orthographes
      voisines — « A 6 year old child » et « A 6-year-old child » — et un doublon
      réparti de part et d'autre du découpage rend le jeu de test complaisant.
    """
    examples = [from_clinical_case(c) for c in generated]
    examples += [from_corpus_case(c) for c in from_corpora]

    uniques: list[TriageExample] = []
    vus = {empreinte_de_cas(tour) for tour in excluded_user_turns}
    for example in examples:
        empreinte = empreinte_de_cas(example.user_turn)
        if empreinte in vus:
            continue
        vus.add(empreinte)
        uniques.append(example)

    rng.shuffle(uniques)
    return uniques
