"""Tests des diagnostics du jeu produit.

Ces mesures disent ce que valent les volumes du dataset : combien de réponses
différentes le modèle a vues, quelle part du jeu de test est de la restitution,
et ce que le niveau doit aux métadonnées du générateur. Elles sont publiées dans
la carte du dataset et reprises par le rapport, donc elles doivent être justes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from clinical_triage.data.diagnostics import (
    completions_distinctes,
    fuite_par_metadonnees,
    separabilite,
)


@dataclass(frozen=True)
class ExempleFictif:
    """Le strict nécessaire pour éprouver les diagnostics."""

    user_turn: str
    assistant_turn: str
    level: str
    source: str
    presentation_id: str


@dataclass(frozen=True)
class FicheFictive:
    id: str
    onset: str
    vitals_profile: str


def _vignettes(nombre_par_presentation: int, presentations: list[str]) -> list[ExempleFictif]:
    """Vignettes fictives, mélangées comme le sont celles du jeu livré.

    Le mélange n'est pas cosmétique. Laissées dans l'ordre des présentations, les
    vignettes font qu'un découpage stratifié prend des blocs contigus, c'est-à-dire
    une présentation entière par pli : il devient alors un découpage groupé, et la
    mesure ne peut plus montrer l'écart qu'elle sert à montrer.
    """
    exemples = []
    for rang, presentation in enumerate(presentations):
        for index in range(nombre_par_presentation):
            exemples.append(
                ExempleFictif(
                    user_turn=f"Patient {index} avec {presentation}",
                    assistant_turn=f"Réponse pour {presentation}",
                    level="URGENCE_VITALE" if rang % 2 else "URGENCE_MODEREE",
                    source="vignette_clinique",
                    presentation_id=presentation,
                )
            )
    random.Random(0).shuffle(exemples)
    return exemples


# --- Diversité des réponses attendues ---


def test_les_vignettes_d_une_meme_presentation_partagent_leur_reponse():
    """C'est le fait que la mesure doit exposer : 60 exemples, 3 réponses."""
    exemples = _vignettes(20, ["abc", "bcd", "cde"])
    mesure = completions_distinctes(exemples)
    assert mesure["total"] == 3
    vignettes = mesure["par_origine"]["vignettes"]
    assert vignettes["exemples"] == 60
    assert vignettes["reponses_distinctes"] == 3
    assert vignettes["repetition_moyenne"] == 20.0


def test_les_cas_de_corpus_sont_comptes_a_part():
    exemples = _vignettes(5, ["abc"]) + [
        ExempleFictif("Un patient tousse.", "Réponse unique", "URGENCE_MODEREE", "mediqal", "")
    ]
    mesure = completions_distinctes(exemples)
    assert mesure["par_origine"]["corpus"]["exemples"] == 1
    assert mesure["par_origine"]["vignettes"]["exemples"] == 5


def test_une_ligne_jsonl_est_acceptee_comme_un_objet():
    """Les exemples circulent sous deux formes ; la mesure doit lire les deux."""
    ligne = {
        "user_turn": "Patient",
        "completion": "Réponse",
        "level": "URGENCE_VITALE",
        "source": "vignette_clinique",
        "presentation_id": "abc",
    }
    assert completions_distinctes([ligne])["total"] == 1


# --- Séparabilité : découpage aléatoire contre découpage par présentation ---


def test_le_decoupage_groupe_est_plus_severe_que_le_decoupage_aleatoire():
    """Un découpage aléatoire laisse des reformulations d'un même cas des deux côtés.

    Avec des présentations dont le vocabulaire ne se recoupe pas, un classifieur
    reconnaît parfaitement celles qu'il a vues et ne transfère pas à celles qu'il
    n'a pas vues. L'écart entre les deux chiffres est ce que la mesure publie.
    """
    presentations = [f"motif{lettre}" for lettre in "abcdefghij"]
    exemples = _vignettes(30, presentations)
    mesure = separabilite(exemples)
    assert mesure["presentations"] == 10
    assert mesure["decoupage_aleatoire"] > mesure["decoupage_par_presentation"]
    assert mesure["ecart"] > 0


def test_la_separabilite_renonce_sur_un_jeu_trop_petit():
    assert separabilite(_vignettes(1, ["abc"])) == {}


# --- Fuite du niveau par les métadonnées du générateur ---


def test_une_metadonnee_parfaitement_correlee_au_niveau_est_detectee():
    """Si chaque profil de constantes ne va qu'avec un niveau, le vote est parfait."""
    exemples = _vignettes(10, ["abc", "bcd"])
    fiches = [
        FicheFictive(id="abc", onset="minutes", vitals_profile="critique"),
        FicheFictive(id="bcd", onset="jours", vitals_profile="normal"),
    ]
    mesure = fuite_par_metadonnees(exemples, fiches)
    assert mesure["exactitude_du_vote_majoritaire"] == 1.0
    assert mesure["part_en_cellule_homogene"] == 1.0


def test_une_metadonnee_sans_lien_avec_le_niveau_ne_predit_rien_de_plus_que_la_majorite():
    """Deux niveaux dans la même cellule : le vote majoritaire plafonne à la majorité."""
    exemples = _vignettes(10, ["abc", "bcd"])
    fiches = [
        FicheFictive(id="abc", onset="heures", vitals_profile="normal"),
        FicheFictive(id="bcd", onset="heures", vitals_profile="normal"),
    ]
    mesure = fuite_par_metadonnees(exemples, fiches)
    assert mesure["cellules"] == 1
    assert mesure["exactitude_du_vote_majoritaire"] == 0.5
    assert mesure["part_en_cellule_homogene"] == 0.0


def test_la_fuite_renonce_quand_aucune_presentation_ne_correspond():
    assert fuite_par_metadonnees(_vignettes(2, ["abc"]), []) == {}
