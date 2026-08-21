"""Tests du chargement des corpus publics.

Ils portent sur les deux conduites que le module promet face à une panne : une
source injoignable à l'ouverture est ignorée, une lecture interrompue en route
arrête tout. La seconde n'allait pas de soi — le `try` d'origine n'entourait que
l'ouverture, alors qu'en streaming tout le trafic réseau a lieu pendant
l'itération.
"""

from __future__ import annotations

import pytest

from clinical_triage.data.corpus_sources import (
    LectureInterrompue,
    _lignes,
    _plafond_atteint,
)

# --- Une lecture interrompue ne doit pas passer pour une lecture complète ---


def _flux_coupe_apres(nombre: int):
    """Un flux qui livre `nombre` lignes puis se coupe, comme le fait le réseau."""

    def flux():
        for index in range(nombre):
            yield {"question": f"ligne {index}"}
        raise ConnectionError("Server disconnected without sending a response.")

    return flux()


def test_une_coupure_en_cours_de_lecture_arrete_tout():
    """Le rendement publié se calcule sur le nombre d'entrées lues.

    Absorber la coupure produirait un tableau qui décrit une lecture partielle
    en la présentant comme le corpus entier.
    """
    with pytest.raises(LectureInterrompue) as erreur:
        list(_lignes(_flux_coupe_apres(3), "openlifescienceai/medmcqa"))
    message = str(erreur.value)
    assert "openlifescienceai/medmcqa" in message
    assert "3 entrées" in message


def test_une_lecture_complete_ne_leve_rien():
    lignes = list(_lignes(iter([{"a": 1}, {"a": 2}, {"a": 3}]), "essai"))
    assert len(lignes) == 3


def test_un_arret_volontaire_n_est_pas_une_coupure():
    """Le plafond de lecture ferme le générateur ; ce n'est pas une panne."""
    lues = []
    for ligne in _lignes(iter([{"a": 1}, {"a": 2}, {"a": 3}]), "essai"):
        lues.append(ligne)
        if len(lues) == 2:
            break
    assert len(lues) == 2


# --- Le plafond de lecture, et son absence ---


@pytest.mark.parametrize(
    ("deja_lues", "limite", "attendu"),
    [
        (5, 0, False),  # zéro ne borne rien
        (5, -1, False),  # une limite négative non plus
        (5, 10, False),
        (10, 10, True),
        (11, 10, True),
        (0, 1, False),
        (1, 1, True),
    ],
)
def test_le_plafond_de_lecture(deja_lues: int, limite: int, attendu: bool):
    assert _plafond_atteint([None] * deja_lues, limite) is attendu
