"""Tests de la mesure de préférence sur un jeu externe.

Aucun modèle n'est chargé : les vraisemblances sont fournies directement. Ce qui
est vérifié, c'est la lecture qu'on en fait — l'ordre des deux réponses, la marge,
et le détail par difficulté annoncée par le corpus.
"""

from __future__ import annotations

import pytest

from clinical_triage.evaluation.preference import PreferenceScore, summarize


def test_une_paire_bien_ordonnee():
    score = PreferenceScore(chosen=-1.20, rejected=-1.80, label_type="hard")
    assert score.bien_ordonnee
    assert score.marge == pytest.approx(0.6)


def test_une_paire_mal_ordonnee():
    score = PreferenceScore(chosen=-2.0, rejected=-1.5, label_type="easy")
    assert not score.bien_ordonnee
    assert score.marge < 0


def test_la_synthese_compte_les_paires_bien_ordonnees():
    scores = [
        PreferenceScore(-1.0, -1.5, "hard"),
        PreferenceScore(-1.0, -1.2, "hard"),
        PreferenceScore(-2.0, -1.0, "easy"),
        PreferenceScore(-1.0, -1.1, "easy"),
    ]
    resume = summarize(scores)
    assert resume["n"] == 4
    assert resume["part_bien_ordonnees"] == 0.75


def test_la_synthese_detaille_par_difficulte():
    scores = [
        PreferenceScore(-1.0, -1.5, "hard"),
        PreferenceScore(-2.0, -1.0, "hard"),
        PreferenceScore(-1.0, -1.5, "easy"),
        PreferenceScore(-1.0, -1.5, "easy"),
    ]
    detail = summarize(scores)["par_difficulte"]
    assert detail["hard"]["part_bien_ordonnees"] == 0.5
    assert detail["easy"]["part_bien_ordonnees"] == 1.0


def test_la_synthese_sur_un_jeu_vide():
    resume = summarize([])
    assert resume["n"] == 0
    assert resume["part_bien_ordonnees"] == 0.0
