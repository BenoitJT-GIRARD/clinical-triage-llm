"""Tests du calcul des pas de chauffe.

Une chauffe trop courte fait diverger le début de l'entraînement ; une chauffe
plus longue que l'entraînement lui-même le fait s'arrêter avant d'avoir atteint
le taux d'apprentissage visé. Ce calcul est le seul endroit du dépôt où la
proportion de chauffe devient un nombre de pas.
"""

from __future__ import annotations

from clinical_triage.training.schedule import warmup_steps


def test_un_nombre_de_pas_impose_l_emporte_sur_la_taille_du_corpus():
    """`max_steps` fixe l'entraînement : la chauffe se calcule sur lui."""
    assert (
        warmup_steps(
            nombre_exemples=100_000, lot_effectif=8, epochs=3, max_steps=200, proportion=0.1
        )
        == 20
    )


def test_la_chauffe_se_deduit_du_corpus_et_du_lot_effectif():
    # 4 000 exemples par lots de 16 font 250 pas par époque, soit 500 sur deux
    # époques ; 5 % de chauffe font 25 pas.
    assert (
        warmup_steps(nombre_exemples=4000, lot_effectif=16, epochs=2, max_steps=0, proportion=0.05)
        == 25
    )


def test_un_corpus_partiel_compte_un_pas_entier():
    """Le dernier lot, incomplet, est un pas comme les autres."""
    assert (
        warmup_steps(nombre_exemples=17, lot_effectif=8, epochs=1, max_steps=0, proportion=1.0) == 3
    )


def test_la_chauffe_ne_descend_jamais_a_zero():
    """Zéro pas de chauffe expose la première mise à jour au taux plein."""
    assert (
        warmup_steps(nombre_exemples=10, lot_effectif=8, epochs=1, max_steps=0, proportion=0.01)
        == 1
    )


def test_un_lot_ou_un_nombre_d_epochs_nul_ne_divise_pas_par_zero():
    assert (
        warmup_steps(nombre_exemples=100, lot_effectif=0, epochs=0, max_steps=0, proportion=0.1)
        == 10
    )
