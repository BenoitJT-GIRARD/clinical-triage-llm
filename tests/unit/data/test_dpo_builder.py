"""Tests des paires de préférence servant à l'alignement.

Le point vérifié ici n'est pas cosmétique. Le modèle apprendra la différence
qu'il trouve entre la réponse préférée et la réponse rejetée, pas celle qu'on
avait en tête. Si cette différence est systématiquement la longueur, c'est la
longueur qu'il apprend — et il cesse de savoir s'arrêter.
"""

from __future__ import annotations

import random
from collections import Counter

from clinical_triage.config import TRIAGE
from clinical_triage.data.case_generator import generate_cases
from clinical_triage.data.dataset_io import dpo_record
from clinical_triage.data.dpo_builder import build_preference_pairs, length_balance
from clinical_triage.data.sft_builder import assemble
from clinical_triage.prompts import extract_level


def paires_de_test(nombre: int = 200):
    """Construit un jeu de préférences à partir de vignettes générées."""
    generateur = random.Random(12)
    vignettes = []
    for niveau in TRIAGE.levels:
        vignettes += generate_cases(nombre, niveau, "fr", generateur)
    exemples = assemble(vignettes, [], set(), generateur)
    return exemples, build_preference_pairs(exemples, nombre, generateur)


def test_les_reponses_rejetees_ne_sont_jamais_un_surclassement():
    """Surclasser est la bonne réaction en cas de doute : ce ne peut pas être un contre-exemple."""
    _, paires = paires_de_test()
    for paire in paires:
        niveau_rejete = extract_level(paire.rejected)
        if niveau_rejete is None:
            continue
        assert TRIAGE.severity[niveau_rejete] <= TRIAGE.severity[paire.level]


def test_les_paires_ne_presentent_pas_de_biais_de_longueur():
    _, paires = paires_de_test()
    equilibre = length_balance(paires)
    assert 0.35 <= equilibre["part_chosen_plus_long"] <= 0.65
    ratio = equilibre["rejected_moyenne"] / equilibre["chosen_moyenne"]
    assert 0.85 <= ratio <= 1.15


def test_les_reponses_rejetees_respectent_la_structure_attendue():
    """Une réponse rejetée mal formée ferait apprendre le format au lieu du fond."""
    _, paires = paires_de_test()
    structurees = [p for p in paires if p.strategie != "reponse_en_anglais"]
    for paire in structurees:
        assert paire.rejected.startswith("Niveau de priorité :")
        assert "Justification :" in paire.rejected
        assert "Recommandation :" in paire.rejected


def test_les_paires_couvrent_les_quatre_defauts():
    _, paires = paires_de_test()
    assert len({p.strategie for p in paires}) == 4


def test_les_cas_graves_sont_sur_representes():
    _, paires = paires_de_test()
    repartition = Counter(p.level for p in paires)
    assert repartition["URGENCE_VITALE"] > repartition["CONSULTATION_DIFFEREE"]


def test_un_meme_cas_ne_donne_pas_deux_paires():
    _, paires = paires_de_test()
    tours = [p.user_turn for p in paires]
    assert len(tours) == len(set(tours))


def test_l_enregistrement_de_preference_expose_les_trois_colonnes_attendues():
    _, paires = paires_de_test(30)
    enregistrement = dpo_record(paires[0])
    assert set(enregistrement) >= {"prompt", "chosen", "rejected", "strategie"}
    assert enregistrement["prompt"].endswith("<|im_start|>assistant\n")
