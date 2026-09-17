"""Tests de la configuration et des utilitaires transverses.

La configuration se présente comme la source unique de vérité du projet : ces
tests vérifient qu'elle est cohérente, et que la graine reproduit réellement les
tirages — sans quoi la reproductibilité annoncée n'en serait pas une.
"""

from __future__ import annotations

import logging
import random
import warnings
from pathlib import Path

from clinical_triage.config import DATA, MODEL, PATHS, SEED, SERVING, TRAINING, TRIAGE
from clinical_triage.utils import path_for_log, get_logger, set_seed


def test_la_taxonomie_compte_trois_niveaux_tous_libelles():
    assert len(TRIAGE.levels) == 3
    for niveau in TRIAGE.levels:
        assert TRIAGE.labels_fr[niveau]
        assert niveau in TRIAGE.severity


def test_les_niveaux_sont_ordonnes_par_gravite():
    assert (
        TRIAGE.severity["CONSULTATION_DIFFEREE"]
        < TRIAGE.severity["URGENCE_MODEREE"]
        < TRIAGE.severity["URGENCE_VITALE"]
    )


def test_les_proportions_de_decoupage_sont_coherentes():
    assert 0 < DATA.val_ratio < 1
    assert 0 < DATA.test_ratio < 1
    assert DATA.val_ratio + DATA.test_ratio < 1
    assert 0 < DATA.french_share < 1
    assert 0 < DATA.max_corpus_share < 1


def test_les_corpus_du_cahier_des_charges_sont_declares():
    assert {"medquad", "frenchmedmcqa", "medmcqa", "ultramedical_pref"} <= set(DATA.corpora)
    for identifiant in DATA.corpora.values():
        assert "/" in identifiant  # identifiant de dépôt Hugging Face


def test_lora_adapte_l_attention_et_le_perceptron():
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= set(TRAINING.lora_target_modules)
    assert {"gate_proj", "up_proj", "down_proj"} <= set(TRAINING.lora_target_modules)


def test_l_alignement_reste_prudent():
    """Un taux d'apprentissage d'alignement élevé défait le format appris au fine-tuning."""
    assert TRAINING.dpo_lr < TRAINING.sft_lr
    assert TRAINING.dpo_rpo_alpha is not None


def test_les_identifiants_de_depot_suivent_le_compte(monkeypatch):
    monkeypatch.setenv("HF_NAMESPACE", "un-autre-compte")
    assert MODEL.hub_dataset_id.startswith("un-autre-compte/")
    assert MODEL.hub_merged_model_id.startswith("un-autre-compte/")
    assert MODEL.hub_sft_merged_model_id.startswith("un-autre-compte/")


def test_un_compte_vide_retombe_sur_le_compte_par_defaut(monkeypatch):
    """Une variable de dépôt GitHub non définie arrive vide, pas absente."""
    monkeypatch.setenv("HF_NAMESPACE", "")
    assert not MODEL.hub_dataset_id.startswith("/")
    assert MODEL.hub_namespace == "BenoitJT-GIRARD"


def test_les_chemins_du_pipeline_sont_sous_la_racine():
    for chemin in (PATHS.data_processed, PATHS.models, PATHS.reports, PATHS.logs, PATHS.tracking):
        assert PATHS.root in chemin.parents or chemin == PATHS.root
    assert PATHS.sft_merged.name.endswith("-sft-merged")
    assert PATHS.dpo_merged.name.endswith("-dpo-merged")


def test_la_generation_du_service_est_deterministe_par_defaut():
    assert SERVING.temperature == 0.0
    assert SERVING.max_new_tokens > 0


def test_la_graine_reproduit_les_tirages():
    set_seed(SEED, include_torch=False)
    premier = [random.random() for _ in range(5)]
    set_seed(SEED, include_torch=False)
    assert [random.random() for _ in range(5)] == premier


def test_la_graine_s_applique_aussi_a_numpy():
    import numpy as np

    set_seed(SEED, include_torch=False)
    premier = np.random.rand(5).tolist()
    set_seed(SEED, include_torch=False)
    assert np.random.rand(5).tolist() == premier


def test_le_logger_est_configure_une_seule_fois():
    logger = get_logger("essai_de_journalisation")
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert get_logger("essai_de_journalisation") is logger
    assert len(logger.handlers) == 1


def test_les_chemins_du_projet_sont_journalises_relativement_a_la_racine():
    assert path_for_log(PATHS.sft_merged) == "models/qwen3-1.7b-triage-sft-merged"
    assert path_for_log(str(PATHS.data_processed)) == "data/processed"


def test_ce_qui_n_est_pas_un_chemin_du_projet_est_journalise_tel_quel():
    assert path_for_log(MODEL.base_model) == MODEL.base_model
    ailleurs = Path(PATHS.root.anchor) / "modeles" / "qwen3"
    assert path_for_log(ailleurs) == str(ailleurs)


def test_un_avertissement_de_bibliotheque_perd_son_chemin_d_installation():
    # Importer le paquet installe le formateur : c'est ce que fait la première
    # cellule de chaque carnet, dont les sorties sont versionnées.
    import clinical_triage  # noqa: F401

    texte = warnings.formatwarning(
        "essai", UserWarning, "/ailleurs/venv/Lib/site-packages/peft/tuners.py", 1463
    )
    assert texte.startswith("peft/tuners.py:1463: UserWarning: essai")
