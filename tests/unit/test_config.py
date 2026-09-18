"""Tests of the configuration and of the cross-cutting helpers.

The configuration presents itself as the project's single source of truth: these tests check
that it is coherent, and that the seed really reproduces the draws — without which the
reproducibility announced would not be one.
"""

from __future__ import annotations

import logging
import random
import warnings
from pathlib import Path

from clinical_triage.config import DATA, MODEL, PATHS, SEED, SERVING, TRAINING, TRIAGE
from clinical_triage.utils import get_logger, path_for_log, set_seed


def test_the_taxonomy_has_three_levels_all_labelled():
    assert len(TRIAGE.levels) == 3
    for level in TRIAGE.levels:
        assert TRIAGE.labels_fr[level]
        assert level in TRIAGE.severity


def test_the_levels_are_ordered_by_severity():
    assert (
        TRIAGE.severity["CONSULTATION_DIFFEREE"]
        < TRIAGE.severity["URGENCE_MODEREE"]
        < TRIAGE.severity["URGENCE_VITALE"]
    )


def test_the_split_ratios_are_coherent():
    assert 0 < DATA.val_ratio < 1
    assert 0 < DATA.test_ratio < 1
    assert DATA.val_ratio + DATA.test_ratio < 1
    assert 0 < DATA.french_share < 1
    assert 0 < DATA.max_corpus_share < 1


def test_the_required_corpora_are_declared():
    assert {"medquad", "frenchmedmcqa", "medmcqa", "ultramedical_pref"} <= set(DATA.corpora)
    for identifier in DATA.corpora.values():
        assert "/" in identifier  # a Hugging Face repository identifier


def test_lora_adapts_the_attention_and_the_perceptron():
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= set(TRAINING.lora_target_modules)
    assert {"gate_proj", "up_proj", "down_proj"} <= set(TRAINING.lora_target_modules)


def test_the_alignment_stays_conservative():
    """A high alignment learning rate undoes the format learnt during fine-tuning."""
    assert TRAINING.dpo_lr < TRAINING.sft_lr
    assert TRAINING.dpo_rpo_alpha is not None


def test_the_repository_identifiers_follow_the_account(monkeypatch):
    monkeypatch.setenv("HF_NAMESPACE", "another-account")
    assert MODEL.hub_dataset_id.startswith("another-account/")
    assert MODEL.hub_merged_model_id.startswith("another-account/")
    assert MODEL.hub_sft_merged_model_id.startswith("another-account/")


def test_an_empty_account_falls_back_to_the_default_one(monkeypatch):
    """An undefined GitHub repository variable arrives empty, not absent."""
    monkeypatch.setenv("HF_NAMESPACE", "")
    assert not MODEL.hub_dataset_id.startswith("/")
    assert MODEL.hub_namespace == "BenoitJT-GIRARD"


def test_the_pipeline_paths_sit_under_the_root():
    for path in (PATHS.data_processed, PATHS.models, PATHS.reports, PATHS.logs, PATHS.tracking):
        assert PATHS.root in path.parents or path == PATHS.root
    assert PATHS.sft_merged.name.endswith("-sft-merged")
    assert PATHS.dpo_merged.name.endswith("-dpo-merged")


def test_what_a_run_produces_lives_under_var():
    """Four root folders a clone never carries would be four folders a reader must open."""
    for path in (PATHS.models, PATHS.logs, PATHS.tracking):
        assert path.relative_to(PATHS.root).parts[0] == "var"


def test_the_service_generation_is_deterministic_by_default():
    assert SERVING.temperature == 0.0
    assert SERVING.max_new_tokens > 0


def test_the_seed_reproduces_the_draws():
    set_seed(SEED, include_torch=False)
    first = [random.random() for _ in range(5)]
    set_seed(SEED, include_torch=False)
    assert [random.random() for _ in range(5)] == first


def test_the_seed_also_applies_to_numpy():
    import numpy as np

    set_seed(SEED, include_torch=False)
    first = np.random.rand(5).tolist()
    set_seed(SEED, include_torch=False)
    assert np.random.rand(5).tolist() == first


def test_the_logger_is_configured_only_once():
    logger = get_logger("logging_trial")
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert get_logger("logging_trial") is logger
    assert len(logger.handlers) == 1


def test_project_paths_are_logged_relative_to_the_root():
    assert path_for_log(PATHS.sft_merged) == "var/models/qwen3-1.7b-triage-sft-merged"
    assert path_for_log(str(PATHS.data_processed)) == "data/processed"


def test_what_is_not_a_project_path_is_logged_as_is():
    assert path_for_log(MODEL.base_model) == MODEL.base_model
    elsewhere = Path(PATHS.root.anchor) / "models" / "qwen3"
    assert path_for_log(elsewhere) == str(elsewhere)


def test_a_library_warning_loses_its_installation_path():
    # Importing the package installs the formatter: that is what the first cell of every
    # notebook does, and notebook outputs are versioned.
    import clinical_triage  # noqa: F401

    text = warnings.formatwarning(
        "trial", UserWarning, "/elsewhere/venv/Lib/site-packages/peft/tuners.py", 1463
    )
    assert text.startswith("peft/tuners.py:1463: UserWarning: trial")
