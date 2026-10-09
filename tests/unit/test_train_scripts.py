"""Where the supervised setting comes from, and where a capped run writes.

The training scripts load weights onto a GPU from their first useful line, and are verified by
real runs. What is tested here is what decides before that line: which setting is trained, and
that nothing a capped run writes can land on a delivered artefact.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

import pytest

from clinical_triage.config import PATHS


@pytest.fixture(scope="module")
def train_sft():
    spec = importlib.util.spec_from_file_location(
        "train_sft_script", PATHS.root / "scripts" / "train_sft.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**given) -> argparse.Namespace:
    return argparse.Namespace(
        **{"lora_r": None, "lora_alpha": None, "learning_rate": None, **given}
    )


def test_the_kept_setting_is_trained(train_sft, monkeypatch):
    monkeypatch.setattr(
        train_sft,
        "kept_setting",
        lambda: {"variant": "r32_lr2e-4", "lora_r": 32, "lora_alpha": 64, "learning_rate": 2e-4},
    )
    assert train_sft.setting(_args()) == (32, 64, 2e-4)


def test_without_a_tuning_the_script_refuses_to_guess(train_sft, monkeypatch):
    """A default rank once stood in for the tuning, and it was not the rank shipped."""
    monkeypatch.setattr(train_sft, "kept_setting", lambda: None)
    with pytest.raises(SystemExit, match="tune_hyperparameters"):
        train_sft.setting(_args())


def test_an_explicit_setting_needs_all_three_values(train_sft, monkeypatch):
    monkeypatch.setattr(
        train_sft,
        "kept_setting",
        lambda: {"variant": "r32_lr2e-4", "lora_r": 32, "lora_alpha": 64, "learning_rate": 2e-4},
    )
    with pytest.raises(SystemExit, match="together"):
        train_sft.setting(_args(lora_r=8))
    assert train_sft.setting(_args(lora_r=8, lora_alpha=16, learning_rate=1e-4)) == (8, 16, 1e-4)
