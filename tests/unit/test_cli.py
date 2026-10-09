"""The project summary must describe what exists, not what was planned.

``uv run clinical-triage`` is the first command run on a fresh machine. If it announces the
default LoRA setting while an adapter trained with another sits on disk, it misleads from the
first second.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from clinical_triage import cli


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    """Point ``PATHS`` at a disposable root and return the adapter folder."""
    monkeypatch.setattr(cli, "PATHS", dataclasses.replace(cli.PATHS, models=tmp_path))
    return cli.PATHS.sft_adapter


def test_without_an_adapter_the_kept_setting_is_announced_as_not_trained(adapter, monkeypatch):
    monkeypatch.setattr(
        cli, "kept_setting", lambda: {"lora_r": 32, "lora_alpha": 64, "learning_rate": 2e-4}
    )
    summary = cli._lora()
    assert "r=32" in summary
    assert "not trained yet" in summary


def test_without_a_tuning_no_setting_is_invented(adapter, monkeypatch):
    """A default rank once stood here, and it was not the rank of the shipped model."""
    monkeypatch.setattr(cli, "kept_setting", lambda: None)
    assert cli._lora().startswith("not tuned yet")


def test_with_an_adapter_its_own_rank_is_shown(adapter):
    """Hyper-parameter tuning may keep a rank other than the default."""
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"r": 32, "lora_alpha": 64}), encoding="utf-8"
    )

    summary = cli._lora()
    assert "r=32" in summary
    assert "alpha=64" in summary
    assert "default" not in summary


def test_the_summary_prints_without_a_dataset_or_a_model(adapter, capsys):
    """On a fresh machine nothing has been produced yet: the command must hold."""
    cli.main()
    output = capsys.readouterr().out
    assert "Emergency triage assistant" in output
    assert "SFT adapter" in output
