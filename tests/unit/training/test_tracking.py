"""Experiment tracking must never fail a training run.

Training costs hours of GPU; tracking costs a few kilobytes of JSON. If the second falls over,
the first carries on. These tests force the failures ``tracking`` claims to absorb, because a
promise written in a docstring and never exercised is only an intention.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import types

import pytest

from clinical_triage.training import tracking


@pytest.fixture
def reports(tmp_path, monkeypatch):
    """Redirect the JSON summaries to a disposable folder.

    ``PATHS`` is a frozen dataclass: the whole object is replaced rather than one field.
    """
    monkeypatch.setattr(tracking, "PATHS", dataclasses.replace(tracking.PATHS, reports=tmp_path))
    return tmp_path


def _broken_mlflow(message: str) -> types.ModuleType:
    """A fake MLflow module that fails on opening, like a locked store."""
    fake = types.ModuleType("mlflow")

    def fail(*_args, **_kwargs):
        raise RuntimeError(message)

    fake.set_tracking_uri = fail
    fake.set_experiment = fail
    fake.start_run = fail
    fake.log_params = fail
    fake.log_metric = fail
    fake.end_run = fail
    return fake


def test_a_locked_store_lets_training_carry_on(reports, monkeypatch):
    """This is the failure observed when the repository sits on a synchronised folder."""
    monkeypatch.setitem(sys.modules, "mlflow", _broken_mlflow("database is locked"))

    with tracking.track("trial", {"lr": 2e-4}) as summary:
        summary.metrics["accuracy"] = 0.9

    written = json.loads((reports / "training" / "trial.json").read_text(encoding="utf-8"))
    assert written["metrics"]["accuracy"] == 0.9


def test_an_absent_mlflow_lets_training_carry_on(reports, monkeypatch):
    """A ``None`` entry in ``sys.modules`` makes the import raise ``ImportError``."""
    monkeypatch.setitem(sys.modules, "mlflow", None)

    with tracking.track("without_mlflow", {}) as summary:
        summary.metrics["accuracy"] = 0.5

    written = json.loads((reports / "training" / "without_mlflow.json").read_text(encoding="utf-8"))
    assert written["metrics"]["accuracy"] == 0.5


def test_a_failing_run_still_leaves_its_summary(reports, monkeypatch):
    """What was measured before the error is the first piece of the diagnosis."""
    monkeypatch.setitem(sys.modules, "mlflow", _broken_mlflow("database is locked"))

    with (
        pytest.raises(ValueError, match="diverging loss"),
        tracking.track("interrupted", {}) as summary,
    ):
        summary.metrics["last_loss"] = 42.0
        raise ValueError("diverging loss")

    written = json.loads((reports / "training" / "interrupted.json").read_text(encoding="utf-8"))
    assert written["metrics"]["last_loss"] == 42.0


def test_an_interrupted_run_is_marked_as_failed(reports, monkeypatch):
    """Otherwise the MLflow interface would mix completed and interrupted runs."""
    statuses = []
    fake = types.ModuleType("mlflow")
    fake.set_tracking_uri = lambda *_a, **_k: None
    fake.set_experiment = lambda *_a, **_k: None
    fake.start_run = lambda *_a, **_k: None
    fake.log_params = lambda *_a, **_k: None
    fake.log_metric = lambda *_a, **_k: None
    fake.end_run = lambda status="FINISHED": statuses.append(status)
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    with tracking.track("completed", {}):
        pass
    with pytest.raises(ValueError, match="boom"), tracking.track("interrupted", {}):
        raise ValueError("boom")

    assert statuses == ["FINISHED", "FAILED"]


def test_the_store_address_follows_the_mlflow_variable(monkeypatch):
    """On a machine whose repository is synchronised, that is the only way out."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///elsewhere/mlflow.db")
    assert tracking.tracking_uri() == "sqlite:///elsewhere/mlflow.db"


def test_the_environment_description_degrades_rather_than_raising(monkeypatch):
    """Continuous integration installs the project without torch: a summary must still exist.

    These very tests run in that environment. A description that raised there would take down
    the four tests above, which exist to prove that tracking never fails a run.
    """
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "trl", None)

    described = tracking.describe_environment()

    assert described["torch"] == tracking.UNAVAILABLE
    assert described["gpu"] == "none"
