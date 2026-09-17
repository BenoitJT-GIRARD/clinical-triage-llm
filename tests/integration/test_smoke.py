"""The run evidence says what really happened, or it is not written.

``reports/run-evidence.json`` is what replaces a green tick nobody can date: the command, the
day, the revision, the versions, the cost, and the files the run reproduced. Its value rests
entirely on one property — that it is **not** written when the run failed to reproduce what the
repository publishes. A piece of evidence that is written either way says "something ran", which
is the one thing nobody needed to know.

These tests drive the script's own functions, with the figure step replaced: redrawing the ten
figures is the business of ``test_build_figures.py``, and doing it twice would double the
suite's cost for nothing.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from clinical_triage.config import PATHS

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def smoke():
    """Load ``scripts/smoke.py``, which is not importable as a module."""
    path = PATHS.root / "scripts" / "smoke.py"
    spec = importlib.util.spec_from_file_location("smoke_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["smoke_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def a_run_that_reproduces(smoke, monkeypatch, tmp_path):
    """A run whose figure step succeeds and whose outputs match what is committed."""
    monkeypatch.setattr(smoke, "steps", lambda: None)
    monkeypatch.setattr(smoke, "reproduced", list)
    monkeypatch.setattr(smoke, "ROOT_DIR", tmp_path)
    return tmp_path / smoke.EVIDENCE


def test_the_evidence_carries_everything_needed_to_replay_the_run(smoke, a_run_that_reproduces):
    assert smoke.main() == 0

    evidence = json.loads(a_run_that_reproduces.read_text(encoding="utf-8"))
    assert evidence["command"] == "uv run python scripts/smoke.py"
    assert evidence["ran_at"].count("-") == 2
    assert evidence["duration_seconds"] >= 0
    assert evidence["key_output"] == list(smoke.KEY_OUTPUT)
    assert evidence["tool_versions"]["python"].startswith("3.12")
    assert set(smoke.TOOLS) <= set(evidence["tool_versions"])
    # Null and said so: the workflows are frozen to a manual trigger, so there is no run to
    # point at, and a missing key would read as an oversight.
    assert "workflow_run_url" in evidence
    assert evidence["workflow_run_url"] is None


def test_nothing_is_written_when_the_run_does_not_reproduce_what_is_published(
    smoke, monkeypatch, tmp_path, capsys
):
    """The whole point of the exercise: evidence that costs nothing proves nothing."""
    monkeypatch.setattr(smoke, "steps", lambda: None)
    monkeypatch.setattr(
        smoke,
        "reproduced",
        lambda: ["reports/figures/systems_comparison.png — differs from the committed version"],
    )
    monkeypatch.setattr(smoke, "ROOT_DIR", tmp_path)

    assert smoke.main() == 1
    assert not (tmp_path / smoke.EVIDENCE).exists()
    printed = capsys.readouterr().err
    assert "systems_comparison.png" in printed
    assert "not written" in printed


def test_a_failed_step_stops_the_run_instead_of_being_recorded_as_a_success(smoke, monkeypatch):
    """A step that failed must not leave evidence saying the run succeeded."""

    def explode():
        raise RuntimeError("scripts/build_figures.py failed with code 1")

    monkeypatch.setattr(smoke, "steps", explode)
    with pytest.raises(RuntimeError, match="build_figures"):
        smoke.main()


def test_the_key_outputs_are_the_figures_the_repository_publishes(smoke):
    """The list is what the evidence claims to have reproduced: it must be the real one."""
    published = {path.name for path in PATHS.figures.glob("*.png")}
    declared = {name.rsplit("/", 1)[-1] for name in smoke.KEY_OUTPUT if name.endswith(".png")}
    assert declared == published
    assert "reports/figures/MANIFEST.json" in smoke.KEY_OUTPUT


def test_an_output_that_is_not_tracked_counts_as_not_reproduced(smoke, monkeypatch, tmp_path):
    """An untracked file cannot be compared with anything, so it proves nothing.

    Produced but never committed, the ten figures would be redrawn at every run and would match
    themselves every time.
    """
    monkeypatch.setattr(smoke, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(smoke, "KEY_OUTPUT", ("reports/figures/systems_comparison.png",))
    produced = tmp_path / "reports" / "figures" / "systems_comparison.png"
    produced.parent.mkdir(parents=True)
    produced.write_bytes(b"\x89PNG")
    # No repository under ``tmp_path``: every git question comes back empty, which is what a
    # file git does not know looks like.
    monkeypatch.setattr(smoke, "_git", lambda *arguments: "")

    assert smoke.reproduced() == [
        "reports/figures/systems_comparison.png — produced but not tracked by git"
    ]


def test_a_missing_output_is_named_as_missing(smoke, monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(smoke, "KEY_OUTPUT", ("reports/figures/absent.png",))

    assert smoke.reproduced() == ["reports/figures/absent.png — not produced"]


def test_the_versions_recorded_are_the_ones_that_would_explain_a_different_number(smoke):
    versions = smoke.tool_versions()
    assert versions["matplotlib"] != "not installed"
    assert versions["numpy"] != "not installed"
    assert "python" in versions
