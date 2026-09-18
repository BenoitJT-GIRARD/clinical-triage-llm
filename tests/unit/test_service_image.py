"""Consistency of the service image's dependencies.

``requirements-api.txt`` is generated from ``requirements-api.in``: the first carries the full
pinned closure, the second the constraints. The two files are edited separately, and nothing
stops the generated one being edited by hand — in which case the image would install something
other than what was resolved and audited.

These tests are static: they compare the two files, resolving nothing and touching no network.
"""

from __future__ import annotations

import re

import pytest

from clinical_triage.config import PATHS

CONSTRAINTS = PATHS.root / "infra" / "requirements-api.in"
CLOSURE = PATHS.root / "infra" / "requirements-api.txt"


def _pinned(path) -> dict[str, str]:
    """Pinned packages of a dependency file, names normalised."""
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([a-zA-Z0-9._-]+)(?:\[[^\]]+\])?==(\S+)", line)
        if match:
            found[match.group(1).lower().replace("_", "-")] = match.group(2)
    return found


@pytest.fixture(scope="module")
def constraints() -> dict[str, str]:
    return _pinned(CONSTRAINTS)


@pytest.fixture(scope="module")
def closure() -> dict[str, str]:
    return _pinned(CLOSURE)


def test_the_closure_is_much_wider_than_the_constraints(constraints, closure):
    """Guard rail of the test itself: a broken read would leave it always green."""
    assert len(constraints) >= 7
    assert len(closure) > 3 * len(constraints)


def test_every_constraint_appears_at_the_same_version(constraints, closure):
    """The generated file must not drift from what was asked for."""
    gaps = {
        name: (version, closure.get(name))
        for name, version in constraints.items()
        if closure.get(name) != version
    }
    assert gaps == {}


def test_both_spacy_models_are_in_the_closure(closure):
    """Presidio masks nothing without them, and the log anonymisation falls over."""
    text = CLOSURE.read_text(encoding="utf-8")
    assert "fr_core_news_md-3.8.0" in text
    assert "en_core_web_sm-3.8.0" in text


def test_the_image_carries_neither_torch_nor_transformers(closure):
    """Generation is delegated to vLLM: the image has no business weighing three gigabytes."""
    assert "torch" not in closure
    assert "transformers" not in closure
    assert "datasets" not in closure


def test_the_generated_file_declares_itself_as_such():
    """Without that warning, the next fix would be made in the wrong file."""
    assert "GENERATED FILE" in CLOSURE.read_text(encoding="utf-8")
