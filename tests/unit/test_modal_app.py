"""The Modal deployment descriptor must load, and read its revision.

``modal deploy`` starts by importing ``infra/modal_app.py``: a decorator error, a keyword gone
from the API or a missing import fails there before ever reaching the network. That file being
executed nowhere else, nothing would check it without these tests.

Continuous deployment passes the model revision through the environment, at ``modal deploy``
time. It is then baked into the engine image, because the container reimports this file in an
environment that holds nothing of the deployment's.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from clinical_triage.config import MODEL, PATHS

PATH = PATHS.root / "infra" / "modal_app.py"


def _load():
    """Import the descriptor as ``modal deploy`` would."""
    spec = importlib.util.spec_from_file_location("modal_app_under_test", PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["modal_app_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def descriptor(monkeypatch):
    monkeypatch.delenv("TRIAGE_MODEL_ID", raising=False)
    monkeypatch.delenv("TRIAGE_MODEL_REVISION", raising=False)
    return _load()


def test_the_descriptor_loads(descriptor):
    """First step of ``modal deploy``: if it passes here, it will pass there."""
    assert descriptor.app.name == "clinical-triage"
    assert descriptor.Engine is not None
    assert descriptor.gateway is not None


def test_the_engine_serves_the_projects_final_model(descriptor):
    assert MODEL.hub_merged_model_id == descriptor.MODEL_ID


def test_the_model_references_are_baked_into_the_image(descriptor):
    """The container reimports this file without the deployment's environment.

    What is not written into the image at deployment time is lost.
    """
    variables = descriptor.ENGINE_ENVIRONMENT
    assert variables["TRIAGE_MODEL_ID"] == descriptor.MODEL_ID
    assert variables["TRIAGE_MODEL_REVISION"] == descriptor.REVISION


def test_the_revision_follows_the_continuous_deployment_variable(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL_ID", "another-account/a-model")
    monkeypatch.setenv("TRIAGE_MODEL_REVISION", "model-v9.9.9")
    descriptor = _load()
    assert descriptor.MODEL_ID == "another-account/a-model"
    assert descriptor.REVISION == "model-v9.9.9"


def test_an_empty_variable_falls_back_to_the_pinned_version(monkeypatch):
    """An undefined GitHub repository variable arrives empty, not absent."""
    monkeypatch.setenv("TRIAGE_MODEL_ID", "")
    monkeypatch.setenv("TRIAGE_MODEL_REVISION", "")
    descriptor = _load()
    assert MODEL.hub_merged_model_id == descriptor.MODEL_ID
    assert descriptor.REVISION.startswith("model-v")


def test_an_empty_account_does_not_produce_an_identifier_with_a_leading_slash(monkeypatch):
    """The deployment chain passes only the account: empty, it must fall back.

    Composed by interpolation, an undefined account gave "/qwen3-1.7b-clinical-triage", which the
    Hub refuses.
    """
    monkeypatch.delenv("TRIAGE_MODEL_ID", raising=False)
    monkeypatch.setenv("HF_NAMESPACE", "")
    descriptor = _load()

    assert not descriptor.MODEL_ID.startswith("/")
    assert descriptor.MODEL_ID == "BenoitJT-GIRARD/qwen3-1.7b-clinical-triage"


def test_the_continuous_deployment_account_is_used(monkeypatch):
    """Another account publishes under its own name with no change to the code."""
    monkeypatch.delenv("TRIAGE_MODEL_ID", raising=False)
    monkeypatch.setenv("HF_NAMESPACE", "another-account")
    descriptor = _load()

    assert descriptor.MODEL_ID == "another-account/qwen3-1.7b-clinical-triage"
