"""Tests of the triage API.

These tests need no model: the vLLM engine is replaced by a controlled generation function.
What is checked here is not the quality of the answers — it is the service contract:
authentication, rate limiting, health probe, traceability and the shape of the reply.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from clinical_triage.serving import api as module_api
from clinical_triage.serving.api import RateLimiter

pytestmark = pytest.mark.integration

KEY = "test-key"

# The answer is in French because the service answers in French: it is the model's output, not
# prose about the project.
MODEL_ANSWER = (
    "Niveau de priorité : URGENCE_VITALE\n"
    "Justification : Douleur thoracique avec sueurs chez un patient à risque.\n"
    "Recommandation : Prise en charge immédiate et appel du 15 (SAMU)."
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The service configured in vLLM mode, with a simulated generation."""
    monkeypatch.setenv("TRIAGE_BACKEND", "vllm")
    monkeypatch.setenv("TRIAGE_API_KEY", KEY)
    monkeypatch.setenv("TRIAGE_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("TRIAGE_RATE_LIMIT", "5")
    monkeypatch.setattr(
        module_api,
        "_generate_vllm",
        lambda request, symptoms: (MODEL_ANSWER, "URGENCE_VITALE", 42.0, False),
    )
    with TestClient(module_api.app) as tester:
        yield tester


def test_the_service_refuses_to_start_without_a_key(monkeypatch):
    """An authentication that switches itself off when a variable is missing is not one."""
    monkeypatch.setenv("TRIAGE_BACKEND", "vllm")
    monkeypatch.delenv("TRIAGE_API_KEY", raising=False)
    monkeypatch.delenv("TRIAGE_ALLOW_ANONYMOUS", raising=False)
    with pytest.raises(RuntimeError, match="TRIAGE_API_KEY"), TestClient(module_api.app):
        pass


def test_open_mode_has_to_be_asked_for_explicitly(monkeypatch, tmp_path):
    monkeypatch.setenv("TRIAGE_BACKEND", "vllm")
    monkeypatch.delenv("TRIAGE_API_KEY", raising=False)
    monkeypatch.setenv("TRIAGE_ALLOW_ANONYMOUS", "true")
    monkeypatch.setenv("TRIAGE_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    with TestClient(module_api.app) as tester:
        assert (
            tester.post("/questionnaire/next", json={"chief_complaint": "toux"}).status_code == 200
        )


def test_the_probe_reports_an_unreachable_engine(client):
    """A probe that is always green triggers no restart on the day of the outage."""
    reply = client.get("/health")
    assert reply.status_code == 200
    body = reply.json()
    assert body["backend"] == "vllm"
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["detail"]


def test_the_probe_does_not_publish_the_engines_address(client):
    """The probe asks for neither key nor quota: it says there is an outage, never where.

    The httpx message carries the URL it queried. On Modal that URL is precisely what protects
    the engine, and the probe was making it public.
    """
    detail = client.get("/health").json()["detail"]
    address = module_api.app.state.settings.vllm_url
    assert address not in detail
    assert "http" not in detail


def test_triage_requires_a_key(client):
    assert client.post("/triage", json={"symptoms": "douleur thoracique"}).status_code == 401
    assert (
        client.post(
            "/triage", json={"symptoms": "douleur thoracique"}, headers={"X-API-Key": "wrong"}
        ).status_code
        == 401
    )


def test_triage_returns_a_structured_reply(client):
    reply = client.post(
        "/triage",
        json={"symptoms": "Douleur thoracique et sueurs depuis 20 minutes.", "patient_age": 62},
        headers={"X-API-Key": KEY},
    )
    assert reply.status_code == 200
    body = reply.json()
    assert body["level"] == "URGENCE_VITALE"
    assert body["level_label"].startswith("Urgence maximale")
    assert body["justification"]
    assert body["recommendation"]
    assert body["request_id"]
    assert body["latency_ms"] == 42.0


def test_the_reply_exposes_what_the_rule_decided(client):
    """When the model and the rule diverge, reception must be able to see it."""
    reply = client.post(
        "/triage",
        json={"symptoms": "Douleur thoracique et sueurs depuis 20 minutes."},
        headers={"X-API-Key": KEY},
    ).json()
    assert reply["rule_level"] == "URGENCE_VITALE"
    assert reply["rule_reasons"]
    assert reply["agreement"] is True


def test_a_disagreement_between_the_model_and_the_rule_is_flagged(client):
    reply = client.post(
        "/triage",
        json={"symptoms": "Rhume banal depuis deux jours, pas de fièvre."},
        headers={"X-API-Key": KEY},
    ).json()
    assert reply["level"] == "URGENCE_VITALE"  # the simulated answer
    assert reply["rule_level"] == "CONSULTATION_DIFFEREE"
    assert reply["agreement"] is False


def test_every_triage_leaves_a_trace(client, tmp_path):
    client.post("/triage", json={"symptoms": "Douleur thoracique."}, headers={"X-API-Key": KEY})
    rows = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    trace = json.loads(rows[0])
    assert trace["level"] == "URGENCE_VITALE"
    assert trace["engine"] == "vllm"
    assert trace["anonymised_symptoms"]


def test_an_empty_description_is_refused(client):
    reply = client.post("/triage", json={"symptoms": "  "}, headers={"X-API-Key": KEY})
    assert reply.status_code == 422


def test_the_rate_limit_protects_the_gpu(client):
    """Every call takes a GPU: an endpoint without a quota can be saturated."""
    codes = [
        client.post(
            "/triage", json={"symptoms": "Douleur thoracique."}, headers={"X-API-Key": KEY}
        ).status_code
        for _ in range(7)
    ]
    assert codes[:5] == [200] * 5
    assert codes[5] == 429


def test_the_limiter_forgets_silent_callers():
    """A counter that protects the service must not grow without end.

    In open mode the key is the caller's address: without forgetting, the service would keep one
    entry per address seen since it started.
    """
    limiter = RateLimiter(requests_per_minute=10)
    for number in range(50):
        limiter.allows(f"10.0.0.{number}")
    assert len(limiter._history) == 50

    # Move a minute and a second ahead without waiting: the limiter reads the monotonic clock,
    # so shifting it is enough.
    later = time.monotonic() + 61
    limiter._forget_silent_callers(later)
    assert limiter._history == {}


def test_the_questionnaire_adapts_its_questions_to_the_complaint(client):
    reply = client.post(
        "/questionnaire/next",
        json={
            "chief_complaint": "gêne dans la poitrine à l'effort",
            "answers": {"conscience": "oui", "respiration": "non", "saignement": "non"},
        },
        headers={"X-API-Key": KEY},
    ).json()
    assert reply["theme"] == "douleur_thoracique"
    assert reply["next_question_id"] == "irradiation"
    assert reply["finished"] is False


def test_the_questionnaire_stops_on_a_vital_sign(client):
    reply = client.post(
        "/questionnaire/next",
        json={"chief_complaint": "douleur thoracique violente"},
        headers={"X-API-Key": KEY},
    ).json()
    assert reply["finished"] is True
    assert reply["next_question_id"] is None


def test_the_openapi_contract_is_published(client):
    schema = client.get("/openapi.json").json()
    assert "/triage" in schema["paths"]
    assert "/questionnaire/next" in schema["paths"]
    assert "/health" in schema["paths"]


def test_the_gateway_presents_a_key_to_the_engine(monkeypatch):
    """Without it, the engine's address alone buys GPU inference.

    Free, with no quota, no anonymisation and no line in the audit log. Locally the engine asks
    for none and the header is ignored; at a host it demands one, and the gateway must have it.
    """
    monkeypatch.setenv("TRIAGE_API_KEY", "service-key")
    monkeypatch.delenv("TRIAGE_VLLM_API_KEY", raising=False)
    settings = module_api.Settings.from_env()
    assert settings.vllm_headers == {"Authorization": "Bearer service-key"}


def test_a_key_of_the_engines_own_wins_over_the_services(monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEY", "service-key")
    monkeypatch.setenv("TRIAGE_VLLM_API_KEY", "engine-key")
    assert module_api.Settings.from_env().vllm_headers["Authorization"] == "Bearer engine-key"


def test_with_no_key_at_all_no_header_is_sent(monkeypatch):
    """The open demonstration mode must not send an empty "Bearer"."""
    monkeypatch.delenv("TRIAGE_API_KEY", raising=False)
    monkeypatch.delenv("TRIAGE_VLLM_API_KEY", raising=False)
    assert module_api.Settings.from_env().vllm_headers == {}


def test_in_open_mode_the_quota_is_not_evaded_by_changing_header(monkeypatch, tmp_path):
    """The header counts as an identity only once it has been verified.

    In open mode nobody verifies it: it still served as the key of the counting bucket, so a
    caller changing it on every request got a fresh bucket every time. The quota is the last
    guard rail of that mode, and it was counting nothing.
    """
    monkeypatch.setenv("TRIAGE_BACKEND", "vllm")
    monkeypatch.delenv("TRIAGE_API_KEY", raising=False)
    monkeypatch.setenv("TRIAGE_ALLOW_ANONYMOUS", "true")
    monkeypatch.setenv("TRIAGE_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("TRIAGE_RATE_LIMIT", "3")
    with TestClient(module_api.app) as tester:
        codes = [
            tester.post(
                "/questionnaire/next",
                json={"chief_complaint": "toux"},
                headers={"X-API-Key": f"forged-{number}"},
            ).status_code
            for number in range(6)
        ]
    assert codes[:3] == [200, 200, 200]
    assert 429 in codes[3:]


def test_the_questionnaire_refuses_an_oversized_body(client):
    """The answers field was the only text of the API without a bound.

    Everything in it is concatenated into a single string, read back by the triage rule: one
    request carrying a few hundred megabytes of JSON was enough to saturate the container, which
    has no memory limit either.
    """
    oversized = {"chief_complaint": "toux", "answers": {f"q{i}": "oui" for i in range(64)}}
    assert (
        client.post("/questionnaire/next", json=oversized, headers={"X-API-Key": KEY}).status_code
        == 422
    )
    too_long = {"chief_complaint": "toux", "answers": {"q1": "o" * 5000}}
    assert (
        client.post("/questionnaire/next", json=too_long, headers={"X-API-Key": KEY}).status_code
        == 422
    )


def test_a_description_that_is_too_long_is_bounded_and_said_to_be(client, monkeypatch):
    """A description overflowing the model window used to cut the generation short.

    The contract accepted four thousand characters, the window holds far fewer, and the outcome
    was a tensor dimension error at the precise moment a clinician is waiting for an answer. It
    is bounded now — and the reply says so, because a clinical narrative silently truncated is
    exactly what a decision-support system must not produce.
    """
    received: list[str] = []

    def _generate(request, symptoms):
        bounded, truncated = module_api.bound_in_characters(symptoms)
        received.append(bounded)
        return MODEL_ANSWER, "URGENCE_VITALE", 42.0, truncated

    monkeypatch.setattr(module_api, "_generate_vllm", _generate)
    long_one = "Le patient décrit une gêne diffuse et variable depuis plusieurs jours. " * 40
    reply = client.post("/triage", json={"symptoms": long_one}, headers={"X-API-Key": KEY})
    assert reply.status_code == 200
    assert reply.json()["description_truncated"] is True
    assert len(received[0]) == module_api.MAX_CHARACTERS


def test_a_normal_description_is_not_flagged_as_bounded(client):
    reply = client.post(
        "/triage",
        json={"symptoms": "Homme de 62 ans, douleur thoracique depuis 20 minutes."},
        headers={"X-API-Key": KEY},
    )
    assert reply.json()["description_truncated"] is False
