"""The product, started and questioned from outside its process.

What this tier proves that the others cannot: that the gateway really starts under ``uvicorn``,
that it presents its key to the engine over the network, that it refuses to start without one,
that the quota holds across the process boundary, and that the file a medical audit would read
lands on the disk with nothing identifying left in it.

Everything here goes through HTTP. Nothing imports the service.
"""

from __future__ import annotations

import subprocess
import time

import httpx
import pytest

pytestmark = [pytest.mark.system, pytest.mark.slow]


def test_the_probe_answers_without_a_key_and_finds_the_engine(service):
    """A probe that demanded a key would be useless to whatever restarts the container."""
    reply = httpx.get(f"{service.url}/health", timeout=10)
    assert reply.status_code == 200
    body = reply.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["backend"] == "vllm"


def test_a_triage_crosses_the_two_processes_and_comes_back_structured(service):
    """The whole chain: HTTP in, engine call out, parsed answer and rule verdict back."""
    reply = httpx.post(
        f"{service.url}/triage",
        json={"symptoms": "Douleur thoracique et sueurs depuis 20 minutes.", "patient_age": 62},
        headers=service.headers(),
        timeout=60,
    )
    assert reply.status_code == 200
    body = reply.json()
    assert body["level"] == "URGENCE_VITALE"
    assert body["rule_level"] == "URGENCE_VITALE"
    assert body["agreement"] is True
    assert body["recommendation"]
    assert body["request_id"]

    # The engine received the framed prompt, not the raw field: the gateway owns the format.
    prompt = service.prompts()[-1]
    assert "triage" in prompt.lower()
    assert "Douleur thoracique" in prompt


def test_the_gateway_presents_its_key_to_the_engine(service):
    """Without that header the engine answers 401 and every triage would come back as 503.

    The stand-in engine demands the key, exactly as vLLM started with ``--api-key`` does. The
    test questions it from outside first, to show it really refuses, and then reads the
    gateway's probe: the only way that probe reports the engine as reachable is by presenting
    the key itself.
    """
    unauthenticated = httpx.get(f"{service.engine_url}/v1/models", timeout=10)
    assert unauthenticated.status_code == 401

    presented = httpx.get(
        f"{service.engine_url}/v1/models",
        headers={"Authorization": f"Bearer {service.key}"},
        timeout=10,
    )
    assert presented.status_code == 200

    assert httpx.get(f"{service.url}/health", timeout=10).json()["model_loaded"] is True


def test_a_request_without_the_service_key_never_reaches_the_engine(service):
    before = len(service.prompts())
    refused = httpx.post(
        f"{service.url}/triage",
        json={"symptoms": "Douleur thoracique."},
        headers=service.headers("wrong"),
        timeout=30,
    )
    assert refused.status_code == 401
    assert len(service.prompts()) == before, "the engine was called for a rejected request"


def test_the_audit_file_lands_on_the_disk_with_nothing_identifying_in_it(service):
    """The log is the one place patient data would settle, and it is kept for a year.

    Both texts are masked before being written — the description and the answer — and that is
    checked here on the file the container really writes, not on the return value of a function.
    """
    httpx.post(
        f"{service.url}/triage",
        json={
            "symptoms": (
                "Mme Martine Dupont, douleur thoracique depuis 20 minutes. "
                "Joindre au 06 11 22 33 44."
            )
        },
        headers=service.headers(),
        timeout=90,
    ).raise_for_status()

    trace = service.traces()[-1]
    assert trace["engine"] == "vllm"
    assert trace["level"] == "URGENCE_VITALE"
    assert trace["retention_days"] == 365
    written = trace["anonymised_symptoms"] + trace["anonymised_answer"]
    assert "Martine" not in written
    assert "Dupont" not in written
    assert "06 11 22 33 44" not in written
    assert "<PERSON>" in trace["anonymised_symptoms"]


def test_the_quota_holds_across_the_process_boundary(start_service):
    """Every call takes a GPU. The counter lives in the served process, not in the client."""
    quota = start_service(TRIAGE_RATE_LIMIT="3")
    codes = [
        httpx.post(
            f"{quota.url}/questionnaire/next",
            json={"chief_complaint": "toux"},
            headers=quota.headers(),
            timeout=30,
        ).status_code
        for _ in range(5)
    ]
    assert codes[:3] == [200, 200, 200]
    assert 429 in codes[3:]


def test_the_service_refuses_to_start_without_a_key(gateway_without_a_key):
    """Checked where it matters: the process really exits, rather than serving in the open.

    The integration tier reads the same requirement through the lifespan; here it is the exit
    code, which is what an orchestrator sees.
    """
    refused = subprocess.run(  # the command line comes from the fixture, no shell
        gateway_without_a_key.command,
        cwd=gateway_without_a_key.cwd,
        env=gateway_without_a_key.environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=gateway_without_a_key.timeout,
        check=False,
    )

    assert refused.returncode != 0
    assert "TRIAGE_API_KEY" in refused.stdout + refused.stderr


def test_the_contract_is_published_at_the_port(service):
    """An integrator reads the schema from the running service, not from the source."""
    started = time.monotonic()
    schema = httpx.get(f"{service.url}/openapi.json", timeout=15).json()
    assert time.monotonic() - started < 15
    assert set(schema["paths"]) >= {"/triage", "/questionnaire/next", "/health"}
    assert schema["info"]["version"] == "1.0.0"
