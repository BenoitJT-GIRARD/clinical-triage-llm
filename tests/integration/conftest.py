"""Isolation of the integration tier.

This tier drives the assembled service in the test process, with the inference engine replaced
by a controlled function. Two settings therefore have to be forced rather than inherited:

- the audit log, which would otherwise be the checkout's own;
- the engine's address, which defaults to ``http://localhost:8000`` — the port the local stack
  of ``infra/docker-compose.yml`` uses. A developer who happens to have that stack running
  would see the health-probe tests reverse their verdict, and the suite would depend on what
  else is running on the machine. It is pinned here to a port nothing listens on.
"""

from __future__ import annotations

import pytest

#: Reserved by convention and never served: connecting to it is refused immediately.
NO_ENGINE = "http://127.0.0.1:1"


@pytest.fixture(autouse=True)
def a_service_that_depends_on_nothing_else(tmp_path, monkeypatch):
    monkeypatch.setenv("TRIAGE_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("TRIAGE_VLLM_URL", NO_ENGINE)
