"""Isolation of the unit tier.

The unit tier exercises the package inside the test process, one module at a time. The one
thing it can reach without meaning to is the checkout's own audit log: any call to the tracing
module appends to ``var/logs/audit_triage.jsonl`` unless told otherwise, and a test suite has no
business writing into the file a medical audit would later read.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def audit_log_outside_the_checkout(tmp_path, monkeypatch):
    """Send every trace written by a unit test to a temporary file."""
    monkeypatch.setenv("TRIAGE_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
