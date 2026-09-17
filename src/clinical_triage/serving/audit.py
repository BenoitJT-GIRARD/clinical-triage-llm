"""Audit log of triage interactions.

The requirement is to trace every interaction for medical audits. Concretely: to be able,
months later, to find what was asked, what the agent answered, with which model, how fast, and
what the explicit rule would have said about it.

Two compliance points that are not obvious:

- **both texts are anonymised before being written.** The log is the only place in the system
  where real patient data would settle durably. The description received therefore goes through
  GDPR masking before reaching the disk — **and so does the answer**, because the justification
  the model produces echoes the nurse's narrative and would bring back through one door a name
  masked at the other. A module that claims to write anonymised data must anonymise it itself,
  not trust its caller;
- **the model version is the one actually loaded**, passed by the service at call time. A
  configuration constant could describe a model that is not the one that answered, and
  falsifiable traceability traces nothing.

The file is JSONL: one line per interaction, readable without a tool, and appended atomically
as long as writes come from a single process.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from clinical_triage.config import PATHS
from clinical_triage.utils import get_logger

logger = get_logger("audit")

# Retention period announced to users of the service. It is written into every line so that the
# obligation is carried by the data itself.
RETENTION_DAYS = 365


def audit_log_path() -> Path:
    """Path of the audit log, overridable for container deployment."""
    configured = os.getenv("TRIAGE_AUDIT_LOG")
    return Path(configured) if configured else PATHS.logs / "audit_triage.jsonl"


def check_log_is_writable(log_path: Path | None = None) -> Path:
    """Open the log for writing at startup, and refuse to serve otherwise.

    Traceability is a compliance requirement: a service that answers without being able to
    record what it answered does not meet the contract. Without this check, an unavailable log
    only shows up at the first request, after the decision has been produced, as a 500 whose
    cause appears only in the container logs.

    The case occurs as soon as a mounted volume belongs to a user other than the one running
    the service.
    """
    path = log_path or audit_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8"):
            pass
    except OSError as error:
        raise RuntimeError(
            f"Audit log not writable ({path}): {error}. Tracing every interaction is a "
            "requirement of this service: it does not start without it."
        ) from error
    return path


def new_request_id() -> str:
    """Generate a unique interaction id."""
    return uuid.uuid4().hex[:16]


@dataclass(frozen=True)
class AuditEntry:
    """One line of the audit log."""

    request_id: str
    timestamp: str
    anonymised_symptoms: str
    level: str | None
    rule_level: str | None
    rule_reasons: list[str]
    anonymised_answer: str
    # True when the description exceeds the model window and is bounded before the call: the
    # model then reads only part of the narrative. A triage decision taken on an incomplete
    # narrative must leave a trace — that is the very purpose of this log.
    description_truncated: bool
    latency_ms: float
    model: str
    engine: str
    retention_days: int


def mask(text: str) -> str:
    """Mask personal data in a text without knowing which language it is in.

    The API accepts free text and the model is bilingual: a description arrives in English as
    readily as in French. Anonymising with the French engine alone would leave "Mr Jenkins" in
    the clear in the log — the one place in the system where personal data is kept, and for a
    year.

    Rather than guessing the language, both passes are chained. Both spaCy engines are already
    loaded in the service image, and the second pass sees of the first's text only what it did
    not mask: the placeholders one leaves behind are not proper nouns to the other.
    """
    from clinical_triage.data.anonymize import anonymize_text

    return anonymize_text(anonymize_text(text, "fr"), "en")


def record_interaction(
    request_id: str,
    symptoms: str,
    level: str | None,
    answer: str,
    latency_ms: float,
    model: str,
    engine: str,
    rule_level: str | None = None,
    rule_reasons: list[str] | None = None,
    description_truncated: bool = False,
    log_path: Path | None = None,
) -> AuditEntry:
    """Anonymise then record one interaction, and return the line written."""
    entry = AuditEntry(
        request_id=request_id,
        timestamp=datetime.now(UTC).isoformat(),
        anonymised_symptoms=mask(symptoms),
        level=level,
        rule_level=rule_level,
        rule_reasons=rule_reasons or [],
        anonymised_answer=mask(answer),
        description_truncated=description_truncated,
        latency_ms=round(latency_ms, 1),
        model=model,
        engine=engine,
        retention_days=RETENTION_DAYS,
    )
    path = log_path or audit_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    return entry


def read_entries(log_path: Path | None = None, limit: int | None = None) -> list[dict]:
    """Read the audit log back, for verification or export."""
    path = log_path or audit_log_path()
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    return lines[-limit:] if limit else lines
