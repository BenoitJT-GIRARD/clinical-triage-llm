"""Audit log of triage interactions.

The requirement is to trace every interaction for medical audits. Concretely: to be able,
months later, to find what was asked, what the agent answered, with which model, how fast, and
what the explicit rule would have said about it.

Five properties, none of which goes without saying:

- **both texts are anonymised before being written**, each by the engine of its own language.
  The log is the only place in the system where real patient data would settle durably. The
  description received goes through GDPR masking before reaching the disk — and so does the
  answer, because the justification echoes the nurse's narrative and would bring back through
  one door a name masked at the other;
- **the model version is the one actually loaded**, passed by the service at call time. A
  configuration constant could describe a model that is not the one that answered;
- **the retention is applied, not only announced.** One file per day; a file past the retention period goes when the service
  starts, or when a new day's first line is written;
- **a modification shows.** Every line holds the SHA-256 digest of its predecessor, even across a
  day's boundary, so a line edited, inserted or deleted breaks the chain from there on, and
  :func:`verify` names the place. It detects, it does not prevent — a log a user can delete is still
  a file — and that is the property an audit needs;
- **one writer.** Only within one process is an append atomic, which is why the deployment
  runs a single gateway container.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from clinical_triage.config import PATHS
from clinical_triage.utils import get_logger

logger = get_logger("audit")

# Retention period announced to the users of the service, written into every line so that the
# obligation travels with the data, and applied by `purge_expired`.
RETENTION_DAYS = 365

#: The link of the first line ever written: no line precedes it.
GENESIS = "0" * 64


def audit_directory() -> Path:
    """Directory of the audit log, overridable for container deployment."""
    configured = os.getenv("TRIAGE_AUDIT_DIR")
    return Path(configured) if configured else PATHS.logs / "audit"


def _day_file(directory: Path, day: date) -> Path:
    return directory / f"{day.isoformat()}.jsonl"


def _files(directory: Path) -> list[Path]:
    """The day files, oldest first: their names sort as their dates do."""
    return sorted(directory.glob("????-??-??.jsonl"))


def purge_expired(directory: Path, today: date, retention_days: int = RETENTION_DAYS) -> list[Path]:
    """Delete the day files older than the retention period, and return them."""
    limit = today - timedelta(days=retention_days)
    expired = [path for path in _files(directory) if date.fromisoformat(path.stem) < limit]
    for path in expired:
        path.unlink()
    if expired:
        logger.info("Audit log: %d day file(s) past the retention deleted.", len(expired))
    return expired


def check_log_is_writable(directory: Path | None = None) -> Path:
    """Open the log for writing at startup, purge it, and refuse to serve otherwise.

    Traceability is a requirement of this service: answering without being able to record the
    answer does not meet it. Without this check, an unwritable log only shows up at the first
    request, after the decision has been produced, as a 500 whose cause is in the container logs.
    The case occurs as soon as a mounted volume belongs to another user.
    """
    directory = directory or audit_directory()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        purge_expired(directory, datetime.now(UTC).date())
        with _day_file(directory, datetime.now(UTC).date()).open("a", encoding="utf-8"):
            pass
    except OSError as error:
        raise RuntimeError(
            f"Audit log not writable ({directory}): {error}. Tracing every interaction is a "
            "requirement of this service: it does not start without it."
        ) from error
    return directory


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
    # True when the local engine bounded a description longer than its window: the model then
    # read part of the narrative only, and a decision taken that way must leave a trace.
    description_truncated: bool
    latency_ms: float
    model: str
    engine: str
    retention_days: int
    previous_digest: str


def mask(text: str, lang: str | None = None) -> str:
    """Mask personal data in a text, with the engine of the text's language."""
    from clinical_triage.data.anonymize import anonymize_text, detect_language

    return anonymize_text(text, lang or detect_language(text))


def _digest(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def _last_line(directory: Path) -> str | None:
    for path in reversed(_files(directory)):
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return None


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
    directory: Path | None = None,
) -> AuditEntry:
    """Anonymise then record one interaction, chained to the line before it."""
    directory = directory or audit_directory()
    directory.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).date()
    day_file = _day_file(directory, today)
    if not day_file.exists():
        purge_expired(directory, today)
    previous = _last_line(directory)
    entry = AuditEntry(
        request_id=request_id,
        timestamp=datetime.now(UTC).isoformat(),
        anonymised_symptoms=mask(symptoms),
        level=level,
        rule_level=rule_level,
        rule_reasons=rule_reasons or [],
        # The model is asked to answer in French; an answer that breaks that contract is masked
        # in its own language all the same.
        anonymised_answer=mask(answer),
        description_truncated=description_truncated,
        latency_ms=round(latency_ms, 1),
        model=model,
        engine=engine,
        retention_days=RETENTION_DAYS,
        previous_digest=_digest(previous) if previous is not None else GENESIS,
    )
    with day_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    return entry


def read_entries(directory: Path | None = None, limit: int | None = None) -> list[dict]:
    """Read the audit log back, oldest first, for verification or export."""
    directory = directory or audit_directory()
    if not directory.exists():
        return []
    rows = [
        json.loads(line)
        for path in _files(directory)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[-limit:] if limit else rows


def verify(directory: Path | None = None) -> list[str]:
    """Walk the chain and name every line whose link does not match the line before it.

    The first line kept is the anchor: after a purge, the line it points to is legitimately
    gone.
    """
    directory = directory or audit_directory()
    problems = []
    previous: str | None = None
    for path in _files(directory) if directory.exists() else []:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            link = json.loads(line).get("previous_digest")
            if previous is not None and link != _digest(previous):
                problems.append(f"{path.name}, line {number}: the chain breaks here")
            previous = line
    return problems
