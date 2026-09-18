"""Tests of the audit log.

The log carries the traceability requirement. Two properties matter as much as its content: it
anonymises what it writes, and it records the model actually used rather than a configuration
value.
"""

from __future__ import annotations

import json

import pytest

from clinical_triage.serving.audit import (
    RETENTION_DAYS,
    check_log_is_writable,
    new_request_id,
    read_entries,
    record_interaction,
)


def test_the_identifiers_are_unique():
    identifiers = {new_request_id() for _ in range(500)}
    assert len(identifiers) == 500


def test_an_interaction_is_written_with_every_field(tmp_path):
    log = tmp_path / "audit.jsonl"
    entry = record_interaction(
        request_id="abc123",
        symptoms="Homme de 62 ans, douleur thoracique.",
        level="URGENCE_VITALE",
        answer="Niveau de priorité : URGENCE_VITALE",
        latency_ms=421.37,
        model="qwen3-1.7b-clinical-triage",
        engine="vllm",
        rule_level="URGENCE_VITALE",
        rule_reasons=["douleur thoracique"],
        log_path=log,
    )
    row = json.loads(log.read_text(encoding="utf-8").strip())
    assert row["request_id"] == "abc123"
    assert row["level"] == "URGENCE_VITALE"
    assert row["rule_level"] == "URGENCE_VITALE"
    assert row["rule_reasons"] == ["douleur thoracique"]
    assert row["latency_ms"] == 421.4
    assert row["model"] == "qwen3-1.7b-clinical-triage"
    assert row["engine"] == "vllm"
    assert row["retention_days"] == RETENTION_DAYS
    assert row["timestamp"].endswith("+00:00")
    assert entry.request_id == "abc123"


def test_interactions_append_without_overwriting(tmp_path):
    log = tmp_path / "audit.jsonl"
    for index in range(3):
        record_interaction(
            request_id=f"id{index}",
            symptoms="Rhume banal.",
            level="CONSULTATION_DIFFEREE",
            answer="Niveau de priorité : CONSULTATION_DIFFEREE",
            latency_ms=10.0,
            model="model",
            engine="vllm",
            log_path=log,
        )
    rows = read_entries(log)
    assert [row["request_id"] for row in rows] == ["id0", "id1", "id2"]
    assert read_entries(log, limit=1)[0]["request_id"] == "id2"


def test_an_absent_log_reads_as_empty(tmp_path):
    assert read_entries(tmp_path / "nonexistent.jsonl") == []


def test_the_patient_name_is_masked_before_writing(tmp_path):
    """The masking belongs to the writer, and is not left to the caller, without
    trusting its caller."""
    log = tmp_path / "audit.jsonl"
    record_interaction(
        request_id="pii1",
        symptoms="Madame Dupont, 72 ans, joignable au 06 12 34 56 78, douleur thoracique.",
        level="URGENCE_VITALE",
        answer="Niveau de priorité : URGENCE_VITALE",
        latency_ms=12.0,
        model="model",
        engine="vllm",
        log_path=log,
    )
    written = json.loads(log.read_text(encoding="utf-8").strip())["anonymised_symptoms"]
    assert "Dupont" not in written
    assert "06 12 34 56 78" not in written
    # The clinical information, on the other hand, must survive the masking.
    assert "douleur thoracique" in written


def test_the_model_answer_is_masked_too(tmp_path):
    """The justification echoes the nurse's narrative, name included.

    Masking the description and letting the answer through would bring back through the door
    what was chased out of the window.
    """
    log = tmp_path / "audit.jsonl"
    record_interaction(
        request_id="pii2",
        symptoms="Madame Dupont, 72 ans, douleur thoracique.",
        level="URGENCE_VITALE",
        answer=(
            "Niveau de priorité : URGENCE_VITALE\n"
            "Justification : Madame Dupont présente une douleur thoracique.\n"
            "Recommandation : orientation immédiate."
        ),
        latency_ms=12.0,
        model="model",
        engine="vllm",
        log_path=log,
    )
    row = json.loads(log.read_text(encoding="utf-8").strip())
    assert "Dupont" not in row["anonymised_answer"]
    assert "URGENCE_VITALE" in row["anonymised_answer"]
    assert "douleur thoracique" in row["anonymised_answer"]


def test_an_english_text_is_masked_like_a_french_one(tmp_path):
    """The service receives free text, and the model is bilingual.

    Masking always started from the French engine, which does not spot a name in English
    syntax: "Mr Jenkins" stayed in the clear in the only file of the system that keeps personal
    data, and for a year.
    """
    log = tmp_path / "audit.jsonl"
    record_interaction(
        request_id="pii3",
        symptoms="Mr Jenkins, 72, reachable at +33 6 12 34 56 78, reports chest pain.",
        level="URGENCE_VITALE",
        answer="Priority level: URGENCE_VITALE — Mr Jenkins needs immediate care.",
        latency_ms=12.0,
        model="model",
        engine="vllm",
        log_path=log,
    )
    row = json.loads(log.read_text(encoding="utf-8").strip())
    assert "Jenkins" not in row["anonymised_symptoms"]
    assert "Jenkins" not in row["anonymised_answer"]
    assert "chest pain" in row["anonymised_symptoms"]


def test_a_decision_taken_on_a_truncated_narrative_leaves_a_trace(tmp_path):
    """That is the very purpose of this log.

    A description longer than the model window is bounded before inference — without which
    generation stops. The triage is then given on an incomplete narrative, and an audit must be
    able to find that out months later.
    """
    log = tmp_path / "audit.jsonl"
    record_interaction(
        request_id="trq1",
        symptoms="Description très longue " * 200,
        level="URGENCE_MODEREE",
        answer="Niveau de priorité : URGENCE_MODEREE",
        latency_ms=12.0,
        model="model",
        engine="vllm",
        description_truncated=True,
        log_path=log,
    )
    row = json.loads(log.read_text(encoding="utf-8").strip())
    assert row["description_truncated"] is True


def test_a_normal_interaction_is_not_marked_truncated(tmp_path):
    log = tmp_path / "audit.jsonl"
    record_interaction(
        request_id="trq2",
        symptoms="Homme de 62 ans, douleur thoracique.",
        level="URGENCE_VITALE",
        answer="Niveau de priorité : URGENCE_VITALE",
        latency_ms=12.0,
        model="model",
        engine="vllm",
        log_path=log,
    )
    assert json.loads(log.read_text(encoding="utf-8").strip())["description_truncated"] is False


def test_a_writable_log_is_created_empty(tmp_path):
    """The startup check prepares the file without writing a line into it."""
    log = tmp_path / "traces" / "audit_triage.jsonl"

    assert check_log_is_writable(log) == log
    assert log.exists()
    assert log.read_text(encoding="utf-8") == ""


def test_an_unreachable_log_stops_the_startup(tmp_path):
    """An impossible path fails at startup, not at the first request.

    The parent folder here is a file: the case occurs in a container when the mounted volume
    belongs to another user, with the same consequence — impossible to write — and it is the
    startup that must stop.
    """
    obstacle = tmp_path / "logs"
    obstacle.write_text("this is not a folder", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Audit log not writable"):
        check_log_is_writable(obstacle / "audit_triage.jsonl")
