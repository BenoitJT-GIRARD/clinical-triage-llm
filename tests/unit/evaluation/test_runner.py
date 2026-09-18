"""Tests of running an evaluation and of the performance bench.

No model is loaded: a fake agent produces controlled answers. What is checked is the chaining —
batched generation, metric computation, safety checks, subgroup detail and error table.
"""

from __future__ import annotations

from dataclasses import dataclass

from clinical_triage.evaluation.latency import Measure
from clinical_triage.evaluation.runner import error_table, evaluate_agent, evaluate_predictions
from clinical_triage.prompts import completion_payload

CASES = [
    {
        "id": "ev01",
        "user_turn": "Situation clinique.\nDouleur thoracique et sueurs.\nNiveau ?",
        "description": "Homme de 67 ans, douleur thoracique et sueurs depuis 20 minutes.",
        "level": "URGENCE_VITALE",
        "lang": "fr",
        "case_type": "",
        "clinical_note": "Syndrome coronarien aigu probable.",
    },
    {
        "id": "ev02",
        "user_turn": "Situation clinique.\nRhume banal.\nNiveau ?",
        "description": "Femme de 34 ans, nez bouché depuis deux jours, pas de fièvre.",
        "level": "CONSULTATION_DIFFEREE",
        "lang": "fr",
        "case_type": "falsely_alarming",
        "clinical_note": "Rhinopharyngite virale bénigne.",
    },
]

RIGHT_ANSWER = (
    "Niveau de priorité : URGENCE_VITALE\n"
    "Justification : Douleur thoracique avec sueurs chez un patient à risque.\n"
    "Recommandation : Prise en charge immédiate et appel du 15 (SAMU)."
)
WRONG_ANSWER = (
    "Niveau de priorité : URGENCE_VITALE\n"
    "Justification : Symptômes inquiétants.\n"
    "Recommandation : Prise en charge immédiate et appel du 15 (SAMU)."
)


@dataclass
class FakeAnswer:
    text: str
    level: str | None
    latency_ms: float
    generated_tokens: int
    clean_stop: bool


class FakeAgent:
    """An agent that always answers "life-threatening": right one time out of two."""

    def generate_batch(self, symptoms: list[str]) -> list[FakeAnswer]:
        return [
            FakeAnswer(
                text=RIGHT_ANSWER if "thoracique" in s else WRONG_ANSWER,
                level="URGENCE_VITALE",
                latency_ms=120.0,
                generated_tokens=64,
                clean_stop=True,
            )
            for s in symptoms
        ]


def test_evaluating_an_agent():
    outcome = evaluate_agent(FakeAgent(), CASES, batch_size=1)
    assert outcome.predictions == ["URGENCE_VITALE", "URGENCE_VITALE"]
    assert outcome.metrics["accuracy"] == 0.5
    assert outcome.metrics["format_compliance"] == 1.0
    assert outcome.metrics["clean_stops"] == 1.0
    assert outcome.metrics["overtriage"] == 0.5
    assert outcome.metrics["mean_generated_tokens"] == 64.0


def test_the_evaluation_produces_the_safety_checks():
    outcome = evaluate_agent(FakeAgent(), CASES, batch_size=2)
    assert outcome.safety["n"] == 2
    assert 0.0 <= outcome.safety["flawless_share"] <= 1.0


def test_the_evaluation_details_the_subgroups():
    outcome = evaluate_agent(FakeAgent(), CASES, batch_size=2)
    assert outcome.per_language["fr"]["n"] == 2
    assert "falsely_alarming" in outcome.per_case_type
    assert "direct_presentation" in outcome.per_case_type


def test_evaluating_predictions_without_generation():
    """The path used by the baselines: a level, with no text and no latency."""
    summary = evaluate_predictions(CASES, ["URGENCE_VITALE", "CONSULTATION_DIFFEREE"])
    assert summary["accuracy"] == 1.0
    assert "latency" not in summary
    assert summary["per_language"]["fr"]["accuracy"] == 1.0


def test_the_error_table_keeps_only_the_misclassified_cases():
    errors = error_table(CASES, ["URGENCE_VITALE", "URGENCE_VITALE"], [RIGHT_ANSWER, WRONG_ANSWER])
    assert len(errors) == 1
    error = errors[0]
    assert error["id"] == "ev02"
    assert error["expected"] == "CONSULTATION_DIFFEREE"
    assert error["predicted"] == "URGENCE_VITALE"
    assert error["clinical_note"]


# --- Performance bench ---


def test_the_request_body_forces_the_stop_on_the_end_token():
    """The bench and the gateway now assemble the same request.

    Each had its own version: identical field for field, but nothing held them together, and
    the published latencies would eventually have described a request the service no longer
    sends.
    """
    payload = completion_payload("Douleur thoracique.", "qwen3-1.7b-clinical-triage")
    assert payload["model"] == "qwen3-1.7b-clinical-triage"
    assert payload["prompt"].endswith("<|im_start|>assistant\n")
    assert payload["stop"] == ["<|im_end|>"]
    assert payload["temperature"] == 0.0


def test_a_measurement_carries_the_extracted_level():
    measure = Measure(latency_ms=100.0, tokens=50, level="URGENCE_VITALE")
    assert measure.level == "URGENCE_VITALE"
