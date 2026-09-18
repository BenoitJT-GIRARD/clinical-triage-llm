"""Running an evaluation: generation over the clinical set, then the metrics.

The evaluation set is the hand-written one that ``clinical_eval_set`` produces. It never served
for training, and its expected answers were not manufactured by the rule the model is compared
against.

Generation runs in batches, so that passing sixty cases takes minutes rather than hours. The
latency reported on that basis is therefore a per-case latency under batching: it serves to
compare models with each other, not to describe what a user feels. Perceived latency is
measured separately, request by request, by the bench in ``latency.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from clinical_triage.evaluation import safety
from clinical_triage.evaluation.metrics import subgroup_accuracy, summarize
from clinical_triage.utils import get_logger

logger = get_logger("evaluation")


@dataclass
class EvaluationOutcome:
    """The result of one evaluation: predictions, metrics and safety checks."""

    predictions: list[str | None]
    answers: list[str]
    metrics: dict
    safety: dict
    per_language: dict
    per_case_type: dict


def evaluate_predictions(cases: list[dict], predictions: list[str | None]) -> dict:
    """Compute the metrics of a list of predictions, with no generation.

    This is the path used by the baselines: they produce a level without producing text, so
    without latency and without safety checks.
    """
    expected = [c["level"] for c in cases]
    summary = summarize(expected, predictions)
    summary["per_language"] = subgroup_accuracy(expected, predictions, [c["lang"] for c in cases])
    summary["per_case_type"] = subgroup_accuracy(
        expected, predictions, [c["case_type"] for c in cases]
    )
    return summary


def evaluate_agent(agent, cases: list[dict], batch_size: int = 8) -> EvaluationOutcome:
    """Have the agent generate on every case, then compute metrics and safety."""
    descriptions = [c["description"] for c in cases]
    turns = [c["user_turn"] for c in cases]
    expected = [c["level"] for c in cases]

    answers = []
    for start in range(0, len(turns), batch_size):
        batch = turns[start : start + batch_size]
        answers.extend(agent.generate_batch(batch))
        logger.info("  %d/%d cases evaluated", min(start + batch_size, len(turns)), len(turns))

    predictions = [r.level for r in answers]
    texts = [r.text for r in answers]
    metrics = summarize(
        expected,
        predictions,
        latencies_ms=[r.latency_ms for r in answers],
        clean_stops=[r.clean_stop for r in answers],
    )
    metrics["mean_generated_tokens"] = round(
        sum(r.generated_tokens for r in answers) / max(1, len(answers)), 1
    )

    reports = [
        safety.check(description, text, prediction)
        for description, text, prediction in zip(descriptions, texts, predictions, strict=True)
    ]

    return EvaluationOutcome(
        predictions=predictions,
        answers=texts,
        metrics=metrics,
        safety=safety.summarize(reports),
        per_language=subgroup_accuracy(expected, predictions, [c["lang"] for c in cases]),
        per_case_type=subgroup_accuracy(expected, predictions, [c["case_type"] for c in cases]),
    )


def error_table(cases: list[dict], predictions: list[str | None], answers: list[str]) -> list[dict]:
    """Detail the misclassified cases, for the error analysis."""
    errors = []
    for case, prediction, answer in zip(cases, predictions, answers, strict=True):
        if prediction == case["level"]:
            continue
        errors.append(
            {
                "id": case["id"],
                "language": case["lang"],
                "case_type": case["case_type"],
                "expected": case["level"],
                "predicted": prediction,
                "description": case["description"],
                "clinical_note": case["clinical_note"],
                "answer": answer,
            }
        )
    return errors
