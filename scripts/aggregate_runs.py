"""Put the evaluations side by side: the spread of the training seeds, and the before/after.

Two published files, each read from committed results and never from a model:

- ``reports/seed_study.json`` — for each seed, the supervised and the aligned model on the
  clinical and internal sets, their paired comparison, and the task-format preference share;
  then, per measure, the
  spread over the seeds. One training run says what one run gave; three say how much of a gap
  between two systems the training alone can produce. The shipped model is seed 42's, whatever
  its rank among the three: choosing the best on the clinical set would make that set a
  selection set;
- ``reports/before_after.json`` — the served system of the previous release against this one,
  on the clinical set, with a paired test on the urgent cases. The previous release's numbers
  come from ``evaluation_previous-release.json``, written by ``run_evaluation.py --model-path``.

Usage::

    uv run python scripts/aggregate_runs.py
"""

from __future__ import annotations

from clinical_triage.bootstrap import use_utf8_console

use_utf8_console()

import json
import statistics
from pathlib import Path

from clinical_triage.config import PATHS, SEED, TRIAGE
from clinical_triage.data.dataset_io import read_jsonl
from clinical_triage.evaluation.metrics import mcnemar_exact
from clinical_triage.utils import get_logger

logger = get_logger("aggregate")

SEED_RESULTS = PATHS.reports / "seeds"
PREVIOUS_RELEASE = PATHS.reports / "evaluation_previous-release.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _measures(results: dict, model: str) -> dict:
    clinical = results["clinical_set"]["models"][model]
    detail = clinical["undertriage_detail"]
    internal = results["internal_set"]["models"].get(model, {})
    return {
        "clinical_accuracy": clinical["accuracy"],
        "clinical_undertriage_misses": detail["misses"],
        "urgent_cases": detail["urgent_cases"],
        "format_compliance": clinical["format_compliance"],
        "internal_accuracy": internal.get("accuracy"),
    }


def _spread(values: list[float]) -> dict:
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 4),
        "sd": round(statistics.stdev(values), 4) if len(values) > 1 else None,
    }


def seed_study(runs: dict[int, dict]) -> dict:
    """The per-seed measures of the supervised and the aligned model, and their spread."""
    per_seed = {}
    for seed, results in sorted(runs.items()):
        per_seed[str(seed)] = {
            model: _measures(results, model)
            for model in ("sft", "dpo-merged")
            if model in results["clinical_set"]["models"]
        }
        task = results.get("task_preferences") or {}
        per_seed[str(seed)]["task_preferences_share"] = task.get("correctly_ordered_share")
        # The protocol compares the two models on every seed, not on the shipped one alone: one
        # seed's verdict on what the alignment changes could be that seed's luck.
        per_seed[str(seed)]["aligned_versus_supervised"] = results.get("aligned_versus_supervised")
    spread = {}
    for model in ("sft", "dpo-merged"):
        rows = [per_seed[s][model] for s in per_seed if model in per_seed[s]]
        spread[model] = {
            key: _spread([row[key] for row in rows])
            for key in ("clinical_accuracy", "clinical_undertriage_misses", "internal_accuracy")
            if rows and all(row[key] is not None for row in rows)
        }
    return {"shipped_seed": SEED, "seeds": per_seed, "spread": spread}


def _predictions(results: dict, model: str, cases: list[dict]) -> list[str | None]:
    """The predictions of a model, rebuilt from its error table: a case not in it was right."""
    wrong = {error["id"]: error["predicted"] for error in results["errors"][model]}
    return [wrong.get(case["id"], case["level"]) for case in cases]


def _undertriaged(expected: str, predicted: str | None) -> bool:
    rank = TRIAGE.severity.get(predicted, -1) if predicted is not None else -1
    return rank < TRIAGE.severity[expected]


def before_after(previous: dict, current: dict, cases: list[dict]) -> dict:
    """The previous release's served system against this release's, on the same cases."""
    label = next(iter(previous["clinical_set"]["models"]))
    before = _predictions(previous, label, cases)
    after = _predictions(current, "dpo-merged", cases)
    urgent = [i for i, case in enumerate(cases) if TRIAGE.severity[case["level"]] >= 1]
    after_only = sum(
        _undertriaged(cases[i]["level"], before[i])
        and not _undertriaged(cases[i]["level"], after[i])
        for i in urgent
    )
    before_only = sum(
        _undertriaged(cases[i]["level"], after[i])
        and not _undertriaged(cases[i]["level"], before[i])
        for i in urgent
    )
    return {
        "before": {"label": label, **_measures(previous, label)},
        "after": {"label": "dpo-merged", **_measures(current, "dpo-merged")},
        "undertriage": {
            "urgent_cases": len(urgent),
            "rescued_by_this_release": after_only,
            "lost_by_this_release": before_only,
            "p_mcnemar": round(mcnemar_exact(after_only, before_only), 4),
        },
    }


def main() -> None:
    current = _read(PATHS.reports / "evaluation_results.json")
    runs = {SEED: current}
    for path in sorted(SEED_RESULTS.glob("seed-*.json")):
        runs[int(path.stem.split("-")[1])] = _read(path)
    study = seed_study(runs)
    (PATHS.reports / "seed_study.json").write_text(
        json.dumps(study, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Seed study over %d seed(s) written.", len(runs))

    if PREVIOUS_RELEASE.exists():
        cases = read_jsonl(PATHS.data_processed / "clinical_eval.jsonl")
        comparison = before_after(_read(PREVIOUS_RELEASE), current, cases)
        (PATHS.reports / "before_after.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Before/after written.")


if __name__ == "__main__":
    main()
