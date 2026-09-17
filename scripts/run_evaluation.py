"""Comparative clinical evaluation: baselines, base model, SFT and DPO.

The evaluation covers two sets, deliberately:

- the **independent clinical set**, written by hand, which never served for training and whose
  labels do not come from the rule. It is the only set on which the published figures mean
  something;
- the **internal test split**, produced by the same generator as the training data. It measures
  the model's ability to reproduce what it was shown. The gap between the two is precisely what
  to look at: a model that excels on the second and is mediocre on the first has learnt
  templates, not triage.

Four baselines frame the results: always the majority class, always the life-threatening level,
the explicit rule, and an ordinary classifier trained on the same pairs as the model. The last
two are the ones to beat, and the fourth is the most demanding: it says what fine-tuning adds
over ordinary learning on the same data.

Usage::

    uv run python scripts/run_evaluation.py
    uv run python scripts/run_evaluation.py --models sft dpo --internal 0
"""

from __future__ import annotations

from clinical_triage.bootstrap import use_utf8_console

use_utf8_console()

import argparse
import json
import random
from pathlib import Path

from clinical_triage.config import MODEL, PATHS, SEED, TRIAGE
from clinical_triage.data.corpus_sources import load_ultramedical_preferences
from clinical_triage.data.dataset_io import read_jsonl
from clinical_triage.evaluation import preference, robustness
from clinical_triage.evaluation.baselines import (
    always_critical,
    classical_classifier,
    explicit_rule,
    majority_class,
)
from clinical_triage.evaluation.metrics import mcnemar_exact
from clinical_triage.evaluation.runner import (
    error_table,
    evaluate_agent,
    evaluate_predictions,
)
from clinical_triage.inference import TriageAgent
from clinical_triage.utils import free_gpu_memory, get_logger, set_seed

logger = get_logger("evaluation")


# The models that can be evaluated, in progression order. `argparse` refuses a name absent from
# this list: without that check, a typo would fail on a `KeyError` once the other models had been
# evaluated, and the results file would never be written.
EVALUABLE_MODELS = ("base", "sft", "dpo", "dpo-merged")


def _specification(name: str) -> tuple[str, str | None]:
    """Return (base model, adapter) for each model to evaluate.

    DPO having been trained above the merged SFT model, it must be evaluated with that model as
    its base: applying the DPO adapter to the base model would produce an incoherent composition
    of weights.

    ``dpo-merged`` designates the **same weights** as ``dpo``, merged into a single model instead
    of applied hot. Both are evaluated side by side because they are two ways of shipping the
    model, and the choice between them is settled on measurements, not on principle.
    """
    return {
        "base": (MODEL.base_model, None),
        "sft": (MODEL.base_model, str(PATHS.sft_adapter)),
        "dpo": (str(PATHS.sft_merged), str(PATHS.dpo_adapter)),
        "dpo-merged": (str(PATHS.dpo_merged), None),
    }[name]


def _weights_on_disk(path: Path) -> float:
    """Size of a model folder's weight files, in megabytes."""
    if not path.exists():
        return 0.0
    patterns = ("*.safetensors", "*.bin")
    size = sum(f.stat().st_size for pattern in patterns for f in path.glob(pattern))
    return round(size / 1024**2, 1)


def _compare_deliveries(results: dict) -> dict:
    """Compare delivery by adapter with delivery as a merged model.

    Both carry the same weights, to the bf16 rounding. What separates them is operational: what
    has to be downloaded, what the inference engine can load, and what applying the adapter costs
    on every token.
    """
    models = results["clinical_set"]["models"]
    if "dpo" not in models or "dpo-merged" not in models:
        return {}

    adapter, merged = models["dpo"], models["dpo-merged"]
    adapter_predictions = results["_predictions"].get("dpo", [])
    merged_predictions = results["_predictions"].get("dpo-merged", [])
    agreement = (
        round(
            sum(a == f for a, f in zip(adapter_predictions, merged_predictions, strict=True))
            / len(adapter_predictions),
            4,
        )
        if adapter_predictions and len(adapter_predictions) == len(merged_predictions)
        else None
    )
    return {
        "prediction_agreement": agreement,
        "adapter": {
            "accuracy": adapter["accuracy"],
            "accuracy_ci95": adapter["accuracy_ci95"],
            "median_latency_ms": adapter.get("latency", {}).get("p50_ms"),
            "download_size_mb": _weights_on_disk(PATHS.dpo_adapter),
            "base_model_required": str(PATHS.sft_merged.name),
        },
        "merged": {
            "accuracy": merged["accuracy"],
            "accuracy_ci95": merged["accuracy_ci95"],
            "median_latency_ms": merged.get("latency", {}).get("p50_ms"),
            "download_size_mb": _weights_on_disk(PATHS.dpo_merged),
            "base_model_required": None,
        },
    }


def _internal_cases(count: int) -> list[dict]:
    """A random sample of the internal test split, in the shape of the clinical set."""
    rows = read_jsonl(PATHS.data_processed / "sft_test.jsonl")
    draw = random.Random(SEED)
    draw.shuffle(rows)
    return [
        {
            "id": f"internal_{index}",
            "user_turn": row["user_turn"],
            "description": row["user_turn"],
            "level": row["level"],
            "lang": row["lang"],
            "case_type": "",
            "clinical_note": "",
        }
        for index, row in enumerate(rows[:count])
    ]


def _baselines(
    cases: list[dict],
    training_levels: list[str],
    training: list[dict],
    validation: list[dict],
    with_rule: bool = True,
) -> dict:
    """Evaluate the baselines on a set of cases.

    ``with_rule`` exists because the explicit rule is not a legitimate baseline everywhere. Data
    preparation uses it to discard from the corpus the cases whose label it contradicts: on the
    part of the internal set that comes from that corpus, it recovers the label 100% of the time
    by construction. Comparing it to the model there would measure that filter, not the rule. The
    clinical set, on the other hand, is written by hand and never passes through ``classify``:
    the comparison is honest there, and that is the one the published protocol reports.

    The classical baseline applies everywhere: it learns from the same pairs as the model, and
    none of the lines it judges was in its training. On the internal set it has nonetheless seen
    rewordings of them — exactly as the model has, and that is what makes the comparison fair
    there.
    """
    baselines = {
        "majority_class": evaluate_predictions(cases, majority_class(training_levels, len(cases))),
        "always_critical": evaluate_predictions(cases, always_critical(len(cases))),
    }
    if with_rule:
        baselines["explicit_rule"] = evaluate_predictions(
            cases, explicit_rule([c["description"] for c in cases])
        )
    predictions, trace = classical_classifier(
        [e["user_turn"] for e in training],
        [e["level"] for e in training],
        [e["user_turn"] for e in validation],
        [e["level"] for e in validation],
        [c["description"] for c in cases],
    )
    baselines["linear_classifier"] = {
        **evaluate_predictions(cases, predictions),
        "selection": trace,
        # Kept for the duration of the paired comparison, then removed before writing: raw
        # predictions have no place in the results file.
        "_predictions": predictions,
    }
    return baselines


def _compare_to_baseline(
    cases: list[dict],
    predictions: list[str | None],
    model_name: str,
    baseline: list[str | None],
    baseline_name: str,
) -> dict | None:
    """Compare the model to a baseline on the **same** cases.

    The central claim — "the model does better than what it would replace" — is settled by a
    paired test, not by the overlap of two confidence intervals: non-overlap proves a difference,
    overlap proves nothing. On **paired** data that overlap is besides systematically too
    cautious, because the information is carried by the cases alone where the two systems
    diverge — two nearly superposed intervals are perfectly compatible with a real gap.

    Both systems see exactly the same cases here. The test that suits them is McNemar's, exact,
    also used to separate the supervised model from the aligned one.

    Two baselines are compared this way, and both are needed. The explicit rule says what a
    department can deploy in one afternoon. The classical classifier says what fine-tuning adds
    over ordinary learning on the same data: beating the rule without beating the classifier does
    not establish that a language model was needed.
    """
    if not predictions:
        return None
    expected = [c["level"] for c in cases]

    model_only = sum(
        p == a and r != a for p, r, a in zip(predictions, baseline, expected, strict=True)
    )
    baseline_only = sum(
        r == a and p != a for p, r, a in zip(predictions, baseline, expected, strict=True)
    )
    return {
        "model": model_name,
        "baseline": baseline_name,
        "accuracy": {
            "n": len(cases),
            "model_only": model_only,
            "baseline_only": baseline_only,
            "agreements": len(cases) - model_only - baseline_only,
            "p_mcnemar": round(mcnemar_exact(model_only, baseline_only), 4),
        },
        # Overall accuracy is not the deciding measure. A system can be wrong often without
        # danger — overtriaging a cold costs a waiting-room place — and rarely with danger. The
        # same paired test, restricted to urgent cases and to the one failure that kills, says
        # what accuracy dilutes: a triage tool is judged on undertriage.
        "undertriage": _compare_undertriage(cases, predictions, baseline),
    }


def _compare_undertriage(
    cases: list[dict], predictions: list[str | None], baseline: list[str | None]
) -> dict:
    """Compare model and baseline on the dangerous failure alone, urgent cases only."""

    def undertriages(expected: str, predicted: str | None) -> bool:
        rank = TRIAGE.severity.get(predicted, -1) if predicted is not None else -1
        return rank < TRIAGE.severity[expected]

    urgent = [
        (c["level"], p, r)
        for c, p, r in zip(cases, predictions, baseline, strict=True)
        if TRIAGE.severity[c["level"]] >= 1
    ]
    model_only = sum(not undertriages(a, p) and undertriages(a, r) for a, p, r in urgent)
    baseline_only = sum(undertriages(a, p) and not undertriages(a, r) for a, p, r in urgent)
    return {
        "urgent_cases": len(urgent),
        "model_undertriages": sum(undertriages(a, p) for a, p, _ in urgent),
        "baseline_undertriages": sum(undertriages(a, r) for a, _, r in urgent),
        "model_only": model_only,
        "baseline_only": baseline_only,
        "p_mcnemar": round(mcnemar_exact(model_only, baseline_only), 4),
    }


def _compare_preferences(results: dict) -> dict | None:
    """Compare the supervised model and the aligned model on the **same** pairs.

    Both were measured on the same external set: what separates them is the pairs alone where
    one succeeds and the other fails. Comparing their two proportions would throw that
    information away, and would declare a gain on a difference of two pairs out of a hundred and
    fifty.
    """
    preferences = results.get("external_preferences", {})
    supervised = preferences.get("sft", {}).get("correctly_ordered")
    aligned = preferences.get("dpo", {}).get("correctly_ordered")
    if not supervised or not aligned or len(supervised) != len(aligned):
        return None

    aligned_only = sum(a and not s for s, a in zip(supervised, aligned, strict=True))
    supervised_only = sum(s and not a for s, a in zip(supervised, aligned, strict=True))
    return {
        "n": len(supervised),
        "aligned_only": aligned_only,
        "supervised_only": supervised_only,
        "agreements": len(supervised) - aligned_only - supervised_only,
        "p_mcnemar": round(mcnemar_exact(aligned_only, supervised_only), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=EVALUABLE_MODELS,
        default=list(EVALUABLE_MODELS),
        help="models to evaluate, in this order",
    )
    parser.add_argument("--internal", type=int, default=120, help="cases of the internal split")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--preferences",
        type=int,
        default=150,
        help="pairs of the external preference set (0 to skip)",
    )
    args = parser.parse_args()

    set_seed(SEED)
    clinical_cases = read_jsonl(PATHS.data_processed / "clinical_eval.jsonl")
    internal_cases = _internal_cases(args.internal) if args.internal else []
    # The classical baseline learns from the same pairs as the model: the training split to
    # learn, the validation split to choose its setting. It never sees the cases it is judged on.
    training = read_jsonl(PATHS.data_processed / "sft_train.jsonl")
    validation = read_jsonl(PATHS.data_processed / "sft_validation.jsonl")
    training_levels = [row["level"] for row in training]
    external_pairs = (
        load_ultramedical_preferences(limit=args.preferences) if args.preferences else []
    )
    logger.info(
        "Independent clinical set: %d cases. Internal test split: %d cases. "
        "External preferences: %d pairs.",
        len(clinical_cases),
        len(internal_cases),
        len(external_pairs),
    )

    results: dict = {
        "clinical_set": {
            "n": len(clinical_cases),
            "baselines": _baselines(clinical_cases, training_levels, training, validation),
            "models": {},
        },
        "internal_set": {"n": len(internal_cases), "baselines": {}, "models": {}},
        "robustness": {},
        "external_preferences": {},
        "errors": {},
        # Kept for the duration of the delivery comparison, then removed: raw predictions have
        # no place in the results file.
        "_predictions": {},
    }
    if internal_cases:
        results["internal_set"]["baselines"] = _baselines(
            internal_cases, training_levels, training, validation, with_rule=False
        )

    for name in args.models:
        base, adapter = _specification(name)
        if adapter is not None and not Path(adapter).exists():
            logger.warning("Adapter %s not found (%s) — model skipped.", name, adapter)
            continue
        # `base` is either the identifier of the base model on the Hub — which cannot be looked
        # for on disk — or one of the two folders produced by the merge, which is missing if that
        # step was not run.
        local_bases = {str(PATHS.sft_merged), str(PATHS.dpo_merged)}
        if base in local_bases and not Path(base).exists():
            logger.warning("Model %s not found (%s) — model skipped.", name, base)
            continue
        logger.info("=== Evaluating model: %s ===", name)
        free_gpu_memory()
        agent = TriageAgent(adapter_dir=adapter, base_model=base)

        clinical = evaluate_agent(agent, clinical_cases, batch_size=args.batch_size)
        results["clinical_set"]["models"][name] = {
            **clinical.metrics,
            "safety": clinical.safety,
            "per_language": clinical.per_language,
            "per_case_type": clinical.per_case_type,
        }
        results["errors"][name] = error_table(
            clinical_cases, clinical.predictions, clinical.answers
        )
        results["_predictions"][name] = list(clinical.predictions)

        if internal_cases:
            internal = evaluate_agent(agent, internal_cases, batch_size=args.batch_size)
            results["internal_set"]["models"][name] = {
                **internal.metrics,
                "safety": internal.safety,
            }

        # Degraded inputs: a three-letter entry, a two-page paste, an off-domain question, a
        # hijacked prompt. None has a right triage answer; what is checked is that the agent
        # keeps its output contract.
        checks = robustness.run(agent)
        results["robustness"][name] = robustness.summarize(checks)

        # An independent measurement of the alignment, on a corpus of human preferences kept out
        # of training.
        if external_pairs:
            scores = preference.score_pairs(
                agent.model, agent.tokenizer, external_pairs, agent.device
            )
            results["external_preferences"][name] = preference.summarize(scores)

        del agent
        free_gpu_memory()

    shipped = next(
        (n for n in ("dpo-merged", "dpo", "sft", "base") if n in results["_predictions"]), None
    )
    if shipped:
        shipped_predictions = results["_predictions"][shipped]
        results["model_versus_rule"] = _compare_to_baseline(
            clinical_cases,
            shipped_predictions,
            shipped,
            explicit_rule([c["description"] for c in clinical_cases]),
            "explicit_rule",
        )
        # The same paired comparison against the classical classifier. Both sets of predictions
        # have just been computed on these same cases: the baseline's is read back rather than
        # the classifier retrained.
        classical_predictions = results["clinical_set"]["baselines"]["linear_classifier"][
            "_predictions"
        ]
        results["model_versus_classifier"] = _compare_to_baseline(
            clinical_cases,
            shipped_predictions,
            shipped,
            classical_predictions,
            "linear_classifier",
        )
    results["adapter_versus_merged"] = _compare_deliveries(results)
    results["paired_preferences"] = _compare_preferences(results)
    del results["_predictions"]
    for block in ("clinical_set", "internal_set"):
        classical = results[block]["baselines"].get("linear_classifier")
        if classical:
            classical.pop("_predictions", None)
    # The pair-by-pair detail serves the paired comparison; it has no place in the results file,
    # where it would add hundreds of booleans nobody will read.
    for measures in results["external_preferences"].values():
        measures.pop("correctly_ordered", None)

    PATHS.reports.mkdir(parents=True, exist_ok=True)
    destination = PATHS.reports / "evaluation_results.json"
    destination.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Results written → %s", destination)

    print("\n=== Independent clinical set ===")
    header = (
        f"{'system':24s} | {'accuracy':>18s} | {'undertriage':>11s} | "
        f"{'format':>6s} | {'overtriage':>10s}"
    )
    print(header)
    print("-" * len(header))
    rows = [
        *results["clinical_set"]["baselines"].items(),
        *results["clinical_set"]["models"].items(),
    ]
    for name, measures in rows:
        low, high = measures["accuracy_ci95"]
        print(
            f"{name:24s} | {measures['accuracy']:.3f} [{low:.2f}-{high:.2f}] | "
            f"{measures['undertriage']:>11.3f} | {measures['format_compliance']:>6.3f} | "
            f"{measures['overtriage']:>10.3f}"
        )

    if results["robustness"]:
        print("\n=== Robustness on degraded inputs ===")
        for name, measures in results["robustness"].items():
            failures = ", ".join(measures["non_compliant_cases"]) or "none"
            print(f"{name:24s} | compliant {measures['compliant_share']:.3f} | failing: {failures}")

    if results["external_preferences"]:
        print("\n=== External preferences (UltraMedical, outside training) ===")
        for name, measures in results["external_preferences"].items():
            print(
                f"{name:24s} | correctly ordered {measures['correctly_ordered_share']:.3f} "
                f"on {measures['n']} pairs | margin {measures['mean_margin']:+.4f}"
            )


if __name__ == "__main__":
    main()
