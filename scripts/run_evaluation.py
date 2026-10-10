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
the explicit rule, and an ordinary classifier trained on the same examples as the model. The last
two are the ones to beat, and the fourth is the most demanding: it says what fine-tuning adds
over ordinary learning on the same data.

Three analyses are fixed before any result exists, and published whatever they show: the
supervised model against the aligned one, on the same cases; the **more urgent of the two opinions**,
model or rule, against the model alone; and the cause of each urgent case the rule misses.

``--seed`` evaluates another training seed of the same method, whose results go under
``reports/seeds/``; ``--model-path`` evaluates one merged model given by its folder, on the
clinical set alone — that is how the served system of an earlier version is measured.

Usage::

    uv run python scripts/run_evaluation.py
    uv run python scripts/run_evaluation.py --seed 43 --models sft dpo dpo-merged --preferences 0
    uv run python scripts/run_evaluation.py --model-path <folder> --label previous-release
"""

from __future__ import annotations

from clinical_triage.bootstrap import use_utf8_console

use_utf8_console()

import argparse
import json
import random
from pathlib import Path

from clinical_triage.config import MODEL, PATHS, SEED, TRIAGE, Paths
from clinical_triage.data.corpus_sources import load_ultramedical_preferences
from clinical_triage.data.dataset_io import read_jsonl
from clinical_triage.evaluation import preference, robustness
from clinical_triage.evaluation.baselines import (
    always_critical,
    classical_classifier,
    explicit_rule,
    majority_class,
)
from clinical_triage.evaluation.metrics import (
    cases_needed_without_miss,
    mcnemar_exact,
    upper_bound_without_miss,
)
from clinical_triage.evaluation.rule_anatomy import rule_misses
from clinical_triage.evaluation.runner import (
    error_table,
    evaluate_agent,
    evaluate_predictions,
)
from clinical_triage.evaluation.task_preferences import task_preference_pairs
from clinical_triage.inference import TriageAgent
from clinical_triage.training.export import weight_digests
from clinical_triage.utils import free_gpu_memory, get_logger, set_seed

logger = get_logger("evaluation")


# The models that can be evaluated, in progression order. `argparse` refuses a name absent from
# this list: without that check, a typo would fail on a `KeyError` once the other models had been
# evaluated, and the results file would never be written.
EVALUABLE_MODELS = ("base", "sft", "dpo", "dpo-merged")


def _specification(name: str, paths: Paths = PATHS) -> tuple[str, str | None]:
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
        "sft": (MODEL.base_model, str(paths.sft_adapter)),
        "dpo": (str(paths.sft_merged), str(paths.dpo_adapter)),
        "dpo-merged": (str(paths.dpo_merged), None),
    }[name]


def _weights_on_disk(path: Path) -> float:
    """Size of a model folder's weight files, in megabytes."""
    if not path.exists():
        return 0.0
    patterns = ("*.safetensors", "*.bin")
    size = sum(f.stat().st_size for pattern in patterns for f in path.glob(pattern))
    return round(size / 1024**2, 1)


def _compare_deliveries(results: dict, paths: Paths = PATHS) -> dict:
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
            "download_size_mb": _weights_on_disk(paths.dpo_adapter),
            "base_model_required": str(paths.sft_merged.name),
        },
        "merged": {
            "accuracy": merged["accuracy"],
            "accuracy_ci95": merged["accuracy_ci95"],
            "median_latency_ms": merged.get("latency", {}).get("p50_ms"),
            "download_size_mb": _weights_on_disk(paths.dpo_merged),
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

    The classical baseline applies everywhere: it learns from the same examples as the model, and
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


def worse_of_two(model: list[str | None], rule: list[str]) -> list[str]:
    """For each case, the more urgent of the two opinions; the rule's alone when the model gave none.

    Pre-registered on 2026-10-09: an earlier version imagined this combination after seeing its
    errors, which makes a result unreadable. It is now computed on every run, before anyone looks.
    """
    combined = []
    for predicted, ruled in zip(model, rule, strict=True):
        if predicted is None or TRIAGE.severity[ruled] > TRIAGE.severity[predicted]:
            combined.append(ruled)
        else:
            combined.append(predicted)
    return combined


def _shipped(predictions: dict) -> str | None:
    return next((n for n in ("dpo-merged", "dpo", "sft", "base") if n in predictions), None)


def _paired_analyses(results: dict, cases: list[dict]) -> None:
    """The comparisons fixed before the results: against the baselines, between the two training
    stages, and the more urgent of the two opinions."""
    predictions = results["_predictions"]
    shipped = _shipped(predictions)
    if shipped is None:
        return
    rule = explicit_rule([c["description"] for c in cases])
    results["model_versus_rule"] = _compare_to_baseline(
        cases, predictions[shipped], shipped, rule, "explicit_rule"
    )
    # Both sets of predictions were just computed on these same cases: the classifier's is read
    # back rather than the classifier retrained.
    classical = results["clinical_set"]["baselines"]["linear_classifier"]["_predictions"]
    results["model_versus_classifier"] = _compare_to_baseline(
        cases, predictions[shipped], shipped, classical, "linear_classifier"
    )
    aligned = next((n for n in ("dpo-merged", "dpo") if n in predictions), None)
    if "sft" in predictions and aligned:
        results["aligned_versus_supervised"] = _compare_to_baseline(
            cases, predictions[aligned], aligned, predictions["sft"], "sft"
        )
    combined = worse_of_two(predictions[shipped], rule)
    results["worse_of_two"] = {
        "model": shipped,
        **evaluate_predictions(cases, combined),
        "versus_model_alone": _compare_to_baseline(
            cases, combined, "worse_of_two", predictions[shipped], shipped
        ),
    }


def resolution(cases: list[dict], target: float = 0.05) -> dict:
    """What the clinical set can show about undertriage, whatever the model.

    Even a model that misses no urgent case leaves an exact upper bound above zero; the number of
    urgent cases fixes it. Written beside the results, so that a safety claim is read against
    what the set could have shown.
    """
    urgent = sum(TRIAGE.severity[case["level"]] >= 1 for case in cases)
    return {
        "urgent_cases": urgent,
        "upper_bound_without_miss": upper_bound_without_miss(urgent),
        "target": target,
        "urgent_cases_needed_for_target": cases_needed_without_miss(target),
    }


def _default_output(seed: int, label: str | None) -> Path:
    if label:
        return PATHS.reports / f"evaluation_{label}.json"
    if seed == SEED:
        return PATHS.reports / "evaluation_results.json"
    return PATHS.reports / "seeds" / f"seed-{seed}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=EVALUABLE_MODELS,
        default=list(EVALUABLE_MODELS),
        help="models to evaluate, in this order",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="training seed whose models to read")
    parser.add_argument("--model-path", type=Path, help="one merged model to evaluate instead")
    parser.add_argument("--label", help="the name of the model given by --model-path")
    parser.add_argument("--internal", type=int, default=120, help="cases of the internal split")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--preferences",
        type=int,
        default=777,
        help="pairs of the external preference set, 777 in all (0 to skip)",
    )
    parser.add_argument("--task-preferences", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--robustness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.model_path) != bool(args.label):
        parser.error("--model-path and --label go together")

    set_seed(SEED)
    paths = PATHS.for_seed(args.seed)
    clinical_cases = read_jsonl(PATHS.data_processed / "clinical_eval.jsonl")
    if args.model_path:
        # A model given by its folder is measured where published numbers are read: the clinical
        # set, against the same baselines, and nothing else.
        args.models, args.internal, args.preferences = [args.label], 0, 0
        args.task_preferences = args.robustness = False
    internal_cases = _internal_cases(args.internal) if args.internal else []
    # The classical baseline learns from the same examples as the model: the training split to
    # learn, the validation split to choose its setting. It never sees the cases it is judged on.
    training = read_jsonl(PATHS.data_processed / "sft_train.jsonl")
    validation = read_jsonl(PATHS.data_processed / "sft_validation.jsonl")
    training_levels = [row["level"] for row in training]
    external_pairs = (
        load_ultramedical_preferences(limit=args.preferences) if args.preferences else []
    )
    task_pairs = task_preference_pairs() if args.task_preferences else []
    logger.info(
        "Seed %d. Clinical set: %d cases. Internal split: %d cases. External preferences: %d "
        "pairs. Task-format preferences: %d pairs.",
        args.seed,
        len(clinical_cases),
        len(internal_cases),
        len(external_pairs),
        len(task_pairs),
    )

    results: dict = {
        "seed": args.seed,
        "clinical_set": {
            "n": len(clinical_cases),
            "baselines": _baselines(clinical_cases, training_levels, training, validation),
            "models": {},
        },
        "internal_set": {"n": len(internal_cases), "baselines": {}, "models": {}},
        "robustness": {},
        "external_preferences": {},
        "task_preferences": {},
        "errors": {},
        # Kept for the duration of the paired comparisons, then removed: raw predictions have no
        # place in the results file.
        "_predictions": {},
    }
    if internal_cases:
        results["internal_set"]["baselines"] = _baselines(
            internal_cases, training_levels, training, validation, with_rule=False
        )

    for name in args.models:
        if args.model_path:
            base, adapter = str(args.model_path), None
        else:
            base, adapter = _specification(name, paths)
        if adapter is not None and not Path(adapter).exists():
            logger.warning("Adapter %s not found (%s) — model skipped.", name, adapter)
            continue
        # `base` is either the identifier of the base model on the Hub — which cannot be looked
        # for on disk — or a folder, which is missing if the step that writes it was not run.
        if base != MODEL.base_model and not Path(base).exists():
            logger.warning("Model %s not found (%s) — model skipped.", name, base)
            continue
        logger.info("=== Evaluating model: %s ===", name)
        if adapter is None and base != MODEL.base_model:
            # A merged model is what gets published and served: the publication compares what
            # it uploads with the digests of the files measured here.
            results["evaluated_weights"] = weight_digests(Path(base))
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

        if args.robustness:
            results["robustness"][name] = robustness.summarize(robustness.run(agent))

        # What the alignment changed is read on the model that carries the DPO adapter alone:
        # each answer is read with and without it, that is after and before the alignment.
        if name == "dpo":
            if external_pairs:
                effects = preference.measure_alignment(
                    agent.model, agent.tokenizer, external_pairs, agent.device
                )
                results["external_preferences"] = {
                    **preference.summarize(effects),
                    "clipped_answers": preference.clipped_answers(agent.tokenizer, external_pairs),
                    "preferred_is_longer_outside_length_group": (
                        preference.preferred_longer_share(external_pairs)
                    ),
                }
            if task_pairs:
                effects = preference.measure_alignment(
                    agent.model, agent.tokenizer, task_pairs, agent.device, separator=""
                )
                results["task_preferences"] = preference.summarize(effects)

        del agent
        free_gpu_memory()

    _paired_analyses(results, clinical_cases)
    results["rule_anatomy"] = rule_misses(clinical_cases)
    results["clinical_set"]["resolution"] = resolution(clinical_cases)
    if not args.model_path:
        results["adapter_versus_merged"] = _compare_deliveries(results, paths)
    del results["_predictions"]
    for block in ("clinical_set", "internal_set"):
        classical = results[block]["baselines"].get("linear_classifier")
        if classical:
            classical.pop("_predictions", None)

    destination = args.output or _default_output(args.seed, args.label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Results written → %s", destination)
    _print_summary(results)


def _print_summary(results: dict) -> None:
    print("\n=== Independent clinical set ===")
    header = f"{'system':24s} | {'accuracy':>8s} | {'undertriage':>11s} | {'format':>6s}"
    print(header)
    print("-" * len(header))
    rows = {**results["clinical_set"]["baselines"], **results["clinical_set"]["models"]}
    if "worse_of_two" in results:
        rows["more urgent of the two opinions"] = results["worse_of_two"]
    for name, measures in rows.items():
        print(
            f"{name:24s} | {measures['accuracy']:8.3f} | {measures['undertriage']:11.3f} | "
            f"{measures['format_compliance']:6.3f}"
        )
    for key in ("model_versus_rule", "model_versus_classifier", "aligned_versus_supervised"):
        comparison = results.get(key)
        if comparison:
            print(
                f"{key:24s} | undertriage p = {comparison['undertriage']['p_mcnemar']:.4f} "
                f"(McNemar, n = {comparison['undertriage']['urgent_cases']} urgent cases)"
            )
    for key in ("external_preferences", "task_preferences"):
        block = results.get(key)
        if block:
            print(
                f"{key:24s} | preferred raised more {block['correctly_ordered_share']:.3f} on {block['n']} pairs"
            )


if __name__ == "__main__":
    main()
