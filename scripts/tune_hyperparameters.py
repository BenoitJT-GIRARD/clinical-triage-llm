"""Tune the hyper-parameters of the supervised fine-tuning.

Before launching the full training, a few settings are compared on a subset of the corpus. Every
variant is trained under the same conditions — same seed, same subset, same number of steps —
then judged on two complementary criteria:

- the **validation loss**, which says whether the model learns the distribution;
- the **triage accuracy** measured on cases from the validation split, which says whether that
  learning turns into the right clinical decision. The two do not always agree, and it is the
  second that matters to a department.

The variants cover the rank of the LoRA adaptation and the learning rate: the two settings that
weigh most on the trade-off between adaptation capacity and overfitting.

The table produced is written to ``reports/training/hyperparameter_comparison.json`` and is read
back by ``train_sft.py``.

Usage::

    uv run python scripts/tune_hyperparameters.py
    uv run python scripts/tune_hyperparameters.py --steps 40 --eval-cases 40
"""

from __future__ import annotations

from clinical_triage.bootstrap import use_utf8_console

use_utf8_console()

import argparse
import json
import shutil

from clinical_triage.config import MODEL, PATHS, SEED
from clinical_triage.data.dataset_io import read_jsonl, write_jsonl
from clinical_triage.evaluation.metrics import accuracy, wilson_interval
from clinical_triage.inference import TriageAgent
from clinical_triage.training.sft import train_sft
from clinical_triage.utils import free_gpu_memory, get_logger, set_seed

logger = get_logger("tune")

# The variants compared. One setting varies at a time around the reference configuration, so
# that every gap observed can be attributed.
VARIANTS = (
    {"name": "r8_lr2e-4", "lora_r": 8, "lora_alpha": 16, "learning_rate": 2e-4},
    {"name": "r16_lr2e-4", "lora_r": 16, "lora_alpha": 32, "learning_rate": 2e-4},
    {"name": "r32_lr2e-4", "lora_r": 32, "lora_alpha": 64, "learning_rate": 2e-4},
    {"name": "r16_lr1e-4", "lora_r": 16, "lora_alpha": 32, "learning_rate": 1e-4},
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-subset", type=int, default=1200)
    parser.add_argument("--steps", type=int, default=75)
    parser.add_argument("--eval-cases", type=int, default=60)
    args = parser.parse_args()

    set_seed(SEED)
    processed = PATHS.data_processed
    workspace = PATHS.models / "tuning"
    workspace.mkdir(parents=True, exist_ok=True)

    # A frozen subset, identical for every variant.
    subset = read_jsonl(processed / "sft_train.jsonl")[: args.train_subset]
    subset_path = workspace / "sft_train_subset.jsonl"
    write_jsonl(subset, subset_path)

    validation_cases = read_jsonl(processed / "sft_validation.jsonl")[: args.eval_cases]
    symptoms = [row["user_turn"] for row in validation_cases]
    expected = [row["level"] for row in validation_cases]

    results = []
    for variant in VARIANTS:
        logger.info("=== Variant %s ===", variant["name"])
        output = workspace / variant["name"]
        run = train_sft(
            train_path=subset_path,
            eval_path=processed / "sft_validation.jsonl",
            output_dir=output,
            run_name=f"tuning_{variant['name']}",
            epochs=1,
            max_steps=args.steps,
            lora_r=variant["lora_r"],
            lora_alpha=variant["lora_alpha"],
            learning_rate=variant["learning_rate"],
        )

        # The card must be given back before loading the next variant: four training +
        # evaluation cycles in the same process otherwise saturate the 16 GB, and loading falls
        # back onto the CPU.
        free_gpu_memory()
        agent = TriageAgent(adapter_dir=output, base_model=MODEL.base_model)
        answers = agent.generate_batch(symptoms)
        triage_accuracy = accuracy(expected, [r.level for r in answers])
        clean_stops = sum(r.clean_stop for r in answers) / len(answers)
        del agent
        free_gpu_memory()

        results.append(
            {
                "variant": variant["name"],
                "lora_r": variant["lora_r"],
                "lora_alpha": variant["lora_alpha"],
                "learning_rate": variant["learning_rate"],
                "trainable_parameters": run.trainable_params,
                "train_loss": run.train_loss,
                "eval_loss": run.eval_loss,
                "triage_accuracy": round(triage_accuracy, 3),
                "clean_stop_share": round(clean_stops, 3),
                "duration_s": run.duration_s,
                "peak_gpu_memory_gb": run.peak_gpu_memory_gb,
            }
        )
        # Clean stops are logged at every variant, not only in the final table: it is the metric
        # that revealed the model could not emit its end token, and a zero must jump out during
        # the run rather than an hour later.
        logger.info("  triage accuracy: %.3f | clean stops: %.3f", triage_accuracy, clean_stops)
        # The tuning adapters are not meant to be kept.
        shutil.rmtree(output, ignore_errors=True)

    results.sort(key=lambda r: (-r["triage_accuracy"], r["eval_loss"] or 9.9))
    destination = PATHS.reports / "training" / "hyperparameter_comparison.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "protocol": {
                    "training_examples": len(subset),
                    "steps": args.steps,
                    "evaluated_cases": len(validation_cases),
                    "seed": SEED,
                },
                "variants": results,
                "kept": results[0]["variant"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(workspace, ignore_errors=True)

    # The confidence interval is shown with the accuracy, never without: on sixty validation
    # cases, three points of gap are two cases, and reading a column of bare accuracies gives
    # the illusion of a ranking.
    print("\n=== Variant comparison ===")
    header = (
        f"{'variant':12s} | {'eval_loss':9s} | {'accuracy [95% CI]':22s} | "
        f"{'clean stops':11s} | {'GPU mem':8s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        loss = f"{r['eval_loss']:.4f}" if r["eval_loss"] is not None else "n/a"
        low, high = wilson_interval(
            round(r["triage_accuracy"] * len(validation_cases)), len(validation_cases)
        )
        shown = f"{r['triage_accuracy']:.3f} [{low:.2f} - {high:.2f}]"
        print(
            f"{r['variant']:12s} | {loss:9s} | {shown:22s} | "
            f"{r['clean_stop_share']:<11.3f} | {r['peak_gpu_memory_gb']:.1f} GB"
        )
    print(f"\nSetting kept: {results[0]['variant']}")
    print("The intervals overlap: the comparison ranks, it does not decide.")


if __name__ == "__main__":
    main()
