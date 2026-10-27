"""The three studies that decide how the supervised pass is trained, each replayed by one command.

``--study settings`` (the default) compares the LoRA settings of
``clinical_triage.training.selection.SETTINGS`` on a subset of the corpus — same seed, same
subset, same number of steps — and judges each on two criteria: the **validation loss**, which
says whether the model learns the distribution, and the **triage accuracy** on cases of the
validation split, which says whether that learning turns into the right decision. The setting
with the best accuracy is kept, and ``train_sft.py`` reads it back. Its accuracy is a
**selection score**: measured on the cases it was chosen on, it is optimistic by construction,
and the model is judged on the clinical set and nowhere else.

``--study modules`` compares three sets of adapted modules (projections only; projections and
embeddings; projections and output head) on the same protocol, adding the share of answers that
stop cleanly. It is the study that showed the output head must be trained.

``--study engines`` trains the same configuration for a few steps with Unsloth and with
Transformers and PEFT alone, and compares memory and speed. Each engine runs in a process of its
own: Unsloth rewrites parts of `transformers` as soon as it is imported, and the second run would
otherwise measure the first engine under the second's name.

Each study writes its comparison under ``reports/training/``.

Usage::

    uv run python scripts/tune_hyperparameters.py
    uv run python scripts/tune_hyperparameters.py --study modules
    uv run python scripts/tune_hyperparameters.py --study engines
"""

from __future__ import annotations

from clinical_triage.bootstrap import use_utf8_console

use_utf8_console()

import argparse
import json
import shutil
import statistics
import subprocess  # nosec B404 - calls this very script back, once per engine
import sys
import time
from pathlib import Path

from clinical_triage.config import MODEL, PATHS, SEED, TRAINING
from clinical_triage.data.dataset_io import read_jsonl, write_jsonl
from clinical_triage.evaluation.metrics import accuracy, wilson_interval
from clinical_triage.training.selection import COMPARISON, SETTINGS
from clinical_triage.utils import free_gpu_memory, get_logger, set_seed

logger = get_logger("tune")

PROJECTIONS = tuple(TRAINING.lora_target_modules)

#: The adapted modules compared by ``--study modules``.
MODULE_CONFIGURATIONS = (
    ("projections only", PROJECTIONS),
    ("projections and embeddings", (*PROJECTIONS, "embed_tokens")),
    ("projections and output head", (*PROJECTIONS, "lm_head")),
)

#: The reference setting of the modules and engines studies: the middle of the settings grid.
REFERENCE = {"lora_r": 16, "lora_alpha": 32, "learning_rate": 2e-4}

WORKSPACE = PATHS.models / "tuning"
RESULTS = PATHS.reports / "training"


def _write(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    destination = RESULTS / name
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Comparison written → %s", destination.name)
    return destination


def _subset(size: int) -> Path:
    """A frozen subset of the training split, identical for every run of a study."""
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    path = WORKSPACE / "sft_train_subset.jsonl"
    write_jsonl(read_jsonl(PATHS.data_processed / "sft_train.jsonl")[:size], path)
    return path


def _judge(adapter: Path, cases: list[dict]) -> dict:
    """Triage accuracy, clean stops and answer length of an adapter on validation cases."""
    from clinical_triage.inference import TriageAgent

    free_gpu_memory()
    agent = TriageAgent(adapter_dir=adapter, base_model=MODEL.base_model)
    answers = agent.generate_batch([row["user_turn"] for row in cases])
    del agent
    free_gpu_memory()
    return {
        "accuracy": round(accuracy([row["level"] for row in cases], [a.level for a in answers]), 3),
        "clean_stop_share": round(sum(a.clean_stop for a in answers) / len(answers), 3),
        "median_tokens": statistics.median(a.generated_tokens for a in answers),
    }


def study_settings(args: argparse.Namespace) -> None:
    from clinical_triage.training.sft import train_sft

    subset = _subset(args.train_subset)
    cases = read_jsonl(PATHS.data_processed / "sft_validation.jsonl")[: args.eval_cases]
    results = []
    for setting in SETTINGS:
        logger.info("=== Setting %s ===", setting["name"])
        output = WORKSPACE / setting["name"]
        run = train_sft(
            train_path=subset,
            eval_path=PATHS.data_processed / "sft_validation.jsonl",
            output_dir=output,
            run_name=f"tuning_{setting['name']}",
            epochs=1,
            max_steps=args.steps,
            lora_r=setting["lora_r"],
            lora_alpha=setting["lora_alpha"],
            learning_rate=setting["learning_rate"],
        )
        judged = _judge(output, cases)
        # Clean stops are logged for every setting, not only in the final table: a zero must jump
        # out during the run rather than an hour later.
        logger.info(
            "  accuracy %.3f | clean stops %.3f", judged["accuracy"], judged["clean_stop_share"]
        )
        results.append(
            {
                "variant": setting["name"],
                "lora_r": setting["lora_r"],
                "lora_alpha": setting["lora_alpha"],
                "learning_rate": setting["learning_rate"],
                "trainable_parameters": run.trainable_params,
                "train_loss": run.train_loss,
                "eval_loss": run.eval_loss,
                "selection_accuracy": judged["accuracy"],
                "clean_stop_share": judged["clean_stop_share"],
                "duration_s": run.duration_s,
                "peak_gpu_memory_gb": run.peak_gpu_memory_gb,
            }
        )
        shutil.rmtree(output, ignore_errors=True)

    results.sort(key=lambda r: (-r["selection_accuracy"], r["eval_loss"] or 9.9))
    _write(
        COMPARISON.name,
        {
            "protocol": {
                "training_examples": args.train_subset,
                "steps": args.steps,
                "evaluated_cases": len(cases),
                "seed": SEED,
                "selection": "highest accuracy on the evaluated cases, then lowest validation loss",
            },
            "variants": results,
            "kept": results[0]["variant"],
        },
    )
    # The interval goes with the accuracy, never without: on sixty validation cases, three points
    # are two cases, and a column of bare accuracies gives the illusion of a ranking.
    print(
        f"\n{'setting':12s} | {'eval_loss':9s} | {'selection accuracy [95% CI]':28s} | clean stops"
    )
    for r in results:
        low, high = wilson_interval(round(r["selection_accuracy"] * len(cases)), len(cases))
        loss = f"{r['eval_loss']:.4f}" if r["eval_loss"] is not None else "n/a"
        print(
            f"{r['variant']:12s} | {loss:9s} | {r['selection_accuracy']:.3f} [{low:.2f} - {high:.2f}]"
            f"{'':10s} | {r['clean_stop_share']:.3f}"
        )
    print(
        f"\nSetting kept: {results[0]['variant']}. The intervals overlap: the study ranks, it does not decide."
    )


def study_modules(args: argparse.Namespace) -> None:
    from clinical_triage.training.sft import train_sft

    subset = _subset(args.train_subset)
    cases = read_jsonl(PATHS.data_processed / "sft_validation.jsonl")[: args.eval_cases]
    configurations = []
    for name, modules in MODULE_CONFIGURATIONS:
        logger.info("=== Modules: %s ===", name)
        output = WORKSPACE / name.replace(" ", "_")
        run = train_sft(
            train_path=subset,
            eval_path=None,
            output_dir=output,
            run_name=f"modules_{name.replace(' ', '_')}",
            epochs=1,
            max_steps=args.steps,
            modules=modules,
            summary_dir=PATHS.training_summaries(capped=True),
            **REFERENCE,
        )
        configurations.append(
            {
                "configuration": name,
                "modules": list(modules),
                "trainable_parameters": run.trainable_params,
                **_judge(output, cases),
                "duration_s": run.duration_s,
                "peak_gpu_memory_gb": run.peak_gpu_memory_gb,
            }
        )
        shutil.rmtree(output, ignore_errors=True)
    _write(
        "module_comparison.json",
        {
            "protocol": {
                "steps": args.steps,
                "training_examples": args.train_subset,
                "validation_cases": len(cases),
                "seed": SEED,
                **REFERENCE,
            },
            "configurations": configurations,
        },
    )


def _train_engine(engine: str, subset: Path, steps: int) -> dict:
    """Train the reference configuration for ``steps`` steps with one engine, in this process."""
    import torch

    modules = (*PROJECTIONS, "lm_head")
    if engine == "unsloth":
        from clinical_triage.training.sft import train_sft

        run = train_sft(
            train_path=subset,
            eval_path=None,
            output_dir=WORKSPACE / "engine_unsloth",
            run_name="engine_unsloth",
            epochs=1,
            max_steps=steps,
            modules=modules,
            summary_dir=PATHS.training_summaries(capped=True),
            **REFERENCE,
        )
        trainable, duration, memory = run.trainable_params, run.duration_s, run.peak_gpu_memory_gb
        final_loss = run.train_loss
    else:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        from clinical_triage.prompts import IM_END, prepare_tokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL.base_model)
        prepare_tokenizer(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(MODEL.base_model, dtype=torch.bfloat16)
        model = get_peft_model(
            model,
            LoraConfig(
                r=REFERENCE["lora_r"],
                lora_alpha=REFERENCE["lora_alpha"],
                lora_dropout=TRAINING.lora_dropout,
                target_modules=list(PROJECTIONS),
                # PEFT trains a whole module through `modules_to_save`, as Unsloth does with the
                # output head it is asked to adapt.
                modules_to_save=["lm_head"],
                task_type="CAUSAL_LM",
            ),
        )
        model.enable_input_require_grads()
        dataset = load_dataset("json", data_files={"train": str(subset)})["train"]
        trainer = SFTTrainer(
            model=model,
            args=SFTConfig(
                output_dir=str(WORKSPACE / "engine_transformers"),
                per_device_train_batch_size=TRAINING.sft_batch_size,
                gradient_accumulation_steps=TRAINING.sft_grad_accum,
                learning_rate=REFERENCE["learning_rate"],
                max_steps=steps,
                logging_steps=10,
                save_strategy="no",
                bf16=True,
                max_length=MODEL.max_seq_length,
                gradient_checkpointing=True,
                seed=SEED,
                data_seed=SEED,
                eos_token=IM_END,
                loss_type="nll",
                report_to=[],
            ),
            train_dataset=dataset.select_columns(["prompt", "completion"]),
            processing_class=tokenizer,
        )
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = trainer.train()
        duration = round(time.perf_counter() - started, 1)
        memory = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        final_loss = float(result.training_loss)
    return {
        "engine": engine,
        "steps": steps,
        "trainable_parameters": trainable,
        "peak_gpu_memory_gb": memory,
        "duration_s": duration,
        "seconds_per_step": round(duration / steps, 2),
        "final_loss": round(final_loss, 4),
        "torch": torch.__version__,
    }


def study_engines(args: argparse.Namespace) -> None:
    subset = _subset(args.train_subset)
    measured = {}
    for engine in ("unsloth", "transformers"):
        logger.info("=== Engine: %s (own process) ===", engine)
        output = WORKSPACE / f"engine_{engine}.json"
        subprocess.run(  # nosec B603 - fixed arguments: this very script and its own options
            [
                sys.executable,
                __file__,
                "--study",
                "engines",
                "--engine",
                engine,
                "--steps",
                str(args.engine_steps),
                "--subset-file",
                str(subset),
                "--output",
                str(output),
            ],
            check=True,
        )
        measured[engine] = json.loads(output.read_text(encoding="utf-8"))
    _write(
        "engine_comparison.json",
        {
            "protocol": {
                "steps": args.engine_steps,
                "training_examples": args.train_subset,
                "modules": [*PROJECTIONS, "lm_head"],
                "seed": SEED,
                **REFERENCE,
            },
            **measured,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("settings", "modules", "engines"), default="settings")
    parser.add_argument("--train-subset", type=int, default=1200)
    parser.add_argument("--steps", type=int, default=75)
    parser.add_argument("--engine-steps", type=int, default=30)
    parser.add_argument("--eval-cases", type=int, default=60)
    # The engines study calls this script back once per engine, with these three.
    parser.add_argument("--engine", choices=("unsloth", "transformers"), help=argparse.SUPPRESS)
    parser.add_argument("--subset-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    set_seed(SEED)
    if args.engine:
        measurement = _train_engine(args.engine, args.subset_file, args.steps)
        args.output.write_text(json.dumps(measurement, indent=2) + "\n", encoding="utf-8")
        return
    {"settings": study_settings, "modules": study_modules, "engines": study_engines}[args.study](
        args
    )
    shutil.rmtree(WORKSPACE, ignore_errors=True)


if __name__ == "__main__":
    main()
