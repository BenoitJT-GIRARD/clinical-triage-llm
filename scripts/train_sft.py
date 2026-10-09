"""Supervised fine-tuning (SFT + LoRA) of Qwen3-1.7B-Base.

Trains the adapter on the whole training split, with the setting ``tune_hyperparameters.py``
kept — read from its comparison file, never copied by hand. An experiment whose result depends on
a copy ends up deciding nothing: the published protocol would announce a winning setting while
the shipped model used another. Without that file the script refuses to train, unless the three
values are given explicitly.

``--seed`` trains another seed of the same method: its weights go under ``var/models/seed-<n>/``
and its summary under ``reports/training/seeds/``. ``--max-steps`` checks the chain: the run
writes under ``var/``, and never over a delivered artefact.

Usage::

    uv run python scripts/train_sft.py
    uv run python scripts/train_sft.py --seed 43
    uv run python scripts/train_sft.py --max-steps 20     # chain check
"""

from __future__ import annotations

from clinical_triage.bootstrap import use_utf8_console

use_utf8_console()

import argparse
import json

from clinical_triage.config import PATHS, SEED, TRAINING
from clinical_triage.training.selection import kept_setting
from clinical_triage.training.sft import train_sft
from clinical_triage.utils import get_logger, set_seed

logger = get_logger("train_sft")


def setting(args: argparse.Namespace) -> tuple[int, int, float]:
    """The rank, scaling and learning rate to train with, and where they come from."""
    explicit = (args.lora_r, args.lora_alpha, args.learning_rate)
    if all(value is not None for value in explicit):
        logger.info("Setting given on the command line: r=%d, alpha=%d, lr=%g", *explicit)
        return explicit
    kept = kept_setting()
    if kept is None:
        raise SystemExit(
            "No setting to train with: run scripts/tune_hyperparameters.py first, or give "
            "--lora-r, --lora-alpha and --learning-rate together."
        )
    if any(value is not None for value in explicit):
        raise SystemExit("Give the three settings together, or none: the tuning keeps all three.")
    logger.info(
        "Setting kept by the tuning: %s (r=%d, alpha=%d, lr=%g)",
        kept["variant"],
        kept["lora_r"],
        kept["lora_alpha"],
        kept["learning_rate"],
    )
    return kept["lora_r"], kept["lora_alpha"], kept["learning_rate"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=TRAINING.sft_epochs)
    parser.add_argument("--batch-size", type=int, default=TRAINING.sft_batch_size)
    parser.add_argument("--grad-accum", type=int, default=TRAINING.sft_grad_accum)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--no-grad-checkpointing", action="store_true")
    args = parser.parse_args()

    lora_r, lora_alpha, learning_rate = setting(args)
    capped = args.max_steps > 0
    paths = PATHS.capped() if capped else PATHS.for_seed(args.seed)

    set_seed(args.seed)
    processed = PATHS.data_processed
    result = train_sft(
        train_path=processed / "sft_train.jsonl",
        eval_path=processed / "sft_validation.jsonl",
        output_dir=paths.sft_adapter,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        gradient_checkpointing=not args.no_grad_checkpointing,
        seed=args.seed,
        summary_dir=PATHS.training_summaries(args.seed, capped=capped),
    )
    logger.info("Result: %s", json.dumps(result.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
