"""Supervised fine-tuning (SFT + LoRA) of Qwen3-1.7B-Base.

Trains the final adapter on the whole training split, with the setting that
``tune_hyperparameters.py`` kept — read from its comparison file, not copied by hand. An
experiment whose result depends on a copy ends up deciding nothing: the published protocol would
announce a winning variant while the shipped model used another.

Failing a comparison on disk, the values from ``config.py`` apply. The values actually used are
logged into the run summary: that file is what counts.

Usage::

    uv run python scripts/train_sft.py
    uv run python scripts/train_sft.py --max-steps 20     # pipeline check
    uv run python scripts/train_sft.py --epochs 3
"""

from __future__ import annotations

from clinical_triage.bootstrap import use_utf8_console

use_utf8_console()

import argparse
import json

from clinical_triage.config import PATHS, SEED, TRAINING
from clinical_triage.training.sft import train_sft
from clinical_triage.utils import get_logger, set_seed

logger = get_logger("train_sft")


def kept_configuration() -> dict:
    """Read back the variant the hyper-parameter tuning designated.

    Returns an empty dictionary when the comparison was not produced: training then uses the
    values from ``config.py``, and says so.
    """
    path = PATHS.reports / "training" / "hyperparameter_comparison.json"
    if not path.exists():
        logger.info("No hyper-parameter comparison: using the values from config.py.")
        return {}

    comparison = json.loads(path.read_text(encoding="utf-8"))
    name = comparison["kept"]
    kept = next(v for v in comparison["variants"] if v["variant"] == name)
    logger.info(
        "Setting kept by the tuning: %s (r=%d, alpha=%d, lr=%g, accuracy %.3f)",
        name,
        kept["lora_r"],
        kept["lora_alpha"],
        kept["learning_rate"],
        kept["triage_accuracy"],
    )
    return kept


def main() -> None:
    kept = kept_configuration()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=TRAINING.sft_epochs)
    parser.add_argument("--batch-size", type=int, default=TRAINING.sft_batch_size)
    parser.add_argument("--grad-accum", type=int, default=TRAINING.sft_grad_accum)
    parser.add_argument(
        "--learning-rate", type=float, default=kept.get("learning_rate", TRAINING.sft_lr)
    )
    parser.add_argument("--lora-r", type=int, default=kept.get("lora_r", TRAINING.lora_r))
    parser.add_argument(
        "--lora-alpha", type=int, default=kept.get("lora_alpha", TRAINING.lora_alpha)
    )
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--no-grad-checkpointing", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    processed = PATHS.data_processed

    result = train_sft(
        train_path=processed / "sft_train.jsonl",
        eval_path=processed / "sft_validation.jsonl",
        output_dir=PATHS.sft_adapter,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_steps=args.max_steps,
        gradient_checkpointing=not args.no_grad_checkpointing,
    )
    logger.info("Result: %s", json.dumps(result.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
