"""Preference alignment (DPO) above the supervised model.

DPO starts from the **merged** SFT model, produced by ``merge_adapter.py``. That is deliberate:
the library then takes the model without its adapter as the reference, that is to say the SFT
model itself. With the base model as reference, the alignment would be free to drift away from
the format just learnt.

``--seed`` aligns the supervised model of that seed; ``--max-steps`` checks the chain and writes
under ``var/`` only, as ``train_sft.py`` does.

Usage::

    uv run python scripts/train_dpo.py
    uv run python scripts/train_dpo.py --seed 43
    uv run python scripts/train_dpo.py --max-steps 20   # does the chain hold?
"""

from __future__ import annotations

from clinical_triage.bootstrap import use_utf8_console

use_utf8_console()

import argparse
import json

from clinical_triage.config import PATHS, SEED, TRAINING
from clinical_triage.training.dpo import train_dpo
from clinical_triage.utils import get_logger, set_seed

logger = get_logger("train_dpo")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=TRAINING.dpo_epochs)
    parser.add_argument("--batch-size", type=int, default=TRAINING.dpo_batch_size)
    parser.add_argument("--grad-accum", type=int, default=TRAINING.dpo_grad_accum)
    parser.add_argument("--learning-rate", type=float, default=TRAINING.dpo_lr)
    parser.add_argument("--beta", type=float, default=TRAINING.dpo_beta)
    parser.add_argument("--rpo-alpha", type=float, default=TRAINING.dpo_rpo_alpha)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    capped = args.max_steps > 0
    paths = PATHS.capped() if capped else PATHS.for_seed(args.seed)
    set_seed(args.seed)
    if not paths.sft_merged.exists():
        raise SystemExit(
            "Merged SFT model not found. Run first:\n"
            f"  uv run python scripts/merge_adapter.py --adapter sft --seed {args.seed}"
        )

    result = train_dpo(
        train_path=PATHS.data_processed / "dpo_train.jsonl",
        sft_merged_dir=paths.sft_merged,
        output_dir=paths.dpo_adapter,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        beta=args.beta,
        rpo_alpha=args.rpo_alpha,
        max_steps=args.max_steps,
        seed=args.seed,
        summary_dir=PATHS.training_summaries(args.seed, capped=capped),
    )
    logger.info("Result: %s", json.dumps(result.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
