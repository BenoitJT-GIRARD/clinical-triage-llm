"""Alignement par préférences (DPO) au-dessus du modèle SFT.

Le DPO part du modèle SFT **fusionné**, produit par `04_merge_and_export.py`.
C'est volontaire : la bibliothèque prend alors le modèle sans adaptateur comme
référence, c'est-à-dire le modèle SFT lui-même. Avec le modèle de base comme
référence, l'alignement serait libre de s'éloigner du format tout juste appris.

Usage :
    uv run python scripts/05_train_dpo.py
    uv run python scripts/05_train_dpo.py --max-steps 20   # vérification du pipeline
    uv run python scripts/05_train_dpo.py --beta 0.2
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
    parser.add_argument("--epochs", type=int, default=TRAINING.dpo_epochs)
    parser.add_argument("--batch-size", type=int, default=TRAINING.dpo_batch_size)
    parser.add_argument("--grad-accum", type=int, default=TRAINING.dpo_grad_accum)
    parser.add_argument("--learning-rate", type=float, default=TRAINING.dpo_lr)
    parser.add_argument("--beta", type=float, default=TRAINING.dpo_beta)
    parser.add_argument("--rpo-alpha", type=float, default=TRAINING.dpo_rpo_alpha)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    set_seed(SEED)
    if not PATHS.sft_merged.exists():
        raise SystemExit(
            "Modèle SFT fusionné introuvable. Lancez d'abord :\n"
            "  uv run python scripts/04_merge_and_export.py --adapter sft"
        )

    resultat = train_dpo(
        train_path=PATHS.data_processed / "dpo_train.jsonl",
        sft_merged_dir=PATHS.sft_merged,
        output_dir=PATHS.dpo_adapter,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        beta=args.beta,
        rpo_alpha=args.rpo_alpha,
        max_steps=args.max_steps,
    )
    logger.info("Résultat : %s", json.dumps(resultat.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
