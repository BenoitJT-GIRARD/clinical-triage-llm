"""Fine-tuning supervisé (SFT + LoRA) du Qwen3-1.7B-Base.

Entraîne l'adaptateur définitif sur l'ensemble du jeu d'entraînement, avec la
configuration que `02_tune_hyperparameters.py` a retenue — lue dans son fichier
de comparaison, et non recopiée à la main. Une expérience dont le résultat
dépend d'une recopie finit par ne plus décider de rien : le rapport annoncerait
une variante gagnante et le modèle livré en utiliserait une autre.

À défaut de comparaison sur le disque, les valeurs de `config.py` s'appliquent.
Les valeurs réellement utilisées sont journalisées dans le résumé d'exécution :
c'est ce fichier qui fait foi pour le rapport.

Usage :
    uv run python scripts/03_train_sft.py
    uv run python scripts/03_train_sft.py --max-steps 20     # vérification du pipeline
    uv run python scripts/03_train_sft.py --epochs 3
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


def configuration_retenue() -> dict:
    """Relit la variante que le réglage des hyperparamètres a désignée.

    Renvoie un dictionnaire vide si la comparaison n'a pas été produite : on
    entraîne alors avec les valeurs de `config.py`, en le disant.
    """
    fichier = PATHS.reports / "training" / "comparaison_hyperparametres.json"
    if not fichier.exists():
        logger.info("Pas de comparaison d'hyperparamètres : valeurs de config.py.")
        return {}

    comparaison = json.loads(fichier.read_text(encoding="utf-8"))
    nom = comparaison["retenue"]
    retenue = next(v for v in comparaison["variantes"] if v["variante"] == nom)
    logger.info(
        "Configuration retenue par le réglage : %s (r=%d, alpha=%d, lr=%g, exactitude %.3f)",
        nom,
        retenue["lora_r"],
        retenue["lora_alpha"],
        retenue["learning_rate"],
        retenue["exactitude_triage"],
    )
    return retenue


def main() -> None:
    retenue = configuration_retenue()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=TRAINING.sft_epochs)
    parser.add_argument("--batch-size", type=int, default=TRAINING.sft_batch_size)
    parser.add_argument("--grad-accum", type=int, default=TRAINING.sft_grad_accum)
    parser.add_argument(
        "--learning-rate", type=float, default=retenue.get("learning_rate", TRAINING.sft_lr)
    )
    parser.add_argument("--lora-r", type=int, default=retenue.get("lora_r", TRAINING.lora_r))
    parser.add_argument(
        "--lora-alpha", type=int, default=retenue.get("lora_alpha", TRAINING.lora_alpha)
    )
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--no-grad-checkpointing", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    processed = PATHS.data_processed

    resultat = train_sft(
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
    logger.info("Résultat : %s", json.dumps(resultat.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
