"""Alignement par préférences (DPO) au-dessus du modèle fine-tuné, avec Unsloth.

Le SFT apprend à répondre ; le DPO apprend à préférer. Ici, la préférence porte
sur la sécurité clinique : entre deux réponses au même cas, on veut celle qui ne
sous-évalue pas l'urgence, ne retarde pas la prise en charge, n'affirme pas de
diagnostic et répond dans la langue attendue.

Quatre choix de méthode, tous destinés à empêcher l'alignement de défaire le
travail du SFT :

1. **La référence est le modèle SFT, pas le modèle de base.** On charge le modèle
   SFT fusionné et on y attache un *nouvel* adaptateur LoRA. La bibliothèque
   utilise alors le modèle adaptateur désactivé comme référence : c'est bien le
   modèle SFT. Prendre le modèle de base comme référence autoriserait le DPO à
   s'éloigner du format tout juste appris.
2. **`rpo_alpha` ajoute la perte supervisée sur la réponse préférée.** Le DPO seul
   n'optimise qu'un rapport de vraisemblances : il peut faire baisser la
   probabilité des deux réponses pourvu que l'écart se creuse. Ce second terme
   maintient la réponse préférée à un niveau de vraisemblance élevé.
3. **L'adaptateur ne touche pas à la tête de sortie.** Le SFT s'en est déjà
   chargé, et le modèle fusionné qui sert de base sait donc s'arrêter. Laisser la
   tête hors de l'adaptateur garde celui-ci petit — et surtout servable à chaud
   par vLLM, qui refuse les adaptateurs portant un `modules_to_save`.
4. **Un jeu de validation est mis de côté.** Sans lui, les seules courbes
   disponibles sont celles de l'entraînement, et rien ne permet de distinguer un
   alignement qui généralise d'un alignement qui apprend par cœur.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from clinical_triage.config import MODEL, SEED, TRAINING
from clinical_triage.prompts import prepare_tokenizer
from clinical_triage.training.schedule import warmup_steps
from clinical_triage.training.tracking import track
from clinical_triage.utils import chemin_pour_journal, get_logger

logger = get_logger("dpo")


def _referencer_le_modele_de_base(output_dir: Path) -> None:
    """Remplace le chemin local du modèle de référence par son identifiant public.

    L'adaptateur est entraîné au-dessus du modèle SFT fusionné, présent sur le
    disque. La bibliothèque inscrit donc le **chemin absolu** de ce dossier dans
    la configuration de l'adaptateur, ce qui rend les poids inchargeables sur
    toute autre machine — y compris après publication sur le Hub. On y substitue
    l'identifiant du dépôt public correspondant.
    """
    configuration = output_dir / "adapter_config.json"
    if not configuration.exists():
        return
    contenu = json.loads(configuration.read_text(encoding="utf-8"))
    contenu["base_model_name_or_path"] = MODEL.hub_sft_merged_model_id
    configuration.write_text(
        json.dumps(contenu, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Modèle de référence de l'adaptateur : %s", contenu["base_model_name_or_path"])


@dataclass
class DPOResult:
    """Résultat d'un alignement par préférences."""

    output_dir: str
    train_loss: float | None
    eval_reward_accuracy: float | None
    eval_reward_margin: float | None
    duree_s: float
    memoire_gpu_max_go: float


def train_dpo(
    train_path: Path,
    sft_merged_dir: Path,
    output_dir: Path,
    run_name: str = "dpo",
    epochs: int = TRAINING.dpo_epochs,
    batch_size: int = TRAINING.dpo_batch_size,
    grad_accum: int = TRAINING.dpo_grad_accum,
    learning_rate: float = TRAINING.dpo_lr,
    beta: float = TRAINING.dpo_beta,
    rpo_alpha: float | None = TRAINING.dpo_rpo_alpha,
    eval_ratio: float = 0.1,
    max_steps: int = -1,
) -> DPOResult:
    """Lance l'alignement DPO au-dessus du modèle SFT fusionné."""
    # Même contrainte qu'au SFT : Unsloth doit précéder `trl`. `PatchDPOTrainer`
    # doit en plus être appelé avant l'import de `DPOTrainer`, sinon c'est la
    # version non corrigée qui est chargée.
    from unsloth import FastLanguageModel, PatchDPOTrainer

    PatchDPOTrainer()

    import torch
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    output_dir.mkdir(parents=True, exist_ok=True)

    hyperparametres = {
        "modele_de_reference": str(sft_merged_dir.name),
        "moteur": "unsloth",
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lot_effectif": batch_size * grad_accum,
        "learning_rate": learning_rate,
        "beta": beta,
        "rpo_alpha": rpo_alpha,
        "lora_r": TRAINING.lora_r,
        "lora_alpha": TRAINING.lora_alpha,
        "max_length": MODEL.max_seq_length,
        "max_steps": max_steps,
        "graine": SEED,
    }

    with track(run_name, hyperparametres) as resume:
        logger.info(
            "Chargement du modèle SFT fusionné (base et référence) : %s",
            chemin_pour_journal(sft_merged_dir),
        )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(sft_merged_dir),
            max_seq_length=MODEL.max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=False,
        )
        prepare_tokenizer(tokenizer)

        model = FastLanguageModel.get_peft_model(
            model,
            r=TRAINING.lora_r,
            lora_alpha=TRAINING.lora_alpha,
            lora_dropout=TRAINING.lora_dropout,
            target_modules=list(TRAINING.lora_target_modules),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
        )

        dataset = load_dataset("json", data_files={"train": str(train_path)})["train"]
        dataset = dataset.select_columns(["prompt", "chosen", "rejected"])
        decoupage = dataset.train_test_split(test_size=eval_ratio, seed=SEED)
        logger.info(
            "Paires : %d pour l'entraînement, %d pour la validation.",
            len(decoupage["train"]),
            len(decoupage["test"]),
        )

        config = DPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            beta=beta,
            rpo_alpha=rpo_alpha,
            max_steps=max_steps,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            eval_strategy="steps",
            eval_steps=25,
            bf16=True,
            max_length=MODEL.max_seq_length,
            warmup_steps=warmup_steps(
                len(decoupage["train"]), batch_size * grad_accum, epochs, max_steps, 0.05
            ),
            lr_scheduler_type="cosine",
            seed=SEED,
            data_seed=SEED,
            report_to=[],
        )

        # `ref_model=None` : la référence est le modèle courant, adaptateur
        # désactivé — c'est-à-dire le modèle SFT fusionné.
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=config,
            train_dataset=decoupage["train"],
            eval_dataset=decoupage["test"],
            processing_class=tokenizer,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        debut = time.perf_counter()
        resultat = trainer.train()
        duree = time.perf_counter() - debut
        memoire = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0

        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        trainer.save_state()
        _referencer_le_modele_de_base(output_dir)

        historique = trainer.state.log_history
        derniere_eval = next(
            (e for e in reversed(historique) if "eval_rewards/accuracies" in e), {}
        )
        precision = derniere_eval.get("eval_rewards/accuracies")
        marge = derniere_eval.get("eval_rewards/margins")

        resume.historique = historique
        resume.metriques = {
            "train_loss": float(resultat.training_loss),
            # Effectif du jeu de validation : sans lui, la part de paires bien
            # ordonnées est une proportion dont personne ne peut estimer
            # l'incertitude, et la figure la trace nue.
            "paires_entrainement": len(decoupage["train"]),
            "paires_validation": len(decoupage["test"]),
            "eval_reward_accuracy": float(precision) if precision is not None else None,
            "eval_reward_margin": float(marge) if marge is not None else None,
            "duree_s": round(duree, 1),
            "memoire_gpu_max_go": round(memoire, 2),
        }
        logger.info(
            "DPO terminé en %.0f s. Préférences correctement ordonnées sur le jeu de validation : %s",
            duree,
            f"{precision:.3f}" if precision is not None else "non mesuré",
        )

        return DPOResult(
            output_dir=str(output_dir),
            train_loss=float(resultat.training_loss),
            eval_reward_accuracy=float(precision) if precision is not None else None,
            eval_reward_margin=float(marge) if marge is not None else None,
            duree_s=round(duree, 1),
            memoire_gpu_max_go=round(memoire, 2),
        )
