"""Fine-tuning supervisé du Qwen3-1.7B-Base avec Unsloth et LoRA.

Le modèle de base ne sait pas dialoguer : il complète du texte. Le SFT lui
apprend deux choses à la fois — le format ChatML de l'agent, et la tâche de
triage elle-même.

Quatre points méritent d'être explicités, parce qu'ils conditionnent la suite :

1. **Le jeu de données expose `prompt` et `completion`**, pas une colonne
   `messages`. Cela évite que la bibliothèque d'entraînement ne re-sérialise les
   exemples avec le gabarit de dialogue natif du modèle — celui de Qwen3 insère
   un bloc `<think></think>` devant la réponse, et le modèle apprendrait alors un
   format que l'inférence ne reproduit jamais. Le couple prompt/completion permet
   en prime de ne calculer la perte que sur la réponse, sans collateur maison.
2. **Le jeton de fin de séquence est corrigé** avant l'entraînement : le
   tokenizer de Qwen3-1.7B-Base sort d'usine avec `<|endoftext|>` alors que le
   format appris se termine par `<|im_end|>`.
3. **La tête de sortie est entraînée avec les projections.** C'est la correction
   la moins évidente et la plus décisive du projet — voir plus bas.
4. **LoRA n'entraîne que les matrices d'adaptation** des projections — quelques
   millions de paramètres — auxquelles s'ajoute la tête de sortie, entraînée en
   entier au point précédent. Elle pèse à elle seule près de 311 millions de
   paramètres : le total entraîné avoisine donc 14 % du modèle, et non le
   pourcent que LoRA seul laisserait attendre. C'est le prix de la correction du
   jeton de fin, et le résumé d'exécution publie le compte exact.

## Pourquoi la tête de sortie doit être entraînée

Qwen3-1.7B-**Base** porte les vingt-cinq jetons ChatML (151644 à 151668) comme un
seul et même vecteur, jamais entraîné : leurs normes sont identiques au bit près,
et `<|im_start|>` et `<|im_end|>` ont une similarité cosinus de 1,000. Le modèle
lie par ailleurs sa tête de sortie à sa matrice d'embeddings
(`tie_word_embeddings`), que LoRA gèle.

Déclarer `<|im_end|>` comme fin de séquence ne suffit donc pas : sa ligne de
sortie reste figée sur une valeur d'initialisation partagée avec vingt-quatre
autres jetons, et le modèle ne peut **structurellement pas** l'émettre de
préférence à eux. Mesuré : 0 % d'arrêts nets, et les 220 jetons du budget
consommés à chaque réponse. Aucun nombre d'époques supplémentaires n'y change
quoi que ce soit.

Ajouter `lm_head` aux modules adaptés lève le verrou. Trois configurations ont été
comparées à protocole identique — 75 pas, 60 cas de validation :

| Configuration | Exactitude | Arrêts nets | Jetons générés |
|---|---|---|---|
| projections seules | 0,850 | 0,000 | 220 |
| + embeddings (`modules_to_save`) | 0,850 | 0,000 | 220 |
| **+ tête de sortie** | **0,900** | **1,000** | **97** |

Les chiffres exacts vivent dans `reports/training/comparaison_modules.json`, que
le rapport lit ; ceux-ci les reprennent pour que la démonstration se suive sans
quitter le fichier.

Entraîner les seuls embeddings ne sert à rien : PEFT en recopie le module et
rompt le lien avec la tête de sortie, qui reste gelée. C'est bien la tête qu'il
faut adapter.

Unsloth le fait nativement : il déplace `lm_head` vers `modules_to_save` et gère
le déliage des poids. Son correctif `fix_untrained_tokens`, lui, ne détecte que
les lignes d'embedding exactement nulles : sur ce modèle, dont les jetons ChatML
sont des vecteurs dupliqués de faible norme, il resterait sans effet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from clinical_triage.config import MODEL, SEED, TRAINING
from clinical_triage.prompts import IM_END, prepare_tokenizer
from clinical_triage.training.schedule import warmup_steps
from clinical_triage.training.tracking import track
from clinical_triage.utils import chemin_pour_journal, get_logger

logger = get_logger("sft")


@dataclass
class SFTResult:
    """Résultat d'un entraînement supervisé."""

    output_dir: str
    trainable_params: int
    total_params: int
    train_loss: float | None
    eval_loss: float | None
    duree_s: float
    memoire_gpu_max_go: float


def train_sft(
    train_path: Path,
    eval_path: Path | None,
    output_dir: Path,
    run_name: str = "sft",
    epochs: int = TRAINING.sft_epochs,
    batch_size: int = TRAINING.sft_batch_size,
    grad_accum: int = TRAINING.sft_grad_accum,
    learning_rate: float = TRAINING.sft_lr,
    lora_r: int = TRAINING.lora_r,
    lora_alpha: int = TRAINING.lora_alpha,
    max_steps: int = -1,
    max_length: int = MODEL.max_seq_length,
    gradient_checkpointing: bool = True,
) -> SFTResult:
    """Lance le fine-tuning supervisé et renvoie le résumé de l'entraînement."""
    # Unsloth réécrit des fonctions internes de `transformers` et de `trl` au
    # moment de son import. Il doit donc être importé AVANT elles : importé après,
    # ses correctifs s'appliquent à des références déjà copiées et n'ont plus
    # aucun effet — sans le moindre message pour le signaler.
    # L'ordre ci-dessous est fonctionnel : le trier alphabétiquement placerait
    # `unsloth` en dernier et neutraliserait ses correctifs, d'où la dérogation.
    from unsloth import FastLanguageModel  # noqa: I001

    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    output_dir.mkdir(parents=True, exist_ok=True)

    # La tête de sortie rejoint les projections adaptées : sans elle, le modèle
    # ne peut pas apprendre à s'arrêter (voir la docstring du module).
    modules_adaptes = [*TRAINING.lora_target_modules, "lm_head"]

    hyperparametres = {
        "modele_de_base": MODEL.base_model,
        "moteur": "unsloth",
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lot_effectif": batch_size * grad_accum,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": TRAINING.lora_dropout,
        "lora_modules": ", ".join(modules_adaptes),
        "max_length": max_length,
        "gradient_checkpointing": gradient_checkpointing,
        "max_steps": max_steps,
        "graine": SEED,
    }

    with track(run_name, hyperparametres) as resume:
        logger.info("Chargement du tokenizer et du modèle de base : %s", MODEL.base_model)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL.base_model,
            max_seq_length=max_length,
            dtype=torch.bfloat16,
            load_in_4bit=False,
        )
        prepare_tokenizer(tokenizer)

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=TRAINING.lora_dropout,
            target_modules=modules_adaptes,
            bias="none",
            # Le point de reprise d'Unsloth décharge les activations vers la
            # mémoire centrale : même calcul, empreinte GPU nettement moindre.
            use_gradient_checkpointing="unsloth" if gradient_checkpointing else False,
            random_state=SEED,
        )

        data_files = {"train": str(train_path)}
        if eval_path is not None:
            data_files["eval"] = str(eval_path)
        dataset = load_dataset("json", data_files=data_files)
        # L'entraînement n'a besoin que de ces deux colonnes ; les métadonnées
        # cliniques du dataset ne doivent pas atteindre le collateur.
        dataset = dataset.select_columns(["prompt", "completion"])

        config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            max_steps=max_steps,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            eval_strategy="epoch" if eval_path is not None else "no",
            load_best_model_at_end=eval_path is not None,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=True,
            max_length=max_length,
            warmup_steps=warmup_steps(
                len(dataset["train"]),
                batch_size * grad_accum,
                epochs,
                max_steps,
                TRAINING.sft_warmup_ratio,
            ),
            lr_scheduler_type="cosine",
            seed=SEED,
            data_seed=SEED,
            eos_token=IM_END,
            # La perte par morceaux économise de la mémoire en découpant le calcul
            # sur le vocabulaire, mais elle suppose une tête de sortie intacte.
            # Elle est donc incompatible avec l'adaptation de `lm_head`.
            loss_type="nll",
            report_to=[],
        )

        # Le modèle porte déjà son adaptateur : pas de `peft_config` ici, sans
        # quoi TRL en grefferait un second par-dessus celui d'Unsloth.
        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("eval"),
            processing_class=tokenizer,
        )

        entrainables = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in trainer.model.parameters())
        logger.info(
            "Paramètres entraînables : %s sur %s (%.2f %%)",
            f"{entrainables:,}".replace(",", " "),
            f"{total:,}".replace(",", " "),
            100 * entrainables / total,
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

        historique = trainer.state.log_history
        perte_eval = next((e["eval_loss"] for e in reversed(historique) if "eval_loss" in e), None)
        resume.historique = historique
        resume.metriques = {
            "train_loss": float(resultat.training_loss),
            "eval_loss": float(perte_eval) if perte_eval is not None else None,
            "parametres_entrainables": entrainables,
            "parametres_total": total,
            "part_entrainable_pct": round(100 * entrainables / total, 3),
            "duree_s": round(duree, 1),
            "memoire_gpu_max_go": round(memoire, 2),
            "exemples_par_seconde": round(resultat.metrics.get("train_samples_per_second", 0.0), 2),
        }
        logger.info(
            "SFT terminé en %.0f s. Adaptateur sauvegardé dans %s",
            duree,
            chemin_pour_journal(output_dir),
        )

        return SFTResult(
            output_dir=str(output_dir),
            trainable_params=entrainables,
            total_params=total,
            train_loss=float(resultat.training_loss),
            eval_loss=float(perte_eval) if perte_eval is not None else None,
            duree_s=round(duree, 1),
            memoire_gpu_max_go=round(memoire, 2),
        )
