"""Preference alignment (DPO) above the fine-tuned model, with Unsloth.

SFT teaches the model to answer; DPO teaches it to prefer. Here the preference is about clinical
safety: between two answers to the same case, the wanted one does not underestimate the urgency,
does not delay care, does not assert a diagnosis and answers in the expected language.

Four method choices, all meant to stop the alignment from undoing the work of the SFT:

1. **the reference is the SFT model, not the base model.** The merged SFT model is loaded and a
   *new* LoRA adapter is attached to it. The library then uses the model with its adapter
   disabled as the reference: that is exactly the SFT model. Taking the base model as reference
   would allow DPO to drift away from the format just learnt.
2. **``rpo_alpha`` keeps a supervised term on the answer that was preferred.** DPO alone optimises a
   likelihood ratio: it can lower the probability of both answers as long as the gap widens.
   This second term keeps the preferred answer at a high likelihood.
3. **the adapter does not touch the output head.** SFT already did, and the merged model that
   serves as the base therefore knows how to stop. Leaving the head out of the adapter keeps it
   small — and above all servable hot by vLLM, which refuses adapters carrying a
   ``modules_to_save``.
4. **a validation split is held out.** Without it the only curves available are the training
   ones, and nothing distinguishes an alignment that generalises from one that memorises.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from clinical_triage.config import MODEL, SEED, TRAINING
from clinical_triage.prompts import prepare_tokenizer
from clinical_triage.training.schedule import warmup_steps
from clinical_triage.training.tracking import track
from clinical_triage.utils import get_logger, path_for_log

logger = get_logger("dpo")


@dataclass
class DPOResult:
    """The result of one preference alignment run."""

    output_dir: str
    train_loss: float | None
    eval_reward_accuracy: float | None
    eval_reward_margin: float | None
    duration_s: float
    peak_gpu_memory_gb: float


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
    seed: int = SEED,
    summary_dir: Path | None = None,
) -> DPOResult:
    """Run the DPO alignment above the merged SFT model.

    ``seed`` sets how the adapter starts and in what order the pairs come. The split between
    training and validation pairs stays the project's: every seed is judged on the same pairs.
    """
    # Same constraint as in SFT: Unsloth must come before `trl`. `PatchDPOTrainer` must also be
    # called before importing `DPOTrainer`, otherwise the unpatched version is loaded.
    from unsloth import FastLanguageModel, PatchDPOTrainer

    PatchDPOTrainer()

    import torch
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    output_dir.mkdir(parents=True, exist_ok=True)

    hyperparameters = {
        "reference_model": str(sft_merged_dir.name),
        "engine": "unsloth",
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "learning_rate": learning_rate,
        "beta": beta,
        "rpo_alpha": rpo_alpha,
        "lora_r": TRAINING.dpo_lora_r,
        "lora_alpha": TRAINING.dpo_lora_alpha,
        "max_length": MODEL.max_seq_length,
        "max_steps": max_steps,
        "seed": seed,
    }

    with track(run_name, hyperparameters, summary_dir) as summary:
        logger.info(
            "Loading the merged SFT model (base and reference): %s",
            path_for_log(sft_merged_dir),
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
            r=TRAINING.dpo_lora_r,
            lora_alpha=TRAINING.dpo_lora_alpha,
            lora_dropout=TRAINING.lora_dropout,
            target_modules=list(TRAINING.lora_target_modules),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )

        dataset = load_dataset("json", data_files={"train": str(train_path)})["train"]
        dataset = dataset.select_columns(["prompt", "chosen", "rejected"])
        split = dataset.train_test_split(test_size=eval_ratio, seed=SEED)
        logger.info(
            "Pairs: %d for training, %d for validation.", len(split["train"]), len(split["test"])
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
                len(split["train"]), batch_size * grad_accum, epochs, max_steps, 0.05
            ),
            lr_scheduler_type="cosine",
            seed=seed,
            data_seed=seed,
            report_to=[],
        )

        # `ref_model=None`: the reference is the current model with its adapter disabled — that
        # is, the merged SFT model.
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=config,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            processing_class=tokenizer,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = trainer.train()
        duration = time.perf_counter() - started
        memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0

        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        trainer.save_state()

        history = trainer.state.log_history
        last_eval = next((e for e in reversed(history) if "eval_rewards/accuracies" in e), {})
        accuracy = last_eval.get("eval_rewards/accuracies")
        margin = last_eval.get("eval_rewards/margins")

        summary.history = history
        summary.metrics = {
            "train_loss": float(result.training_loss),
            # Size of the validation split: without it, the share of correctly ordered pairs is a
            # proportion whose uncertainty nobody can estimate, and a figure would draw it bare.
            "training_pairs": len(split["train"]),
            "validation_pairs": len(split["test"]),
            "eval_reward_accuracy": float(accuracy) if accuracy is not None else None,
            "eval_reward_margin": float(margin) if margin is not None else None,
            "duration_s": round(duration, 1),
            "peak_gpu_memory_gb": round(memory, 2),
        }
        logger.info(
            "DPO finished in %.0f s. Preferences correctly ordered on the validation split: %s",
            duration,
            f"{accuracy:.3f}" if accuracy is not None else "not measured",
        )

        return DPOResult(
            output_dir=str(output_dir),
            train_loss=float(result.training_loss),
            eval_reward_accuracy=float(accuracy) if accuracy is not None else None,
            eval_reward_margin=float(margin) if margin is not None else None,
            duration_s=round(duration, 1),
            peak_gpu_memory_gb=round(memory, 2),
        )
