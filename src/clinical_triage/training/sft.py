"""Supervised fine-tuning of Qwen3-1.7B-Base with Unsloth and LoRA.

The base model cannot hold a dialogue: it completes text. SFT teaches it two things at once —
the agent's ChatML format, and the triage task itself.

Four points deserve stating, because they condition everything after:

1. **the dataset exposes ``prompt`` and ``completion``**, not a ``messages`` column. That keeps
   the training library from re-serialising the examples with the model's native chat template —
   Qwen3's inserts a ``<think></think>`` block before the answer, and the model would then learn
   a format inference never reproduces. The prompt/completion pair also makes it possible to
   compute the loss on the answer only, with no hand-written collator.
2. **the end-of-sequence token is fixed** before training: the Qwen3-1.7B-Base tokenizer ships
   with ``<|endoftext|>`` while the learnt format ends with ``<|im_end|>``.
3. **the output head is trained with the projections.** That is the least obvious and most
   decisive correction of the project — see below.
4. **LoRA trains only the adapter matrices** of the projections — a few million parameters — to
   which the output head is added, trained in full by the previous point. It alone weighs nearly
   311 million parameters: the total trained is therefore around 14% of the model, not the one
   percent LoRA alone would suggest. That is the price of the end-token fix, and the run summary
   publishes the exact count.

## Why the output head must be trained

Qwen3-1.7B-**Base** carries the twenty-five ChatML tokens (151644 to 151668) as one and the same
vector, never trained: their norms are identical bit for bit, and ``<|im_start|>`` and
``<|im_end|>`` have a cosine similarity of 1.000. The model also ties its output head to its
embedding matrix (``tie_word_embeddings``), which LoRA freezes.

Declaring ``<|im_end|>`` as end of sequence is therefore not enough: its output row stays frozen
on an initialisation value shared with twenty-four other tokens, and the model **structurally
cannot** emit it in preference to them. Measured: 0% clean stops, and the whole 220-token budget
consumed on every answer. No amount of extra epochs changes anything.

Adding ``lm_head`` to the adapted modules lifts the lock. Three configurations were compared
under an identical protocol — 75 steps, 60 validation cases:

| Configuration | Accuracy | Clean stops | Tokens generated |
|---|---|---|---|
| projections only | 0.850 | 0.000 | 220 |
| + embeddings (`modules_to_save`) | 0.850 | 0.000 | 220 |
| **+ output head** | **0.900** | **1.000** | **97** |

The exact figures live in ``reports/training/module_comparison.json``; these repeat them so that
the argument can be followed without leaving the file.

Training the embeddings alone achieves nothing: PEFT copies the module and breaks the link with
the output head, which stays frozen. It is indeed the head that must be adapted.

Unsloth does this natively: it moves ``lm_head`` into ``modules_to_save`` and handles untying the
weights. Its ``fix_untrained_tokens`` patch, on the other hand, only detects embedding rows that
are exactly zero: on this model, whose ChatML tokens are duplicated vectors of small norm, it has
no effect.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from clinical_triage.config import MODEL, SEED, TRAINING
from clinical_triage.prompts import IM_END, prepare_tokenizer
from clinical_triage.training.schedule import warmup_steps
from clinical_triage.training.tracking import track
from clinical_triage.utils import get_logger, path_for_log

logger = get_logger("sft")


@dataclass
class SFTResult:
    """The result of one supervised training run."""

    output_dir: str
    trainable_params: int
    total_params: int
    train_loss: float | None
    eval_loss: float | None
    duration_s: float
    peak_gpu_memory_gb: float


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
    """Run the supervised fine-tuning and return the run summary."""
    # Unsloth rewrites internal functions of `transformers` and `trl` when it is imported. It
    # must therefore be imported BEFORE them: imported after, its patches apply to references
    # that have already been copied and have no effect at all — with no message to say so.
    # The order below is functional: sorting it alphabetically would put `unsloth` last and
    # neutralise its patches, hence the waiver.
    from unsloth import FastLanguageModel  # noqa: I001

    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    output_dir.mkdir(parents=True, exist_ok=True)

    # The output head joins the adapted projections: without it the model cannot learn to stop
    # (see the module docstring).
    adapted_modules = [*TRAINING.lora_target_modules, "lm_head"]

    hyperparameters = {
        "base_model": MODEL.base_model,
        "engine": "unsloth",
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": TRAINING.lora_dropout,
        "lora_modules": ", ".join(adapted_modules),
        "max_length": max_length,
        "gradient_checkpointing": gradient_checkpointing,
        "max_steps": max_steps,
        "seed": SEED,
    }

    with track(run_name, hyperparameters) as summary:
        logger.info("Loading the tokenizer and the base model: %s", MODEL.base_model)
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
            target_modules=adapted_modules,
            bias="none",
            # Unsloth's checkpointing offloads activations to main memory: same computation, a
            # markedly smaller GPU footprint.
            use_gradient_checkpointing="unsloth" if gradient_checkpointing else False,
            random_state=SEED,
        )

        data_files = {"train": str(train_path)}
        if eval_path is not None:
            data_files["eval"] = str(eval_path)
        dataset = load_dataset("json", data_files=data_files)
        # Training needs these two columns only; the clinical metadata of the dataset must not
        # reach the collator.
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
            # Chunked loss saves memory by splitting the computation over the vocabulary, but it
            # assumes an untouched output head. It is therefore incompatible with adapting
            # `lm_head`.
            loss_type="nll",
            report_to=[],
        )

        # The model already carries its adapter: no `peft_config` here, without which TRL would
        # graft a second one on top of Unsloth's.
        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("eval"),
            processing_class=tokenizer,
        )

        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in trainer.model.parameters())
        logger.info(
            "Trainable parameters: %s out of %s (%.2f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100 * trainable / total,
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
        eval_loss = next((e["eval_loss"] for e in reversed(history) if "eval_loss" in e), None)
        summary.history = history
        summary.metrics = {
            "train_loss": float(result.training_loss),
            "eval_loss": float(eval_loss) if eval_loss is not None else None,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_share_pct": round(100 * trainable / total, 3),
            "duration_s": round(duration, 1),
            "peak_gpu_memory_gb": round(memory, 2),
            "examples_per_second": round(result.metrics.get("train_samples_per_second", 0.0), 2),
        }
        logger.info(
            "SFT finished in %.0f s. Adapter saved to %s", duration, path_for_log(output_dir)
        )

        return SFTResult(
            output_dir=str(output_dir),
            trainable_params=trainable,
            total_params=total,
            train_loss=float(result.training_loss),
            eval_loss=float(eval_loss) if eval_loss is not None else None,
            duration_s=round(duration, 1),
            peak_gpu_memory_gb=round(memory, 2),
        )
