"""Merge a LoRA adapter into its base model.

Two uses:

- ``--adapter sft`` merges the SFT adapter into Qwen3-1.7B-Base and produces
  ``var/models/qwen3-1.7b-triage-sft-merged``. That model is both the base **and** the reference
  of the DPO alignment;
- ``--adapter dpo`` merges the DPO adapter into the merged SFT model and produces
  ``var/models/qwen3-1.7b-triage-dpo-merged``, the final model served by vLLM.

The tokenizer exported with the model carries the project's chat template and ``<|im_end|>`` as
end-of-sequence token. That is what makes the inference server stop in the right place with no
extra setting: the generation configuration is written into the model, not into the caller.

An export that fails the check of :mod:`clinical_triage.training.export` is deleted, not left
beside the delivered models for something to pick up.

Usage::

    uv run python scripts/merge_adapter.py --adapter sft
    uv run python scripts/merge_adapter.py --adapter dpo --seed 43
"""

from __future__ import annotations

from clinical_triage.bootstrap import use_utf8_console

use_utf8_console()

import argparse
import shutil
from pathlib import Path

from clinical_triage.config import MODEL, PATHS, SEED
from clinical_triage.prompts import prepare_tokenizer
from clinical_triage.training.export import export_problems
from clinical_triage.utils import get_logger, path_for_log

logger = get_logger("merge")


def keep_or_delete(destination: Path) -> None:
    """Keep a checked export, or delete it and stop."""
    problems = export_problems(destination)
    if problems:
        shutil.rmtree(destination, ignore_errors=True)
        raise SystemExit("Export refused and deleted: " + "; ".join(problems))
    logger.info("Export checked: output head untied and present in the weights.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=["sft", "dpo"], required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    paths = PATHS.for_seed(args.seed)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.adapter == "sft":
        base = MODEL.base_model
        adapter = paths.sft_adapter
        destination = paths.sft_merged
    else:
        base = str(paths.sft_merged)
        adapter = paths.dpo_adapter
        destination = paths.dpo_merged

    if not adapter.exists():
        raise SystemExit(f"Adapter not found: {adapter}")

    destination.mkdir(parents=True, exist_ok=True)

    # Merging in float32, not in bfloat16. A LoRA delta is small against the weights it
    # corrects; added on eight bits of mantissa, part of it disappears outright in the rounding.
    # So the merge happens in single precision, and bfloat16 comes back only at write time.
    logger.info("Loading the base model in float32: %s", base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)

    if getattr(model.config, "tie_word_embeddings", False):
        # The base model shares a single tensor between its embedding matrix and its output
        # head. Our adapter corrects the head: merging without untying would write the delta
        # into the shared tensor, hence into the model's input as much as into its output, and
        # would corrupt it silently. The head is given its own copy before the merge.
        logger.info("Tied weights detected: untying the output head before merging.")
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        model.config.tie_word_embeddings = False

    logger.info("Merging adapter %s", adapter)
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    merged = merged.to(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter))
    end_token = prepare_tokenizer(tokenizer)
    # The generation configuration travels with the model: any inference engine will stop on the
    # right token with no setting from the caller.
    merged.generation_config.eos_token_id = end_token
    merged.generation_config.pad_token_id = end_token
    merged.config.eos_token_id = end_token

    merged.save_pretrained(str(destination), safe_serialization=True)
    tokenizer.save_pretrained(str(destination))
    keep_or_delete(destination)
    logger.info("Merged model saved to %s", path_for_log(destination))


if __name__ == "__main__":
    main()
