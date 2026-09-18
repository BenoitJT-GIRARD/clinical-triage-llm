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

Usage::

    uv run python scripts/merge_adapter.py --adapter sft
    uv run python scripts/merge_adapter.py --adapter dpo
"""

from __future__ import annotations

from clinical_triage.bootstrap import use_utf8_console

use_utf8_console()

import argparse

from clinical_triage.config import MODEL, PATHS
from clinical_triage.prompts import prepare_tokenizer
from clinical_triage.utils import get_logger

logger = get_logger("merge")


def _check_export(destination) -> None:
    """Refuse an export whose output head did not survive the untying.

    Two ways the export can be wrong without raising anything: the configuration re-enables the
    tie between embeddings and output head, or the weight file contains no distinct
    ``lm_head.weight`` tensor. In both cases the model served is not the one that was trained.
    """
    import json

    configuration = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    if configuration.get("tie_word_embeddings"):
        raise SystemExit(
            f"{destination}: the weights are still tied, the merged output head is lost."
        )

    index = destination / "model.safetensors.index.json"
    if index.exists():
        keys = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    else:
        from safetensors import safe_open

        with safe_open(str(destination / "model.safetensors"), framework="pt") as handle:
            keys = set(handle.keys())
    if "lm_head.weight" not in keys:
        raise SystemExit(f"{destination}: `lm_head.weight` absent from the exported weights.")
    logger.info("Export checked: output head untied and present in the weights.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=["sft", "dpo"], required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.adapter == "sft":
        base = MODEL.base_model
        adapter = PATHS.sft_adapter
        destination = PATHS.sft_merged
    else:
        base = str(PATHS.sft_merged)
        adapter = PATHS.dpo_adapter
        destination = PATHS.dpo_merged

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
    _check_export(destination)
    logger.info("Merged model saved to %s", destination)


if __name__ == "__main__":
    main()
