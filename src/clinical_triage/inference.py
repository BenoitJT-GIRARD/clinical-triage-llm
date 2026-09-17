"""Triage inference: loading the model and generating.

Shared by the evaluation and by the service in demonstration mode. Production deployment goes
through vLLM (see ``infra/``), but both paths produce exactly the same prompt and the same
answer: the same ChatML template, the same end-of-sequence token and the same safety
truncation.

That truncation deserves a word. A 1.7-billion-parameter model sometimes forgets to emit its
end token and keeps writing. Without a net, the text returned to the nurse and written to the
audit log then contains the rest of the generation, system prompt included.
:func:`truncate_to_answer` cuts after the recommendation line, whatever happens.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from clinical_triage.config import MODEL, SERVING
from clinical_triage.prompts import (
    bound_description,
    build_messages,
    description_budget,
    extract_level,
    format_chatml,
    prepare_tokenizer,
    truncate_to_answer,
)
from clinical_triage.utils import get_logger, path_for_log

logger = get_logger("inference")


@dataclass
class TriageResponse:
    """The result of one triage inference."""

    text: str
    level: str | None
    latency_ms: float
    generated_tokens: int
    clean_stop: bool  # the model emitted its end token before the cap
    # The description exceeded the model's window and was cut. This flag travels all the way to
    # the returned answer: the nurse alone knows whether what is missing mattered, and silently
    # truncating a clinical narrative is exactly what a decision-support system must not do.
    description_truncated: bool = False


def _check_gpu_placement(model) -> None:
    """Check that Unsloth really put the whole model on the GPU.

    When video memory runs short, loading silently spreads part of the layers onto the CPU.
    Unsloth memorises the location of each layer at that moment, and a layer left on the CPU
    gets an empty device index. The error only surfaces at the first generation, as an
    "Invalid target device: None" that says nothing about its cause. Better to say it here,
    before the loading time has been spent, and with what to do about it.
    """
    locations = {parameter.device.type for parameter in model.parameters()}
    if locations <= {"cuda"}:
        return
    raise RuntimeError(
        "Not enough GPU memory: the model was loaded partly on "
        f"{', '.join(sorted(locations - {'cuda'}))}. Free the card — one GPU job at a time — "
        "then run again."
    )


class TriageAgent:
    """Triage agent: wraps the model, the tokenizer and generation."""

    def __init__(
        self,
        adapter_dir: str | Path | None = None,
        base_model: str = MODEL.base_model,
        device: str | None = None,
    ):
        # Unsloth replaces the `forward` of Qwen3's attention layers on import, and its version
        # expects attributes it only sets on models it loaded itself. A model loaded by bare
        # `transformers` in the same process would therefore hit a `forward` that cannot handle
        # it. So Unsloth loads whenever it is present, and `transformers` takes over where it is
        # not — the service image, which carries neither torch nor Unsloth.
        import importlib.util

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with_unsloth = importlib.util.find_spec("unsloth") is not None and torch.cuda.is_available()

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        logger.info(
            "Loading model %s on %s (%s)",
            path_for_log(base_model),
            self.device,
            "unsloth" if with_unsloth else "transformers",
        )

        if with_unsloth:
            from unsloth import FastLanguageModel

            model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(base_model),
                max_seq_length=MODEL.max_seq_length,
                dtype=dtype,
                load_in_4bit=False,
            )
            _check_gpu_placement(model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(str(base_model))
            model = AutoModelForCausalLM.from_pretrained(str(base_model), dtype=dtype)

        self.end_token_id = prepare_tokenizer(self.tokenizer)
        # Padding goes on the left: in batched generation every answer must start right after
        # the last token of its own prompt.
        self.tokenizer.padding_side = "left"

        if adapter_dir is not None:
            from peft import PeftModel

            logger.info("Applying the LoRA adapter: %s", path_for_log(adapter_dir))
            model = PeftModel.from_pretrained(model, str(adapter_dir))

        if with_unsloth:
            from unsloth import FastLanguageModel

            FastLanguageModel.for_inference(model)

        self.model = model.to(self.device).eval()
        self.description = (
            f"{Path(str(base_model)).name}+{Path(str(adapter_dir)).name}"
            if adapter_dir
            else Path(str(base_model)).name
        )

    def _prompts(self, symptoms: list[str]) -> tuple[list[str], list[bool]]:
        """Build the ChatML prompts, bounding descriptions that are too long.

        A description that overflows the model's window does not produce a degraded answer
        there: it interrupts generation with a dimension error. It is bounded here, and each
        case says whether it was.
        """
        budget = description_budget(self.tokenizer, MODEL.max_seq_length, SERVING.max_new_tokens)
        bounded = [bound_description(s, self.tokenizer, budget) for s in symptoms]
        prompts = [
            format_chatml(build_messages(text), add_generation_prompt=True) for text, _ in bounded
        ]
        return prompts, [cut for _, cut in bounded]

    def generate_batch(
        self,
        symptoms: list[str],
        max_new_tokens: int = SERVING.max_new_tokens,
        temperature: float = SERVING.temperature,
    ) -> list[TriageResponse]:
        """Generate the triage answers of a batch of descriptions.

        Batching exists only for the evaluation: it divides by several the time needed to pass
        the whole clinical set. The latency returned is then the mean latency per case of the
        batch, which ``latency.py`` states explicitly to avoid any confusion with the latency a
        user perceives.
        """
        prompts, cuts = self._prompts(symptoms)
        if any(cuts):
            logger.warning(
                "%d description(s) out of %d exceeded the model window and were bounded.",
                sum(cuts),
                len(cuts),
            )
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        prompt_length = inputs["input_ids"].shape[1]

        started = time.perf_counter()
        with self._torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.end_token_id,
            )
        total_latency = (time.perf_counter() - started) * 1000
        latency_per_case = total_latency / max(1, len(symptoms))

        answers: list[TriageResponse] = []
        for row, cut in zip(output, cuts, strict=True):
            new_tokens = row[prompt_length:].tolist()
            clean_stop = self.end_token_id in new_tokens
            if clean_stop:
                new_tokens = new_tokens[: new_tokens.index(self.end_token_id)]
            text = truncate_to_answer(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
            answers.append(
                TriageResponse(
                    text=text,
                    level=extract_level(text),
                    latency_ms=latency_per_case,
                    generated_tokens=len(new_tokens),
                    clean_stop=clean_stop,
                    description_truncated=cut,
                )
            )
        return answers

    def generate(
        self,
        symptoms: str,
        max_new_tokens: int = SERVING.max_new_tokens,
        temperature: float = SERVING.temperature,
    ) -> TriageResponse:
        """Generate the triage answer of a single description."""
        return self.generate_batch([symptoms], max_new_tokens, temperature)[0]
