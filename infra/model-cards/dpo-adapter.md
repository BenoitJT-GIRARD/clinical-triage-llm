---
license: mit
language:
  - fr
  - en
base_model: BenoitJT-GIRARD/qwen3-1.7b-clinical-triage-sft-merged
library_name: peft
tags:
  - medical
  - triage
  - emergency-department
  - lora
  - dpo
---

# Emergency triage assistant — LoRA adapter (preference alignment)

A LoRA adapter from the **preference alignment (DPO)** of the project's triage model. It applies
on top of the **merged supervised model**, not on `Qwen3-1.7B-Base`: the alignment was trained
above the first, and applying it to the second would compose weights that do not go together.

For direct use, prefer the
[final merged model](https://huggingface.co/BenoitJT-GIRARD/qwen3-1.7b-clinical-triage), which
already combines both stages.

> **Teaching prototype.** What this adapter corrects, it corrects on synthetic data. The
> presentations it learnt from **were never reviewed by a physician**, and a real triage desk is
> not a place for it. Call the emergency services on any life-threatening sign.

## What the alignment learns

Between two answers to the same case, prefer the one that:

- does not **understate** the level of urgency;
- does not suggest a course of action that **delays** care;
- does not **assert** a diagnosis, which the system prompt forbids;
- keeps to the **language** it was asked for.

The preference pairs share their format and are of comparable length. Without that precaution the
alignment learns length rather than substance, and the model stops emitting its end-of-turn
token.

What it does **not** learn is worth stating here rather than in a report: on an external
preference set it never saw, this adapter orders the pairs no better than the supervised model
does, and both are below chance. The benefit measured on the clinical cases is real; it does not
generalise beyond the defects the pairs were built from.

## Use

This adapter applies on the **merged supervised model**, not on the base model: it was trained
above it, and applying it directly would compose weights that do not go together. That
intermediate model is published for this precise reason, and the adapter's configuration names
it, so nothing has to be rebuilt.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "BenoitJT-GIRARD/qwen3-1.7b-clinical-triage-sft-merged", device_map="auto"
)
model = PeftModel.from_pretrained(base, "BenoitJT-GIRARD/qwen3-1.7b-clinical-triage-dpo")
tokenizer = AutoTokenizer.from_pretrained("BenoitJT-GIRARD/qwen3-1.7b-clinical-triage-dpo")
```

**To serve the model, prefer the [final merged
model](https://huggingface.co/BenoitJT-GIRARD/qwen3-1.7b-clinical-triage)**: it needs no
rebuilding, and it is one reference to pin, where this adapter asks the server to host the merged
supervised model as well and to pair the two.

## Evaluation

Sixty cases **composed for the evaluation**, unseen in training. Nearly half of them are
atypical on purpose, because that is where a triage system earns its place or loses it.

{{EVALUATION}}

What the alignment moved and what it did not: the protocol page has the paired comparison.

## Data, evaluation and limits

Read them on the card of the
[final model](https://huggingface.co/BenoitJT-GIRARD/qwen3-1.7b-clinical-triage), or in the
repository itself: <https://github.com/BenoitJT-GIRARD/clinical-triage>

## Licence

MIT.
