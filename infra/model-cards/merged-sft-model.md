---
license: mit
language:
  - fr
  - en
base_model: Qwen/Qwen3-1.7B-Base
library_name: transformers
pipeline_tag: text-generation
tags:
  - medical
  - triage
  - emergency-department
  - lora
---

# Emergency triage assistant — merged supervised model

`Qwen3-1.7B-Base` after **supervised fine-tuning** on the project's triage corpus, with the LoRA
adapter merged into the weights. This is the project's **intermediate** model: it triages and
respects the output format, but it has not yet been aligned on preferences.

For direct use, prefer the
[final model](https://huggingface.co/BenoitJT-GIRARD/qwen3-1.7b-clinical-triage), which carries
that alignment.

This repository exists for two precise reasons:

1. **it is the base model of the DPO adapter.** The alignment was trained on top of these
   weights; without them,
   [the DPO adapter](https://huggingface.co/BenoitJT-GIRARD/qwen3-1.7b-clinical-triage-dpo)
   cannot be loaded at all;
2. **it makes the alignment measurable**, by comparing the two models on the same evaluation set.

> **Teaching prototype.** An intermediate artefact of a teaching project, published so that
> the adapter above it can load. Its levels come from a catalogue **no emergency physician has
> read**, and it has no place in a real department. Call the emergency services on any
> life-threatening sign.

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repository = "BenoitJT-GIRARD/qwen3-1.7b-clinical-triage-sft-merged"
model = AutoModelForCausalLM.from_pretrained(repository, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(repository)
```

The tokenizer in this repository is the project's own: it carries the dialogue template the
model was trained on, and `<|im_end|>` is declared as its stop token.

The output head was **untied** from the embedding matrix before the merge. Base and head share
one tensor in the original model; since the fine-tuning changes the head, a merge that left them
tied would have applied the correction to what the model reads as well as to what it writes. This
repository therefore carries an `lm_head.weight` of its own, and `tie_word_embeddings` set to
`false`.

## Output format

```
Niveau de priorité : URGENCE_VITALE | URGENCE_MODEREE | CONSULTATION_DIFFEREE
Justification : <short clinical explanation>
Recommandation : <what to do next>
```

## Evaluation

The same sixty cases as every other card here: **written one by one**, held out of every
training set, and deliberately weighted towards presentations that read as the opposite of what
they are.

{{EVALUATION}}

Where the errors land, and what separates this stage from the aligned one: see the protocol
page.

## Data, evaluation and limits

Both are written out on the
[final model](https://huggingface.co/BenoitJT-GIRARD/qwen3-1.7b-clinical-triage)'s card. Code and
pipeline: <https://github.com/BenoitJT-GIRARD/clinical-triage>

## Licence

MIT.
