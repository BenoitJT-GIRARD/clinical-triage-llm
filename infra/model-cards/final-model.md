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
  - dpo
---

# Emergency triage assistant — final model

`Qwen3-1.7B-Base`, specialised for emergency triage by supervised fine-tuning with LoRA and then
aligned on preferences with DPO. Both adapters are **merged**: the model loads like any causal
model, with no adapter library.

> **Teaching prototype.** Decision support for clinical staff, under mandatory human
> supervision. The catalogue of clinical presentations behind the training data **has not been
> validated by an emergency physician**: this model must not be used in a real setting. On any
> life-threatening sign, call the emergency services.

## What the model does

From a patient description, complaint, symptoms, history and vital signs taken at the desk, in
French or in English, it produces an answer in French, always shaped like this:

```
Niveau de priorité : URGENCE_VITALE | URGENCE_MODEREE | CONSULTATION_DIFFEREE
Justification : <short clinical explanation>
Recommandation : <what to do next>
```

| Level | Care within | FRENCH scale |
|---|---|---|
| `URGENCE_VITALE` | immediately | sorts 1 and 2 |
| `URGENCE_MODEREE` | a few hours | sorts 3 and 4 |
| `CONSULTATION_DIFFEREE` | a scheduled consultation | sort 5 |

## Use

The dialogue template and the end-of-sequence token are written into the exported tokenizer:
the caller has nothing to configure.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "BenoitJT-GIRARD/qwen3-1.7b-clinical-triage"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Homme de 67 ans, douleur thoracique et sueurs depuis 20 minutes."},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

The exact system prompt is published in the dataset card and in its `metadata.json`. Served by
vLLM:

```bash
vllm serve BenoitJT-GIRARD/qwen3-1.7b-clinical-triage \
  --revision {{REVISION}} \
  --served-model-name qwen3-1.7b-clinical-triage --max-model-len 1024
```

`--revision` pins a citable version: the default branch moves at every publication, the tag does
not.

### One thing worth knowing

The base model carries its ChatML tokens as a **single untrained vector**, and ties its output
head to its embeddings. A LoRA fine-tuning that adapts the projections alone therefore produces
a model unable to emit its end-of-turn token: it generates to the cap, every time. This model was
trained **with its output head**, and its weights are exported with that tie undone
(`tie_word_embeddings: false`). It stops on its own.

Anyone starting again from the base model will meet the same trap. It is described, with the
measurements, in the project's protocol page.

## Training data

Dataset [`BenoitJT-GIRARD/clinical-triage-bilingual`](https://huggingface.co/datasets/BenoitJT-GIRARD/clinical-triage-bilingual):
a corpus balanced over three triage levels and two languages, built from a catalogue of clinical
presentations written for the project and completed with filtered cases from **MediQAl** (French
clinical vignettes), MedQuAD and MedMCQA. **No real patient data.**

## Evaluation

Sixty cases **written one by one** for this evaluation, none of them seen in training, nearly
half built to mislead. Four baselines stand beside the model, and the two that matter are the
keyword rule a department could deploy tomorrow and a linear classifier fitted on the very same
examples.

{{EVALUATION}}

The failure modes, subgroup by subgroup, and the paired tests behind each comparison: the
repository's protocol page.

## Limits

- **The clinical catalogue was not validated by an emergency physician** — the main limit.
- The training data is mostly synthetic: it lacks the disorder of real language, and the
  generated vignettes share a small number of expected answers.
- An evaluation set of a few dozen cases: at that effective, no difference in accuracy between
  two systems is conclusive.
- No external ground truth: the catalogue, the triage rule and the evaluation set have the same
  author.
- 1.7 billion parameters: a format and three classes, not clinical reasoning.

## Reproducing it

Code, hyper-parameters, seed and the whole pipeline:
<https://github.com/BenoitJT-GIRARD/clinical-triage>

## Licence

MIT, like the base model `Qwen/Qwen3-1.7B-Base`.
