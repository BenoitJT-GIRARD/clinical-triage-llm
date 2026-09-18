---
license: mit
language:
  - fr
  - en
base_model: Qwen/Qwen3-1.7B-Base
library_name: peft
tags:
  - medical
  - triage
  - emergency-department
  - lora
---

# Emergency triage assistant — LoRA adapter (supervised fine-tuning)

A LoRA adapter from the **supervised fine-tuning** of `Qwen3-1.7B-Base` on the project's triage
corpus. It holds no model: it applies on top of the base model.

For direct use, prefer the
[final merged model](https://huggingface.co/BenoitJT-GIRARD/qwen3-1.7b-clinical-triage), which
carries the preference alignment as well.

> **Teaching prototype.** This adapter proposes a triage level; a clinician decides it. The
> catalogue its training data comes from **was written by a developer and reviewed by no
> physician**, so nothing here belongs anywhere near a patient. Call the emergency services on
> any life-threatening sign.

## Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B-Base", device_map="auto")
model = PeftModel.from_pretrained(base, "BenoitJT-GIRARD/qwen3-1.7b-clinical-triage-sft")
tokenizer = AutoTokenizer.from_pretrained("BenoitJT-GIRARD/qwen3-1.7b-clinical-triage-sft")
```

The tokenizer published here carries the **project's dialogue template** and declares
`<|im_end|>` as the end-of-sequence token. Loading the base model's tokenizer instead would
generate without ever stopping: the format the adapter learnt is not the original one.

## Output format

```
Niveau de priorité : URGENCE_VITALE | URGENCE_MODEREE | CONSULTATION_DIFFEREE
Justification : <short clinical explanation>
Recommandation : <what to do next>
```

## Evaluation

Sixty **hand-written** cases the model never met, forty of them urgent, with a good half built
to mislead a reader who goes by keywords.

{{EVALUATION}}

How it stands against the explicit rule and against an ordinary classifier: the repository's
protocol page.

## Data, evaluation and limits

The [final model](https://huggingface.co/BenoitJT-GIRARD/qwen3-1.7b-clinical-triage) carries
the full account, and the code lives at <https://github.com/BenoitJT-GIRARD/clinical-triage>

## Licence

MIT.
