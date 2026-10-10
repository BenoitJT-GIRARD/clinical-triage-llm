---
license: apache-2.0
language:
  - fr
  - en
base_model: Qwen/Qwen3-1.7B-Base
base_model_relation: finetune
datasets:
  - BenoitJT-GIRARD/clinical-triage-bilingual
library_name: transformers
pipeline_tag: text-generation
tags:
  - medical
  - triage
  - emergency-department
  - lora
  - dpo
model-index:
  - name: qwen3-1.7b-clinical-triage
    results:
      - task:
          type: text-generation
          name: Emergency triage level
        dataset:
          name: Clinical evaluation set
          type: BenoitJT-GIRARD/clinical-triage-bilingual
          config: clinical_eval
          split: test
          revision: model-v2.0.0
        metrics:
          - type: accuracy
            name: Triage level accuracy
            value: 0.7167
          - type: undertriage
            name: Urgent cases sent to a lower level
            value: 0.275
          - type: format_compliance
            name: Answers the expected format can parse
            value: 1.0
---

# Emergency triage, fine-tuned then preference-aligned

`Qwen3-1.7B-Base`, fine-tuned with LoRA to read a patient description and answer with a triage
level, a justification and a course of action, then aligned with DPO on safety preferences. Both
steps are merged into these weights: the model loads like any causal language model.

## Intended use

This model exists to evaluate a method on a corpus: what supervised fine-tuning, then preference
alignment, change in a small model's triage of written cases, against an explicit rule and an
ordinary classifier. It is not a medical device, gives no clinical advice, and must not inform a
decision about a patient. The clinical catalogue behind its training data was not validated by an
emergency physician.

## How to use

The chat template and the end-of-turn token travel with the tokenizer: the caller configures neither.
The model expects the system prompt it was trained with, word for word:

```text
Tu es l'assistant de triage médical du service des urgences. Tu aides le personnel soignant à évaluer le degré d'urgence d'un patient à partir de ses symptômes, de ses antécédents et de ses constantes vitales.

Tu réponds TOUJOURS en français et selon le format suivant :
Niveau de priorité : <URGENCE_VITALE | URGENCE_MODEREE | CONSULTATION_DIFFEREE>
Justification : <explication clinique courte et claire>
Recommandation : <conduite à tenir pour le patient et le personnel>

Règles de sécurité :
- En cas de doute ou de signe de gravité, surclasse le niveau d'urgence.
- Ne pose jamais de diagnostic définitif : tu fournis une aide à la décision.
- Rappelle de contacter le 15 (SAMU) devant tout signe vital engagé.
```

It always answers in French, in three lines:

```text
Niveau de priorité : URGENCE_VITALE | URGENCE_MODEREE | CONSULTATION_DIFFEREE
Justification : <short clinical explanation>
Recommandation : <what to do next>
```

With `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

name, revision = "BenoitJT-GIRARD/qwen3-1.7b-clinical-triage", "model-v2.0.0"
tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
model = AutoModelForCausalLM.from_pretrained(name, revision=revision, device_map="auto")

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},  # the prompt above
    {"role": "user", "content": "Homme de 67 ans, douleur thoracique et sueurs depuis 20 minutes."},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)
answer = model.generate(inputs, max_new_tokens=220, do_sample=False)
print(tokenizer.decode(answer[0, inputs.shape[1] :], skip_special_tokens=True))
```

With vLLM:

```bash
vllm serve BenoitJT-GIRARD/qwen3-1.7b-clinical-triage --revision model-v2.0.0 --max-model-len 1024
```

The revision pins the weights this card describes; the default branch moves with every
publication.

## Training

**Data.** The supervised set of [`BenoitJT-GIRARD/clinical-triage-bilingual`](https://huggingface.co/datasets/BenoitJT-GIRARD/clinical-triage-bilingual):
4,000 training and 500 validation
examples, balanced over the three levels and the two languages, then
2,400 preference pairs. No real patient data.

**Supervised step.** LoRA of rank 32, alpha
64, learning rate 0.0002, over
2 epochs, on every projection **and the output head**. The base model
carries its chat tokens as one untrained vector and ties its output head to its embeddings: an
adapter on the projections alone produces a model that never emits its end-of-turn token. The
exported weights therefore untie the head (`tie_word_embeddings: false`).

**Preference step.** DPO on top of the merged supervised model, which is also its reference, with a
supervised term on the preferred answer that keeps it likely. The rejected answers carry one
defect each: a level too low, an unsafe course of action, an asserted diagnosis, or English.

**Hardware.** One NVIDIA GeForce RTX 4060 Ti: 30 min for the supervised
step, 63 min for the preference step, through Unsloth.

## Evaluation

<!-- source: reports/evaluation_results.json -->
60 cases written one by one for the evaluation, none seen in training,
10 of them falsely reassuring and
11 falsely alarming.

| System | Accuracy | Urgent cases sent lower | Format compliance |
|---|---|---|---|
| **This model** | 0.717 [0.59 – 0.81] | 11 of 40 | 100% |
| Supervised step alone | 0.700 [0.57 – 0.80] | 12 of 40 | 100% |
| Keyword rule | 0.617 [0.49 – 0.73] | 22 of 40 | n/a |
| Linear classifier, same examples | 0.667 [0.54 – 0.77] | 8 of 40 | n/a |
> n = 60 cases, of which n = 40 urgent. The accuracy
> intervals are 95% Wilson intervals.

> **How to read it.** One row per system, all on the same cases. Accuracy is the share filed
> at the reference level written for each case. The urgent cases sent lower are the failure
> that reaches a patient, so they are counted on the urgent cases alone. Format compliance is
> the share of answers the three-line format can parse; the rule and the classifier return a
> level, not a text, so it does not apply to them (n/a).

Every comparison below is paired on the same cases, by an exact McNemar test, and was fixed
before the results. The first row is the primary analysis. "More urgent of the two" keeps, case
by case, the more urgent of this model's level and the rule's, which every reply of the service
carries. Each row counts the cases the first system handles and the second does not, and the
reverse: on the urgent cases, handled means not sent lower; on all cases, filed at the
reference level.

<!-- source: reports/evaluation_results.json -->
| On the 40 urgent cases | First only | Second only | Test |
|---|---|---|---|
| This model against the rule | 17 | 6 | p = 0.035 |
| This model against the classifier | 3 | 6 | p = 0.508 |
| Aligned against supervised | 1 | 0 | p = 1.000 |
| More urgent of the two against this model alone | 6 | 0 | p = 0.031 |
> n = 40 urgent cases, the same in every row.

<!-- source: reports/evaluation_results.json -->
| On all 60 cases, accuracy | First only | Second only | Test |
|---|---|---|---|
| This model against the rule | 16 | 10 | p = 0.327 |
| This model against the classifier | 10 | 7 | p = 0.629 |
| Aligned against supervised | 2 | 1 | p = 1.000 |
> n = 60 cases, the same in every row.

<!-- source: reports/seed_study.json -->
**Seeds.** The method was trained three times, on seeds 42, 43 and 44; these weights are seed 42,
chosen before any result. Across the three, accuracy on the clinical set ranges from
0.700 to
0.783, and the urgent cases missed from
9 to
13. Training alone moves the result
that much, so a gap of that size between two systems trained once each can be the seed.

<!-- source: reports/before_after.json -->
**Against the previous version.** The model this repository served before, version 1.0.0,
evaluated by the same code on the same cases, scores 0.717 and misses
10 urgent cases where this one misses
11:
1 caught by this version alone,
2 by the previous one alone,
p = 1.000. The retraining did not make a better triage decision.
It corrected the data the model learns from and the prompt it was trained with, and measured the
result on three seeds.

<!-- source: reports/evaluation_results.json -->
**Alignment.** On 100 preference pairs of the task, held out of
training, the aligned model ranks the preferred answer first in
84% of pairs. On
777 pairs from a medical preference set built outside the project, the share is
36.5%, and in
50% of the
180 whose two answers are of similar length.
It learnt the defects of the task, not medicine at large, and its score below chance comes from
answer length, which it penalises.

**Robustness.** On 41 inputs outside the nominal case (a
three-letter entry, a two-page paste, a description in another language, an attempt to make it
say something else), the model keeps its contract in
85.4% of cases: the expected format, in
French, its system prompt kept to itself, a clean stop, the identifying data it was given not
echoed, and the level an input announces when it announces one.

**The weights evaluated are these.** The figures above were measured on files whose SHA-256 the
evaluation recorded, and the publication refuses any other:

| File | SHA-256 |
|---|---|
| `model.safetensors` | `a0405c89c8a8a4dc78b04b1b3850e3a4b7c00f0a18dbab171d88c39693f59b4b` |

## Limitations

- The clinical catalogue was not validated by an emergency physician.
- The catalogue, the triage rule and the evaluation set have the same author: there is no
  external ground truth.
- 60 evaluation cases: an interval of accuracy spans about twenty
  points, and most differences between systems are not significant. On
  40 urgent cases, even a model that missed none
  would keep an upper bound of 8.8%
  on its undertriage; it takes
  72 to bring it under
  5%.
- The training data is mostly generated: it lacks the disorder of real language.
- 1.7 billion parameters learn a format and three classes, not clinical reasoning.

## Licence

Apache-2.0, the licence of the base model `Qwen/Qwen3-1.7B-Base`. The dataset is published under
CC BY 4.0, the code under MIT.

## Reproducing

The code, the settings and the pipeline, at the version of these weights:
<https://github.com/BenoitJT-GIRARD/clinical-triage-llm/tree/v2.0.0>. The adapters and the intermediate merged model
are not published: the pipeline rebuilds them from the dataset, at the settings these weights
were trained with.
