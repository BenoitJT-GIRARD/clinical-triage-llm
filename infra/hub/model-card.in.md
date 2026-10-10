---
license: apache-2.0
language:
  - fr
  - en
base_model: Qwen/Qwen3-1.7B-Base
base_model_relation: finetune
datasets:
  - {{release:dataset_id}}
library_name: transformers
pipeline_tag: text-generation
tags:
  - medical
  - triage
  - emergency-department
  - qwen3
  - sft
  - lora
  - dpo
  - trl
  - unsloth
model-index:
  - name: qwen3-1.7b-clinical-triage
    results:
      - task:
          type: text-generation
          name: Emergency triage level
        dataset:
          name: Clinical evaluation set
          type: {{release:dataset_id}}
          config: clinical_eval
          split: test
          revision: {{release:model_tag}}
        metrics:
          - type: accuracy
            name: Triage level accuracy
            value: {{evaluation:clinical_set.models.dpo-merged.accuracy|raw}}
          - type: undertriage
            name: Urgent cases sent to a lower level
            value: {{evaluation:clinical_set.models.dpo-merged.undertriage|raw}}
          - type: format_compliance
            name: Answers the expected format can parse
            value: {{evaluation:clinical_set.models.dpo-merged.format_compliance|raw}}
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
{{data:system_prompt}}
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

name, revision = "{{release:model_id}}", "{{release:model_tag}}"
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
vllm serve {{release:model_id}} --revision {{release:model_tag}} --max-model-len 1024
```

The revision pins the weights this card describes; the default branch moves with every
publication.

## Training

**Data.** The supervised set of [`{{release:dataset_id}}`](https://huggingface.co/datasets/{{release:dataset_id}}):
{{data:statistics.sft_train|int}} training and {{data:statistics.sft_validation|int}} validation
examples, balanced over the three levels and the two languages, then
{{data:statistics.dpo_total|int}} preference pairs. No real patient data.

**Supervised step.** LoRA of rank {{sft:hyperparameters.lora_r}}, alpha
{{sft:hyperparameters.lora_alpha}}, learning rate {{sft:hyperparameters.learning_rate}}, over
{{sft:hyperparameters.epochs}} epochs, on every projection **and the output head**. The base model
carries its chat tokens as one untrained vector and ties its output head to its embeddings: an
adapter on the projections alone produces a model that never emits its end-of-turn token. The
exported weights therefore untie the head (`tie_word_embeddings: false`).

**Preference step.** DPO on top of the merged supervised model, which is also its reference, with a
supervised term on the preferred answer that keeps it likely. The rejected answers carry one
defect each: a level too low, an unsafe course of action, an asserted diagnosis, or English.

**Hardware.** One {{sft:environment.gpu}}: {{sft:metrics.duration_s|minutes}} for the supervised
step, {{dpo:metrics.duration_s|minutes}} for the preference step, through Unsloth.

## Evaluation

<!-- source: reports/evaluation_results.json -->
{{evaluation:clinical_set.n}} cases written one by one for the evaluation, none seen in training,
{{data:statistics.evaluation_case_types.falsely_reassuring}} of them falsely reassuring and
{{data:statistics.evaluation_case_types.falsely_alarming}} falsely alarming.

| System | Accuracy | Urgent cases sent lower | Format compliance |
|---|---|---|---|
| **This model** | {{evaluation:clinical_set.models.dpo-merged.accuracy|f3}} {{evaluation:clinical_set.models.dpo-merged.accuracy_ci95|ci}} | {{evaluation:clinical_set.models.dpo-merged.undertriage_detail.misses}} of {{evaluation:clinical_set.models.dpo-merged.undertriage_detail.urgent_cases}} | {{evaluation:clinical_set.models.dpo-merged.format_compliance|pct}} |
| Supervised step alone | {{evaluation:clinical_set.models.sft.accuracy|f3}} {{evaluation:clinical_set.models.sft.accuracy_ci95|ci}} | {{evaluation:clinical_set.models.sft.undertriage_detail.misses}} of {{evaluation:clinical_set.models.sft.undertriage_detail.urgent_cases}} | {{evaluation:clinical_set.models.sft.format_compliance|pct}} |
| Keyword rule | {{evaluation:clinical_set.baselines.explicit_rule.accuracy|f3}} {{evaluation:clinical_set.baselines.explicit_rule.accuracy_ci95|ci}} | {{evaluation:clinical_set.baselines.explicit_rule.undertriage_detail.misses}} of {{evaluation:clinical_set.baselines.explicit_rule.undertriage_detail.urgent_cases}} | n/a |
| Linear classifier, same examples | {{evaluation:clinical_set.baselines.linear_classifier.accuracy|f3}} {{evaluation:clinical_set.baselines.linear_classifier.accuracy_ci95|ci}} | {{evaluation:clinical_set.baselines.linear_classifier.undertriage_detail.misses}} of {{evaluation:clinical_set.baselines.linear_classifier.undertriage_detail.urgent_cases}} | n/a |
> n = {{evaluation:clinical_set.n}} cases, of which n = {{evaluation:clinical_set.models.dpo-merged.undertriage_detail.urgent_cases}} urgent. The accuracy
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
| On the {{evaluation:model_versus_rule.undertriage.urgent_cases}} urgent cases | First only | Second only | Test |
|---|---|---|---|
| This model against the rule | {{evaluation:model_versus_rule.undertriage.model_only}} | {{evaluation:model_versus_rule.undertriage.baseline_only}} | {{evaluation:model_versus_rule.undertriage.p_mcnemar|p}} |
| This model against the classifier | {{evaluation:model_versus_classifier.undertriage.model_only}} | {{evaluation:model_versus_classifier.undertriage.baseline_only}} | {{evaluation:model_versus_classifier.undertriage.p_mcnemar|p}} |
| Aligned against supervised | {{evaluation:aligned_versus_supervised.undertriage.model_only}} | {{evaluation:aligned_versus_supervised.undertriage.baseline_only}} | {{evaluation:aligned_versus_supervised.undertriage.p_mcnemar|p}} |
| More urgent of the two against this model alone | {{evaluation:worse_of_two.versus_model_alone.undertriage.model_only}} | {{evaluation:worse_of_two.versus_model_alone.undertriage.baseline_only}} | {{evaluation:worse_of_two.versus_model_alone.undertriage.p_mcnemar|p}} |
> n = {{evaluation:model_versus_rule.undertriage.urgent_cases}} urgent cases, the same in every row.

<!-- source: reports/evaluation_results.json -->
| On all {{evaluation:clinical_set.n}} cases, accuracy | First only | Second only | Test |
|---|---|---|---|
| This model against the rule | {{evaluation:model_versus_rule.accuracy.model_only}} | {{evaluation:model_versus_rule.accuracy.baseline_only}} | {{evaluation:model_versus_rule.accuracy.p_mcnemar|p}} |
| This model against the classifier | {{evaluation:model_versus_classifier.accuracy.model_only}} | {{evaluation:model_versus_classifier.accuracy.baseline_only}} | {{evaluation:model_versus_classifier.accuracy.p_mcnemar|p}} |
| Aligned against supervised | {{evaluation:aligned_versus_supervised.accuracy.model_only}} | {{evaluation:aligned_versus_supervised.accuracy.baseline_only}} | {{evaluation:aligned_versus_supervised.accuracy.p_mcnemar|p}} |
> n = {{evaluation:clinical_set.n}} cases, the same in every row.

<!-- source: reports/seed_study.json -->
**Seeds.** The method was trained three times, on seeds 42, 43 and 44; these weights are seed 42,
chosen before any result. Across the three, accuracy on the clinical set ranges from
{{seeds:spread.dpo-merged.clinical_accuracy.min|f3}} to
{{seeds:spread.dpo-merged.clinical_accuracy.max|f3}}, and the urgent cases missed from
{{seeds:spread.dpo-merged.clinical_undertriage_misses.min|int}} to
{{seeds:spread.dpo-merged.clinical_undertriage_misses.max|int}}. Training alone moves the result
that much, so a gap of that size between two systems trained once each can be the seed.

<!-- source: reports/before_after.json -->
**Against the previous version.** The model this repository served before, version 1.0.0,
evaluated by the same code on the same cases, scores {{before_after:before.clinical_accuracy|f3}} and misses
{{before_after:before.clinical_undertriage_misses|int}} urgent cases where this one misses
{{before_after:after.clinical_undertriage_misses|int}}:
{{before_after:undertriage.rescued_by_this_release|int}} caught by this version alone,
{{before_after:undertriage.lost_by_this_release|int}} by the previous one alone,
{{before_after:undertriage.p_mcnemar|p}}. The retraining did not make a better triage decision.
It corrected the data the model learns from and the prompt it was trained with, and measured the
result on three seeds.

<!-- source: reports/evaluation_results.json -->
**Alignment.** On {{evaluation:task_preferences.n}} preference pairs of the task, held out of
training, the aligned model ranks the preferred answer first in
{{evaluation:task_preferences.correctly_ordered_share|pct}} of pairs. On
{{evaluation:external_preferences.n}} pairs from a medical preference set built outside the project, the share is
{{evaluation:external_preferences.correctly_ordered_share|pct}}, and in
{{evaluation:external_preferences.per_group.length.correctly_ordered_share|pct}} of the
{{evaluation:external_preferences.per_group.length.n}} whose two answers are of similar length.
It learnt the defects of the task, not medicine at large, and its score below chance comes from
answer length, which it penalises.

**Robustness.** On {{evaluation:robustness.dpo-merged.n}} inputs outside the nominal case (a
three-letter entry, a two-page paste, a description in another language, an attempt to make it
say something else), the model keeps its contract in
{{evaluation:robustness.dpo-merged.compliant_share|pct}} of cases: the expected format, in
French, its system prompt kept to itself, a clean stop, the identifying data it was given not
echoed, and the level an input announces when it announces one.

**The weights evaluated are these.** The figures above were measured on files whose SHA-256 the
evaluation recorded, and the publication refuses any other:

| File | SHA-256 |
|---|---|
{{evaluation:evaluated_weights|digests}}

## Limitations

- The clinical catalogue was not validated by an emergency physician.
- The catalogue, the triage rule and the evaluation set have the same author: there is no
  external ground truth.
- {{evaluation:clinical_set.n}} evaluation cases: an interval of accuracy spans about twenty
  points, and most differences between systems are not significant. On
  {{evaluation:clinical_set.resolution.urgent_cases}} urgent cases, even a model that missed none
  would keep an upper bound of {{evaluation:clinical_set.resolution.upper_bound_without_miss|pct}}
  on its undertriage; it takes
  {{evaluation:clinical_set.resolution.urgent_cases_needed_for_target}} to bring it under
  {{evaluation:clinical_set.resolution.target|pct}}.
- The training data is mostly generated: it lacks the disorder of real language.
- 1.7 billion parameters learn a format and three classes, not clinical reasoning.

## Licence

Apache-2.0, the licence of the base model `Qwen/Qwen3-1.7B-Base`. The dataset is published under
CC BY 4.0, the code under MIT.

## Reproducing

The code, the settings and the pipeline, at the version of these weights:
<{{release:repository}}/tree/{{release:tag}}>. The adapters and the intermediate merged model
are not published: the pipeline rebuilds them from the dataset, at the settings these weights
were trained with.
