---
license: cc-by-4.0
language:
  - fr
  - en
task_categories:
  - text-generation
tags:
  - medical
  - triage
  - emergency-department
  - synthetic
size_categories:
  - {{data:statistics.sft_total+statistics.dpo_total+statistics.clinical_evaluation|size}}
pretty_name: Bilingual emergency triage
configs:
  - config_name: sft
    data_files:
      - split: train
        path: sft_train.jsonl
      - split: validation
        path: sft_validation.jsonl
      - split: test
        path: sft_test.jsonl
  - config_name: dpo
    data_files:
      - split: train
        path: dpo_train.jsonl
  - config_name: clinical_eval
    data_files:
      - split: test
        path: clinical_eval.jsonl
---

{{include:data/README.md}}
