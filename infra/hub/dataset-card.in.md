---
license: cc-by-4.0
language:
  - fr
  - en
multilinguality:
  - multilingual
task_categories:
  - text-generation
annotations_creators:
  - machine-generated
  - other
language_creators:
  - machine-generated
  - found
  - other
source_datasets:
  - ANR-MALADES/MediQAl
  - openlifescienceai/medmcqa
  - keivalya/MedQuad-MedicalQnADataset
tags:
  - medical
  - triage
  - emergency-department
  - synthetic
  - sft
  - dpo
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
