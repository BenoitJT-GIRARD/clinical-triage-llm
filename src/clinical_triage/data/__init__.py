"""Building the triage corpus, from ground truth to the delivered files.

- ``vital_signs``       : vital signs, alert thresholds and reading them out of a text;
- ``triage_rules``      : the explicit triage rule, the baseline to beat at evaluation;
- ``clinical_catalogue``: typical clinical presentations, which carry the ground truth;
- ``case_generator``    : clinical vignettes composed from the catalogue;
- ``corpus_sources``    : loading the public medical corpora;
- ``corpus_cases``      : extracting the cases actually usable for triage;
- ``clinical_eval_set`` : the hand-written evaluation set, kept out of training;
- ``anonymize``         : GDPR masking and an independent quality check;
- ``sft_builder``       : assembling the supervised examples;
- ``dpo_builder``       : clinical-safety preference pairs;
- ``dataset_io``        : serialisation, leak-free splitting and the dataset card.
"""

from __future__ import annotations
