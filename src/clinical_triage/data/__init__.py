"""Construction du corpus de triage, de la vérité terrain jusqu'aux fichiers livrés.

- `vital_signs`       : constantes vitales, seuils d'alerte et lecture dans un texte ;
- `triage_rules`      : règle de triage explicite, référence à battre en évaluation ;
- `clinical_catalogue`: présentations cliniques types, qui portent la vérité terrain ;
- `case_generator`    : vignettes cliniques composées à partir du catalogue ;
- `corpus_sources`    : chargement des corpus médicaux publics ;
- `corpus_cases`      : extraction des cas réellement exploitables pour du triage ;
- `clinical_eval_set` : jeu d'évaluation rédigé à la main, tenu hors entraînement ;
- `anonymize`         : masquage RGPD et contrôle qualité indépendant ;
- `sft_builder`       : assemblage des exemples supervisés ;
- `dpo_builder`       : paires de préférence de sécurité clinique ;
- `dataset_io`        : sérialisation, découpage sans fuite et carte du dataset.
"""

from __future__ import annotations
