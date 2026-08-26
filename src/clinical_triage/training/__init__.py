"""Spécialisation du modèle : fine-tuning supervisé puis alignement par préférences.

- `sft`      : fine-tuning supervisé du modèle de base avec adaptation LoRA ;
- `dpo`      : alignement par préférences au-dessus du modèle supervisé fusionné ;
- `tracking` : suivi local des exécutions et résumés JSON versionnés.

Les deux entraînements importent torch tardivement : sous Windows, `datasets`
doit être chargé avant lui, ce dont se charge `clinical_triage.bootstrap` au début
de chaque script.
"""

from __future__ import annotations
