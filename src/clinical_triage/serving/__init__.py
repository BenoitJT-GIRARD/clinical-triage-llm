"""Service d'inférence : API de triage, questionnaire adaptatif et traçabilité.

- `schemas`       : contrats d'échange avec le système d'information hospitalier ;
- `questionnaire` : collecte des symptômes, adaptée au motif de consultation ;
- `audit`         : journal anonymisé de chaque interaction, pour les audits médicaux ;
- `api`           : application FastAPI et garde-fous d'exploitation.
"""

from __future__ import annotations
