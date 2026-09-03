"""Évaluation du modèle : qualité des décisions, sécurité, robustesse et performance.

- `metrics`    : exactitude, sous-triage, F1 par niveau, intervalles de confiance ;
- `safety`     : contrôles du contenu généré (recommandation, diagnostic, langue) ;
- `robustness` : comportement sur des entrées dégradées ou détournées ;
- `baselines`  : références auxquelles comparer le modèle ;
- `runner`     : exécution d'une évaluation complète sur un jeu de cas ;
- `latency`    : latence perçue et débit de l'endpoint d'inférence.
"""

from __future__ import annotations
