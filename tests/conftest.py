"""Réglages communs à toute la suite de tests.

Sous Windows, importer `torch` avant `pyarrow` provoque un conflit de
bibliothèques dynamiques qui termine le processus par une violation d'accès.
L'ordre d'import dépend ici de l'ordre des tests, donc du hasard. On force donc
le bon ordre au démarrage de la session, comme le fait `clinical_triage.bootstrap`
pour les scripts.
"""

from __future__ import annotations

import pyarrow  # noqa: F401  (import volontaire : il doit précéder celui de torch)
