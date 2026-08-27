"""Calcul du nombre de pas de chauffe d'un entraînement.

Les versions récentes de la bibliothèque d'entraînement n'acceptent plus une
*proportion* de chauffe, seulement un nombre de pas. Une proportion reste
pourtant le bon réglage : elle s'adapte à la taille du corpus et au lot effectif,
là où un nombre fixe devient absurde dès que l'un des deux change.

On convertit donc la proportion en pas, ici, à partir de ce que l'entraînement
va réellement exécuter.
"""

from __future__ import annotations

import math


def warmup_steps(
    nombre_exemples: int,
    lot_effectif: int,
    epochs: int,
    max_steps: int,
    proportion: float,
) -> int:
    """Convertit une proportion de chauffe en nombre de pas d'optimisation.

    `max_steps > 0` fixe directement le nombre de pas de l'entraînement, et
    l'emporte donc sur le calcul à partir de la taille du corpus.
    """
    if max_steps > 0:
        pas_total = max_steps
    else:
        pas_total = math.ceil(nombre_exemples / max(1, lot_effectif)) * max(1, epochs)
    return max(1, round(pas_total * proportion))
