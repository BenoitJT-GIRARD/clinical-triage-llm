"""Computing the number of warm-up steps of a training run.

Recent versions of the training library no longer accept a warm-up *ratio*, only a number of
steps. A ratio is nonetheless the right setting: it adapts to the size of the corpus and to the
effective batch, where a fixed number becomes absurd as soon as either changes.

The ratio is therefore converted into steps here, from what the training will actually run.
"""

from __future__ import annotations

import math


def warmup_steps(
    example_count: int,
    effective_batch: int,
    epochs: int,
    max_steps: int,
    ratio: float,
) -> int:
    """Convert a warm-up ratio into a number of optimiser steps.

    ``max_steps > 0`` sets the number of training steps directly, and therefore wins over the
    computation from the corpus size.
    """
    if max_steps > 0:
        total_steps = max_steps
    else:
        total_steps = math.ceil(example_count / max(1, effective_batch)) * max(1, epochs)
    return max(1, round(total_steps * ratio))
