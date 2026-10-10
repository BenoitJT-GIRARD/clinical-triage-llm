"""The settings the supervised pass is tuned over, and the one the tuning kept.

Rank, scaling and learning rate of the supervised LoRA pass are settled by a comparison, not by a default: ``scripts/tune_hyperparameters.py`` trains each setting below
under the same conditions and writes which one it kept. ``scripts/train_sft.py`` and the
project summary read that choice here, and nowhere else.
"""

from __future__ import annotations

import json
from pathlib import Path

from clinical_triage.config import PATHS

# The settings compared. One value varies at a time around the reference, so that every gap
# observed can be attributed.
SETTINGS = (
    {"name": "r8_lr2e-4", "lora_r": 8, "lora_alpha": 16, "learning_rate": 2e-4},
    {"name": "r16_lr2e-4", "lora_r": 16, "lora_alpha": 32, "learning_rate": 2e-4},
    {"name": "r32_lr2e-4", "lora_r": 32, "lora_alpha": 64, "learning_rate": 2e-4},
    {"name": "r16_lr1e-4", "lora_r": 16, "lora_alpha": 32, "learning_rate": 1e-4},
)

#: Where the tuning writes its comparison and the setting it kept.
COMPARISON = PATHS.reports / "training" / "hyperparameter_comparison.json"


def kept_setting(path: Path = COMPARISON) -> dict | None:
    """The setting the tuning kept, or None when the tuning has not run."""
    if not path.exists():
        return None
    comparison = json.loads(path.read_text(encoding="utf-8"))
    return next(s for s in comparison["variants"] if s["variant"] == comparison["kept"])
