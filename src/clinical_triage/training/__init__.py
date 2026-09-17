"""Specialising the model: supervised fine-tuning, then preference alignment.

- ``sft``      : supervised fine-tuning of the base model with a LoRA adapter;
- ``dpo``      : preference alignment above the merged supervised model;
- ``tracking`` : local run tracking and versioned JSON summaries.

Both trainings import torch late: on Windows, ``datasets`` must be loaded before it, which
``clinical_triage.bootstrap`` takes care of at the top of every script.
"""

from __future__ import annotations
