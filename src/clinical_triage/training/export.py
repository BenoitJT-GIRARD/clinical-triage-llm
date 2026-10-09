"""The check an exported model must pass before it is kept, and again before it is published.

Two ways a merged export can be wrong without raising anything: its configuration re-enables the
tie between the embeddings and the output head, or its weight files hold no distinct
``lm_head.weight`` tensor. In both cases the model served is not the one that was trained — its
trained output head, the one that learnt to stop, is gone.

The merge script deletes an export that fails; the publication script refuses to send one. The
same function decides both, so that what is published is what was checked.

The evaluation records the digest of every weight file it read, and the publication refuses
weights whose digests differ: what goes to the Hub is what was measured, not a later export.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def export_problems(folder: Path) -> list[str]:
    """What is wrong with an exported model folder, or an empty list."""
    configuration = folder / "config.json"
    if not configuration.exists():
        return [f"{folder.name}: no config.json"]
    problems = []
    if json.loads(configuration.read_text(encoding="utf-8")).get("tie_word_embeddings"):
        problems.append(
            f"{folder.name}: the weights are still tied, the trained output head is lost"
        )

    index = folder / "model.safetensors.index.json"
    single = folder / "model.safetensors"
    if index.exists():
        keys = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    elif single.exists():
        from safetensors import safe_open

        with safe_open(str(single), framework="pt") as handle:
            keys = set(handle.keys())
    else:
        return [*problems, f"{folder.name}: no weight file"]
    if "lm_head.weight" not in keys:
        problems.append(f"{folder.name}: `lm_head.weight` absent from the exported weights")
    return problems


def weight_digests(folder: Path) -> dict[str, str]:
    """The SHA-256 of every weight file of a model folder, by file name.

    The Hub reports the same digest for every file it stores in large-file storage: the two
    compare without downloading anything.
    """
    digests = {}
    for path in sorted(folder.glob("*.safetensors")):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digests[path.name] = digest.hexdigest()
    return digests
