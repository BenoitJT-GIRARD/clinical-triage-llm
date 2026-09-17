"""Cross-cutting helpers: seeds, git revision and logging.

What every stage shares lives here, to avoid duplication and to keep reproducibility uniform.
"""

from __future__ import annotations

import gc
import logging
import random
import subprocess  # nosec B404
from pathlib import Path

from clinical_triage.config import PATHS, SEED


def set_seed(seed: int = SEED, include_torch: bool = True) -> None:
    """Seed ``random``, ``numpy`` and, optionally, ``torch``.

    numpy and torch are imported lazily so that this module stays usable without a heavy
    dependency. ``include_torch=False`` helps the data-preparation stages: on Windows,
    importing torch before pyarrow can cause a DLL conflict, so the data pipeline stays free
    of torch.

    Setting ``PYTHONHASHSEED`` here would achieve nothing: CPython reads that variable only at
    process start, so the seed would be announced without being fixed. The one place in the
    pipeline whose result depends on the order of a set makes itself deterministic
    (``triage_rules._compile``), which is worth more than an environment variable nobody must
    forget.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is always present in practice
        pass
    if not include_torch:
        return
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def free_gpu_memory() -> None:
    """Give the card back the memory of a model that is no longer needed.

    ``del model`` is not enough, for two reasons that compound: PyTorch modules form reference
    cycles that only the garbage collector breaks, and the CUDA allocator then keeps the freed
    blocks in its cache. A script that loads several models in one process ends up saturating
    the card, and the loading library silently moves part of the layers onto the CPU — the
    error only surfaces at the first generation, far from its cause.
    """
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is always present when training
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_logger(name: str = "clinical_triage") -> logging.Logger:
    """Return a simply configured logger (timestamped format, INFO level)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def path_for_log(path: str | Path) -> str:
    """Render a path the way a trace of this project should show it.

    Module logs end up in notebook outputs, which are versioned and published: an absolute
    path would write the directory tree of the machine that ran the code, and a reader gets
    nothing from it. Anything under the repository root is therefore written relative to it,
    with forward slashes whatever the platform. The rest is returned as is — a Hugging Face
    repository id, a model mounted elsewhere in a container: shortening it would make it wrong.
    """
    text = str(path)
    candidate = Path(text)
    if not candidate.is_absolute():
        return text
    try:
        return candidate.relative_to(PATHS.root).as_posix()
    except ValueError:
        return text


def git_revision() -> str:
    """Short revision of the repository, or ``"unknown"`` outside a git repository.

    The dataset card and the published protocol both record this revision: it is what lets a
    reader find the exact code that produced the numbers. Both read it here and nowhere else:
    two copies, under two names and with two fallback values, would let the two traceabilities
    drift apart unnoticed.
    """
    try:
        # A constant argument list, no shell and no outside input. Bandit's three subprocess
        # rules are waived inline, here for the call and at module import for B404, and
        # nowhere else in the package. ``git`` is deliberately looked up on PATH, its location
        # varying from machine to machine.
        completed = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PATHS.root,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
