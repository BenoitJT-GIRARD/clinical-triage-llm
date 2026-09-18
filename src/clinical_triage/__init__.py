"""Emergency triage assistant.

The package holds every reusable piece: configuration, data preparation, training (SFT then
DPO), clinical evaluation and the inference service. The scripts under ``scripts/`` chain
those pieces in pipeline order, and the notebooks under ``notebooks/`` walk through them.
"""

from __future__ import annotations

import linecache
import os
import warnings

__version__ = "1.0.0"

# On its first import MLflow prints a greeting that quotes its installation paths. It has no
# business in a notebook output, which is versioned and published. The variable is set at
# package import: inside a notebook MLflow arrives behind Unsloth, long before the tracking
# module is asked for anything.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# A library warning prints the file that raised it, installation path included: notebook
# outputs, which are versioned and published, would then carry the directory tree of the
# machine that produced them. Only the prefix up to the installation directory is dropped from
# the display — the category, the text and the offending line stay exactly as they are, and no
# warning is silenced. The formatter is replaced process-wide, which is the only level where a
# notebook benefits without having to do anything.
_INSTALL_MARKER = "/site-packages/"
_original_format = warnings.formatwarning


def _warning_without_install_path(message, category, filename, lineno, line=None):
    """Format a warning by naming the module, without its installation path."""
    location = filename.replace("\\", "/")
    if _INSTALL_MARKER in location:
        # The offending line is read while the full path is still known: the formatter could
        # no longer find it from the shortened one.
        line = line or linecache.getline(filename, lineno)
        location = location.split(_INSTALL_MARKER)[-1]
    return _original_format(message, category, location, lineno, line)


warnings.formatwarning = _warning_without_install_path
