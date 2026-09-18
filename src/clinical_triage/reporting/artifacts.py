"""Reading the result artefacts the figures are drawn from.

One door, so that a missing artefact is reported where it is looked for rather than where a
reader finds a dash in a table.
"""

from __future__ import annotations

import json
from pathlib import Path

from clinical_triage.utils import get_logger, path_for_log

logger = get_logger("artifacts")


def read_results(path: Path) -> dict | None:
    """Read a result file, or return None when it has not been produced yet.

    The absence is logged: a figure built on missing measurements must say so at the moment it
    looks for them.
    """
    if not path.exists():
        logger.warning("File absent: %s", path_for_log(path))
        return None
    return json.loads(path.read_text(encoding="utf-8"))
