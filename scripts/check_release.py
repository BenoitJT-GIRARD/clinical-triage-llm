"""The gate a model version passes before its cards are published and its tag is served.

Two criteria, fixed on 2026-10-09 before the model they judge existed, and read from the committed
evaluation of the shipped model:

- **every answer is usable**: format compliance of 1.0 on the clinical set. An answer the
  information system cannot read is a triage that did not happen;
- **no more urgent cases missed than the explicit rule**: a model that sends more urgent patients
  down the queue than a keyword rule has no business beside it.

They are a floor, not a claim: passing them says the model is fit to be shown, not fit to be
used. The continuous deployment runs this script before it publishes anything.

Usage::

    uv run python scripts/check_release.py
"""

from __future__ import annotations

from clinical_triage.bootstrap import use_utf8_console

use_utf8_console()

import json
import sys
from pathlib import Path

from clinical_triage.config import PATHS

SHIPPED = "dpo-merged"


def release_problems(results: dict) -> list[str]:
    """What keeps this evaluation from releasing its model, or an empty list."""
    model = results.get("clinical_set", {}).get("models", {}).get(SHIPPED)
    versus_rule = (results.get("model_versus_rule") or {}).get("undertriage")
    if model is None or versus_rule is None:
        return [f"no evaluation of `{SHIPPED}` against the explicit rule"]
    problems = []
    if model["format_compliance"] < 1.0:
        problems.append(f"format compliance {model['format_compliance']} on the clinical set")
    if versus_rule["model_undertriages"] > versus_rule["baseline_undertriages"]:
        problems.append(
            f"{versus_rule['model_undertriages']} urgent cases missed, the explicit rule misses "
            f"{versus_rule['baseline_undertriages']}"
        )
    return problems


def main(path: Path = PATHS.reports / "evaluation_results.json") -> int:
    problems = release_problems(json.loads(path.read_text(encoding="utf-8")))
    if problems:
        print("Release refused: " + "; ".join(problems))
        return 1
    print("Release gate passed: every answer usable, no more urgent cases missed than the rule.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
