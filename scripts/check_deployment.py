"""Check that the deployed service serves the model that was evaluated, not merely a model.

A health probe says the engine answers; it does not say what it answers with. A wrong revision,
weights garbled in transit or an engine started on the base model would all pass it. This check
sends three cases of the clinical set — for each level, the first direct case the evaluated model
classified right — and requires the level the evaluation recorded. Temperature is zero, so the
deployed engine has no reason to answer otherwise.

The cases are read from the committed evaluation, never typed here: when the model changes,
the check follows it.

Usage::

    uv run python scripts/check_deployment.py --url https://... --key "$TRIAGE_API_KEY"
"""

from __future__ import annotations

from clinical_triage.bootstrap import use_utf8_console

use_utf8_console()

import argparse
import json
import sys

import httpx

from clinical_triage.config import PATHS, TRIAGE
from clinical_triage.data.dataset_io import read_jsonl

SHIPPED = "dpo-merged"


def reference_cases(results: dict, cases: list[dict]) -> list[dict]:
    """One direct case per level that the shipped model classified right, in set order."""
    missed = {error["id"] for error in results["errors"][SHIPPED]}
    chosen = []
    for level in TRIAGE.levels:
        case = next(
            (
                c
                for c in cases
                if c["level"] == level and not c["case_type"] and c["id"] not in missed
            ),
            None,
        )
        if case is None:
            raise SystemExit(f"No direct {level} case the evaluated model got right.")
        chosen.append(case)
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--key", required=True)
    args = parser.parse_args()

    results = json.loads((PATHS.reports / "evaluation_results.json").read_text(encoding="utf-8"))
    cases = read_jsonl(PATHS.data_processed / "clinical_eval.jsonl")
    failures = []
    with httpx.Client(timeout=120, headers={"X-API-Key": args.key}) as client:
        for case in reference_cases(results, cases):
            reply = client.post(f"{args.url}/triage", json={"symptoms": case["description"]})
            reply.raise_for_status()
            level = reply.json()["level"]
            print(f"{case['id']}: expected {case['level']}, served {level}")
            if level != case["level"]:
                failures.append(case["id"])
    if failures:
        print(f"The deployed model does not answer as the evaluated one on: {', '.join(failures)}")
        return 1
    print("The deployed model answers as the evaluated one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
