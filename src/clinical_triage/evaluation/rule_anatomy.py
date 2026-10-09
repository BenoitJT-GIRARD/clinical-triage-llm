"""Why the explicit rule misses the urgent cases it misses.

A rate says how often the rule sends an urgent patient down the queue; a head of department wants
to know why, because that is what tells whether the rule can be repaired. The rule is
deterministic Python and the clinical set is versioned, so each miss can be traced to what the
rule read, and to what it did not.

Three causes, exclusive and in this order:

- **a sign negated away** — the narrative names a life-threatening sign, and the rule discarded
  it because a negation stood in front of it;
- **only a lesser sign read** — the rule found something, a warning sign or a moderate vital
  sign, and nothing at the case's level;
- **nothing read** — no sign and no vital sign the rule knows: it falls back on the lowest
  level, which is what a keyword rule does with a narrative written in other words.
"""

from __future__ import annotations

from collections import Counter

from clinical_triage.config import TRIAGE
from clinical_triage.data.triage_rules import (
    MODERATE,
    VITAL,
    classify,
    matched_flags,
    negated_flags,
)
from clinical_triage.data.vital_signs import critical_findings, parse, parse_age, warning_findings

CAUSES = ("a sign negated away", "only a lesser sign read", "nothing read")


def _severity(level: str) -> int:
    return TRIAGE.levels[::-1].index(level)


def cause_of_miss(description: str) -> str:
    """Name what made the rule file this narrative too low."""
    if negated_flags(description, VITAL) and not matched_flags(description, VITAL):
        return CAUSES[0]
    reading, age = parse(description), parse_age(description)
    if warning_findings(reading, age) or critical_findings(reading, age):
        return CAUSES[1]
    if matched_flags(description, MODERATE):
        return CAUSES[1]
    return CAUSES[2]


def rule_misses(cases: list[dict]) -> dict:
    """Every urgent case the rule files below its level, with its cause, and the counts.

    ``cases`` are records of the clinical set: ``id``, ``level``, ``case_type``, ``description``.
    """
    urgent = [case for case in cases if case["level"] in (VITAL, MODERATE)]
    misses = []
    for case in urgent:
        rule_level = classify(case["description"])
        if _severity(rule_level) < _severity(case["level"]):
            misses.append(
                {
                    "id": case["id"],
                    "level": case["level"],
                    "rule_level": rule_level,
                    "case_type": case["case_type"] or "direct",
                    "cause": cause_of_miss(case["description"]),
                }
            )
    return {
        "urgent_cases": len(urgent),
        "missed": len(misses),
        "per_cause": {cause: sum(m["cause"] == cause for m in misses) for cause in CAUSES},
        "per_case_type": dict(sorted(Counter(m["case_type"] for m in misses).items())),
        "cases": misses,
    }
