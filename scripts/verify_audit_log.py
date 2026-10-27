"""Check the chain of digests that reveals a line of the audit log edited, inserted or deleted.

Each line carries the digest of the one before it. The check walks the chain across the day
files and names every line where it breaks; it exits with a non-zero status if it finds one.
Two changes leave the chain intact and go unreported: a change to the last line, and the
deletion of the first lines.

Usage::

    uv run python scripts/verify_audit_log.py                 # the directory TRIAGE_AUDIT_DIR names
    uv run python scripts/verify_audit_log.py var/logs/audit
"""

from __future__ import annotations

from clinical_triage.bootstrap import use_utf8_console

use_utf8_console()

import argparse
import sys
from pathlib import Path

from clinical_triage.serving.audit import audit_directory, read_entries, verify


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", nargs="?", type=Path, default=None)
    directory = parser.parse_args().directory or audit_directory()
    problems = verify(directory)
    for problem in problems:
        print(problem)
    entries = len(read_entries(directory))
    if problems:
        print(f"{len(problems)} break(s) in the chain of {entries} line(s).")
        return 1
    print(f"{entries} line(s), chain intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
