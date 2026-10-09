"""Regenerate the gateway image's pinned closure, with a digest for every package.

``infra/requirements-api.in`` carries the constraints a person writes; this script turns them
into ``infra/requirements-api.txt``, the file the Dockerfile installs. Two properties make that
file worth trusting:

- **its versions are the lock file's.** The resolution is constrained by an export of
  ``uv.lock``: without it, the image would drift from the repository, and the deployed image
  would no longer hold the versions continuous integration audits;
- **every package carries its SHA-256 digests**, and the Dockerfile installs with
  ``--require-hashes``: a package replaced on the index, or in transit, is refused rather than
  installed.

Run it after every ``uv lock``::

    uv run python scripts/compile_image_requirements.py
"""

from __future__ import annotations

import subprocess  # nosec B404 - runs uv, by name, with arguments fixed in this file
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONSTRAINTS = Path("var") / "constraints-from-lock.txt"
SOURCE = Path("infra") / "requirements-api.in"
TARGET = Path("infra") / "requirements-api.txt"

HEADER = """\
# Full pinned closure of the gateway image's dependencies, with their digests.
#
# GENERATED FILE -- do not edit. The constraints live in `requirements-api.in`; regenerate with
# `uv run python scripts/compile_image_requirements.py` after every `uv lock`. The versions are
# those of the project lock file, so what continuous integration audits is what the image
# installs, and the Dockerfile installs with `--require-hashes`.

"""


def _uv(*arguments: str) -> str:
    done = subprocess.run(  # nosec B603 B607 - uv from the PATH, fixed arguments
        ["uv", *arguments], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return done.stdout


def main() -> int:
    exported = _uv(
        "export",
        "--frozen",
        "--no-emit-project",
        "--all-groups",
        "--no-hashes",
        "--format",
        "requirements-txt",
    )
    # Plain `name==version` lines only, with the local CUDA suffix of the training wheels
    # dropped: they constrain nothing the image installs, and pip would not parse them.
    pins = [
        line.split("+cu")[0]
        for line in exported.splitlines()
        if line[:1].isalpha() and "==" in line
    ]
    (ROOT / CONSTRAINTS).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / CONSTRAINTS).write_text("\n".join(pins) + "\n", encoding="utf-8")
    _uv(
        "pip",
        "compile",
        SOURCE.as_posix(),
        "--python-version",
        "3.12",
        "--python-platform",
        "x86_64-unknown-linux-gnu",
        "--constraint",
        CONSTRAINTS.as_posix(),
        "--generate-hashes",
        "--no-header",
        "--output-file",
        TARGET.as_posix(),
    )
    body = (ROOT / TARGET).read_text(encoding="utf-8")
    (ROOT / TARGET).write_text(HEADER + body, encoding="utf-8", newline="\n")
    print(f"{TARGET.as_posix()} regenerated from {SOURCE.as_posix()} and uv.lock.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
