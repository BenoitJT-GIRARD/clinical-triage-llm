"""Settings to apply at the very top of a script, before the heavy imports.

Two constraints of the development machine, gathered here so that scripts do not repeat them:

- **import order.** On Windows, importing ``torch`` before ``pyarrow`` (which ``datasets``
  loads) causes a dynamic-library conflict that ends the process with a segmentation fault and
  no message. Importing this module first loads ``datasets`` in the right order.
- **console encoding.** Windows standard output is cp1252; any trace containing an accented
  character or a symbol interrupts the script.

A training script therefore starts with::

    from clinical_triage.bootstrap import use_utf8_console

    use_utf8_console()
"""

from __future__ import annotations

import sys

import datasets  # noqa: F401  (deliberate import: it must precede torch)


def use_utf8_console() -> None:
    """Switch standard output to UTF-8, replacing what cannot be encoded."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
