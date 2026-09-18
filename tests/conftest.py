"""Settings shared by the whole test suite.

On Windows the two libraries have to be imported in one particular order, for the reason
``clinical_triage.bootstrap`` sets out. Under pytest the order would otherwise follow the order
of the tests, which is to say chance, so it is forced here at session start.
"""

from __future__ import annotations

import pyarrow  # noqa: F401  (deliberate import: it must precede torch)
