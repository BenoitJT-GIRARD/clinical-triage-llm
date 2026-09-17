"""Settings shared by the whole test suite.

On Windows, importing ``torch`` before ``pyarrow`` causes a dynamic-library conflict that ends
the process with an access violation. The import order here depends on the order of the tests,
hence on chance. The right order is therefore forced at session start, as
``clinical_triage.bootstrap`` does for the scripts.
"""

from __future__ import annotations

import pyarrow  # noqa: F401  (deliberate import: it must precede torch)
