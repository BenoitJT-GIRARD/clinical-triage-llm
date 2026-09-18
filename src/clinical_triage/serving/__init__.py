"""The inference service: triage API, adaptive questionnaire and traceability.

- ``schemas``       : the exchange contract with the hospital information system;
- ``questionnaire`` : symptom collection, adapted to the presenting complaint;
- ``audit``         : an anonymised log of every interaction, for medical audits;
- ``api``           : the FastAPI application and its operational guard rails.
"""

from __future__ import annotations
