"""Evaluating the model: decision quality, safety, robustness and performance.

- ``metrics``    : accuracy, undertriage, per-level F1, confidence intervals;
- ``safety``     : checks on the generated content (recommendation, diagnosis, language);
- ``robustness`` : behaviour on degraded or hijacked inputs;
- ``baselines``  : the references the model is compared against;
- ``runner``     : running a full evaluation over a set of cases;
- ``latency``    : perceived latency and throughput of the inference endpoint.
"""

from __future__ import annotations
