"""Performance bench of the triage endpoint.

The clinical evaluation measures decision quality; this module measures what a decision costs.
The two numbers that matter to an emergency department are not the same:

- **perceived latency**, request by request, with no concurrency: the time the triage nurse
  waits in front of the screen;
- **throughput under load**, several requests in parallel: what says how many reception desks
  one server can feed at peak.

Both are measured on the **gateway**, ``POST /triage``, because that is what the nurse calls.
Measuring them on the vLLM engine alone would leave out everything the gateway does on every
request: the explicit rule, then the Presidio anonymisation of both texts before writing to the
audit log, all of it synchronously. The go-live criterion covers what the service delivers, not
what the engine produces.

The cost of the engine alone stays measurable (``target="engine"``), and the published protocol
presents it for what it is: a decomposition. The gateway also reports its own inference duration
in every reply, which isolates its overhead without re-running anything.

In both cases the real endpoint served by vLLM is measured, never the development engine: a
figure obtained with ``transformers`` in sequential generation has nothing to do with it, vLLM
batching requests and reusing the attention cache.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from clinical_triage.evaluation.metrics import latency_summary
from clinical_triage.prompts import completion_payload, extract_level
from clinical_triage.utils import get_logger

logger = get_logger("latency")


@dataclass
class Measure:
    """One timed request.

    ``inference_ms`` is filled in for the gateway only: it publishes the duration of its own
    generation, and the difference with the total latency gives its overhead — explicit rule,
    anonymisation, writing to the log.
    """

    latency_ms: float
    tokens: int
    level: str | None
    inference_ms: float | None = None


def _call_engine(client, url: str, symptoms: str, model: str) -> Measure:
    """Query the vLLM engine directly and time the complete answer."""
    started = time.perf_counter()
    response = client.post(f"{url}/v1/completions", json=completion_payload(symptoms, model))
    response.raise_for_status()
    latency = (time.perf_counter() - started) * 1000
    body = response.json()
    text = body["choices"][0]["text"]
    tokens = body.get("usage", {}).get("completion_tokens", 0)
    return Measure(latency_ms=latency, tokens=tokens, level=extract_level(text))


def _call_gateway(client, url: str, symptoms: str, model: str) -> Measure:
    """Query ``POST /triage`` and time the complete answer.

    ``model`` is unused: the gateway serves the model it loaded. The parameter is there so that
    both call functions share a signature, and ``benchmark`` can pick either without detours.
    """
    started = time.perf_counter()
    response = client.post(f"{url}/triage", json={"symptoms": symptoms})
    if response.status_code == 429:
        # The quota is a service policy, not an engine limit: a throughput measurement
        # necessarily hits it, since it holds a single key. The raw HTTP client message does not
        # say what to do about it.
        raise SystemExit(
            "The gateway is applying its per-caller quota (429). The bench uses a single key: "
            "raise TRIAGE_RATE_LIMIT for the duration of the measurement, or lower "
            "--concurrency and --repetitions."
        )
    response.raise_for_status()
    latency = (time.perf_counter() - started) * 1000
    body = response.json()
    return Measure(
        latency_ms=latency,
        # The gateway does not count the tokens produced: that figure is read on the engine,
        # and the "engine" measurement is there for it.
        tokens=0,
        level=body.get("level"),
        inference_ms=body.get("latency_ms"),
    )


CALLS = {"gateway": _call_gateway, "engine": _call_engine}


def benchmark(
    url: str,
    model: str,
    cases: list[str],
    api_key: str | None = None,
    concurrencies: tuple[int, ...] = (1, 4, 8),
    repetitions: int = 2,
    target: str = "gateway",
) -> dict:
    """Measure latency and throughput of an endpoint at several concurrency levels.

    ``target`` is "gateway" — ``POST /triage``, what the service delivers, and what the go-live
    criterion covers — or "engine", which queries vLLM directly and gives the cost of generation
    alone.
    """
    import httpx

    if target not in CALLS:
        raise ValueError(f"Unknown measurement target: {target!r} (expected: {set(CALLS)})")
    call = CALLS[target]

    # The two targets do not share an authentication scheme: the gateway expects its own key in
    # ``X-API-Key``, the engine the one vLLM receives through ``--api-key`` and checks as a
    # bearer token.
    if not api_key:
        headers: dict[str, str] = {}
    elif target == "gateway":
        headers = {"X-API-Key": api_key}
    else:
        headers = {"Authorization": f"Bearer {api_key}"}
    requests = [cases[i % len(cases)] for i in range(len(cases) * repetitions)]
    results: dict[str, dict] = {}

    with httpx.Client(timeout=120, headers=headers) as client:
        # Warm-up: the first request pays for loading the weights into GPU memory.
        call(client, url, cases[0], model)

        for concurrency in concurrencies:
            started = time.perf_counter()
            if concurrency == 1:
                measures = [call(client, url, case, model) for case in requests]
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    measures = list(pool.map(lambda c: call(client, url, c, model), requests))
            total_duration = time.perf_counter() - started

            summary = latency_summary([m.latency_ms for m in measures])
            summary["requests"] = len(measures)
            summary["requests_per_s"] = round(len(measures) / total_duration, 2)
            counted = sum(m.tokens for m in measures)
            if counted:
                # Written only when tokens are actually counted. The gateway returns none, and
                # a throughput of 0.0 there would announce a null rate where the quantity is
                # simply not measured.
                summary["tokens_per_s"] = round(counted / total_duration, 1)
            summary["level_extracted_share"] = round(
                sum(m.level is not None for m in measures) / len(measures), 3
            )
            overheads = [m.latency_ms - m.inference_ms for m in measures if m.inference_ms]
            if overheads:
                # What the gateway adds to generation: explicit rule, anonymisation of both
                # texts, writing to the audit log.
                summary["gateway_overhead_ms"] = latency_summary(overheads)["p50_ms"]
            results[f"concurrency_{concurrency}"] = summary
            logger.info(
                "  concurrency %d: p50 %.0f ms, p95 %.0f ms, %.2f req/s",
                concurrency,
                summary["p50_ms"],
                summary["p95_ms"],
                summary["requests_per_s"],
            )

    return results
