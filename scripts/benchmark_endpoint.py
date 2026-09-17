"""Performance bench of the triage endpoint.

Measures the perceived latency and the throughput under load of ``POST /triage``, that is, of
what the triage nurse actually calls: the FastAPI gateway, which queries the engine then applies
the explicit rule, anonymises both texts and writes to the audit log before answering. That is
the figure the go-live criterion keeps.

With ``--engine-url``, a second series queries vLLM directly. It does not replace the first: it
decomposes it, giving the cost of generation alone. Both are published under their real names.

Run it against the local stack (``docker compose -f infra/docker-compose.yml up``) or against a
deployed endpoint: the command is the same, only the URL changes.

Usage::

    uv run python scripts/benchmark_endpoint.py --api-key "$TRIAGE_API_KEY"
    uv run python scripts/benchmark_endpoint.py --url https://my-endpoint --api-key "$KEY"
"""

from __future__ import annotations

import argparse
import json

from clinical_triage.config import PATHS, SERVING
from clinical_triage.data.dataset_io import read_jsonl
from clinical_triage.evaluation.latency import benchmark
from clinical_triage.utils import get_logger

logger = get_logger("bench")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="http://localhost:8080", help="triage gateway (POST /triage)"
    )
    parser.add_argument(
        "--engine-url",
        default=None,
        help="vLLM engine, to additionally measure the cost of generation alone",
    )
    parser.add_argument("--model", default=SERVING.served_model_name)
    parser.add_argument("--api-key", default=None, help="gateway key (X-API-Key)")
    parser.add_argument(
        "--vllm-key", default=None, help="engine key, if vLLM was started with --api-key"
    )
    parser.add_argument("--cases", type=int, default=20, help="distinct cases used")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=[1, 4, 8], help="load levels tested"
    )
    args = parser.parse_args()

    cases = [
        row["description"]
        for row in read_jsonl(PATHS.data_processed / "clinical_eval.jsonl")[: args.cases]
    ]
    if not cases:
        raise SystemExit("No evaluation case: run scripts/build_dataset.py first.")

    logger.info(
        "Performance bench on the gateway %s (%d cases, %d repetitions)",
        args.url,
        len(cases),
        args.repetitions,
    )
    results = benchmark(
        url=args.url,
        model=args.model,
        cases=cases,
        api_key=args.api_key,
        concurrencies=tuple(args.concurrency),
        repetitions=args.repetitions,
        target="gateway",
    )

    engine_measures = None
    engine_version = None
    if args.engine_url:
        engine_version = _engine_version(args.engine_url, args.vllm_key)
        logger.info("vLLM engine %s", engine_version or "of unknown version")
        logger.info("Decomposition: cost of the engine alone on %s", args.engine_url)
        engine_measures = benchmark(
            url=args.engine_url,
            model=args.model,
            cases=cases,
            api_key=args.vllm_key,
            concurrencies=tuple(args.concurrency),
            repetitions=args.repetitions,
            target="engine",
        )

    destination = PATHS.reports / "benchmark_endpoint.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "endpoint": args.url,
                "target": "gateway (POST /triage)",
                "model": args.model,
                "measures": results,
                "engine_endpoint": args.engine_url,
                "engine_version": engine_version,
                "engine_measures": engine_measures,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Measurements written → %s", destination)

    _show("Latency and throughput of the gateway (POST /triage)", results)
    if engine_measures:
        _show("Decomposition: cost of the engine alone (vLLM)", engine_measures)


def _engine_version(url: str, api_key: str | None) -> str | None:
    """Ask the engine which version it is running.

    A latency only holds for a given engine. The stack pins its image by digest, but that image
    is overridable — vLLM's V1 engine does not start under WSL 2 — and the published protocol
    must give the version actually measured, not the one the compose file announces.
    """
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = httpx.get(f"{url}/version", headers=headers, timeout=10)
    response.raise_for_status()
    return response.json().get("version")


def _show(title: str, results: dict) -> None:
    """Print a table of measurements to the console."""
    print(f"\n=== {title} ===")
    header = (
        f"{'load':12s} | {'p50 (ms)':>9s} | {'p95 (ms)':>9s} | {'req/s':>6s} | {'gateway':>10s}"
    )
    print(header)
    print("-" * len(header))
    for name, measures in results.items():
        overhead = measures.get("gateway_overhead_ms")
        print(
            f"{name:12s} | {measures['p50_ms']:>9.0f} | {measures['p95_ms']:>9.0f} | "
            f"{measures['requests_per_s']:>6.2f} | "
            f"{(f'{overhead:.0f} ms' if overhead is not None else '-'):>10s}"
        )


if __name__ == "__main__":
    main()
