"""What it takes to start the product, for real, on this machine.

The other two tiers import the package. This one does not: it starts the gateway with
``uvicorn`` in its own process, puts a stand-in inference engine in front of it in a second
one, and talks to both over HTTP. Everything the in-process tiers cannot see is exactly what
lives between those processes — the startup that refuses to serve without a key, the header the
gateway presents to the engine, the audit file that has to land on the disk of whoever runs the
container.

No GPU and no weights: the stand-in engine answers the OpenAI-compatible route vLLM serves,
with a fixed answer. What is under test is the service, not the model.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from clinical_triage.config import PATHS

#: The stand-in engine, written to a temporary file and started as its own process. It is kept
#: here rather than in a module of the tier so that it stays what it is — a fixture — and so
#: that nothing imports it by accident.
STUB_ENGINE = '''
"""Stand-in for the vLLM server: the two routes the gateway actually calls."""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ANSWER = os.environ["STUB_ANSWER"]
EXPECTED_KEY = os.environ["STUB_KEY"]
SEEN = Path(os.environ["STUB_SEEN"])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _reply(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self):
        return self.headers.get("Authorization") == f"Bearer {EXPECTED_KEY}"

    def do_GET(self):
        if not self._authorised():
            self._reply(401, {"error": "no key presented"})
        elif self.path == "/v1/models":
            self._reply(200, {"data": [{"id": "qwen3-1.7b-clinical-triage"}]})
        else:
            self._reply(404, {"error": "unknown route"})

    def do_POST(self):
        if not self._authorised():
            self._reply(401, {"error": "no key presented"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        with SEEN.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(request, ensure_ascii=False) + "\\n")
        self._reply(200, {"choices": [{"text": ANSWER}]})

    def log_message(self, *args):
        """Silence: the server's own log would drown the test output."""


HTTPServer(("127.0.0.1", int(os.environ["STUB_PORT"])), Handler).serve_forever()
'''

#: What the stand-in engine answers, in the shape the model produces. French, like the service.
ENGINE_ANSWER = (
    "Niveau de priorité : URGENCE_VITALE\n"
    "Justification : Douleur thoracique avec sueurs chez un patient à risque.\n"
    "Recommandation : Prise en charge immédiate et appel du 15 (SAMU)."
)

SERVICE_KEY = "system-tier-key"

#: A cold start imports FastAPI, the triage rule and the clinical catalogue.
STARTUP_TIMEOUT = 90.0


def free_port() -> int:
    """Ask the operating system for a port nobody is using."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def gateway_command(port: int) -> list[str]:
    """The command the container runs, pointed at a port of this machine."""
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "clinical_triage.serving.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]


def wait_until_answering(url: str, process: subprocess.Popen, output: Path) -> None:
    """Poll until the process answers, or say what it printed before giving up."""
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{url} died during startup (code {process.returncode}):\n"
                + output.read_text(encoding="utf-8", errors="replace")
            )
        try:
            httpx.get(url, timeout=2.0)
        except httpx.HTTPError:
            time.sleep(0.25)
        else:
            return
    process.kill()
    raise RuntimeError(
        f"{url} did not answer within {STARTUP_TIMEOUT:.0f} s:\n"
        + output.read_text(encoding="utf-8", errors="replace")
    )


@dataclass(frozen=True)
class Service:
    """A gateway running in its own process, and what it needs to be talked to."""

    url: str
    key: str
    #: Address of the stand-in engine, so that a test can question it directly.
    engine_url: str
    audit_log: Path
    #: One JSON line per call the gateway made to the engine.
    engine_calls: Path

    def headers(self, key: str | None = None) -> dict[str, str]:
        return {"X-API-Key": self.key if key is None else key}

    def traces(self) -> list[dict]:
        import json

        if not self.audit_log.exists():
            return []
        lines = self.audit_log.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(line) for line in lines if line]

    def prompts(self) -> list[str]:
        import json

        if not self.engine_calls.exists():
            return []
        lines = self.engine_calls.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(line)["prompt"] for line in lines if line]


@pytest.fixture(scope="session")
def start_service(tmp_path_factory):
    """Start a gateway, and the engine it talks to, both outside this process.

    The engine demands the service key: a gateway that failed to present it would get a 401 and
    answer 503, which is what makes that header verifiable from outside.
    """
    started: list[subprocess.Popen] = []

    def launch(**overrides: str) -> Service:
        workspace = tmp_path_factory.mktemp("service")
        engine_calls = workspace / "engine-calls.jsonl"
        audit_log = workspace / "audit.jsonl"
        engine_port, gateway_port = free_port(), free_port()

        stub = workspace / "stub_engine.py"
        stub.write_text(STUB_ENGINE, encoding="utf-8")
        engine_output = workspace / "engine.log"
        with engine_output.open("w", encoding="utf-8") as handle:
            engine = subprocess.Popen(  # fixed argument list, no shell
                [sys.executable, str(stub)],
                env=os.environ
                | {
                    "STUB_PORT": str(engine_port),
                    "STUB_ANSWER": ENGINE_ANSWER,
                    "STUB_KEY": SERVICE_KEY,
                    "STUB_SEEN": str(engine_calls),
                    "PYTHONIOENCODING": "utf-8",
                },
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        started.append(engine)
        wait_until_answering(f"http://127.0.0.1:{engine_port}/v1/models", engine, engine_output)

        gateway_output = workspace / "gateway.log"
        with gateway_output.open("w", encoding="utf-8") as handle:
            gateway = subprocess.Popen(  # fixed argument list, no shell
                gateway_command(gateway_port),
                cwd=str(PATHS.root),
                env=os.environ
                | {
                    "PYTHONPATH": str(PATHS.root / "src"),
                    "PYTHONIOENCODING": "utf-8",
                    "TRIAGE_BACKEND": "vllm",
                    "TRIAGE_API_KEY": SERVICE_KEY,
                    "TRIAGE_VLLM_URL": f"http://127.0.0.1:{engine_port}",
                    "TRIAGE_AUDIT_LOG": str(audit_log),
                    "TRIAGE_RATE_LIMIT": "60",
                }
                | overrides,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        started.append(gateway)
        url = f"http://127.0.0.1:{gateway_port}"
        wait_until_answering(f"{url}/health", gateway, gateway_output)
        return Service(
            url=url,
            key=overrides.get("TRIAGE_API_KEY", SERVICE_KEY),
            engine_url=f"http://127.0.0.1:{engine_port}",
            audit_log=audit_log,
            engine_calls=engine_calls,
        )

    yield launch

    for process in reversed(started):
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - a hung child
            process.kill()


@pytest.fixture(scope="session")
def service(start_service) -> Service:
    """The gateway the read-only tests share, started once."""
    return start_service()


@dataclass(frozen=True)
class Launch:
    """What it takes to start the gateway once, and how long to give it."""

    command: list[str]
    cwd: str
    environment: dict[str, str]
    timeout: float = STARTUP_TIMEOUT


@pytest.fixture
def gateway_without_a_key(tmp_path) -> Launch:
    """The gateway, with nothing that would let it serve.

    Nothing of the security configuration is inherited from the developer's environment: the
    two variables that decide whether the service may serve are removed here, so that a ``.env``
    or an exported key cannot turn this case green.
    """
    environment = os.environ | {
        "PYTHONPATH": str(PATHS.root / "src"),
        "PYTHONIOENCODING": "utf-8",
        "TRIAGE_BACKEND": "vllm",
        "TRIAGE_AUDIT_LOG": str(tmp_path / "audit.jsonl"),
    }
    for name in ("TRIAGE_API_KEY", "TRIAGE_ALLOW_ANONYMOUS"):
        environment.pop(name, None)
    return Launch(
        command=gateway_command(free_port()),
        cwd=str(PATHS.root),
        environment=environment,
    )
