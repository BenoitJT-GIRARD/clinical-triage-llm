"""The triage service API.

Three entry points: the health probe, the adaptive questionnaire and the triage itself. Two
interchangeable inference engines behind one contract:

- ``vllm`` — the deployment mode. The API is a thin gateway, torch-free, that delegates
  generation to a vLLM server;
- ``transformers`` — the demonstration mode. The model is loaded in-process, which lets the
  demonstration run on a machine without vLLM.

The operational precautions are gathered at startup rather than scattered across the entry
points:

- **the API key is mandatory.** The service refuses to start without one, unless the operator
  explicitly asks for open mode. An authentication that disables itself when a variable is
  missing is not an authentication;
- **key comparison is constant-time**, so the key cannot leak byte by byte through response
  timing;
- **throughput is capped per caller**, because every request occupies a GPU;
- **the health probe really questions the engine.** A probe that always answers "ok" triggers
  no restart the day inference falls over.

Clinical text — the justification and the recommendation — stays in French, because that is
what the model produces and what the staff reads. HTTP error details are in English: they are
read by whoever integrates the service, not by a nurse.
"""

from __future__ import annotations

import os
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from clinical_triage import __version__
from clinical_triage.config import PATHS, SERVING, TRIAGE
from clinical_triage.data.triage_rules import classify, explain
from clinical_triage.prompts import (
    completion_payload,
    extract_level,
    parse_response,
    truncate_to_answer,
)
from clinical_triage.serving.audit import (
    check_log_is_writable,
    new_request_id,
    record_interaction,
)
from clinical_triage.serving.questionnaire import compile_symptoms, next_question
from clinical_triage.serving.schemas import (
    HealthReply,
    QuestionnaireReply,
    QuestionnaireRequest,
    TriageReply,
    TriageRequest,
)
from clinical_triage.utils import get_logger, path_for_log

logger = get_logger("api")


@dataclass(frozen=True)
class Settings:
    """Service configuration, read once at startup."""

    backend: str
    base_model: str
    adapter_dir: str
    model_version: str
    api_key: str
    allow_anonymous: bool
    vllm_url: str
    vllm_model: str
    vllm_api_key: str
    requests_per_minute: int

    @property
    def vllm_headers(self) -> dict[str, str]:
        """Headers sent to the inference engine.

        When the engine is reachable other than through the loopback — which is the case on an
        on-demand host — it must ask for a key, otherwise it offers free GPU inference, with no
        quota and no audit trail, to whoever finds its address. vLLM can do that with
        ``--api-key``; the gateway presents it here.
        """
        return {"Authorization": f"Bearer {self.vllm_api_key}"} if self.vllm_api_key else {}

    @staticmethod
    def from_env() -> Settings:
        """Build the configuration from the environment variables."""
        return Settings(
            backend=os.getenv("TRIAGE_BACKEND", "vllm"),
            base_model=os.getenv("TRIAGE_BASE_MODEL", str(PATHS.sft_merged)),
            adapter_dir=os.getenv("TRIAGE_ADAPTER_DIR", str(PATHS.dpo_adapter)),
            model_version=os.getenv("TRIAGE_MODEL_VERSION", SERVING.served_model_name),
            api_key=os.getenv("TRIAGE_API_KEY", ""),
            allow_anonymous=os.getenv("TRIAGE_ALLOW_ANONYMOUS", "").lower() == "true",
            vllm_url=os.getenv("TRIAGE_VLLM_URL", "http://localhost:8000"),
            vllm_model=os.getenv("TRIAGE_VLLM_MODEL", SERVING.served_model_name),
            # Failing a key of its own, the engine is presented with the service's: on the
            # local stack the engine asks for none and the header is ignored, on a host it asks
            # and the gateway has one.
            vllm_api_key=os.getenv("TRIAGE_VLLM_API_KEY") or os.getenv("TRIAGE_API_KEY", ""),
            requests_per_minute=int(os.getenv("TRIAGE_RATE_LIMIT", "60")),
        )


@dataclass
class RateLimiter:
    """Per-caller rate limit, over a sliding one-minute window."""

    requests_per_minute: int
    _history: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))
    _last_sweep: float = field(default_factory=time.monotonic)

    def _forget_silent_callers(self, now: float) -> None:
        """Drop from tracking the callers that have asked nothing for a minute.

        Without this, the dictionary keeps one entry per caller seen since startup. In open
        mode, where the key is the caller's address, it grows without bound: a counter meant to
        protect the service would end up straining it. The sweep costs once a minute.
        """
        if now - self._last_sweep < 60:
            return
        self._last_sweep = now
        silent = [
            caller
            for caller, recent in self._history.items()
            if not recent or now - recent[-1] > 60
        ]
        for caller in silent:
            del self._history[caller]

    def allows(self, caller: str) -> bool:
        """Record a request and say whether it stays inside the quota."""
        now = time.monotonic()
        self._forget_silent_callers(now)
        recent = self._history[caller]
        while recent and now - recent[0] > 60:
            recent.popleft()
        if len(recent) >= self.requests_per_minute:
            return False
        recent.append(now)
        return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Read the configuration, check security and load the inference engine."""
    settings = Settings.from_env()
    if not settings.api_key and not settings.allow_anonymous:
        raise RuntimeError(
            "TRIAGE_API_KEY is not set. Provide a key, or ask explicitly for open mode with "
            "TRIAGE_ALLOW_ANONYMOUS=true (local demonstration only)."
        )
    if not settings.api_key:
        logger.warning("Service started without an API key: open mode, demonstration only.")

    # The audit log is opened here, empty, so that its unavailability stops startup rather than
    # the first request.
    logger.info("Audit log: %s", path_for_log(check_log_is_writable()))

    app.state.settings = settings
    app.state.limiter = RateLimiter(settings.requests_per_minute)
    app.state.agent = None
    app.state.loaded_model = settings.model_version

    if settings.backend == "transformers":
        from clinical_triage.inference import TriageAgent

        base = settings.base_model
        if not os.path.isdir(base):
            raise RuntimeError(
                f"Model not found: {base}. Run scripts/merge_adapter.py, or point "
                "TRIAGE_BASE_MODEL at a model that exists."
            )
        adapter = settings.adapter_dir if os.path.isdir(settings.adapter_dir) else None
        if adapter is None:
            # The service stays usable, but then serves the supervised model alone: the audit
            # log must carry that, otherwise the answers would be attributed to a model that
            # did not produce them.
            logger.warning(
                "Adapter not found (%s): the model is served without alignment.",
                path_for_log(settings.adapter_dir),
            )
        app.state.agent = TriageAgent(adapter_dir=adapter, base_model=base)
        app.state.loaded_model = app.state.agent.description
        logger.info("transformers engine ready: %s", app.state.loaded_model)
    else:
        logger.info("vLLM engine: generation delegated to %s", settings.vllm_url)

    yield
    app.state.agent = None


app = FastAPI(
    title="Emergency triage assistant",
    description=(
        "Decision support for emergency triage. The level proposed does not replace a "
        "clinician's assessment; on any life-threatening sign, call the emergency services."
    ),
    # The version the OpenAPI contract announces is the package's. Written here, it would
    # stay frozen at the next version bump, and an integrator would read a number that no
    # longer designates the service answering them.
    version=__version__,
    lifespan=lifespan,
)


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Check the API key in constant time, then apply the rate limit."""
    settings: Settings = request.app.state.settings
    if settings.api_key:
        provided = x_api_key or ""
        if not secrets.compare_digest(provided, settings.api_key):
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid API key. Set the X-API-Key header.",
            )
    # The header only serves as a counting identity once it has just been verified. In open
    # mode nobody verifies it: a caller changing it on every request would get a fresh bucket
    # each time, and the quota — the only guard rail left in that mode — would count nothing.
    address = request.client.host if request.client else "unknown"
    caller = x_api_key if (settings.api_key and x_api_key) else address
    if not request.app.state.limiter.allows(caller):
        # Announcing the quota saves the integrator from discovering it by trial and error.
        raise HTTPException(
            status_code=429,
            detail=(
                f"Quota of {settings.requests_per_minute} requests per minute exceeded. "
                "Try again in a minute."
            ),
        )


# Number of characters beyond which the gateway bounds the description.
#
# The vLLM engine serves a 1,024-token window, of which 220 are reserved for the answer and
# about 224 for the system prompt: 580 tokens are left for the patient's narrative. On clinical
# French a token is worth about 2.7 characters; 1,500 leaves a comfortable margin under the
# bound.
#
# The gateway carries no tokenizer — that is the whole point of a thin gateway — hence this
# approximation in characters. The local engine bounds in tokens, because it has one.
MAX_CHARACTERS = 1500


def bound_in_characters(text: str) -> tuple[str, bool]:
    """Bound a description that is too long, and say whether it was."""
    if len(text) <= MAX_CHARACTERS:
        return text, False
    return text[:MAX_CHARACTERS], True


def _generate_transformers(request: Request, symptoms: str) -> tuple[str, str | None, float, bool]:
    """Generate the answer with the local engine."""
    answer = request.app.state.agent.generate(symptoms)
    return answer.text, answer.level, answer.latency_ms, answer.description_truncated


def _generate_vllm(request: Request, symptoms: str) -> tuple[str, str | None, float, bool]:
    """Generate the answer through the vLLM server, over its OpenAI-compatible API."""
    import httpx

    settings: Settings = request.app.state.settings
    symptoms, truncated = bound_in_characters(symptoms)
    payload = completion_payload(symptoms, settings.vllm_model)
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{settings.vllm_url}/v1/completions",
                json=payload,
                headers=settings.vllm_headers,
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["text"]
    except httpx.HTTPError as error:
        # The technical cause goes to the log, not to the caller: it contains the internal
        # address of the inference server, which nothing justifies exposing. The caller gets
        # what they can act on — retry.
        logger.error("Call to the vLLM server failed: %s", error)
        raise HTTPException(
            status_code=503,
            detail=(
                "The inference engine is not responding. Triage is temporarily unavailable; "
                "try again in a few moments."
            ),
        ) from error
    latency = (time.perf_counter() - started) * 1000
    text = truncate_to_answer(text)
    return text, extract_level(text), latency, truncated


@app.get("/health", response_model=HealthReply)
def health(request: Request) -> HealthReply:
    """Availability probe: really questions the inference engine."""
    settings: Settings = request.app.state.settings
    if settings.backend == "transformers":
        available = request.app.state.agent is not None
        detail = None if available else "The model is not loaded."
    else:
        import httpx

        try:
            with httpx.Client(timeout=3) as client:
                client.get(
                    f"{settings.vllm_url}/v1/models", headers=settings.vllm_headers
                ).raise_for_status()
            available, detail = True, None
        except Exception as error:  # noqa: BLE001 - any engine failure must be reported
            # httpx's message contains the address it queried — on a managed host, that is the
            # engine's address. The probe requires neither key nor quota: it must therefore say
            # that there is a failure, never where. The technical cause goes to the log, as it
            # does for generation.
            logger.error("vLLM probe failed: %s", error)
            available, detail = False, "The inference engine is not responding."
    return HealthReply(
        status="ok" if available else "degraded",
        backend=settings.backend,
        model_loaded=available,
        model_version=request.app.state.loaded_model,
        detail=detail,
    )


@app.post(
    "/questionnaire/next",
    response_model=QuestionnaireReply,
    dependencies=[Depends(require_api_key)],
)
def questionnaire_next(req: QuestionnaireRequest) -> QuestionnaireReply:
    """Return the next question of the adaptive questionnaire, or the end of collection."""
    step = next_question(req.chief_complaint, req.answers)
    return QuestionnaireReply(
        next_question_id=step.identifier,
        next_question=step.text,
        theme=step.theme,
        finished=step.finished,
        compiled_symptoms=compile_symptoms(req.chief_complaint, req.answers),
    )


@app.post("/triage", response_model=TriageReply, dependencies=[Depends(require_api_key)])
def triage(req: TriageRequest, request: Request) -> TriageReply:
    """Assess the priority level, compare it to the explicit rule and trace the interaction."""
    symptoms = req.symptoms.strip()
    if req.patient_age is not None:
        symptoms = f"Patient de {req.patient_age} ans. {symptoms}"

    generate = (
        _generate_transformers
        if request.app.state.settings.backend == "transformers"
        else _generate_vllm
    )
    text, level, latency, description_truncated = generate(request, symptoms)

    rule_level = classify(symptoms)
    rule_reasons = explain(symptoms)
    parts = parse_response(text)

    identifier = new_request_id()
    record_interaction(
        request_id=identifier,
        symptoms=symptoms,
        level=level,
        answer=text,
        latency_ms=latency,
        model=request.app.state.loaded_model,
        engine=request.app.state.settings.backend,
        rule_level=rule_level,
        rule_reasons=rule_reasons,
        description_truncated=description_truncated,
    )

    return TriageReply(
        level=level,
        level_label=TRIAGE.labels_fr.get(level) if level else None,
        justification=parts["justification"],
        recommendation=parts["recommendation"],
        rule_level=rule_level,
        rule_reasons=rule_reasons,
        agreement=level == rule_level,
        raw_response=text,
        description_truncated=description_truncated,
        latency_ms=round(latency, 1),
        request_id=identifier,
        model_version=request.app.state.loaded_model,
    )
