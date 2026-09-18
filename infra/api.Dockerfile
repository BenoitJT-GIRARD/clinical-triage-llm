# Triage API gateway: FastAPI in front of a vLLM server.
#
# The image contains neither torch nor model weights. It validates requests, runs the adaptive
# questionnaire, applies the explicit control rule, anonymises and logs, then delegates
# generation to vLLM. That separation is what lets the gateway be redeployed without
# redeploying the model, and the other way round.
#
# The base image is pinned by digest and the Python dependencies down to the last transitive
# one: two builds of the same revision produce the same image.
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

# Unbuffered output: the container's traces reach the log collector in real time.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Pinned dependencies, installed before the code: the layer is reused as long as the file does
# not change. That file is generated from `requirements-api.in`, which carries the constraints
# and the procedure.
COPY infra/requirements-api.txt /app/infra/requirements-api.txt
RUN pip install --no-cache-dir -r /app/infra/requirements-api.txt

# The package code.
COPY src/clinical_triage /app/src/clinical_triage
ENV PYTHONPATH=/app/src

# Unprivileged user, owner of the audit log directory.
RUN useradd --create-home --uid 10001 triage \
    && mkdir -p /app/var/logs \
    && chown -R triage:triage /app
USER triage

ENV TRIAGE_BACKEND=vllm \
    TRIAGE_VLLM_URL=http://vllm:8000 \
    TRIAGE_AUDIT_LOG=/app/var/logs/audit_triage.jsonl

EXPOSE 8080

# The probe is green only when the inference engine answers: the gateway reports itself as
# "degraded" for as long as vLLM is not serving, although it has started perfectly well.
# The grace period covers exactly that — uvicorn starting, then the engine — without
# counting a failure.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=60s \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8080/health', timeout=4).json()['status']=='ok' else 1)"

CMD ["uvicorn", "clinical_triage.serving.api:app", "--host", "0.0.0.0", "--port", "8080"]
