"""Deploying the demonstration endpoint on Modal.

Two applications, exactly like the local stack of ``docker-compose.yml``:

- **``Engine``** — a GPU container serving the merged model with vLLM, through its
  OpenAI-compatible API. It shuts down on its own after a few minutes without a request, and
  comes back up on the next one;
- **``gateway``** — the same FastAPI application as the Docker image, exposed as is. No GPU, no
  torch, no weights.

The separation is not a Modal-specific choice: it is the project's own. Nothing is rewritten,
only rewired.

## Stopped by default

Its resting state is stopped: deployed, verified, then stopped, it runs no container and bills
nothing. A demonstration starts it, then stops it. Only the owner of the account can do either —
from their machine with their Modal token, or from the CD workflow, whose deployment job waits
for their approval (see ``.github/workflows/cd.yml``).

While it runs, two doors stay shut. The engine sits behind Modal's proxy authentication: a request
lacking the workspace's proxy token is turned away before a container starts, so knowing the
engine's address is worth nothing. The gateway's public probe, ``/health``, touches nothing
but the gateway; ``/ready``, which questions the engine, asks for the service key.

## Why the weights come from the Hub, and are not baked in

The model lives on the Hugging Face Hub, tagged by version. The engine reads it at a **specific
revision**: the deployment stays reproducible even if the default branch moves afterwards. The
cache sits on a persistent volume, otherwise every cold start re-downloads four gigabytes.

## Deployment

    modal workspace proxy-tokens create          # once: the gateway's pass to the engine
    modal secret create clinical-triage TRIAGE_API_KEY=... HF_TOKEN=... \\
        TRIAGE_ENGINE_PROXY_KEY=wk-... TRIAGE_ENGINE_PROXY_SECRET=ws-...
    modal deploy infra/modal_app.py

The command prints two addresses. Put the engine's into the secret, then deploy again:

    modal secret create clinical-triage --force ... TRIAGE_VLLM_URL=https://...engine.modal.run
    modal deploy infra/modal_app.py

The gateway receives the engine's address and its pass by configuration, as it would with any
host, and not through a Modal-specific API: the same code runs locally, on Modal, or elsewhere.

    modal app stop clinical-triage

stops everything, and is the only command that does: a container stopped by hand is started
again by Modal.

## The audit log

The gateway writes its audit log to the ``clinical-triage-audit`` volume, which survives the
container and every redeployment:

    modal volume get clinical-triage-audit / ./audit
"""

import os
from pathlib import Path

import modal

# Revision published by the `model` job of `.github/workflows/cd.yml`, which passes it in the
# environment at deployment time. Without it, the shipped version is pinned: the deployment stays
# replayable identically.
# Two `or` that look like shortcuts and are not, for the reason `config.hub_namespace` sets
# out: what GitHub hands over for a variable it does not have is an empty string. Without them
# the Hub would be asked for an identifier beginning with a slash.
ACCOUNT = os.environ.get("HF_NAMESPACE") or "BenoitJT-GIRARD"
MODEL_ID = os.environ.get("TRIAGE_MODEL_ID") or f"{ACCOUNT}/qwen3-1.7b-clinical-triage"
REVISION = os.environ.get("TRIAGE_MODEL_REVISION") or "model-v2.0.0"

VLLM_PORT = 8000
MINUTE = 60

app = modal.App("clinical-triage")

# The secret carries the service key, the Hugging Face token — needed even for a public
# repository, without which the download falls into the anonymous quota shared per IP address —,
# the proxy token the gateway presents to the engine, and the engine's address.
secret = modal.Secret.from_name("clinical-triage")

hf_cache = modal.Volume.from_name("clinical-triage-cache-hf", create_if_missing=True)
vllm_cache = modal.Volume.from_name("clinical-triage-cache-vllm", create_if_missing=True)
# The FlashInfer kernels compiled on the engine's first start: without this volume, every cold
# start would compile them again.
flashinfer_cache = modal.Volume.from_name(
    "clinical-triage-cache-flashinfer", create_if_missing=True
)
# A container's disk goes with it, and the gateway shuts down after five minutes without a
# request: the audit log lives on a volume. Modal commits its writes in the background and when
# the container stops.
audit_log = modal.Volume.from_name("clinical-triage-audit", create_if_missing=True)
GATEWAY_VOLUMES = {"/audit": audit_log}
GATEWAY_ENVIRONMENT = {"TRIAGE_AUDIT_DIR": "/audit"}

# Environment baked into the engine image. Both model references are written there, not read by
# the container: it reimports this file at startup, in an environment that holds nothing of the
# deployment's. What is not baked in here would be lost.
ENGINE_ENVIRONMENT = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_HOME": "/cache/hf",
    "TRIAGE_MODEL_ID": MODEL_ID,
    "TRIAGE_MODEL_REVISION": REVISION,
}

# vLLM compiles FlashInfer kernels when the engine starts, which takes the CUDA compiler: on a
# bare Debian image the engine stops on "Could not find nvcc". The base image is therefore
# NVIDIA's development image, in the CUDA version of the torch that vLLM 0.29 requires.
engine_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .pip_install("vllm==0.29.0", "huggingface_hub[hf_transfer]>=1.0")
    .env(ENGINE_ENVIRONMENT)
)

gateway_image = (
    modal.Image.debian_slim(python_version="3.12")
    # Path anchored on this file: `modal deploy` runs from the repository root, not from
    # `infra/`. The list is pinned by digest, and pip refuses anything else.
    .pip_install_from_requirements(str(Path(__file__).parent / "requirements-api.txt"))
    .add_local_python_source("clinical_triage")
)


@app.server(
    image=engine_image,
    gpu="L4",
    # One GPU whatever the load: a burst of requests must not light several cards billed by the
    # hour. vLLM serves simultaneous requests in batches, inside one container.
    max_containers=1,
    # Fifteen minutes without a request and the container shuts down: on a credit account, a
    # demonstration left running costs the rest of the month.
    scaledown_window=15 * MINUTE,
    startup_timeout=10 * MINUTE,
    volumes={
        "/cache/hf": hf_cache,
        "/root/.cache/vllm": vllm_cache,
        "/root/.cache/flashinfer": flashinfer_cache,
    },
    secrets=[secret],
    port=VLLM_PORT,
)
class Engine:
    """vLLM server serving the merged triage model, behind Modal's proxy authentication."""

    @modal.enter()
    def start(self):
        # A fixed command, without a shell: only the model's identifier, its revision and the key
        # come from the environment, baked into the image or read from the secret. `vllm` is
        # looked up on the image's PATH, where pip installed it.
        import subprocess  # nosec B404

        command = [
            "vllm",
            "serve",
            os.environ["TRIAGE_MODEL_ID"],
            "--revision",
            os.environ["TRIAGE_MODEL_REVISION"],
            "--served-model-name",
            "qwen3-1.7b-clinical-triage",
            "--host",
            # Every interface of the container: that is how Modal's proxy reaches the server.
            "0.0.0.0",  # nosec B104
            "--port",
            str(VLLM_PORT),
            "--dtype",
            "bfloat16",
            # The window the gateway gives itself: about 224 tokens of system prompt, at most 580
            # for the patient narrative — which the gateway bounds to the longest narrative of
            # the training set, 800 characters — and 220 for the answer, that is 1024.
            "--max-model-len",
            "1024",
            "--gpu-memory-utilization",
            "0.85",
            # The key the gateway presents, behind the proxy token: a second line, should the
            # proxy authentication ever be lifted.
            "--api-key",
            os.environ["TRIAGE_API_KEY"],
        ]
        # The key does not go to the log: the command is printed without its value.
        print(" ".join(command[:-1] + ["<key from the secret>"]), flush=True)
        self.process = subprocess.Popen(command)  # nosec B603

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.function(
    image=gateway_image,
    secrets=[secret],
    volumes=GATEWAY_VOLUMES,
    env=GATEWAY_ENVIRONMENT,
    # One container, hence one writer of the audit log: two containers appending to the same
    # file of a volume overwrite each other when the volume commits. Fifty simultaneous requests
    # in one container are more than this service needs.
    max_containers=1,
    # Two cores: anonymisation by Presidio and spaCy runs on every request, and the default share
    # of a core made it slower than the rest of the gateway.
    cpu=2.0,
    scaledown_window=5 * MINUTE,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def gateway():
    """Expose the project's triage API, unchanged."""
    from clinical_triage.serving.api import app as api

    return api
