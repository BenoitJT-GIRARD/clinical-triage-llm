"""Deploying the demonstration endpoint on Modal.

Two applications, exactly like the local stack of ``docker-compose.yml``:

- **``Engine``** — a GPU container serving the merged model with vLLM, through its
  OpenAI-compatible API. It shuts down on its own after a few minutes without a request, and
  comes back up on the next one;
- **``gateway``** — the same FastAPI application as the Docker image, exposed as is. No GPU, no
  torch, no weights.

The separation is not a Modal-specific choice: it is the project's own. Nothing is rewritten,
only rewired.

## Why the weights come from the Hub, and are not baked in

The model already lives on the Hugging Face Hub, tagged by the continuous-deployment pipeline.
The container reads it at a **specific revision**: the demonstration stays reproducible even if
the default branch moves afterwards. The cache is mounted on a persistent volume, otherwise every
cold start re-downloads three gigabytes — the difference between forty seconds and four minutes.

## Deployment

    modal secret create clinical-triage TRIAGE_API_KEY=... HF_TOKEN=...
    modal deploy infra/modal_app.py

The command prints two addresses. Put the engine's into the secret, then redeploy the gateway:

    modal secret create clinical-triage --force \\
        TRIAGE_API_KEY=... HF_TOKEN=... TRIAGE_VLLM_URL=https://...engine.modal.run
    modal deploy infra/modal_app.py

The detour through the secret is deliberate: the gateway receives the engine's address by
configuration, as it would with any host, and not through a Modal-specific API. The same code
therefore runs locally, on Modal, or elsewhere.
"""

import os
from pathlib import Path

import modal

# Revision published by the `model` job of `.github/workflows/cd.yml`, which passes it in the
# environment at deployment time. Without it, the shipped version is pinned: the demonstration
# stays replayable identically.
# Two `or` that look like shortcuts and are not, for the reason `config.hub_namespace` sets
# out: what GitHub hands over for a variable it does not have is an empty string. Without them
# the Hub would be asked for an identifier beginning with a slash.
ACCOUNT = os.environ.get("HF_NAMESPACE") or "BenoitJT-GIRARD"
MODEL_ID = os.environ.get("TRIAGE_MODEL_ID") or f"{ACCOUNT}/qwen3-1.7b-clinical-triage"
REVISION = os.environ.get("TRIAGE_MODEL_REVISION") or "model-v1.0.0"

VLLM_PORT = 8000
MINUTE = 60

app = modal.App("clinical-triage")

# The secret carries the service API key, the Hugging Face token — needed even for a public
# repository, without which the download falls into the anonymous quota shared per IP address —
# and the engine's address.
secret = modal.Secret.from_name("clinical-triage")

hf_cache = modal.Volume.from_name("clinical-triage-cache-hf", create_if_missing=True)
vllm_cache = modal.Volume.from_name("clinical-triage-cache-vllm", create_if_missing=True)

# Environment baked into the engine image. Both model references are written there, not read by
# the container: it reimports this file at startup, in an environment that holds nothing of the
# deployment's. What is not baked in here would be lost.
ENGINE_ENVIRONMENT = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_HOME": "/cache/hf",
    "TRIAGE_MODEL_ID": MODEL_ID,
    "TRIAGE_MODEL_REVISION": REVISION,
}

engine_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.29.0", "huggingface_hub[hf_transfer]>=1.0")
    .env(ENGINE_ENVIRONMENT)
)

gateway_image = (
    modal.Image.debian_slim(python_version="3.12")
    # Path anchored on this file: `modal deploy` runs from the repository root, not from
    # `infra/`.
    .pip_install_from_requirements(str(Path(__file__).parent / "requirements-api.txt"))
    .add_local_python_source("clinical_triage")
)


@app.server(
    image=engine_image,
    gpu="L4",
    # Fifteen minutes without a request and the container shuts down: on a credit account, a
    # demonstration left running costs the rest of the month.
    scaledown_window=15 * MINUTE,
    startup_timeout=10 * MINUTE,
    volumes={"/cache/hf": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[secret],
    port=VLLM_PORT,
    # The engine gets a public address, like the gateway: Modal exposes every server that way.
    # That is not an open door for all that — vLLM itself asks for the service key, which the
    # gateway presents to it. Without that, the address alone would buy free GPU inference, with
    # no quota, no anonymisation and no line in the audit log, at the account's expense.
    unauthenticated=True,
)
class Engine:
    """vLLM server serving the merged triage model."""

    @modal.enter()
    def start(self):
        import subprocess

        command = [
            "vllm",
            "serve",
            os.environ["TRIAGE_MODEL_ID"],
            "--revision",
            os.environ["TRIAGE_MODEL_REVISION"],
            "--served-model-name",
            "qwen3-1.7b-clinical-triage",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--dtype",
            "bfloat16",
            # This is the window the gateway gives itself, and it splits it thus: about 224
            # tokens of system prompt, at most 580 for the patient narrative — which
            # `MAX_CHARACTERS`, in `serving/api.py`, bounds to 1,500 characters — and 220
            # for the answer, that is 1024. A wider window would reserve attention cache
            # for nothing.
            "--max-model-len",
            "1024",
            "--gpu-memory-utilization",
            "0.85",
            # The key the gateway will present. It comes from the Modal secret, like the
            # gateway's: it is the same one.
            "--api-key",
            os.environ["TRIAGE_API_KEY"],
        ]
        # The key does not go to the log: the command is printed without its value.
        print(" ".join(command[:-1] + ["<key from the secret>"]), flush=True)
        self.process = subprocess.Popen(command)

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.function(image=gateway_image, secrets=[secret], scaledown_window=5 * MINUTE)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def gateway():
    """Expose the project's triage API, unchanged."""
    from clinical_triage.serving.api import app as api

    return api
