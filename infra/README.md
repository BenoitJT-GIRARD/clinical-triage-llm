# Deploying the triage endpoint

The assistant is served by **vLLM** behind a **FastAPI** gateway. This page covers the three
situations: the local demonstration, the full container stack, and the cloud deployment.

## Shape of it

```
reception desk / hospital IS  ──HTTPS──▶  FastAPI gateway  ──HTTP──▶  vLLM server (GPU)
                                          authentication             merged model
                                          adaptive questionnaire     batching
                                          control rule               attention cache
                                          anonymisation and audit
```

The split is deliberate. The gateway is light, carrying neither torch nor weights, and redeploys
in seconds; the inference server updates without touching the business logic. The OpenAPI contract makes the
integration independent of both.

## 1. Preparing the model

vLLM serves a whole model, so the adapters have to be merged.

```bash
uv run python scripts/merge_adapter.py --adapter sft
uv run python scripts/train_dpo.py
uv run python scripts/merge_adapter.py --adapter dpo
# produces var/models/qwen3-1.7b-triage-dpo-merged
```

The exported tokenizer carries the project's dialogue template and `<|im_end|>` as the
end-of-sequence token. The generation configuration therefore travels with the model: any
inference engine stops in the right place, with nothing to set on the caller's side.

## 2. The full container stack

Requirements: a host with an NVIDIA GPU and `nvidia-container-toolkit`. On Windows, Docker
Desktop with the WSL 2 backend will do.

```bash
export TRIAGE_API_KEY="a-key-of-your-choosing"
docker compose -f infra/docker-compose.yml up --build
```

- gateway: <http://localhost:8080/docs>
- vLLM server: <http://localhost:8000/v1>

8000 and 8080 are common ports: if one of them is already taken on the machine, the stack refuses
to start. Both can be chosen without touching the compose file and without stopping anything
else:

```bash
TRIAGE_VLLM_PORT=8001 TRIAGE_API_PORT=8081 \
  docker compose -f infra/docker-compose.yml up --build
```

Only what is published on the host changes; inside the stack, the gateway reaches the engine on
its internal port, which does not move.

**If the repository sits on a cloud-synchronised disk**, the weights have to live outside it.
Docker Desktop does not mount a synchronisation virtual disk: the container starts, finds nothing
under `/models`, takes the path for a Hugging Face repository identifier and stops on `Repo id
must be in the form 'repo_name' or 'namespace/repo_name'`. The health probe then fails without
saying why.

```bash
cp -r var/models/qwen3-1.7b-triage-dpo-merged ~/.local/share/clinical-triage/models/
TRIAGE_MODELS_DIR=~/.local/share/clinical-triage/models \
  docker compose -f infra/docker-compose.yml up --build
```

**If the host is a Windows workstation**, the pinned engine will not start. vLLM's V1 engine
requires CUDA's unified virtual addressing; WSL 2 does not expose it, and the server stops on
`RuntimeError: UVA is not available` before loading any weights. The `VLLM_USE_V1=0` switch is of
no help: the V0 engine has been removed. An earlier version works, at the price of an engine that
is not the deployment's:

```bash
TRIAGE_VLLM_IMAGE=vllm/vllm-openai:v0.11.0 \
  docker compose -f infra/docker-compose.yml up --build
```

That is the only reason to override this variable. On a Linux host with
`nvidia-container-toolkit` the version pinned by digest works, and it is the one that should
serve: it freezes the inference engine under a given model and a given API contract.

```bash
curl -X POST http://localhost:8080/triage \
  -H 'Content-Type: application/json' -H "X-API-Key: $TRIAGE_API_KEY" \
  -d '{"symptoms": "Homme de 62 ans, douleur thoracique et sueurs depuis 20 minutes. TA 148/92, FC 102."}'
```

Compose holds the gateway back until vLLM reports itself healthy. Started together, the first
requests would fail during the minutes the weights take to load, and the gateway would report
itself green throughout.

Measuring the stack end to end, on the gateway — which is what the service actually delivers,
anonymisation and audit log included:

```bash
uv run python scripts/benchmark_endpoint.py --api-key "$TRIAGE_API_KEY"
```

Add `--engine-url http://localhost:8000` to measure the cost of the engine alone as well, which
the results present as a decomposition and not as the performance of the endpoint.

The bench holds a single API key, hence a single quota bucket: past the first concurrency level
the gateway answers it 429. Raise the quota for the duration of the measurement, without
touching the service value:

```bash
TRIAGE_RATE_LIMIT=6000 docker compose -f infra/docker-compose.yml up -d
```

## 3. A demonstration without a GPU and without Docker

vLLM does not run natively on Windows. For a local demonstration the gateway can load the model
itself with `transformers`, which is slower but enough to show the behaviour.

```powershell
$env:TRIAGE_BACKEND = "transformers"
$env:TRIAGE_BASE_MODEL = "var/models/qwen3-1.7b-triage-sft-merged"
$env:TRIAGE_ADAPTER_DIR = "var/models/qwen3-1.7b-triage-dpo"
$env:TRIAGE_API_KEY = "a-demonstration-key"
uv run uvicorn clinical_triage.serving.api:app --port 8080
```

The latencies of this mode are not comparable with the vLLM endpoint's, and must not be reported
as if they were.

## 4. Cloud deployment, on Modal

### Stopped by default, woken by its owner alone

[Modal](https://modal.com) allocates a GPU on demand, bills by the second and shuts a container
down when nothing asks for it. `modal_app.py` raises the same two pieces as the Docker stack of
this folder: a GPU container serving the merged model under vLLM, and the project's FastAPI
gateway, exposed as is. That is the project's own split, not an adaptation to the host, and not a
line of the service code changes.

The application's normal state is **stopped**. It is deployed and verified, then stopped: stopped,
it costs nothing and answers nothing. A demonstration starts it, and stops it afterwards. Three
actions exist, all through the `CD` workflow, dispatched by hand on a `model-v*` tag:

| Action | What it does |
|---|---|
| `verify` | deploys, waits for `/ready`, sends three fixed cases and requires the level the evaluation recorded for each, then stops |
| `start` | the same, then leaves the service up for a demonstration |
| `stop` | `modal app stop clinical-triage`, the only command that stops everything |

**Only the owner of the repository can run them**, on two paths:

1. **from GitHub.** Any account with write access can dispatch a workflow; none can deploy. The
   jobs that touch the Hub or Modal declare the `production` environment, which waits for the
   owner's approval and accepts `model-v*` tags only. The Modal and Hub tokens live in that
   environment, not in the repository: a job that was not approved cannot read them. A guard on
   `github.actor == github.repository_owner` doubles the whole;
2. **from the owner's workstation**, with their own Modal token, by the commands below.

While it runs, two doors stay shut:

- the **engine** sits behind Modal's proxy authentication. A request that does not carry the
  workspace's proxy token is refused before any container starts: the engine's address alone
  buys nothing. The gateway presents that token, and the vLLM key stays as a second line;
- the gateway's public probe, `/health`, answers whether the gateway is alive and touches
  nothing else. `/ready`, which questions the engine, asks for the service key.

One L4 GPU serves, and the engine shuts down after fifteen minutes without a request. Its image
is a CUDA development image, heavier than a runtime one, because FlashInfer compiles its kernels
at the first start and needs `nvcc` to do it; the compiled kernels, like the Hugging Face cache,
are kept on volumes, so that a cold start neither recompiles nor re-downloads.

### Deploying from the workstation

```bash
uv run modal setup                                  # authentication, opens the browser
uv run modal workspace proxy-tokens create          # once: the gateway's pass to the engine
uv run modal secret create clinical-triage \
    TRIAGE_API_KEY="a-key-of-your-choosing" \
    HF_TOKEN="hf_..." \
    TRIAGE_ENGINE_PROXY_KEY="wk-..." TRIAGE_ENGINE_PROXY_SECRET="ws-..."
uv run modal deploy infra/modal_app.py
```

Two addresses are printed. The engine's belongs in the secret; deploying again is what tells the
gateway where to send its requests:

```bash
uv run modal secret create clinical-triage --force ... \
    TRIAGE_VLLM_URL="https://<account>--clinical-triage-engine.modal.run"
uv run modal deploy infra/modal_app.py
uv run modal app stop clinical-triage --yes         # once the check is done
```

The engine loads the weights **from the Hub, at a tagged revision**: the deployment stays
replayable identically even if the default branch moves afterwards. The repository and the
revision are chosen through `HF_NAMESPACE`, `TRIAGE_MODEL_ID` and `TRIAGE_MODEL_REVISION`, set in
the shell at `modal deploy` time: `infra/modal_app.py` runs on the workstation and reads its
environment at import, before the project's package, and therefore `.env`, is loaded. Without
them, the deployment serves the model and the revision pinned in the file.

`modal setup` writes the account tokens into `~/.modal.toml`. In continuous integration, where
there is neither a browser nor a configuration file, `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`
replace them, from the `production` environment.

### Other hosts

**A managed vLLM container.** Publish the merged model on the Hub, create an inference endpoint at
the host on the model repository with an L4-class GPU and the vLLM container, then deploy the
gateway (the image the pipeline publishes to GHCR) on a container service without a GPU, with
`TRIAGE_VLLM_URL` pointing at the endpoint and `TRIAGE_API_KEY` as a secret.

**A virtual machine with a GPU.** Docker and `nvidia-container-toolkit`, the merged model copied
from `var/models/` or downloaded by vLLM from the Hub, `docker compose -f
infra/docker-compose.yml up -d`, and port 8080 behind a reverse proxy with a TLS certificate.

### Shipping a new model version

The weights and the dataset are uploaded from the training machine; they do not travel through
continuous integration. The **version tag** is what ties the two together.

```bash
uv run python scripts/build_hub_cards.py                  # render both cards, then commit them
uv run python scripts/publish_to_hub.py --what model      # refused unless these are the weights evaluated
uv run python scripts/publish_to_hub.py --what dataset
git tag model-v2.0.0 && git push origin model-v2.0.0
```

A dispatch of `cd.yml` on that tag, with the `verify` action, runs its `model` job: the release
gate, then the two cards, byte for byte, and the `model-v2.0.0` tag on both Hub repositories. The
deployment job then runs `modal deploy` with that revision, checks the served model against the
evaluated one, and stops. What arrived on the Hub is read back, card by card, digest by digest
and subset by subset:

```bash
uv run python scripts/publish_to_hub.py --what verify --tag model-v2.0.0
```

Pinning the revision instead of the default branch is what makes a deployment reproducible:
`main` designates content that changes at every publication, `model-v2.0.0` always designates the
same weights. Rolling back becomes trivial too: dispatch the workflow on the previous tag.

The two cards, [`hub/model-card.md`](hub/model-card.md) and
[`hub/dataset-card.md`](hub/dataset-card.md), are rendered by `scripts/build_hub_cards.py` from
the templates beside them and the committed results; continuous integration fails when either
falls behind those results.

### What the GitHub repository has to know

The `production` environment, with the owner as required reviewer and `model-v*` as the only
tags allowed to deploy, holds the secrets; the repository holds two variables.

| Name | Kind, and where | Role |
|---|---|---|
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | secrets of `production` | the owner's Modal token, from `modal token new` |
| `HF_TOKEN` | secret of `production` | publishing the cards and placing the tag on the Hub |
| `TRIAGE_API_KEY` | secret of `production` | the service key the checks present to `/ready` and `/triage` |
| `DEPLOY_ENDPOINT_URL` | repository variable | address of the gateway, checked after deployment |
| `HF_NAMESPACE` | repository variable | target Hugging Face account, when it is not the default |

The container's Hugging Face token and the engine's proxy token do not go through GitHub: they
live in the Modal secret `clinical-triage`, created above. GitHub needs to know how to deploy,
not what the service handles.

### Restoring the credentials

Nothing below survives a long pause on purpose: every token is deleted once the deployment
it served is stopped, and recreated in a few minutes when the service has to run again.

| Credential | Who issues it | How to get one | Where it goes |
|---|---|---|---|
| Modal account token | Modal | `uv run modal setup` on a workstation; `uv run modal token new` for continuous deployment | `~/.modal.toml`, or `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in `production` |
| Engine proxy token, `wk-…` and `ws-…` | Modal | `uv run modal workspace proxy-tokens create` | the Modal secret `clinical-triage` |
| `TRIAGE_API_KEY` | whoever runs the service | a random value of your own: `openssl rand -hex 32` prints one | the Modal secret, `.env` for the latency bench, and `production` |
| `HF_TOKEN` | Hugging Face | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), fine-grained: *Write access to contents/settings of all repos under your personal namespace* to publish, read access alone for the engine | `.env` to publish from the workstation, the Modal secret for the engine's download, and `production` for the `model` job |

A secret of the `production` environment is set with `gh secret set <NAME> --env production`,
which asks for the value without echoing it. Once the service is stopped again:
`uv run modal workspace proxy-tokens delete <id>` for the proxy token, and the token pages of
[Modal](https://modal.com/settings/tokens) and [Hugging Face](https://huggingface.co/settings/tokens)
for the other two.

## Configuration

Variables the gateway itself reads, from `.env` at the root or from the container's environment:

| Variable | Role | Default |
|---|---|---|
| `TRIAGE_API_KEY` | key expected in the `X-API-Key` header | **none — the service refuses to start without it** |
| `TRIAGE_ALLOW_ANONYMOUS` | explicitly allows open mode, for a demonstration | `false` |
| `TRIAGE_BACKEND` | `vllm` or `transformers` | `vllm` |
| `TRIAGE_VLLM_URL` | address of the inference server | `http://localhost:8000` |
| `TRIAGE_VLLM_MODEL` | name of the model vLLM serves | `qwen3-1.7b-clinical-triage` |
| `TRIAGE_VLLM_API_KEY` | key presented to the inference engine | failing that, `TRIAGE_API_KEY` |
| `TRIAGE_ENGINE_PROXY_KEY`, `TRIAGE_ENGINE_PROXY_SECRET` | token pair of the host's proxy authentication, presented to the engine | empty: none presented |
| `TRIAGE_BASE_MODEL` | model loaded in `transformers` mode | the merged SFT model |
| `TRIAGE_ADAPTER_DIR` | adapter applied in `transformers` mode | the DPO adapter |
| `TRIAGE_MODEL_VERSION` | version written to the audit log | `qwen3-1.7b-clinical-triage` |
| `TRIAGE_AUDIT_DIR` | directory of the audit log, one file per day | `var/logs/audit` |
| `TRIAGE_RATE_LIMIT` | requests per minute and per caller | `60` |

Variables the tools that raise the stack read, and not the service. They do not go through the
`.env` at the root — `HF_NAMESPACE` excepted, which `config.py` also reads for publication:
`docker compose -f infra/docker-compose.yml` looks for its environment file in `infra/`, and the
`modal` command loads `infra/modal_app.py` outside the process that reads `.env`. Both kinds are
set in the shell, as a prefix to the command or through `export`, as in the examples above.

| Variable | Read by | Role | Default |
|---|---|---|---|
| `TRIAGE_VLLM_PORT` | `docker compose` | engine port published on the host | `8000` |
| `TRIAGE_API_PORT` | `docker compose` | gateway port published on the host | `8080` |
| `TRIAGE_MODELS_DIR` | `docker compose` | weights folder on the host, mounted on `/models` | `../var/models` |
| `TRIAGE_VLLM_IMAGE` | `docker compose` | inference engine image | the version pinned by digest |
| `HF_NAMESPACE` | `modal deploy`, and `config.py` for publication | Hugging Face account the engine pulls the weights from | `BenoitJT-GIRARD` |
| `TRIAGE_MODEL_ID` | `modal deploy` | repository of the served model | `<HF_NAMESPACE>/qwen3-1.7b-clinical-triage` |
| `TRIAGE_MODEL_REVISION` | `modal deploy` | pinned revision of the weights | `model-v2.0.0` |

`docker compose` also interpolates `TRIAGE_API_KEY` and `TRIAGE_RATE_LIMIT` before passing them
to the container: on the container stack those two come from the shell as well. Without
`TRIAGE_API_KEY`, the stack stops before starting, and says so.

`.env.example` at the root lists the variables of the first table, along with the Hugging Face
token and account the publication scripts need and the cache and tracking locations. Copied to
`.env`, it is read by `config.py` — never overwriting a variable already set in the environment,
so that a host keeps control of its own secrets. The file is excluded from git and from the
Docker build context.

## Security

- **No key, no service.** Startup fails rather than falling back to an open port. Open mode
  exists for a local demonstration, and has to be asked for by name.
- **Constant-time comparison**, so that the key cannot be reconstructed from the response time.
- **Per-caller quota** over a sliding window: every request takes a GPU, and an endpoint without
  a quota is trivially saturable.
- **Secrets** outside the repository, as encrypted environment variables on the host's side.
- **The container** runs unprivileged, from a base image pinned by digest, with dependencies
  pinned down to the last transitive one and audited in continuous integration, without
  derogation.
- **The inference engine** is bound to the loopback in the container stack, and protected by a key
  as soon as it is reachable otherwise: it has neither quota nor audit log, and the gateway is
  what carries both.
- **TLS** is the reverse proxy's or the host's business.

## Running it

| Indicator | Alert threshold | Reaction |
|---|---|---|
| `/health` probe, the gateway | two consecutive failures | restart the container, raise an alert |
| `/ready` probe, the engine | two consecutive failures | check the engine, then the GPU |
| 95th-percentile latency | twice the reference | check the GPU load |
| Off-format answers | more than 2% over an hour | freeze the version, roll back |
| Model / rule disagreement | more than 25% over a day | clinical review of the sample |
| HTTP error rate | more than 1% | operations alert |

The **audit log** is mounted on a named volume called `audit`: without it, the traceability
required for medical audits would disappear at every redeployment. A host folder does not do —
Docker mounts it as root, and the service, which runs unprivileged, cannot write there; it then
refuses to start, and says so. The volume survives `docker compose down`; `docker compose down
-v` erases it.

```bash
docker compose -f infra/docker-compose.yml cp api:/app/var/logs/audit .
```

Plan for its rotation and its centralisation, and respect the retention period written into every
line.

The **rate of disagreement between the model and the explicit rule** is the most useful
production indicator: it needs no labels, is computed continuously, and catches a drift before a
patient pays for it.

## Limits of use

Decision support for clinical staff, under mandatory human supervision. The assistant makes no
diagnosis and does not replace a clinical assessment. The catalogue of presentations behind the
training data has not been validated by an emergency physician: this service must not be used in
a real setting.
