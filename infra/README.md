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

## 4. Cloud deployment

### The option taken — Modal

[Modal](https://modal.com) allocates a GPU on demand, bills by the second and shuts the container
down when nothing asks for it. The free account receives 30 $ of credit renewed every month,
without a card; a demonstration consumes less than an hour of L4, about 0.80 $.

`modal_app.py` raises the same two pieces as the Docker stack of this folder: a GPU container
serving the merged model under vLLM, and the project's FastAPI gateway exposed as is. That is the
project's own split, not an adaptation to the host, and not a line of the service code changes.

```bash
uv run modal setup                      # authentication, opens the browser
uv run modal secret create clinical-triage \
    TRIAGE_API_KEY="a-key-of-your-choosing" \
    HF_TOKEN="hf_..."
uv run modal deploy infra/modal_app.py
```

Two addresses are printed. The engine's belongs in the secret; redeploying afterwards is what
tells the gateway where to send its requests:

```bash
uv run modal secret create clinical-triage --force \
    TRIAGE_API_KEY="a-key-of-your-choosing" \
    HF_TOKEN="hf_..." \
    TRIAGE_VLLM_URL="https://<account>--clinical-triage-engine.modal.run"
uv run modal deploy infra/modal_app.py
```

`modal setup` writes the account tokens into `~/.modal.toml`. In continuous integration, where
there is neither a browser nor a configuration file, `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`
replace them in the job's environment.

The engine loads the weights **from the Hub, at the revision tagged** by the deployment pipeline:
the demonstration stays replayable identically even if the default branch moves afterwards. The
Hugging Face cache is mounted on a persistent volume, without which every cold start would
re-download three gigabytes.

The repository and the revision are chosen through `HF_NAMESPACE`, `TRIAGE_MODEL_ID` and
`TRIAGE_MODEL_REVISION`, set in the shell at `modal deploy` time: `infra/modal_app.py` runs on the
workstation and reads its environment at import, before the project's package, and therefore
`.env`, is loaded. Without them, the deployment serves the model and the revision pinned in the
file.

**On the day of a demonstration**, wake the endpoint five minutes before going on: a cold start
takes a minute when the cache is warm, and a minute of silence in front of an audience is long.
`GET /health` really questions the engine, and exists for exactly that.

### Option A — a managed vLLM container

1. Publish the merged model on the Hub:
   ```bash
   export HF_TOKEN="hf_..."      # a write token
   export HF_NAMESPACE="your-account"
   uv run python scripts/publish_to_hub.py --what final-model
   ```
2. Create an inference endpoint at the host, choosing the model repository, an L4-class GPU or
   equivalent, and the vLLM container.
3. Deploy the gateway, which is the image the pipeline publishes to GHCR, on a container service
   without a GPU, with `TRIAGE_VLLM_URL` pointing at the inference endpoint and `TRIAGE_API_KEY`
   as a secret.

### Option B — a virtual machine with a GPU

1. Provision a machine with a GPU, Docker and `nvidia-container-toolkit`.
2. Copy `var/models/qwen3-1.7b-triage-dpo-merged` onto the machine, or let vLLM download it from
   the Hub.
3. `docker compose -f infra/docker-compose.yml up -d`
4. Expose port 8080 behind a reverse proxy with a TLS certificate.

### Shipping a new model version

The weights are uploaded from the training machine; they do not travel through continuous
integration. The **version tag** is what ties the two together.

```bash
uv run python scripts/publish_to_hub.py --what all      # weights, dataset, cards
git tag model-v1.0.0 && git push origin model-v1.0.0    # triggers the pipeline
```

The tag triggers the `model` job of `cd.yml`: the model cards are republished, then the
`model-v1.0.0` tag is placed on the Hub repositories, the dataset included. The dataset files
themselves are not republished from there — the pipeline did not rebuild them and holds only two
of the six. They travel with the weights, from the training machine, through the command above.
The deployment job then runs `modal deploy` with that revision, and the inference server restarts
on those weights:

```bash
vllm serve <account>/qwen3-1.7b-clinical-triage --revision model-v1.0.0 \
  --served-model-name qwen3-1.7b-clinical-triage --max-model-len 1024
```

Pinning the revision instead of the default branch is what makes a deployment reproducible:
`main` designates content that changes at every publication, `model-v1.0.0` always designates the
same weights. Rolling back becomes trivial too: redeploy the previous tag.

The four cards published beside the weights are versioned here, and the continuous-deployment job
is what substitutes the evaluation figures and the served revision into them:
[`model-cards/final-model.md`](model-cards/final-model.md) for the model the service serves,
[`model-cards/sft-adapter.md`](model-cards/sft-adapter.md) and
[`model-cards/dpo-adapter.md`](model-cards/dpo-adapter.md) for the two adapters, and
[`model-cards/merged-sft-model.md`](model-cards/merged-sft-model.md) for the intermediate model
the second adapter needs in order to load at all.

### What the GitHub repository has to know

Automatic deployment is **disabled until it is armed**. It runs only if the `DEPLOY_ENABLED`
variable is `true`: without it the chain builds, tests and publishes, but touches nothing alive.

| Name | Type | Role |
|---|---|---|
| `MODAL_TOKEN_ID` | secret | Modal token, from `modal token new` |
| `MODAL_TOKEN_SECRET` | secret | its secret half |
| `HF_TOKEN` | secret | republishing the model cards and placing the tag on the Hub |
| `DEPLOY_ENABLED` | variable | `true` to arm the deployment |
| `DEPLOY_ENDPOINT_URL` | variable | address of the gateway, probed after deployment |
| `HF_NAMESPACE` | variable | target Hugging Face account, when it is not the default |

The service API key and the container's Hugging Face token do not go through GitHub: they live in
the Modal secret `clinical-triage`, created above. GitHub needs to know how to deploy, not what
the service handles.

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
| `TRIAGE_BASE_MODEL` | model loaded in `transformers` mode | the merged SFT model |
| `TRIAGE_ADAPTER_DIR` | adapter applied in `transformers` mode | the DPO adapter |
| `TRIAGE_MODEL_VERSION` | version written to the audit log | `qwen3-1.7b-clinical-triage` |
| `TRIAGE_AUDIT_LOG` | path of the audit log | `var/logs/audit_triage.jsonl` |
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
| `TRIAGE_MODEL_REVISION` | `modal deploy` | pinned revision of the weights | `model-v1.0.0` |

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
| `/health` probe | two consecutive failures | restart the container, raise an alert |
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
docker compose -f infra/docker-compose.yml cp api:/app/var/logs/audit_triage.jsonl .
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
