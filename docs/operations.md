# Operations

This page covers running the service once it is deployed: what it needs, what it costs, how it
fails, and what it leaves behind. The deployment commands are in
[`../infra/README.md`](../infra/README.md).

It is written for whoever is on call for the service.

## The two processes

The gateway answers HTTP requests and holds no model. The engine holds the model and serves the
gateway's generation requests. They run in separate containers, and most failures come from one
of them running while the other is down.

| process | what it needs | what happens without it |
|---|---|---|
| gateway | a service key and a writable audit log | it refuses to start and names the missing one |
| engine | a GPU and the merged weights | the gateway starts, `/ready` answers 503, and every triage request gets a 503 |

The gateway refuses to start without `TRIAGE_API_KEY` unless open mode is requested explicitly
with `TRIAGE_ALLOW_ANONYMOUS=true`. A key check that turned itself off when the key was missing
would leave the service open, and refusing to start reports the missing key at deployment time,
before the first request.

The gateway also opens the audit log at startup, and it does not serve any request when the log
cannot be written. Recording every interaction is a requirement of this service, and an answer
that cannot be recorded would break it. The case occurs in practice, because a volume mounted
from the host arrives owned by root and the service does not run as root.

## The two probes

`GET /health` is the gateway's liveness probe. It answers `alive` whenever the process serves
requests, calls nothing else and needs no key, so an anonymous caller can poll it freely without
reaching the engine.

`GET /ready` checks the engine. It answers `ok` when the engine replies, and `degraded` with a
503 otherwise. In the degraded case the reply says that the engine is not answering and leaves
out the engine's address. The technical cause, which contains that address, goes to the log. The
probe requires the service key, because on a managed host a request that reaches the engine can
start a GPU.

After a container restart the gateway is alive well before it is ready, because the engine takes
minutes to load nearly four gigabytes of weights. The deployment workflow therefore waits on
`/ready`.

## What one decision costs

<!-- source: reports/benchmark_endpoint.json -->
![Latency of POST /triage at three concurrency levels, from the median to the ninety-fifth percentile, with the gateway's throughput beside it, n = 40 requests per level](../reports/figures/endpoint_latency.png)

> **How to read it.** Both panels have the number of concurrent callers on the horizontal axis.
> The left panel shows how long a caller waits, as a bar from the median to the ninety-fifth percentile,
> so the length of the bar is how much longer an unlucky call takes than a typical one. The
> right panel shows how many requests the gateway completes per second. A wait that keeps rising
> while the throughput levels off would mean that requests are queuing.

<!-- source: reports/benchmark_endpoint.json -->
On the Modal deployment of this version, one triage takes **2,095.2** milliseconds at the median
with a single caller, n = 40 requests, and **2,422.7** at the 95th percentile. The engine was
vLLM **0.29.0**, the version the deployment pins, as the engine itself reports it, on one L4 GPU.
Called directly, the engine answers in **1,739.4** milliseconds at the median. At the median,
**388.8** milliseconds of a triage are spent outside the gateway's call to the engine. That time
covers the gateway's own work on the reply (validation, the explicit rule, the anonymisation of
both texts and the audit line) and the round trip between the benchmark machine and the gateway.
The bench ran from a workstation in France, and the engine's address places the deployment in
Modal's us-east region. The repository does not measure how that time splits between the
gateway's work and the network.

<!-- source: reports/benchmark_endpoint.json -->
With four callers at once, n = 40 requests at that level, the median rises to **2,178.6**
milliseconds and the 95th percentile to **2,670.3**, above the two seconds required by the
criteria listed below. With eight callers, the median is **2,227.8** milliseconds and the
throughput **3.23** requests per second. The median changes little as callers are added, and the
95th percentile rises more.

<!-- source: reports/benchmark_endpoint.json -->
The bench ran three times in a row, with n = 40 requests per level in each run, and the figure
shows the first run. Across runs the medians stay within a tenth of a second, from **2,087.9** to
**2,095.2** milliseconds with one caller and from **2,222.7** to **2,309.0** with eight. The 95th
percentile varies more, from **2,604.9** to **2,774.9** with four callers and from **2,731.9** to
**3,083.6** with eight. On forty requests that percentile is the third-slowest request, and the
three runs show how much it moves. These are order statistics from one deployment, measured from
one workstation on one evening, and the repository draws no capacity plan from them.

## The quota

Each triage occupies the GPU for the length of one generation. To keep a single caller from
saturating the endpoint, the gateway counts requests per caller over a sliding one-minute window
and answers 429 above the limit. The error message states the limit, so an integrator can read
it from the first refusal.

Two details of the quota started as defects. The key serves as the caller's identity only when it
has been checked against the configured key. In open mode nothing is checked, so a caller who
changed the header on every request used to receive a fresh quota each time, and in that mode
the client address now serves as the identity. Callers that have been silent for a minute are
also dropped from the table. Without that sweep the table grew by one entry per address seen
since startup, and the quota became a memory cost of its own.

The default is sixty requests a minute, which suits a reception desk. A load bench uses a single
key and sends forty requests per concurrency level, one level after another, so its requests soon
exceed that count within a minute. The compose file therefore lets `TRIAGE_RATE_LIMIT` be raised
for the duration of a measurement while the service value stays unchanged.

## The audit log

The log holds one JSON line per interaction, appended to the file and readable without any tool.
Each line records the request id, the timestamp, the anonymised description, the model's level,
the rule's level and reasons, the anonymised answer, whether the description was truncated, the
latency, the model version, the engine, the retention period in days, and the digest of the
previous line.

Five properties of the log need explaining.

**Both texts are masked.** The model's justification repeats what the nurse wrote, so a name
removed from the description could reappear through the answer. Each text goes once through the
masking engine of its own language. Running the English engine over a French narrative used to
mask clinical words as if they were names.

**The service supplies the model version** with each record, and the audit module writes what it
receives. With the `transformers` backend the version is read from the loaded model. Behind vLLM
it is the `TRIAGE_MODEL_VERSION` setting, which defaults to the name vLLM serves, so in that mode
a wrong setting can name a model other than the one answering.

**The retention period is enforced.** The log is written as one file per day, and the files older
than the retention period are deleted at startup and at the first write of each day. The period
is also written into every line, so the obligation stays attached to the data.

**Changes are detectable.** Each line carries the SHA-256 digest of the line before it, across
day files. Editing, inserting or deleting a line breaks the link held by the following line, and
one command reports where:

```bash
uv run python scripts/verify_audit_log.py var/logs/audit
```

Two changes escape the chain. A change to the last line has no following line to reveal it,
and the next write chains onto the changed line. Deleting the first lines leaves the first
remaining line as the new anchor, as a retention purge does. The chain detects the other changes
and cannot prevent any of them, since the host can still delete the files.

**One writer.** Appends to a file are atomic only within one process, so the deployment runs a
single gateway container. This limits how far the service can scale.

In the container the log is on a named volume. A host folder is mounted as root while the service
runs unprivileged. Before the startup check existed, every request then failed on the write after
its decision had been produced, and the gateway now refuses to start instead.
`docker compose cp api:/app/var/logs/audit .` copies the log out. On Modal the log lives on the
`clinical-triage-audit` volume, and `modal volume get clinical-triage-audit / ./audit` copies it
out.

## When something is wrong

| symptom | what it means | what to do |
|---|---|---|
| the gateway exits at startup | `TRIAGE_API_KEY` is missing, or the audit log is not writable | the message names which; both are configuration problems |
| `/ready` answers 503 | the engine is not answering | read the engine's log; the gateway needs no action and recovers once the engine answers |
| every triage answers 503 | the same failure, seen by the caller | the reply asks the caller to retry and does not contain the engine's address |
| 422 on a long description | the description exceeds 800 characters, the bound of the training narratives | the reply says so; shorten the narrative to the clinical facts |
| 422, "does not fit the model window" | the engine refused a narrative its window cannot hold | the text has to change, and a retry would fail the same way |
| 401 on every call | the key presented does not match | the contract page names the `X-API-Key` header; the comparison runs in constant time |
| 429 during a bench | the quota is working | raise `TRIAGE_RATE_LIMIT` for the duration of the measurement only |
| `verify_audit_log.py` names a line | the line before it was edited, inserted or removed | keep the copy unchanged; the chain locates the change and cannot say who made it |
| accents garbled in Windows PowerShell 5.1 | `Invoke-RestMethod` decodes a JSON reply that declares no charset as Latin-1, while the service sends UTF-8, as JSON requires | call it from PowerShell 7, or decode the raw bytes of `Invoke-WebRequest` as UTF-8 |

<!-- source: data/processed/metadata.json -->
The gateway answers 422 in two cases. Because the service must never shorten a patient's
narrative without saying so, the gateway refuses a description longer than 800 characters. That
is the bound the dataset build places on corpus narratives, and no training narrative exceeds it,
so a longer text would give the model input unlike anything it was trained on. The bound is
counted in characters because the gateway carries no tokenizer, which keeps it light. When the
engine still refuses a text, the reply says that the description does not fit the model window
and asks the caller to shorten it to the clinical facts before sending it again.

## What is deployed

The on-demand deployment runs the same two processes. One container with a GPU runs vLLM over the
merged weights. The gateway runs the same application code, installed from the same pinned list
of requirements as the Docker image. No service code changes for this deployment, and only the
wiring differs. The engine stops after fifteen idle minutes and starts again when the next
request reaches it.

Two protections keep strangers out. The engine sits behind Modal's proxy authentication, which
refuses a request without the workspace's proxy token before any container starts. Knowing the
engine's address is therefore not enough to run generation that would be untraced, unmasked,
uncounted and billed to the account. On the gateway, `/health` is the only one of the four
service routes that answers without a key, and it calls nothing else. The documentation pages
that FastAPI generates are public as well.

**The service is deployed and stopped.** While stopped it costs nothing and answers no request.
Only the owner of the repository starts it, for a demonstration, and stops it afterwards.
[`../infra/README.md`](../infra/README.md#4-cloud-deployment-on-modal) explains how, and who
holds the keys.

## Conditions for clinical use

Ten criteria would have to be met before this service could be used with patients. They were
written down with this version and are kept, so that each later version reports how far it stands
from them.
Four of them, the first two and the last two, lie outside engineering, and the others depend on
them.

The values for this version come from `reports/evaluation_results.json` and, for the latency,
from `reports/benchmark_endpoint.json`.

1. **The clinical catalogue validated by two emergency physicians**, every presentation: not
   done.
2. **The evaluation set labelled independently by two clinicians**, with an agreement of at least
   0.8: not done.
3. **Undertriage**, with the upper bound of its 95% interval at most 0.05: not met; 0.275, with
   an upper bound of 0.439.
4. **Format compliance**, with the lower bound of its 95% interval at least 0.99: not met, and
   this set cannot meet it; 1.000, with a lower bound of 0.940.
5. **Latency at the ninety-fifth percentile with four simultaneous callers**, at most 2,000 ms:
   not met; 2,604.9 to 2,774.9 ms over three runs on the Modal deployment, with vLLM 0.29.0.
6. **An audit log that persists and is anonymised**, verified: met, with one file a day, a chain
   of digests and an enforced retention period.
7. **An endpoint that is authenticated, rate-limited and served over HTTPS**, verified: met on
   Modal, which provides the HTTPS; a workstation serves plain HTTP.
8. **A rollback**, rehearsed in a pilot: not done.
9. **A data protection impact assessment**, approved by the data protection officer: not done.
10. **The regulatory status of the software**, decided: not done.

Criteria 3 and 4 are judged on the unfavourable bound of the interval, and the clinical set
cannot meet them whatever the model does. Zero misses on its forty urgent cases would still leave
an upper bound of 0.088 on undertriage, and sixty answers all in format leave a lower bound of
0.940 on compliance. Both bounds come from the same exact interval: the results file records the
first, and the second is derived from `upper_bound_without_miss` in `evaluation/metrics.py`
applied to sixty answers. Meeting the two criteria takes 72 urgent cases without a miss, a count
the results file records, and 368 answers without a fault, a count derived from
`cases_needed_without_miss` with a bound of 0.01. The first requirement is therefore a larger
evaluation.

On Modal, by default, every request is routed through Virginia, in the United States, and the
container starts wherever the provider has a free GPU, so a narrative sent to the cloud
deployment leaves the European Union. Both can be kept in Europe with `region="eu"` and
`routing_region="eu-west"`, at a higher compute price. This deployment does not use them, because
it serves invented cases. With real patients, criterion 9 would have to settle this question
first.
