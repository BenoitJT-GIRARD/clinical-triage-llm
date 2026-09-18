# Operations

What it takes to run the service, what it costs, how it fails, and what it leaves behind. The
deployment commands themselves are in [`../infra/README.md`](../infra/README.md); this page is
about running the thing once it is up.

The reader this page expects is whoever is on call for it.

## The two processes

The gateway answers HTTP and holds no model. The engine holds the model and answers one route.
They are separate containers, and that separation is what most of this page is about: almost
every failure is one of them being up while the other is not.

| process | what it needs | what happens without it |
|---|---|---|
| gateway | a service key, a writable audit log | refuses to start, and says which of the two is missing |
| engine | a GPU, the weights at a pinned revision | the gateway starts, answers the probe as degraded, and refuses every triage with 503 |

The gateway **refuses to start** without `TRIAGE_API_KEY`, unless open mode is asked for
explicitly. An authentication that switches itself off when a variable is missing is not an
authentication, and the failure to start is the only way to make that visible at deployment
rather than at the first request.

It also opens the audit log at startup, empty, and refuses to serve if it cannot. Tracing every
interaction is a requirement of this service; a service that answers without being able to record
what it answered does not meet it. The case is not theoretical: it happens the moment a mounted
volume belongs to another user than the one running the process.

## The health probe, and what it does not say

`GET /health` questions the engine rather than reporting on itself. It answers `ok` only when the
engine replies, `degraded` otherwise, and in the degraded case it says that the engine is not
answering — never where the engine is. The technical cause, which carries the engine's address,
goes to the log.

That matters on a managed host, where the engine's address is what protects it. The probe needs
neither key nor quota, so anything that reaches the gateway reaches the probe.

A consequence worth knowing when a container restarts: the gateway is up and healthy as a process
long before the probe turns green, because the engine takes minutes to load three gigabytes of
weights. The image's own health check gives it a grace period for exactly that.

## What one decision costs

<!-- source: reports/benchmark_endpoint.json -->
![Perceived latency of POST /triage at three concurrency levels, median to 95th percentile, and the throughput of the gateway beside it, n = 40 requests per level](../reports/figures/endpoint_latency.png)

<!-- source: reports/benchmark_endpoint.json -->
One triage takes **1,473.5** milliseconds at the median with a single caller, n = 40 requests,
and **1,682.6** at the 95th percentile. The gateway's own share of that is **28.3** milliseconds:
validation, the questionnaire, the explicit rule, the anonymisation and the audit line together
cost under two hundredths of the answer. Everything else is generation.

<!-- source: reports/benchmark_endpoint.json -->
With eight concurrent callers, n = 40 requests each, the median moves to **1,592.0** milliseconds
and the throughput to **4.71** requests per second. The engine absorbs concurrency well; what
degrades is the tail, at **2,244.9** milliseconds for the 95th percentile.

These are order statistics on one machine, one run, against one engine version. They are not a
capacity plan, and the repository does not claim one.

## The quota

Every triage takes a GPU for a second and a half. An endpoint without a quota is an endpoint
anyone can saturate, so the gateway counts requests per caller over a sliding minute and answers
429 above the limit — announcing the limit in the message, so that an integrator does not have to
discover it by trial and error.

Two details that were defects first. The counting identity is the API key **only once it has been
verified**: in open mode nobody verifies it, and a caller changing the header on every request
used to get a fresh bucket each time. In open mode the identity is the caller's address instead.
And the counter forgets callers that have gone quiet for a minute, without which it keeps one
entry per address seen since startup — a counter meant to protect the service would end up
straining it.

The default is sixty a minute, which fits a reception desk. A load bench holds a single key and
passes that from its first concurrency level, so the compose file lets the value be raised for
the duration of a measurement without changing the service value.

## The audit log

One JSON line per interaction, appended, readable without a tool. It carries the request id, the
timestamp, the anonymised description, the level, the rule's level and reasons, the anonymised
answer, whether the description was truncated, the latency, the model that actually answered, the
engine, and the retention period in days.

Three things about it are not obvious.

**Both texts are masked, not just the incoming one.** The justification the model writes echoes
the nurse's narrative, and would bring back through one door a name masked at the other.

**The model version is the one actually loaded**, passed by the service at call time rather than
read from a constant. A configuration value could describe a model that is not the one that
answered, and traceability that can be falsified traces nothing.

**The retention is written into every line**, so the obligation travels with the data rather than
living in a document beside it.

In the container the log is a named volume. A host folder is mounted as root, and the service
runs unprivileged: every request then failed on the write, after having produced its decision.
`docker compose cp api:/app/var/logs/audit_triage.jsonl .` takes a copy out.

## When something is wrong

| symptom | what it means | what to do |
|---|---|---|
| the gateway exits at startup | no `TRIAGE_API_KEY`, or the audit log is not writable | the message names which; both are configuration, not code |
| `/health` says `degraded` | the engine is not answering | look at the engine's log; the gateway is fine and will recover on its own |
| every triage answers 503 | the same thing, seen from the caller's side | the caller is told to retry, and is told nothing about the engine |
| 401 on every call | the key presented does not match | the contract page names the header; the service compares in constant time |
| 429 under a bench | the quota, working | raise `TRIAGE_RATE_LIMIT` for the measurement, not for the service |
| answers cut mid-sentence | a description longer than the window | the reply says `description_truncated`; the bound is in characters, in the gateway |

The last one deserves its sentence. A clinical narrative silently truncated is exactly what a
decision-support system must not produce, so the gateway bounds it, says that it did, and the
audit line records it. The bound is in characters because the gateway carries no tokenizer —
that is the point of a thin gateway — and the character figure is set from the measured token
budget.

## What is deployed, and what is not

The on-demand deployment runs the same two applications: a GPU container serving the merged model
with vLLM, and the same FastAPI application as the Docker image, exposed as is. Nothing is
rewritten, only rewired. The engine shuts down after fifteen minutes without a request and comes
back on the next one — on a credit account, a demonstration left running costs the rest of the
month.

The engine gets a public address, like the gateway, and demands the service key: without that,
the address alone would buy free GPU inference, with no quota, no anonymisation and no line in
the audit log, at the account's expense.

**No service is running behind this repository today.** The deployment is a command, described in
[`../infra/README.md`](../infra/README.md), not an address kept alive.
