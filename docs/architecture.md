# Architecture

This page describes six design decisions that the rest of the project depends on, and for each
one it names the alternative that was considered and rejected.

It is written for someone who maintains or extends the service and needs to know which parts the
rest relies on before changing any of them.

## The ground truth comes from a written catalogue

The first plan was to label a public medical corpus with triage levels and train on it. The
corpora do not allow this: none of the five the project read is annotated with triage levels,
and only one of them consists of patient cases. The funnel table in
[`../data/README.md`](../data/README.md) counts what each of the four case corpora yields.

The ground truth therefore comes from a clinical catalogue written for the project. It holds
seventy presentations, each recording what the patient arrives with, what can be measured, and
why that places the patient at that level. A generator draws vignettes from it. The public
corpora contribute cases written outside the project, filtered and labelled by an explicit rule,
and those cases carry a lower confidence grade, stored in every example.

**What was rejected.** Labelling the corpora by prompting a larger model was considered and
dropped, because the labels would then come from a system whose decisions cannot be audited, and
the project relies on a ground truth that can be inspected. Paying for annotation was beyond the
project's means. Training on the rule's labels alone was rejected, because a model trained on a
rule's output learns the rule, blind spots included, and the evaluation would then measure how
well the model copies the rule.

**What it costs.** No emergency physician has validated the catalogue. This is the project's
largest single limitation, the README states it, and further measurement cannot remove it.

## The explicit rule is kept and returned in every reply

A keyword rule was written first, to label the corpus cases, and it could have been discarded
once the model worked. It was kept, wired into the service, and returned in every reply as
`rule_level`, `rule_reasons` and `agreement`.

The rule is deterministic and only ever returns one of its three levels, while the model
sometimes produces an answer that cannot be parsed. When the two agree, the nurse sees two
separate methods reach the same level. When they disagree, the reception screen can show it, and
the nurse decides with the disagreement in view. A reply that hid the disagreement would present
the model's level as settled when a second method had placed the patient elsewhere.

**What was rejected.** A silent fallback to the rule when the model fails. With such a fallback
the caller could not tell which system had answered. The service returns the rule's verdict with
every reply, and the caller decides what to do with it.

## The gateway and the engine run as separate processes

The service consists of a FastAPI gateway and a vLLM engine, in separate containers.

The gateway validates the request, runs the adaptive questionnaire, applies the explicit rule,
masks personal data, writes the audit line and builds the reply. It holds no weights and does not
import torch, so its image is a few hundred megabytes and it starts in seconds. The engine holds
the model, and the gateway sends it generation requests on its OpenAI-compatible completions
route.

With this split, a new model version ships without rebuilding the gateway, and a gateway change
ships without moving nearly four gigabytes of weights. The split also makes the on-demand
deployment possible. On Modal the engine shuts down after fifteen idle minutes and the gateway
after five. A call to the liveness probe starts only the CPU gateway, and a call to the readiness
probe wakes the GPU.

**What was rejected.** Serving with `transformers` inside the gateway, the shortest path and the
one the project used first. It was dropped when the image reached three gigabytes and the process
needed a GPU to answer a health check. The code remains, behind `TRIAGE_BACKEND=transformers`,
because it runs the model on a machine without vLLM, and because an evaluation loop that makes no
HTTP calls runs much faster.

## LoRA on a small model, with the output head trained

The model is `Qwen3-1.7B-Base`, fine-tuned with LoRA adapters. On one consumer GPU a LoRA run
finishes in an evening, while a full fine-tuning run would not even start.

The least obvious finding of the project is an addition to the usual recipe. In this base model
the twenty-five ChatML control tokens share one untrained vector, with `im_start` and `im_end`
identical to the third decimal by cosine similarity. The output head is tied to the embedding
matrix, which LoRA keeps frozen. The model therefore cannot produce its end-of-sequence token, and
it runs to the generation cap on every answer. Training the output head together with the
projections removes the problem. The share of answers that stop by themselves goes from zero to
one, and the median answer falls from the cap of 220 tokens to 96, as measured in
[`protocol.md`](protocol.md#the-end-of-sequence-token).

Unsloth's kernels replace the plain Transformers ones for the same reason, the single GPU. Over
the same thirty steps they run nearly twice as fast and use about three gigabytes less memory,
and the two final losses are close
([`protocol.md`](protocol.md#the-training-engine)).

**What was rejected.** A larger base model was not tried, and it was rejected on hardware
grounds. The project targets one GPU, and a model that needs two would not show what a department
could run on one. Full fine-tuning was
rejected for the same reason.

## Preference alignment on deliberate defects

The alignment pairs are built from the project's own training examples. Each pair holds one
correct answer and one answer with exactly one deliberate defect: an undertriage, an unsafe
course of action, a diagnosis asserted where it should only be suggested, or an answer in the
wrong language. The `strategy` field of each pair names the defect, so what the alignment teaches
can be read from the data.

**What was rejected.** An external preference dataset as the training signal, rejected after
measurement. On its expert-reviewed test split, 470 of the 597 pairs that were not built to
compare answers of similar length favour the longer answer, and an alignment trained on them
would teach the model that longer answers are better. The set serves as an evaluation set, and
the model's poor score on it is reported.

## Weights, logs and experiment runs live under `var/`

The model weights, the audit log and the MLflow experiment store are untracked and sit under one
directory, `var/`, which `.gitignore` excludes in a single line with no exception. Apart from
Python environments, the `.env` file of secrets, editor folders, operating-system files, caches
and the outputs of development tools, the ignore file lists only the four training files of
`data/processed/`, which the dataset build regenerates.

**What was rejected.** Keeping the model weights in the image, which would have made the
deployment a single artefact. Every gateway change would then rebuild an image carrying nearly
four gigabytes of weights, and a model version could only be replaced by a new image. On Modal
the engine reads the weights from the Hub at a pinned revision, which the deployment workflow
passes to it. The compose stack mounts them from `var/models/`.

## What runs where

<!-- source: docs/images/MANIFEST.json -->
![The triage operation of the published contract, opened on its reply schema, in which the model's level and the rule's level are two fields of one object](images/triage-contract.png)

| piece | where it runs | what it holds |
|---|---|---|
| gateway | container, no GPU | the questionnaire, the rule, the anonymisation, the audit log |
| engine | container with a GPU, on a workstation or on demand on Modal | the merged model, mounted from `var/models/` locally or read from the Hub at a pinned revision on Modal |
| training | one workstation with a GPU | nothing durable; it writes adapters and an MLflow store under `var/` |
| evaluation | the same workstation | the result files under `reports/`, which the figures, the cards and the documentation read |

[`operations.md`](operations.md) describes the deployment: the two containers, the on-demand
variant, and what each of them costs.
