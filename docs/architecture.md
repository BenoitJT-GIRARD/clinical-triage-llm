# Architecture

Six decisions hold this project up. Each one ruled out an alternative, and this page names that
alternative beside the choice, because once a system exists every part of it looks inevitable.

This page is for someone who has to maintain or extend the service and wants to know which pieces
are load-bearing before touching any of them.

## 1. The ground truth is written, not found

The obvious plan was to label a public medical corpus in triage levels and train on that, but the
corpora do not allow it: none of the five is annotated in levels, and only one of them is made of
patient cases. The funnel in [`../data/README.md`](../data/README.md)
counts what each one really yields.

The ground truth therefore comes from a clinical catalogue written for the project: seventy
presentations, each carrying what a patient arrives with, what can be measured on them, and why
that puts them at that level. A generator draws vignettes from it. The public corpora contribute
authentic cases, filtered and labelled by an explicit rule, and carry an explicitly lower
confidence that travels with every example.

**What was rejected.** Labelling the corpora by prompting a larger model was considered and
dropped: the labels would have come from a system nobody can audit, and the project's whole
argument is that the ground truth is inspectable. Buying annotation was out of reach. Training on
the rule's own labels alone was rejected because a model trained on a rule's output learns the
rule, blind spots included, and the evaluation would then measure the copy and not the triage.

**What it costs.** The catalogue was not validated by an emergency physician. That is the
project's largest single limitation, the README states it, and no amount of measurement repairs
it.

## 2. The explicit rule stays, and travels in the answer

A keyword rule was written first, to label the corpus cases, and it could have been thrown away
once the model worked. Instead it was kept, wired into the service, and published in every reply
as `rule_level`, `rule_reasons` and `agreement`.

The rule never hallucinates and never invents a level; the model sometimes does. Where the two
agree, the nurse has two independent reasons to trust the level. Where they disagree, the screen
can say so, and the human decides with the doubt in front of them. A reply that hid the
disagreement would show the nurse a level that looks certain and that nobody has checked.

**What was rejected.** A silent fallback to the rule when the model fails. The caller would have
no way of knowing which system answered. The rule's verdict is published every time, and the
caller decides what to do with it.

## 3. Two processes, not one

The service is a FastAPI gateway and a vLLM engine, in separate containers.

The gateway validates the request, runs the adaptive questionnaire, applies the explicit rule,
masks personal data, writes the audit line and shapes the reply. It holds no weights and does not
import torch: its image is a few hundred megabytes and starts in seconds. The engine holds the
model and answers one OpenAI-compatible route.

Because of that split, a new model version ships without rebuilding the gateway, and the gateway
ships without touching three gigabytes of weights. The split also makes the on-demand deployment
possible: on Modal the engine shuts down after fifteen idle minutes while the gateway answers its
liveness probe, and the readiness probe is the one that wakes the GPU.

**What was rejected.** Serving with `transformers` inside the gateway, which was the shortest
path and the one the project used first. It was dropped when the image reached three gigabytes
and the process needed a GPU to answer a health check. The code for it is still there, behind
`TRIAGE_BACKEND=transformers`, because it is the only way to run the model without a container
runtime, and because an evaluation loop that does not go over HTTP is much faster.

## 4. LoRA on a small model, and the output head with it

The model is a 1.7-billion-parameter base model, fine-tuned with LoRA adapters and not in full.
On one consumer card, a LoRA run finishes in an evening, and a full fine-tuning run would not
start.

The project's least obvious finding is something that had to be added to the usual recipe. In
this base model the twenty-five ChatML control tokens are a single untrained vector, `im_start`
and `im_end` being identical to the third decimal by cosine similarity, and the output head is
tied to the embedding matrix, which LoRA freezes. The model therefore cannot emit the
end-of-sequence token: it runs to the generation cap on every answer. Training the output head
alongside the projections fixes it; clean stops go from none to all, and the median answer falls
from the cap of 220 tokens to 96. The measurement is in
[`protocol.md`](protocol.md#the-end-of-sequence-token).

The kernels are Unsloth's rather than plain Transformers', for the same card: on the same thirty
steps they run nearly twice as fast and need three gigabytes less, for a loss that ends in the
same place ([`protocol.md`](protocol.md#the-training-engine)).

**What was rejected.** A larger base model would have removed the problem by having trained
control tokens, and was rejected on hardware: the project targets one card, and a model that
needs two says nothing about what a department could run. Full fine-tuning was rejected for the
same reason.

## 5. Preference alignment on defects, not on preferences

The alignment set is not a collection of human preferences. Each pair holds one answer that is
right and one that carries exactly one deliberate defect: an undertriage, an unsafe course of
action, a diagnosis asserted where it should be suggested, or an answer in the wrong language. The
defect is named in the example, so what the alignment teaches can be read.

**What was rejected.** Using an external preference dataset as the training signal was rejected
after measuring it: on its expert-reviewed test split, 470 of the 597 pairs not built to compare
answers of similar length favour the longer answer, and an alignment trained on them would have
taught the model that longer is better. It is used as an evaluation set instead, and the model's
poor score on it is reported.

## 6. Everything a run produces lives under `var/`

Weights, audit log, experiment store: none of it is tracked, all of it is under one directory,
and the `.gitignore` holds one line for it. With no exception list in the ignore file, the layout
alone says what a file is.

**What was rejected.** Keeping the model weights in the image, which would have made the
deployment a single artefact. It costs three gigabytes rebuilt on every gateway change, and a
model version that cannot be replaced without a new image. The engine reads the weights from the
Hub at a pinned revision, and the revision is what the deployment publishes.

## What runs where

<!-- source: docs/images/MANIFEST.json -->
![The triage operation of the published contract, expanded on its reply schema: the model's level and the rule's level are two fields of the same object](images/triage-contract.png)

| piece | where it runs | what it holds |
|---|---|---|
| gateway | container, no GPU | the questionnaire, the rule, the anonymisation, the audit log |
| engine | container with a GPU, or on demand | the merged model, at a pinned revision |
| training | one workstation with a GPU | nothing durable; it writes adapters and an MLflow store under `var/` |
| evaluation | the same workstation | the result files under `reports/`, which everything else reads |

The deployment itself, the two containers, the on-demand variant and what each of them costs,
is in [`operations.md`](operations.md).
