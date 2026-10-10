# Where the data comes from

Six sources, and only one of them carries the triage levels the model is trained on. This page
says what each one is, under which licence it is used, what was taken from it and what was left.
The counts are in [`../data/README.md`](../data/README.md), which is the card published beside
the dataset on the Hub; this page is about provenance and rights.

The reader this page expects is someone deciding whether this dataset may be reused, and under
what conditions.

## The corpora, as they were read

Every corpus was read from the Hugging Face Hub on 2026-10-09, the date the published set was built;
the build records that date, with the repository revision, in `metadata.json`. None of them is
redistributed here in its original form: what the dataset carries is derived cases, and the
derivation is in the code.

| source | where | licence | what was taken |
|---|---|---|---|
| Clinical catalogue | written for this project, `src/clinical_triage/data/clinical_catalogue.py` | MIT | the triage levels themselves |
| MediQAl | <https://huggingface.co/datasets/ANR-MALADES/MediQAl> | CC BY 4.0 | French clinical vignettes |
| MedQuAD | <https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset> | CC BY 4.0 | symptom descriptions turned into complaints |
| MedMCQA | <https://huggingface.co/datasets/openlifescienceai/medmcqa> | Apache-2.0 | English clinical vignettes |
| FrenchMedMCQA | <https://huggingface.co/datasets/nthngdy/frenchmedmcqa> | Apache-2.0 | nothing: no case survived the filters |
| UltraMedical-Preference | <https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference> | MIT | nothing for training; an evaluation set only |

Two of the six are read through a mirror rather than the original repository. The licence
recorded for MedQuAD is the one the original `abachaa/MedQuAD` repository carries, and the one
for FrenchMedMCQA is `qanastek/frenchmedmcqa`'s: a mirror does not create rights, so the right
that matters is the original's. Both are recorded in `metadata.json` under `licence_checked_on`.

## What each one gave, and why

**The clinical catalogue is the only source of triage levels.** Seventy presentations, each
written with its complaint, its signs, its history, a vital-sign profile and the clinical reason
for its level. A generator draws vignettes from it, varying the wording, the patient's age, the
measured vitals and the language, and the level travels with the presentation: it is never read
back out of the text. That is what makes it ground truth: no rule, no model and no keyword
decides it.

**MediQAl is the only source corpus that describes patients.** Its clinical-case column carries
French vignettes with a complaint and, for some of them, the vital signs on arrival. It is the
authentic French material of the set.

**MedMCQA was added, and the addition is a decision.** The other source corpora hold no
English clinical vignette at all. Without a second language the dataset would have been French
only, and the service reads both. MedMCQA's yield is poor, a hundred and eighty
thousand rows for twelve hundred cases, but those cases are real presentations written by
someone else, which is exactly what a catalogue cannot provide.

**FrenchMedMCQA gave nothing, and that is the finding.** Its questions are pharmacy questions:
of its entries, six are recognised as describing a patient and none of those carries a sign the
triage rule can read. The corpus is kept in the funnel table rather than dropped from the
document, because "we tried this source and it yields zero" is a result.

**UltraMedical-Preference is never trained on.** Outside the pairs built to compare answers of
similar length, its preferred answer is the longer one in 470 pairs out of 597; an alignment
trained on that signal learns that longer is better. Its test split is used as an external
evaluation set, where the aligned model does badly — which is reported in
[`protocol.md`](protocol.md#what-the-alignment-bought-and-what-it-did-not).

## What the corpus cases are allowed to decide

A case extracted from a public corpus carries no triage level of its own. The explicit rule reads
it and proposes one, and the case is kept only if a sign was recognised. Its label therefore
carries a **`medium` confidence**, written into every example, against `high` for a catalogue
vignette.

Two bounds keep that contribution in its place. Per cell of the level × language grid, a corpus
may not exceed the places that cell has. Globally, corpus cases stay a minority of the set. Both
are there for the same reason: corpus labels come from the rule, and a set where they dominate is
a transcription of the rule, which the evaluation would then flatter.

## Personal data

No real patient data is used. The catalogue vignettes are synthetic, and the public corpora are
research sets with no identifying content. The anonymisation pass runs anyway, on both languages,
and a separate check looks afterwards for what it might have let through. The data card holds the
three answers: the entities masked, the ones left alone on purpose, and the residue the check
still reports — [`../data/README.md`](../data/README.md#personal-data).

## Reusing this dataset

The derived dataset is published on the Hub under the repository named in `config.py`, with the
card of [`../data/README.md`](../data/README.md) beside it. It inherits the constraints of its
sources: the CC BY 4.0 of MediQAl and MedQuAD requires attribution, which the card carries, and
the Apache-2.0 and MIT of the others are compatible with it.

It does not inherit a clinical validity it never had. The levels come from a catalogue written by
a developer, not by a clinician, and nothing in this dataset should be used to train a system
that meets a patient.
