# Where the data comes from

The project read six sources, and four of them contribute to the dataset files. Only one, the
clinical catalogue written for the project, comes with triage levels, and the cases taken from
public corpora receive their level from the project's explicit rule. For each source, this page
gives what it is, the licence it is used under, what was taken from it and what was left out. The
counts are in [`../data/README.md`](../data/README.md), the card published with the dataset on the
Hub, and this page covers provenance and rights.

It is written for someone deciding whether this dataset may be reused, and under what
conditions.

## The sources and their licences

The dataset build read the four case corpora from the Hugging Face Hub on 2026-10-09, the date the
published set was built, and recorded that date in `metadata.json`. The repository revision stored
beside it was entered by hand after the build. UltraMedical-Preference is read by the evaluation
script and contributes nothing to the dataset files. No source is redistributed in its original
form. The dataset carries derived cases, and the code performs the derivation.

| source | where | licence | what was taken |
|---|---|---|---|
| Clinical catalogue | written for this project, `src/clinical_triage/data/clinical_catalogue.py` | MIT | the generated vignettes and their triage levels |
| MediQAl | <https://huggingface.co/datasets/ANR-MALADES/MediQAl> | CC BY 4.0 | French clinical vignettes |
| MedQuAD | <https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset> | CC BY 4.0 | symptom descriptions rewritten as complaints |
| MedMCQA | <https://huggingface.co/datasets/openlifescienceai/medmcqa> | Apache-2.0 | English clinical vignettes |
| FrenchMedMCQA | <https://huggingface.co/datasets/nthngdy/frenchmedmcqa> | Apache-2.0 | nothing, since no case passed the filters |
| UltraMedical-Preference | <https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference> | MIT | nothing for training; it serves as an evaluation set |

Two of the six are read through a mirror. A mirror cannot grant rights of its own, so the licence
that applies is that of the original repository: `abachaa/MedQuAD` for MedQuAD, and
`qanastek/frenchmedmcqa` for FrenchMedMCQA. `metadata.json` records both under
`licence_checked_on`.

## What each source contributed

The clinical catalogue is the only source that comes with triage levels. It holds seventy
presentations, each written with its complaint, its signs, its history, a vital-sign profile and
the clinical reason for its level. A generator draws vignettes from it, varying the wording, the
patient's age, the measured vital signs and the language. Each vignette inherits the level of its
presentation, and the level is never inferred from the generated text. No rule, model or keyword
decides these levels, so they can serve as ground truth.

MediQAl is the only source corpus made of patient cases. Its clinical-case column holds French
vignettes that give a complaint and, in part of them, the vital signs measured on arrival. It
supplies the French cases written outside the project.

MedMCQA supplies the English vignettes. The other source corpora contain no English clinical
vignette, and without a second language the dataset would have been French only, while the
service reads both. The yield of MedMCQA is low, over a hundred and eighty thousand rows for
nearly thirteen hundred cases, but those cases were written by other authors, which the catalogue
cannot provide.

FrenchMedMCQA contributed nothing. Its questions are pharmacy questions. Six of its entries are
recognised as describing a patient, and none of those six carries a sign the triage rule can
read. The corpus remains in the funnel table, because a source that was tried and yielded zero
cases is itself a result.

UltraMedical-Preference is never used for training. In its pairs that were not built to match
answer lengths, the answer marked as preferred is the longer one 470 times out of 597, and an
alignment trained on that signal would learn that longer answers are better. Its test split
serves as an external evaluation set, on which the aligned model scores poorly, as reported in
[`protocol.md`](protocol.md#effect-of-the-alignment-on-preference-pairs).

## How corpus cases are labelled and bounded

A case extracted from a public corpus has no triage level of its own. The explicit rule reads it
and proposes a level, and the case is kept only if the rule recognised a sign. Its label
therefore carries a `medium` confidence, recorded in every example, against `high` for a
catalogue vignette.

Two bounds limit that contribution. In each cell of the level × language grid, corpus cases may
not exceed the number of places the cell has. Over the whole set, corpus cases are capped at 35%
of the examples, which keeps them a minority. Both bounds exist because corpus labels come from
the rule, and a set dominated by them would amount to a transcription of the rule.

## Personal data

No real patient data is used. The catalogue vignettes are synthetic. The public corpora are
research datasets without identifying content. The anonymisation pass still runs on both
languages, and a separate check then looks for anything it missed. The data card lists the
entities masked, the entities deliberately left unmasked, and the residue the check still reports
(see [`../data/README.md`](../data/README.md#personal-data)).

## Reusing this dataset

The derived dataset is published on the Hub under the repository named in `config.py`, with
[`../data/README.md`](../data/README.md) as its card. It inherits the conditions of its sources.
The CC BY 4.0 licence of MediQAl and MedQuAD requires attribution, which the card provides, and
the Apache-2.0 and MIT licences of the other sources are compatible with it.

The dataset has no clinical validity. The developer set its levels through a catalogue and a
keyword rule that no clinician has reviewed, and the dataset must not be used to train a system
that will be used on patients.
