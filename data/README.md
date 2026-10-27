# Bilingual emergency triage dataset

A training corpus for a triage decision-support agent. From a patient description written in
French or in English (complaint, symptoms, history, vital signs taken at reception), the model
must produce a **priority level**, a **clinical justification** and a **course of action**,
always in French.

> This dataset was produced for a prototype. It contains no real patient data, and no emergency
> physician has validated it. It must not be used to train a system intended for real clinical
> use.

## Intended use

The set is used to train and evaluate a small language model on one task: reading a written
patient description and answering in a fixed three-line format. It measures what a method,
supervised fine-tuning followed by preference alignment, changes in such a model, compared with
an explicit triage rule and with an ordinary classifier trained on the same examples.

It is not clinical advice and must not serve as a basis for any decision about a patient. It must
not be used to train, tune or validate a system intended for care, because its labels come from a
catalogue and a keyword rule, both written by the developer and never reviewed by an emergency
physician (see [Limitations](#limitations)).

## Structure

### Files and subsets

<!-- source: data/processed/metadata.json -->
| File | Hub subset and split | Content | n | In the code repository |
|---|---|---|---|---|
| `sft_train.jsonl` | `sft` / `train` | prompt and answer pairs for supervised training | 4,000 | no |
| `sft_validation.jsonl` | `sft` / `validation` | the same format, for tracking convergence | 500 | no |
| `sft_test.jsonl` | `sft` / `test` | the same format, for a test on the training distribution | 500 | no |
| `dpo_train.jsonl` | `dpo` / `train` | a prompt with a chosen and a rejected answer, for preference alignment | 2,400 | no |
| `clinical_eval.jsonl` | `clinical_eval` / `test` | hand-written cases, **used for evaluation only** | 60 | yes |
| `metadata.json` | none | schema, statistics, provenance, personal-data checks | n/a | yes |

The four training files are not versioned in the
[code repository](https://github.com/BenoitJT-GIRARD/clinical-triage-llm). They derive from the
code and the public corpora, `scripts/build_dataset.py` rebuilds them identically with a fixed
seed, and they are published on the Hub. The other two files are versioned in the code repository
as well, so that the annotated evaluation cases and the anonymisation check can be read from a
clone without downloading anything.

### Fields

The supervised files contain these fields:

| Field | Content |
|---|---|
| `prompt` | the full ChatML prompt, ending where the assistant turn opens |
| `completion` | the expected answer, made of a level, a justification and a recommendation |
| `user_turn` | the patient turn on its own, used for deduplication and audit |
| `level` | `URGENCE_VITALE` · `URGENCE_MODEREE` · `CONSULTATION_DIFFEREE` |
| `lang` | the language of the patient description (`fr` or `en`) |
| `source` | where the example comes from |
| `confidence` | `high` for the clinical catalogue, `medium` for a corpus case labelled by the rule |
| `symptoms` | the clinical signs present in the description |
| `medical_history` | the history mentioned in the description |
| `vitals` | the vital-sign reading, empty when the triage is made without measurements |
| `presentation_id` | the catalogue presentation the example was generated from, empty for corpus cases |

The preference file contains:

| Field | Content |
|---|---|
| `prompt` | the ChatML prompt, built as in the supervised files |
| `chosen` | the preferred answer, with the right level, the expected format and a safe course of action |
| `rejected` | the rejected answer, in the same format and of comparable length |
| `user_turn` | the patient turn on its own |
| `level` | the reference triage level of the case |
| `lang` | the language of the description |
| `strategy` | the defect introduced: `undertriage` · `unsafe_recommendation` · `asserted_diagnosis` · `answered_in_english` |
| `source` | always `safety_preference` |

And the clinical evaluation set contains:

| Field | Content |
|---|---|
| `prompt` | the ChatML prompt, built as in the supervised files |
| `completion` | always empty, because only the reference level is provided |
| `user_turn` | the patient turn on its own, exactly as the model receives it |
| `id` | the case identifier |
| `level` | the reference level, written by hand |
| `lang` | the language of the description |
| `case_type` | the kind of difficulty: empty · `falsely_reassuring` · `falsely_alarming` · `negation` · `discordant_vitals` |
| `description` | the patient description without the surrounding prompt |
| `clinical_note` | the clinical reason for the label, kept for audit and error analysis |

`metadata.json` holds these three lists, and a test in the repository checks that they match the
columns written to the files exactly.

### Triage levels

The three levels map onto the five levels of FRENCH (French Emergency Nurses Classification
in-Hospital), the triage scale used in French emergency departments:

| Level | Delay | FRENCH scale |
|---|---|---|
| `URGENCE_VITALE` | immediate care | levels 1 and 2 |
| `URGENCE_MODEREE` | a few hours | levels 3 and 4 |
| `CONSULTATION_DIFFEREE` | scheduled consultation | level 5 |

## Construction

The project read five public corpora. Four were candidate sources of cases: MediQAl, MedMCQA,
FrenchMedMCQA and MedQuAD. The fifth, UltraMedical-Preference, was a candidate source of
preference pairs. The first step was to check what each of them can support.

None of them is annotated with triage levels. They are collections of medical questions and
answers and of exam questions. Used as they are, by wrapping an exam question in a triage
template and labelling it from its keywords, they produce absurd examples such as "A patient
presents with: *Levamisole is used as all except -*", and a circular evaluation in which the
model only relearns the rule that produced its labels.

Five findings, checked on the corpora themselves:

<!-- source: data/processed/metadata.json -->
- **MediQAl is the only corpus made of patient cases.** Its `clinical_case` column holds 3,075
  distinct French vignettes, each with a complaint and a history, and some with the vital signs
  recorded on arrival. It is the only French source of patient cases written outside the project;
- **MedMCQA provides the English vignettes.** Only a small share of its exam questions describe
  a patient, and the corpus is large enough for that share to make it the largest source of
  cases written outside the project. Without it, the
  English half of those cases would contain no clinical vignette, since MedQuAD describes
  conditions;
- **FrenchMedMCQA holds 1,080 questions** across its three splits. All n = 1,080 entries were
  read, six of them are recognised as a patient presentation, and none of those six carries an
  identifiable triage sign. The corpus is a set of pharmacy questions and cannot supply the French
  half of a triage dataset;
- **MedQuAD describes conditions.** Its symptom sheets are rewritten as patient complaints, which
  yields a few cases, and it contains no clinical situations;
- **UltraMedical-Preference favours length.** On its expert-reviewed test split, the preferred
  answer is the longer one in 470 of the 597 pairs that do not belong to the group built to
  compare answers of similar length. Used as a training signal, it would teach the model to
  prefer longer answers. The evaluation recounts these pairs at every run, in
  [`reports/evaluation_results.json`](../reports/evaluation_results.json).

The dataset is therefore built from three contributions, generated vignettes, corpus cases and
preference pairs, and the evaluation relies on a fourth, the hand-written clinical set.

### Generated clinical vignettes, the majority of the corpus

A catalogue of typical presentations at an emergency reception desk was written for this project.
Each presentation has a complaint, associated signs, a plausible history, a vital-sign profile and
a reference triage level. The catalogue covers adults, children, pregnancy, trauma and psychiatry,
along with the general-practice complaints that also reach emergency departments.

A generator turns a presentation into a patient by drawing an age, a sex, a history, an onset
delay, vital signs consistent with the severity, and a wording. **The label comes from the
presentation, and the generated text is never read back to set it.** The model therefore has no
lexical rule to relearn, and the evaluation is not circular.

Confidence level: `high`.

### Cases extracted from the public corpora

The clinical vignettes of MediQAl and MedMCQA, and the symptom descriptions of MedQuAD, are kept
once the exam question is removed, then labelled by the project's explicit triage rule.

The label is checked again after anonymisation. If masking removed the clinical sign that
justified the label, the case is discarded, because its text no longer supports the label.

A safety rule constrains the labels: **a case is kept only if the rule identifies a sign in it
explicitly**. When the rule finds no sign, the case may still be serious: "suspected
pneumoperitoneum" contains no alert word and is a surgical emergency. Labelling a case
"non-urgent" because the rule found nothing in it would be dangerous.

Confidence level: `medium`.

The yield of each corpus, after filtering for the cases usable in triage:

<!-- source: data/processed/metadata.json -->
<!-- yield:start — table written by scripts/build_dataset.py, do not edit by hand -->
| Corpus | Entries read | No patient presentation recognised | Outside length bounds | No sign identified | Duplicates | Cases extracted | Cases delivered | Yield |
|---|---|---|---|---|---|---|---|---|
| MediQAl | 3,075 | 1,408 | 513 | 795 | 0 | 359 | **314** | **314 / 3,075** |
| MedQuAD | 16,407 | 15,909 | 0 | 315 | 0 | 183 | 130 | 130 / 16,407 |
| MedMCQA | 182,822 | 171,590 | 323 | 8,798 | 83 | 2,028 | 1,281 | 1,281 / 182,822 |
| FrenchMedMCQA | 1,080 | 1,074 | 0 | 6 | 0 | 0 | **0** | **0 / 1,080** |
> n = 203,384 entries read across the four corpora, 1,725 cases delivered.
<!-- yield:end -->

> **How to read it.** Each row is one corpus, read from left to right as a funnel. `Entries read`
> counts everything the source holds. The four columns after it count the reasons an entry was
> dropped, and by far the most frequent is that no patient presentation was recognised.
> `Cases extracted` counts the entries that passed these filters. `Cases delivered` counts what
> remains after the per-cell caps, the global cap of 35% of the set, the anonymisation, the label
> review and the final deduplication, and `Yield` sets the cases delivered against the entries
> read.

The four corpora are read in full, with no limit on the number of entries, because a limit would
make the yield describe that limit. In each row, the four loss columns and `Cases extracted` add
up to the entries read.

MediQAl has by far the best yield, as expected, since its entries are patient cases while the two
multiple-choice corpora contain such cases only occasionally. MedMCQA still supplies the largest
number of cases because of its size.

Three of the loss columns reflect choices of form and can be revised: the pattern that recognises
a patient presentation, the length bounds and the deduplication. The fourth reflects a safety
choice, since a corpus case is kept only where the triage rule finds a sign, and no case is
labelled "non-urgent" merely because a keyword rule found nothing in it. For the same reason the
corpora cannot supply the `CONSULTATION_DIFFEREE` class, which comes entirely from the catalogue.

FrenchMedMCQA yields nothing because its entries are pharmacy exam questions. Almost none of them
describes a patient, and the few that do contain no sign the rule can read. Adding them anyway
would bring back the absurd examples that this dataset was rebuilt to remove.

### The clinical evaluation set, written by hand

The cases in `clinical_eval.jsonl` were written one by one, with a syntax and a vocabulary that
differ from the generator's templates. Nearly half of them are atypical presentations: an
emergency that sounds benign, a striking but harmless symptom, a serious sign that the narrative
explicitly denies, or vital signs at odds with the story. Each case records the clinical reason
for its label.

The set **is kept out of training**. The preparation script removes from the supervised set any
patient turn that matches an evaluation case once case, accents and punctuation are ignored,
builds the preference pairs from the training split only, and fails if an evaluation case is still
found in the training split.

### Preference pairs for the alignment

UltraMedical-Preference is not used for training here. Its answers are long English essays,
while the output format fits in three French lines, and among its pairs not built to match
lengths the preferred answer is usually the longer one. It is kept for evaluation, as an
independent measure of the alignment.

The pairs in `dpo_train.jsonl` are built from the **training split only** of the supervised set.
For each prompt, the reference answer is paired with a degraded variant that follows one of four
strategies: a level that is too low, a course of action that delays care, a diagnosis stated as
certain, or an answer outside the expected language.

Three rules govern the construction:

1. **same format, similar length.** Each defect is written in several lengths, the two closest in
   length to the preferred answer are kept, and one of them is drawn at random. Without this,
   length would be the only systematic difference between the two answers, and the alignment
   would learn length;
2. **no escalation as a rejected answer.** The system prompt asks for escalation at the slightest
   doubt, and an over-cautious answer presented as the bad example would teach the opposite;
3. **serious cases are over-represented.** Undertriaging a life-threatening case is the costliest
   failure, so life-threatening cases get three times the sampling weight of deferred cases, and
   moderate cases twice the weight of deferred cases.

## Personal data

**Minimisation by design.** No real patient data enters the project. The vignettes are generated,
and the public corpora are research collections that identify no one.

<!-- source: data/processed/metadata.json -->
**Anonymisation.** Texts from the corpora go through Presidio, configured for medical text. The
default configuration is unusable here, and the build measures the damage it would do at every
run. On a sample of 400 corpus narratives, each analysed in its own language, the `DATE_TIME`,
`LOCATION` and `NRP` entities would mask part of the narrative in 319 of them, or 80%.
`DATE_TIME` alone fires 453 times and removes onset delays and patient ages. `LOCATION` fires 106
times, notably on names of organs and of hospital departments, and once on `TA`, the French
abbreviation for blood pressure. The default configuration also mislabels
the names of molecules, and turns `inhibiteurs de recapture de la sérotonine` into
`inhibiteurs de recapture de la <PERSON>`. Delay, age and vital signs are exactly the information
that decides a triage level. Three decisions follow:

1. only identifying entities are masked: name, phone number, email address, banking identifiers,
   IP address, URL, social security number and date of birth. Five recognisers missing from
   Presidio were added and registered for both languages. They detect the French social security
   number, phone numbers in national and in international format, and dates of birth written
   DD/MM/YYYY or YYYY-MM-DD;
2. the project's clinical vocabulary is protected, so that no term from the catalogue or from the
   rule's keyword lists can be masked. The protection also covers the entity types that stay
   enabled, such as `PERSON`, which can misfire on medical text as well;
3. the quality check is **independent of the detector**. A separate set of regular expressions
   searches the masked text for anything that slipped through, and `metadata.json` publishes the
   results.

The occurrences that this check still reports under the "title followed by a name" pattern are
all the same surname, Shoot, written after a French title, which the MediQAl exam corpus invented
for a vignette of agitation after heroin or cocaine use. The check finds no other name after a
title. The masking is inconsistent on that record, since a later occurrence of the name in it was
replaced by `<PERSON>`.

**Auditability.** Every transformation is in the code and the seed is fixed. `metadata.json`
records the build date and a repository revision, `cb0a0d4`. That revision was written into the
file by hand the day after the build, so it does not identify the commit that ran the build.

**Split separation.** The code enforces it. The preparation script fails if a patient turn
appears in two splits, or if an evaluation case lands in the training split, and `metadata.json`
publishes the overlap counts.

## Limitations

<!-- source: data/processed/metadata.json -->
- **Catalogue not clinically validated.** An engineer wrote it from the triage literature, and
  the same author wrote the triage rule and the evaluation set. There is no external ground
  truth, and this is the main limitation of the dataset.
- **Synthetic vignettes.** They are varied and clinically coherent, and they lack the disorder of
  real language: narratives reported by a third party, contradictory information, patients who
  play down their symptoms.
- **Few distinct answers.** The generated vignettes come from 70 presentations and share 70
  expected answers across 3,275 examples. A model can memorise the answers without learning the
  reasoning, and the build publishes in `metadata.json` the diagnostics that measure this.
- **A level partly predictable from the metadata.** The onset delay and the vital-sign profile
  alone predict the level of a generated vignette in nine cases out of ten. The hand-written
  evaluation set exists for this reason.

## Licences and attribution

The dataset is published under CC BY 4.0. It redistributes text derived from two corpora under
that licence, and the attribution below must accompany every reuse.

| Source | Language | Licence | What the set takes from it, and what was done to it |
|---|---|---|---|
| The project's presentation catalogue | fr + en | MIT, like the project's code | the generated vignettes and their reference level |
| [`ANR-MALADES/MediQAl`](https://huggingface.co/datasets/ANR-MALADES/MediQAl) | fr | CC BY 4.0 | **French clinical vignettes**, from the only source corpus made of patient cases: exam question removed, identifying entities masked, labelled by the rule |
| [`keivalya/MedQuad-MedicalQnADataset`](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) | en | CC BY 4.0 | symptom sheets rewritten as patient complaints, masked, labelled by the rule |
| [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa) | en | Apache-2.0 | clinical vignettes from exams: exam question removed, masked, labelled by the rule |
| [`nthngdy/frenchmedmcqa`](https://huggingface.co/datasets/nthngdy/frenchmedmcqa) | fr | Apache-2.0 | read in full; no case kept, nothing redistributed |
| [`TsinghuaC3I/UltraMedical-Preference`](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference) | en | MIT | evaluation only; nothing redistributed |

Two of these repositories, `MedQuad-MedicalQnADataset` and `frenchmedmcqa`, both mirrors, declare
no licence on the Hub. The licences given above are those of the original repositories,
where they were checked: the `LICENSE.txt` of [`abachaa/MedQuAD`](https://github.com/abachaa/MedQuAD)
is the CC BY 4.0 text, and [`qanastek/frenchmedmcqa`](https://huggingface.co/datasets/qanastek/frenchmedmcqa)
declares Apache-2.0. The mirrors are used because the original FrenchMedMCQA repository provides
its data only through a loading script, which `datasets` no longer runs.

## Reproducing

```bash
uv run python scripts/build_dataset.py
```

The seed is fixed, and the outputs are deterministic as long as the corpus versions stay the
same.
