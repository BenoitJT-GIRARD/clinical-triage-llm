# Bilingual emergency triage dataset

A training corpus for a triage decision-support agent. From a patient description (complaint,
symptoms, history, vital signs taken at reception, in French or in English), the model must produce
a **priority level**, a **clinical justification** and a **course of action**, always in French.

> This dataset is produced for a prototype. It contains no real patient data, and it has not
> been validated by an emergency physician. It must not be used to train a system used in a real
> clinical setting.

## Intended use

The set exists to train and evaluate a small language model on one task: reading a written patient
description and answering in a fixed three-line format. It serves to measure what a method,
supervised fine-tuning then preference alignment, changes in such a model, against an explicit
triage rule and an ordinary classifier trained on the same examples.

It is not clinical advice, and it is not a basis for any decision about a patient. It must not be
used to train, tune or validate a system that will be used in care: its labels come from a
catalogue that no emergency physician has reviewed (see [Limitations](#limitations)).

## Structure

### Files and subsets

<!-- source: data/processed/metadata.json -->
| File | Hub subset and split | Content | n | In the code repository |
|---|---|---|---|---|
| `sft_train.jsonl` | `sft` / `train` | prompt and answer pairs, supervised training | 4,000 | no |
| `sft_validation.jsonl` | `sft` / `validation` | the same, convergence tracking | 500 | no |
| `sft_test.jsonl` | `sft` / `test` | the same, test at the training distribution | 500 | no |
| `dpo_train.jsonl` | `dpo` / `train` | prompt, chosen and rejected answers, preference alignment | 2,400 | no |
| `clinical_eval.jsonl` | `clinical_eval` / `test` | hand-written cases, **evaluation, never training** | 60 | yes |
| `metadata.json` | none | schema, statistics, provenance, personal-data checks | n/a | yes |

The four training files are not versioned in the [code repository](https://github.com/BenoitJT-GIRARD/clinical-triage-llm): they are
derivatives of the code and of the public corpora, which `scripts/build_dataset.py` rebuilds
identically at a fixed seed, and they are published on the Hub. The other two are versioned
there as well: the annotated evaluation cases and the anonymisation check must be readable by
cloning the code repository, with nothing to download.

### Fields

The supervised files carry:

| Field | Content |
|---|---|
| `prompt` | complete ChatML prompt, opening the assistant turn |
| `completion` | expected answer: level, justification, recommendation |
| `user_turn` | the patient turn alone, used for deduplication and audit |
| `level` | `URGENCE_VITALE` · `URGENCE_MODEREE` · `CONSULTATION_DIFFEREE` |
| `lang` | language of the patient description (`fr` or `en`) |
| `source` | origin of the example |
| `confidence` | `high` (clinical catalogue) or `medium` (rule applied to a corpus) |
| `symptoms` | clinical signs present in the description |
| `medical_history` | history mentioned |
| `vitals` | vital-sign reading, empty when triage happens without measurements |
| `presentation_id` | source presentation, to trace back to the reference decision |

The preference file carries:

| Field | Content |
|---|---|
| `prompt` | ChatML prompt, identical to the supervised files' |
| `chosen` | preferred answer: right level, format respected, safe course of action |
| `rejected` | rejected answer, same format and comparable length |
| `user_turn` | the patient turn alone |
| `level` | reference triage level of the case |
| `lang` | language of the description |
| `strategy` | the defect introduced: `undertriage` · `unsafe_recommendation` · `asserted_diagnosis` · `answered_in_english` |
| `source` | always `safety_preference` |

And the clinical evaluation set:

| Field | Content |
|---|---|
| `prompt` | ChatML prompt, identical to the supervised files' |
| `completion` | always empty: the expected answer is not given, only the level is |
| `user_turn` | the patient turn alone, as it is submitted to the model |
| `id` | case identifier |
| `level` | reference level, written by hand |
| `lang` | language of the description |
| `case_type` | nature of the difficulty: empty · `falsely_reassuring` · `falsely_alarming` · `negation` · `discordant_vitals` |
| `description` | the patient description alone, without the framing prompt |
| `clinical_note` | clinical reason for the label, for auditability and error analysis |

These three lists are the ones in `metadata.json`, and a test of the repository checks that they
describe exactly the columns written into the files.

### Triage levels

Three levels, mapped onto the five of FRENCH (French Emergency Nurses Classification
in-Hospital), the triage scale used in French emergency departments:

| Level | Delay | FRENCH scale |
|---|---|---|
| `URGENCE_VITALE` | immediate care | levels 1 and 2 |
| `URGENCE_MODEREE` | a few hours | levels 3 and 4 |
| `CONSULTATION_DIFFEREE` | scheduled consultation | level 5 |

## Construction

Five public corpora were read: MediQAl, MedMCQA, FrenchMedMCQA and MedQuAD as sources of cases,
and UltraMedical-Preference as a source of preferences. The first task was to check what they
actually allow.

**None is annotated with triage levels.** They are sets of medical questions and answers and of exam
questions. Using them as they are (wrapping an exam question in a triage template and labelling it
by keyword presence) produces absurd examples of the kind "A patient presents with: *Levamisole is
used as all except -*", and a circular evaluation where the model merely relearns the rule that
produced the labels.

Five findings, checked on the corpora themselves:

<!-- source: data/processed/metadata.json -->
- **MediQAl is the only one made of patient cases.** Its `clinical_case` column carries 3,075
  distinct French vignettes: complaint, history, and for some of them the vital signs on arrival. It
  is the only authentic French-language source;
- **MedMCQA is the English source of vignettes.** Few of its exam questions describe a patient,
  but its size makes it the largest authentic source, and without it the English half of the
  authentic corpus would have no clinical vignette: MedQuAD describes conditions, not patients;
- **FrenchMedMCQA holds 1,080 questions** across its three splits; n = 1,080 entries read, of
  which only six are recognised as a patient presentation, and none carries an identifiable
  triage sign. It is a set of pharmacy questions; it cannot carry the French half of a triage
  dataset;
- **MedQuAD asks about conditions, not about patients.** Its symptom sheets are turned into
  patient complaints, which yields a few cases, but it does not describe clinical situations;
- **UltraMedical-Preference rewards length.** On its expert-reviewed test split, outside the
  pairs built to compare answers of similar length, the preferred answer is the longer one in
  470 pairs out of 597, a signal which, used for an alignment, teaches the model that "longer
  is better". The evaluation recounts it at every run, in
  [`reports/evaluation_results.json`](../reports/evaluation_results.json).

The corpus therefore rests on three contributions, and the evaluation on a fourth.

### 1. Generated clinical vignettes — the majority of the corpus

A **catalogue of typical presentations** met at an emergency reception desk was written for this
project: complaint, associated signs, plausible history, vital-sign profile, and **reference
triage level**. Adults, children, pregnancy, trauma, psychiatry, and the general-practice
complaints that crowd an emergency department.

A generator then dresses a presentation as a patient: age, sex, history, onset delay, a
vital-sign reading consistent with the severity, a varying wording. **The label comes from the
source presentation, never from re-reading the produced text.** That is what makes the
evaluation honest: there is no lexical rule to relearn.

Confidence level: `high`.

### 2. Cases extracted from the public corpora

The clinical vignettes of MediQAl and MedMCQA, and the symptom descriptions of MedQuAD, are kept
after the exam question is removed, then labelled by the project's explicit triage rule.

That label is **re-examined after anonymisation**: if masking removed the clinical sign that
justified it, the case is discarded rather than delivered with a label its own text no longer
supports.

One safety rule frames the label: **a case is kept only when the rule explicitly identifies a
sign**. The absence of a detected sign does not prove the absence of severity — "suspected
pneumoperitoneum" contains no alert word and remains a surgical emergency. Manufacturing a
"non-urgent" label out of a rule's silence would be dangerous.

Confidence level: `medium`.

Measured yield of each corpus, after filtering for the cases genuinely usable for triage:

<!-- source: data/processed/metadata.json -->
<!-- yield:start — table written by scripts/build_dataset.py, do not edit by hand -->
| Corpus | Entries read | No patient presentation recognised | Outside length bounds | No sign identified | Duplicates | Cases extracted | Cases delivered | Yield |
|---|---|---|---|---|---|---|---|---|
| MediQAl | 3,075 | 1,408 | 513 | 795 | 0 | 359 | **314** | **314 / 3,075** |
| MedQuAD | 16,407 | 15,909 | 0 | 315 | 0 | 183 | 130 | 130 / 16,407 |
| MedMCQA | 182,822 | 171,590 | 323 | 8,798 | 83 | 2,028 | 1,281 | 1,281 / 182,822 |
| FrenchMedMCQA | 1,080 | 1,074 | 0 | 6 | 0 | 0 | **0** | **0 / 1,080** |
n = 203,384 entries read across the four corpora, 1,725 cases delivered.
<!-- yield:end -->

> **How to read it.** One row per corpus, read left to right as a funnel. `Entries read` is
> everything the source holds; the four columns after it are the reasons an entry was dropped, the
> largest by far being that no patient presentation was recognised in the entry. `Cases extracted`
> is what survived, `Cases delivered` is what the balancing caps then let through, and `Yield` sets
> the second against the first.

The four corpora are read **in full**, with no read cap: a cap would give a yield describing the
limit we set ourselves and not the source. The loss columns add up with "cases extracted" to give
back the entries read; "cases delivered" is what remains after per-cell capping, anonymisation,
label review and final deduplication.

MediQAl has by far the best yield, and that is expected: its entries *are* patient cases, where
the two multiple-choice corpora contain them only incidentally. MedMCQA nonetheless supplies the
largest volume, by sheer size.

Three of the loss columns are matters of form, and are revisable: the pattern that recognises a
patient presentation, the length bounds, the deduplication. The fourth is a matter of safety: a
case is kept only when the triage rule explicitly identifies a sign there, because a "non-urgent"
label is not manufactured out of the silence of a keyword rule. That is what makes the
`CONSULTATION_DIFFEREE` class unreachable from the corpora, and therefore entirely drawn from the
catalogue.

FrenchMedMCQA's zero is not an oversight: these are pharmacy exam questions, with no patient
described. Forcing them in would reintroduce exactly the absurd examples this dataset was rebuilt
to eliminate.

### 3. The clinical evaluation set, written by hand

The cases in `clinical_eval.jsonl` are written one by one, with a different syntax and a
different vocabulary from the generator's templates. Nearly half of them are **atypical
presentations**: an emergency that looks benign, a spectacular but harmless symptom, a serious
sign explicitly denied, vital signs that contradict the narrative. Each case carries the clinical
reason for its label.

That set **never enters training**: the preparation script removes from the supervised set and from
the preference set any user turn identical to one of them, then checks again and fails if one
remains.

### 4. Preference pairs for the alignment

UltraMedical-Preference is not used here for training: its answers are long English essays, when
the output contract fits in three French lines, and a large share of its pairs is decided by
answer length alone. It is reserved for evaluation, as an independent measure of alignment.

The pairs in `dpo_train.jsonl` are built from the **training split alone** of the supervised set.
For each prompt, the reference answer is paired with a degraded variant following one of four
strategies: an underestimated level, a course of action that delays care, a diagnosis presented
as certain, an answer outside the expected language.

Three rules frame the construction:

1. **same format, closest length.** Each defect exists in several lengths and the one closest to the
   preferred answer is kept. Without that, the only systematic difference between the two answers
   would be length, and length is what the alignment would learn;
2. **never an escalation as the rejected answer.** The system prompt requires escalating at the
   slightest doubt; presenting an over-cautious answer as the bad example would teach the opposite;
3. **serious cases weigh more**, undertriaging a life-threatening case being the costliest
   failure.

## Personal data

**Minimisation by design.** No real patient data enters the project: the vignettes are synthetic,
and the public corpora are research sets with no identifying data.

<!-- source: data/processed/metadata.json -->
**Anonymisation.** The texts coming from the corpora go through Presidio, with a setting adapted
to medical text. The default setting is unusable here, and the build measures the gap at every
run: on 400 narratives of the corpora, each analysed in its own language, the `DATE_TIME`,
`LOCATION` and `NRP` entities would mask a fragment of the narrative in 319 of them, 80%.
`DATE_TIME` alone fires 453 times: it takes away the onset delays and the patient's age.
`LOCATION` fires 106 times, on the names of organs and of hospital departments. Delay, age and
vital signs are precisely what decides a triage level. Three decisions follow:

1. only genuinely identifying entities are masked: name, phone, email address, banking
   identifiers, IP address, URL, social security number, date of birth. Five recognisers absent
   from Presidio were added and registered in both languages: the social security number, the
   phone number in national and international form, and the date of birth in DD/MM/YYYY as well
   as YYYY-MM-DD;
2. the project's clinical vocabulary is protected — no term of the catalogue or of the triage
   lexicon can be masked. That guard targets the entities that are *kept*, which also derail on
   medical text: without it, "inhibiteurs de recapture de la sérotonine" becomes "inhibiteurs de
   recapture de la `<PERSON>`";
3. the quality check is **independent of the detector**: a separate set of regular expressions
   looks, after masking, for what might have slipped through. Its results are published in
   `metadata.json`.

The occurrences that check still reports under the "title followed by a name" pattern are all
the same surname, "Mr Shoot", invented by the MediQAl examination corpus for a vignette of
agitation after taking heroin or cocaine: no real patient name survives in the set. The masking
is inconsistent all the same — in that very record, a later occurrence was indeed replaced by
`<PERSON>`.

**Auditability.** Every transformation is in the code, the seed is fixed, and the repository
revision that produced the set is recorded in `metadata.json`.

**Split separation.** Checked by code: the preparation script fails if a user turn appears in two
splits, or if an evaluation case ends up in training. The counts are published in `metadata.json`.

## Limitations

<!-- source: data/processed/metadata.json -->
- **Catalogue not clinically validated**: written by an engineer from the triage literature, and
  by the same author as the triage rule and the evaluation set. There is no external ground
  truth. That is the dataset's main limit.
- **Synthetic vignettes**: varied and clinically coherent, but without the disorder of real
  language — narratives reported by a third party, contradictory information, patients who play
  down their symptoms.
- **Few distinct answers**: the generated vignettes come from 70 presentations, and share 70
  expected answers between 3,275 examples. A model can learn the answers rather than the
  reasoning, and the build publishes the diagnostics that measure it in `metadata.json`.
- **A level partly readable from the form**: the onset delay and the vital-sign profile alone
  guess the level of a generated vignette nine times out of ten. The hand-written evaluation set
  exists because of it.

## Licences and attribution

The dataset is published under CC BY 4.0. It redistributes text derived from two corpora under
that licence, and the attribution below is part of every reuse.

| Source | Language | Licence | What the set takes from it, and what was done to it |
|---|---|---|---|
| The project's presentation catalogue | fr + en | MIT, as the project's code | the generated vignettes and their reference level |
| [`ANR-MALADES/MediQAl`](https://huggingface.co/datasets/ANR-MALADES/MediQAl) | fr | CC BY 4.0 | **French clinical vignettes**, the only source corpus made of patient cases: exam question removed, identifying entities masked, labelled by the rule |
| [`keivalya/MedQuad-MedicalQnADataset`](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) | en | CC BY 4.0 | symptom sheets turned into patient complaints, masked, labelled by the rule |
| [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa) | en | Apache-2.0 | exam clinical vignettes: exam question removed, masked, labelled by the rule |
| [`nthngdy/frenchmedmcqa`](https://huggingface.co/datasets/nthngdy/frenchmedmcqa) | fr | Apache-2.0 | read in full; no case kept, nothing redistributed |
| [`TsinghuaC3I/UltraMedical-Preference`](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference) | en | MIT | evaluation only; nothing redistributed |

Two of these repositories declare no licence on the Hub: `MedQuad-MedicalQnADataset` and
`frenchmedmcqa` are mirrors. The licences reported above are those of their original
repositories, where they were read — the `LICENSE.txt` of
[`abachaa/MedQuAD`](https://github.com/abachaa/MedQuAD) is the CC BY 4.0 text, and
[`qanastek/frenchmedmcqa`](https://huggingface.co/datasets/qanastek/frenchmedmcqa) declares
Apache-2.0. The mirrors are used because the original FrenchMedMCQA repository exposes its data
only through a loading script, which `datasets` no longer executes.

## Reproducing

```bash
uv run python scripts/build_dataset.py
```

Fixed seed, deterministic outputs at constant corpus version.
