---
license: mit
language:
  - fr
  - en
task_categories:
  - text-generation
tags:
  - medical
  - triage
  - emergency-department
  - french
size_categories:
  - 1K<n<10K
---

# Bilingual emergency triage dataset

A training corpus for a triage decision-support agent. From a patient description — complaint,
symptoms, history, vital signs taken at reception, in French or in English — the model must
produce a **priority level**, a **clinical justification** and a **course of action**, always in
French.

> This dataset is produced for a prototype. It contains no real patient data, and it has not
> been validated by an emergency physician. It must not be used to train a system used in a real
> clinical setting.

## Files

| File | Content | Use | In the repository |
|---|---|---|---|
| `sft_train.jsonl` | prompt / answer pairs | supervised training | no |
| `sft_validation.jsonl` | idem | convergence tracking | no |
| `sft_test.jsonl` | idem | test at the training distribution | no |
| `dpo_train.jsonl` | prompt / chosen / rejected triples | preference alignment | no |
| `clinical_eval.jsonl` | hand-written cases | **evaluation, never training** | yes |
| `metadata.json` | schema, statistics, provenance, GDPR, checks | auditability | yes |

The four training sets are not versioned: they are derivatives of the code and of the public
corpora, which `scripts/build_dataset.py` rebuilds identically at a fixed seed, and which are
published on the Hub. The other two are versioned: the annotated evaluation cases and the
anonymisation check must be readable by cloning the repository, with nothing to download.

## How this corpus was built, and why

Four public corpora were available: MediQAl, FrenchMedMCQA, MedQuAD and UltraMedical-Preference.
The first task was to check what they actually allow.

**None is annotated with triage levels.** They are sets of medical questions and answers and of
exam questions. Using them as they are — wrapping an exam question in a triage template and
labelling it by keyword presence — produces absurd examples of the kind "A patient presents
with: *Levamisole is used as all except -*", and a circular evaluation where the model merely
relearns the rule that produced the labels.

Four findings, checked on the corpora themselves:

- **MediQAl is the only one that describes patients.** Its `clinical_case` column carries 3,075
  distinct French vignettes — complaint, history, and for 569 of them the vital signs on arrival.
  It is the only authentic French-language source;
- **FrenchMedMCQA holds 1,080 questions** across its three splits, of which only six are
  recognised as a patient presentation — and none carries an identifiable triage sign. It is a
  set of pharmacy questions; it cannot carry the French half of a triage dataset;
- **MedQuAD asks about conditions, not about patients.** Its symptom sheets are turned into
  patient complaints, which yields a few cases, but it does not describe clinical situations;
- **UltraMedical-Preference labels 38.6% of its pairs by answer length alone**, measured on
  5,000 rows — a signal which, used for an alignment, teaches the model that "longer is better".

The corpus therefore rests on three contributions.

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

### 3. The clinical evaluation set, written by hand

The cases in `clinical_eval.jsonl` are written one by one, with a different syntax and a
different vocabulary from the generator's templates. Nearly half of them are **atypical
presentations**: an emergency that looks benign, a spectacular but harmless symptom, a serious
sign explicitly denied, vital signs that contradict the narrative. Each case carries the clinical
reason for its label.

That set **never enters training**: the preparation script removes from the supervised set and
from the preference set any user turn identical to one of them, and fails if one is found there.

### 4. Preference pairs for the alignment

UltraMedical-Preference is not used here for training: its answers are long English essays, when
the output contract fits in three French lines, and 38.6% of its pairs are labelled by answer
length alone. It is reserved for evaluation, as an independent measure of alignment.

The pairs in `dpo_train.jsonl` are built from the **training split alone** of the supervised set.
For each prompt, the reference answer is opposed to a degraded variant following one of four
strategies: an underestimated level, a course of action that delays care, a diagnosis presented
as certain, an answer outside the expected language.

Three rules frame the construction:

1. **same format, same length.** Each defect exists in several lengths and the one closest to the
   preferred answer is kept. Without that, the only systematic difference between the two answers
   would be length, and length is what the alignment would learn;
2. **never an escalation as the rejected answer.** The system prompt requires escalating at the
   slightest doubt; opposing an over-cautious answer as a bad example would teach the opposite;
3. **serious cases weigh more**, undertriaging a life-threatening case being the costliest
   failure.

## Schema

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
| `vitals` | vital-sign reading, empty when sorting happens without measurements |
| `presentation_id` | source presentation, to trace back to the reference decision |

The other two files do not carry those columns. The preference set carries:

| Field | Content |
|---|---|
| `prompt` | ChatML prompt, identical to the supervised set's |
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
| `prompt` | ChatML prompt, identical to the supervised set's |
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

## Taxonomy

Three levels, mapped onto the FRENCH scale used in French emergency departments:

| Level | Delay | FRENCH scale |
|---|---|---|
| `URGENCE_VITALE` | immediate care | sorts 1 and 2 |
| `URGENCE_MODEREE` | a few hours | sorts 3 and 4 |
| `CONSULTATION_DIFFEREE` | scheduled consultation | sort 5 |

## Sources and licences

| Source | Language | Licence | Role |
|---|---|---|---|
| The project's presentation catalogue | fr + en | MIT | triage ground truth |
| [`ANR-MALADES/MediQAl`](https://huggingface.co/datasets/ANR-MALADES/MediQAl) | fr | CC BY 4.0 | **French clinical vignettes** — the only source corpus that describes patients |
| [`keivalya/MedQuad-MedicalQnADataset`](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) | en | CC BY 4.0 | authentic symptom descriptions |
| [`nthngdy/frenchmedmcqa`](https://huggingface.co/datasets/nthngdy/frenchmedmcqa) | fr | Apache-2.0 | the second French-language source corpus |
| [`TsinghuaC3I/UltraMedical-Preference`](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference) | en | MIT | external preference set |
| [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa) | en | Apache-2.0 | exam clinical vignettes *(added)* |

Two of these repositories declare no licence on the Hub: `MedQuad-MedicalQnADataset` and
`frenchmedmcqa` are mirrors. The licences reported above are those of their original
repositories, where they were read — the `LICENSE.txt` of
[`abachaa/MedQuAD`](https://github.com/abachaa/MedQuAD) is the CC BY 4.0 text, and
[`qanastek/frenchmedmcqa`](https://huggingface.co/datasets/qanastek/frenchmedmcqa) declares
Apache-2.0. The mirrors are used because the original FrenchMedMCQA repository exposes its data
only through a loading script, which `datasets` no longer executes.

Measured yield of each corpus, after filtering for the cases genuinely usable for triage:

<!-- yield:start — table written by scripts/build_dataset.py, do not edit by hand -->
| Corpus | Entries read | No patient described | Outside length bounds | No sign identified | Duplicates | Cases extracted | Cases delivered | Yield |
|---|---|---|---|---|---|---|---|---|
| MediQAl | 3,075 | 1,407 | 514 | 792 | 0 | 362 | **313** | **10.2%** |
| MedQuAD | 16,407 | 15,909 | 0 | 315 | 0 | 183 | 120 | 0.7% |
| MedMCQA | 182,822 | 171,251 | 372 | 9,038 | 71 | 2,090 | 1,283 | 0.7% |
| FrenchMedMCQA | 1,080 | 1,074 | 0 | 6 | 0 | 0 | **0** | **0.0%** |
<!-- yield:end -->

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

MedMCQA was added because the three usable corpora available are either French-language —
MediQAl, FrenchMedMCQA — or patient-free — MedQuAD. Without it, the English half of the authentic
corpus would have no clinical vignette at all. The addition is documented rather than passed over.

## GDPR compliance

**Minimisation by design.** No real patient data enters the project: the vignettes are synthetic,
and the public corpora are research sets with no identifying data.

**Anonymisation.** The texts coming from the corpora go through Presidio, with a setting adapted
to medical text. The default setting is unusable here, and the gap was measured: on 400 examples
analysed each in its own language, the `DATE_TIME`, `LOCATION` and `NRP` entities would mask a
fragment of the narrative in 81% of cases. `DATE_TIME` takes away the onset delays and the
patient's age; `LOCATION` takes away `TA`, the French abbreviation for blood pressure, one hundred
and thirty-six times on its own. Delay, age and vital signs are precisely what decides a triage
level. Three decisions follow:

1. only genuinely identifying entities are masked: name, phone, email address, banking
   identifiers, IP address, URL, social security number, date of birth. Four recognisers absent
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

## Reproducing

```bash
uv run python scripts/build_dataset.py
```

Fixed seed, deterministic outputs at constant corpus version.

## Known limits

- **Synthetic vignettes**: varied and clinically coherent, but without the disorder of real
  language — narratives reported by a third party, contradictory information, patients who play
  down their symptoms.
- **Catalogue not clinically validated**: written by an engineer from the triage literature. That
  is the dataset's main limit.
