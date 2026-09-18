# Changelog

Versions follow [SemVer](https://semver.org/). The dataset format and the API contract are part
of this project's public interface: a breaking change to either raises the major version.

## 1.0.0

The first complete version of the prototype.

### Data

- A catalogue of clinical presentations, written for the project, now carries the triage ground
  truth in place of keyword labelling.
- A generator draws bilingual clinical vignettes from it, with an age, a history, an onset delay
  and vital signs consistent with the severity.
- A clinical evaluation set written by hand, nearly half of it atypical presentations, kept out
  of every training set.
- Public corpora are filtered: a case is kept only where a clinical sign is actually identified,
  and it carries an explicit confidence level.
- MediQAl is integrated: 3,075 distinct French clinical vignettes, 569 of them with readable
  vital signs. It is the only source corpus that describes patients, and it now carries the
  authentic French-language volume.
- Anonymisation is restricted to genuinely identifying entities, with the clinical vocabulary
  protected and a quality check independent of the detector.
- Paediatric vignettes carry no self-reported pain before the age of six: the visual analogue
  scale does not exist at that age.
- Split separation is verified by the code, and the build fails on a leak.
- The four corpora are read in full, with no ceiling on entries: 182,822 rows for
  MedMCQA, 16,407 for MedQuAD, the three splits of FrenchMedMCQA. The published yield therefore
  describes whole sources, and the corpora supply a third of the delivered set.
- The marker that recognises a patient presentation accepts the syntax MediQAl uses ("Homme, 58
  ans", "M. Dupont, 30 ans") and not only "homme de 58 ans". Without that, 342 authentic French
  vignettes were discarded unexamined.
- An upper length bound of 800 characters, the value at which no delivered case exceeds the
  service's description budget. It is checked at every build against the real tokenizer, on the
  cases actually delivered, not on the extraction pool, and the build fails if one case passes
  it.
- Symptom sheets are cut at a sentence boundary: 120 of the 172 MedQuAD cases
  delivered used to end mid-word. The preamble that introduces them counts towards the bound,
  without which 178 descriptions exceeded it.
- An examination question is distinguished from a relative pronoun. Dropping every sentence
  containing "which" or "quel" took clinical narrative with it, and let through the commonest
  MedMCQA forms, which use no interrogative word at all. The residue of quiz wording falls to
  half a percent of the set.
- Corpus cases are deduplicated on the description. The patient turn is only a randomly drawn
  wrapper around it.
- A narrative that already ends in a question mark does not receive a full stop as well.
- A corpus read interrupted part-way stops the build. In streaming mode all the traffic happens
  during iteration: a cut would produce a truncated corpus whose published yield described a
  complete read.
- An unavailable corpus fails the build instead of silently producing a truncated set.
- The corpus contribution is bounded per level and language cell before being bounded globally,
  without which the best-supplied English cell overflows the target volume and breaks the balance
  between languages.
- `--sft-size` is validated: below the threshold where the validation and test splits would come
  out empty, the build refuses to start.
- The dataset card records the command actually run, options included, states explicitly when
  anonymisation was turned off, and publishes the extraction funnel corpus by corpus.
- Three diagnostics of the produced set are published: the number of distinct expected answers,
  separability under a split grouped by presentation, and the share of the level predictable from
  the generator's metadata alone.

### Model

- A dialogue template installed by the project, and a corrected end-of-sequence token, shared by
  training, alignment, evaluation and serving.
- The output head is trained alongside the projections. In the base model the ChatML tokens are a
  single untrained vector and the head is tied to the embeddings, which made the end token
  impossible to produce. Clean stops go from none to all, and the answer from 220 tokens to 97.
- Fine-tuning moved to Unsloth: the same configuration in 7.23 GB instead of 10.03.
- Adapter merging corrected. The output head is untied **before** the merge and the computation
  is done in float32, without which the delta was written into the shared embedding matrix. The
  export refuses to finish if the verification fails.
- The supervised set is exposed as `prompt` and `completion` columns: the loss is computed on the
  answer alone, with no special collator.
- Four hyper-parameter settings are compared, judged on the validation loss and on triage
  accuracy, and published with their confidence intervals. On sixty validation cases all four
  overlap, and replaying the comparison changes the order of the middle places. It establishes
  that no setting degrades the result, not that one wins.
- DPO alignment with the supervised model as reference, a preference loss combined with a
  supervised term that holds the drift back, and a preference validation set.
- Preference pairs share their format and are of comparable length; no overtriage is presented as
  a counter-example.

### Evaluation

- Three baselines: majority class, maximum caution, explicit rule.
- No estimate is published bare, and the estimator follows the nature of the quantity. Wilson
  above thirty cases, exact Clopper-Pearson below and on undertriage (the one measure whose
  under-coverage is paid in patients), and nothing at all below six cases, where the raw fraction
  replaces a rate the effective cannot support. Exhaustive corpus counts carry no error bar: they
  are counted, not estimated.
- Comparisons are read from the right test. The overlap of two intervals proves nothing; what
  is used is exact McNemar where the systems see the same cases, which is the model against the
  explicit rule, and a Newcombe interval on the gap where the two sets are independent.
- Figures brought up to publication standard: axes and units on every panel, effective and
  estimator at the foot of every figure, a shared colour scale and a scale bar on the confusion
  matrices, and a missing measurement that is no longer drawn as a zero.
- Safety checks on the generated content: an inconsistent recommendation, an asserted diagnosis,
  an answer in the wrong language, an invented or falsified vital sign, an incomplete answer
  structure, a level announced outside the taxonomy.
- Safety checks validated against the catalogue's own reference answers. The patterns were
  accented while reading an answer strips accents, which wrongly flagged ten compliant answers out
  of seventy. A regression test keeps it from coming back.
- The two possible deliveries of the model, the adapter applied hot and the merged model, are
  compared on accuracy, latency and download size, with the agreement rate between their
  predictions.
- A confusion matrix with a column of its own for off-format answers.
- A breakdown by language and by kind of case, and an annotated error table.
- Latency percentiles computed by nearest rank, with the convention stated. The published 95th
  percentile gates a go-live criterion. The measurement is taken on `POST /triage`, which is what
  the service delivers, rule, anonymisation and log included. The engine alone only decomposes
  that figure.
- The explicit rule is removed from the baselines of the internal test set: it filters the corpus
  upstream, so it would be comparing itself with itself.
- The explicit rule is tightened on two turns of phrase it over-triggered: "ne répond pas" without
  a complement, which is also said of a patient not responding to treatment, and "hémorragie"
  bare, which covers a subconjunctival haemorrhage.
- Degraded inputs: the one describing a coronary syndrome carries the level expected of it, and a
  deferred triage is counted non-compliant there even when well formed. The check used to verify
  the shape alone.
- A fourth baseline: an ordinary classifier, weighted n-grams and a linear separator, trained on
  the same pairs as the model, on the training split alone, and whose setting is chosen on the
  validation split without consulting the clinical set. Beating a keyword rule does not establish
  what fine-tuning adds over ordinary learning.
- The paired McNemar comparison covers both baselines that learn, not the explicit rule alone.
- An unknown model name passed to `--models` is refused at launch. It used to surface after the
  other models had been evaluated, on a `KeyError` that left the results file unwritten.

### Service and deployment

- The API key is mandatory at startup, compared in constant time, with a per-caller quota. The
  header counts as an identity only once it has been verified: in open mode nobody verifies it,
  and changing it at every request used to hand out a fresh counting bucket each time.
- The inference engine is protected by a key of its own as soon as it is reachable other than
  through the loopback, failing which its address alone buys GPU inference with no quota, no
  anonymisation and no audit trail.
- A health probe that really questions the inference engine, and does not publish its address: it
  demands neither key nor quota, and the HTTP client's error message carries the URL it queried.
- Descriptions are bounded to the model window, and the cut is announced. The contract accepted
  four thousand characters for a window that holds far fewer: past it, generation did not degrade,
  it stopped on a tensor dimension error. The robustness check that pastes two pages is what found
  it. The reply and the audit log now carry `description_truncated`, because a triage decision
  taken on an incomplete narrative has to leave a trace, and only a clinician can tell whether
  what is missing mattered.
- The same bound on the preference measurement, whose essays also exceeded the window. The
  comparison stays fair, the same bound on both sides and a score normalised by length, and the
  number of truncated answers is logged.
- Request fields are bounded, the questionnaire's answer dictionary included: everything in it is
  concatenated into a single string, which the triage rule then reads back.
- A questionnaire that adapts to the complaint, compiling its answers into **sentences** and not
  into questions followed by their answers. The description produced is read back by the
  triage rule, and "difficulté à respirer ? non" was read there as the sign itself: a common cold
  with everything denied came out classified life-threatening.
- The alarming-answer rule applies only to questions that expect a yes or a no. Compared without
  that guard, a free-text answer and a question with no alarming answer both counted as nothing,
  so describing how a sprain happened stopped collection on a vital emergency.
- The reply exposes the explicit rule's verdict and its agreement with the model. The free-text
  labels of the questionnaire carry no term from the severity lexicon: "Idées suicidaires : je ne
  sais pas" made that sign appear in the reasons shown to the clinician, for a patient who had
  expressed nothing. The canonical sentences report an answer actually given, and keep the right
  word.
- The audit log is anonymised by the module itself, the description received **and** the answer
  returned, which echoes the clinician's narrative, in both languages: the API accepts free text,
  and the French engine does not spot a name inside English syntax. With the model actually
  loaded, and a retention period.
- The service image is pinned by digest, runs unprivileged, with dependencies pinned to the exact
  version.
- Continuous integration: style, security, dependency audit, tests with a coverage floor, then
  building the image and starting and calling the service.
- Continuous deployment on Modal: an image on every push, a model version frozen by a tag placed
  on the Hub repositories, a redeployment at that revision, and a health check of the endpoint
  that results. Disarmed by default.
- Host ports are configurable, `TRIAGE_VLLM_PORT` and `TRIAGE_API_PORT`: 8000 is a common port,
  and another stack already running on the machine prevented startup.
- The weights folder is configurable, `TRIAGE_MODELS_DIR`: on a workstation whose repository is
  synchronised to the cloud, Docker Desktop mounts an empty folder without saying so, and the
  container starts without seeing the weights.
- The inference engine image can be overridden through `TRIAGE_VLLM_IMAGE`, the version pinned by
  digest remaining the default. vLLM's V1 engine requires CUDA's unified virtual addressing, which
  WSL 2 does not expose: on a Windows workstation the stack stops on "UVA is not available" before
  loading any weights.
- The audit log is written to a named volume. Mounted from a host folder it belongs to root, and
  the service, which runs unprivileged, failed to write it: every request answered 500 after
  producing its decision.
- The service opens its audit log at startup and refuses to serve if it cannot write it.
  Discovering that at the first request means having already answered without a trace.

### What is published

- The merged supervised model is published on the Hub: it is the base model of the DPO adapter,
  which without it cannot be loaded.
- The set's composition, the extraction funnel and the diagnostics are published beside the
  results.
- The limits are stated: no external ground truth, a common author for the catalogue and the
  evaluation set, little diversity among the expected answers, and a share of the level readable
  from the shape of the vignettes.
- The median length of the expected answers is measured when the set is built. Written by hand,
  it had drifted.
- The latency bench records the version of the inference engine it questioned, and the results
  publish it: a latency holds for one engine.
- The estimator table is freed of a contradiction: the line "fewer than six cases, no interval"
  also applied to undertriage, which the line above covers at every effective.
- Three claims the measurements contradicted are corrected: the corpus is no longer called
  "balanced to within one" when the gap is two, MedQuAD is no longer said to supply no case when
  it supplies some, and the count of corpora read no longer confuses the five read with the four
  the project started from.
