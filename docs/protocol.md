# Protocol

This page explains how the published numbers were produced and what each of them can support.
`scripts/build_figures.py` draws the figures below from the result files under `reports/`. Each
figure prints its sample size and a note about its intervals under the plot, and
[`../reports/figures/MANIFEST.json`](../reports/figures/MANIFEST.json) lists, for each image, its
sample size, its interval estimator and its SHA-256 digest.

It is written for a reader who wants to check the claims and to find the places where the
protocol could favour its own results.

## Evaluation sets and what each one measures

Two evaluation sets judge the triage decision, because they answer different questions.

The **clinical set** has sixty cases written by hand, one at a time, each with the clinical
reason for its label recorded beside it. Forty of them are urgent. None was seen during training,
and the explicit rule played no part in their labels, which keeps the comparison with that rule
meaningful. Nearly half of the cases are written to mislead: a serious presentation that sounds
reassuring, a benign one that sounds alarming, a negation, or a vital sign that contradicts the
narrative.

The **internal test split** holds five hundred cases kept out of training, and the evaluation
draws a random sample of a hundred and twenty from it. Most of them were generated from catalogue
presentations that the training split also contains under other wordings. The others come from
the public corpora and carry the rule's label, like the corpus cases used in training. This set
measures restatement, and it is published so that the gap between the two sets can be read.

Three more sets probe what a triage decision cannot show:

- the test split of **UltraMedical-Preference**, 777 pairs of medical answers in which GPT-4
  picked the better answer and biomedical experts checked the pick. It was never used in
  training, and it tests whether the alignment learnt anything about medicine in general;
- a **task-format preference set** of 100 pairs built on the sixty hand-written narratives. Each
  pair holds the reference answer and the same case answered with one defect, and the set tests
  whether the preference learnt on generated narratives carries over to hand-written ones;
- 41 **degraded inputs** in nine families, from an empty description to a prompt injection. They
  test whether the model keeps its output contract on inputs that are not clinical narratives.

<!-- source: data/processed/metadata.json -->
![Composition of the supervised set by source, by triage level, by language and by label confidence, n = 5,000 examples across the training, validation and test splits, exhaustive counts](../reports/figures/dataset_composition.png)

> **How to read it.** The four panels count the same five thousand supervised examples
> (training, validation and test splits together), each panel by a different attribute. Bar
> height is a number of examples. The panels show whether one source, level, language or
> confidence grade dominates the set. The levels and the languages are balanced, and the sources
> and the confidence grades are not.

The training corpus is balanced by construction across the three levels and the two languages.
[`data-source.md`](data-source.md) describes where its cases come from and what the public
corpora yielded.

## Choices made during training, and the data behind them

### The LoRA setting

Four settings were trained on the same twelve hundred examples for the same seventy-five steps,
and judged on the first sixty cases of the validation split. The selection rule was fixed before
the runs: highest accuracy first, then lowest validation loss.

<!-- source: reports/training/hyperparameter_comparison.json -->
![The four LoRA settings on the validation split, with triage accuracy and its interval on the left and validation loss on the right, n = 60 validation cases](../reports/figures/hyperparameter_tuning.png)

> **How to read it.** Each bar is one LoRA setting. LoRA (low-rank adaptation) trains small
> matrices added beside the frozen base weights, and in this project the output head is trained
> as well. The left panel shows triage accuracy on the validation cases, where higher is better,
> with a 95% interval on each bar. The right panel shows validation loss in nats per token, where
> lower is better. With sixty validation cases the accuracy intervals are wide and overlap, and
> the selection rule settles a tie on accuracy with the loss.

<!-- source: reports/training/hyperparameter_comparison.json -->
Two settings tie at **0.833** on the n = 60 validation cases, `r32_lr2e-4` and `r16_lr2e-4`, and
the validation loss separates them, **0.0745** against **0.1328**. `r32_lr2e-4` is kept. The
other two settings reach **0.8**.

Once the intervals are taken into account, this ranking is not established. The gap between
0.833 and 0.8 is two answers in sixty, much less than the width of either interval. The loss
ordering agrees with the ranking, and the loss is computed on the whole validation split of five
hundred examples. The value 0.833 is also the winner's score on the cases used to choose it, so
it is optimistic by construction. The published results come from the clinical set, which played
no part in this choice.

### The end-of-sequence token

The supervised run trains the output head together with the LoRA projections. The usual recipe
leaves the head out, and the departure comes from a property of this base model.

Its twenty-five ChatML control tokens share a single untrained vector, and `im_start` and
`im_end` have a cosine similarity of 1.000 to the third decimal. The output head is tied to the
embedding matrix, and LoRA leaves both frozen. The adapters therefore cannot teach the model to
close its turn, and every answer runs on to the generation cap.

<!-- source: reports/training/module_comparison.json -->
A short study measures the effect. It compares three sets of trained modules at the same rank, on
the same twelve hundred examples and seventy-five steps, judged on the n = 60 validation cases.
With the projections alone, or with the embeddings added, the share of answers that stop by
themselves is **0.0**, and the median answer reaches the cap of **220** tokens. With the output
head added, that share is **1.0** and the median falls to **96** tokens. Accuracy changes little,
**0.833** against **0.817**, so training the output head makes the answers end and leaves the
decision about where it was.

<!-- source: reports/training/sft.json -->
![Training and validation cross-entropy over the optimiser steps of the run that produced the shipped weights, n = 4,000 training examples and 500 validation examples](../reports/figures/sft_training.png)

> **How to read it.** The horizontal axis counts optimiser steps, and both curves show
> cross-entropy in nats per token, which measures how unexpected the correct next token is to the
> model, so lower is better. The validation loss is computed at the end of each of the two
> epochs, and the run keeps the checkpoint with the lower value. A training curve that kept
> falling while the validation curve rose would indicate memorisation. Here the validation loss
> is still lower after the second epoch than after the first.

<!-- source: reports/training/sft.json -->
The run trains **346,030,080** of the model's 2,377,769,984 parameters, or **14.553%**, most of
them in the output head. Over n = 4,000 training examples and two epochs, it takes **1,821.4**
seconds on one RTX 4060 Ti, peaks at **7.43** gigabytes of GPU memory, and ends with a validation
loss of **0.0034**.

### The training engine

<!-- source: reports/training/engine_comparison.json -->
Unsloth and plain Transformers ran the same n = 30 steps on the same examples and the same
modules. Unsloth needs **3.69** seconds a step with a memory peak of **7.22** gigabytes, and
Transformers needs **6.93** seconds with a peak of **10.13** gigabytes. The final losses are
close, **0.9727** against **0.9448**. Over these thirty steps the engine changes the cost of a
run much more than its loss, and on a single GPU the lower cost decided the choice of Unsloth.

### The preference run

<!-- source: reports/training/dpo.json -->
![The alignment run in two panels, its loss falling on the left and the reward margin between the preferred and the rejected answer on the right, n = 2,160 training pairs](../reports/figures/dpo_alignment.png)

> **How to read it.** Both panels show the same alignment run. DPO (direct preference
> optimisation) presents the model with a preferred and a rejected answer and raises the
> likelihood of the first relative to the second. The left panel is the DPO loss, which falls as
> the two answers separate. The right panel shows the reward margin, a scaled log-ratio of how
> much more likely the preferred answer has become than the rejected one compared with the
> starting model, together with the share of pairs ordered correctly. The margin should rise as
> the loss falls, and a falling loss with a flat margin would mean the model had failed to learn
> the preference.

<!-- source: reports/training/dpo.json -->
On its own validation pairs the alignment is complete: n = 240 pairs, **1.0** of them ordered
correctly, with a reward margin of **8.66**. The loss is close to zero by step sixty. Each pair
differs by one deliberate defect, which makes the two answers easy to tell apart. This number
measures the alignment on the kinds of defect it was trained on, and its value on other pairs is
reported further down this page.

## How two systems are compared

Two systems evaluated on the same cases produce paired data, and in paired data the information
lies in the cases on which the systems disagree. Checking whether two marginal confidence
intervals overlap discards that information and is systematically too conservative. Two
intervals that almost coincide can still hide a real difference.

Every comparison between two systems on the clinical set therefore uses an **exact McNemar test**
on the same cases, and reports the counts in each direction: how many cases the model gets right
that the baseline gets wrong, and the reverse. The counts are published beside the p-value,
because the p-value is computed from them and because they can be read on their own.

The same test, restricted to the urgent cases and to undertriage, compares the model with the
rule. Accuracy mixes the two kinds of error. A system can be wrong often in the safe direction
and rarely in the dangerous one, and a triage tool is judged on the dangerous direction.

## What was fixed before the results

These choices were written down and committed before any model of this version was evaluated,
and the results are published whatever they show.

- **The shipped model is fixed in advance**: the model trained with seed 42, whatever its score.
  Seeds 43 and 44 serve only to measure the spread of the method. Keeping the best of three on
  the clinical set would turn that set into a selection set.
- **The primary analysis** is undertriage on the forty urgent cases, the shipped model against
  the explicit rule, by an exact McNemar test.
- **The secondary analyses** are the shipped model against the linear classifier; the supervised
  model against the aligned one, for each seed; the *more urgent of the two opinions* (for each
  case, the more urgent of the model's level and the rule's) against the model alone, on
  undertriage and on overtriage; the model served by version 1.0.0 against this version's model,
  on the same cases; and the spread of the three seeds, as the range and the standard deviation
  of accuracy and of undertriage.
- **The release gate** requires every answer of the clinical set in the expected format, and no
  more urgent cases missed than the explicit rule misses. `scripts/check_release.py` applies it,
  and the deployment workflow runs it before publishing the cards and the version tag on the Hub.
  A version that fails the gate gets neither, and the workflow does not deploy it.

## Results of the registered analyses

### The model against the rule

<!-- source: reports/evaluation_results.json -->
Of the n = 40 urgent cases, the explicit rule misses **22** and the shipped model **11**. The
model catches **17** urgent patients that the rule places too low, and misses **6** that the rule
catches. The exact McNemar test on those counts gives **p = 0.0347**. The model misses fewer
urgent patients than the rule, and a gap of this size is unlikely to arise by chance.

### The model against the classifier

<!-- source: reports/evaluation_results.json -->
The linear classifier misses **8** of the same n = 40 urgent cases, against **11** for the model.
The model alone catches **3** urgent patients and the classifier alone **6**, with **p = 0.5078**.
Accuracy on all n = 60 cases leads to the same reading, **0.7167** against **0.6667**, with
**p = 0.6291**. Neither test separates the two systems. Sixty cases support the statement that
the model is not shown to be better than n-grams here, and they do not support the statement
that it is worse.

### The aligned model against the supervised one

<!-- source: reports/evaluation_results.json -->
With seed 42, of the n = 40 urgent cases, the aligned model misses **11** and the supervised model
**12**. A single case separates them, in one direction, and the test gives **p = 1.0**.

<!-- source: reports/seed_study.json -->
With seeds 43 and 44, the two models miss exactly the same urgent cases among the n = 40: **9**
each with seed 43, and **13** each with seed 44. Over the three seeds, the alignment turns one
urgent miss into a catch, out of a hundred and twenty urgent decisions, and loses none. On
accuracy over the n = 60 cases, the aligned model is right on one or two more cases than the
supervised one with every seed: **0.7833** against **0.75** with seed 43, and **0.7** against
**0.6833** with seed 44. No seed's test comes near significance, and the largest gap, with seed
43, gives **p = 0.5**. None of these tests shows the alignment changing the triage decision. Its
effect is measured where it acts, in
[Effect of the alignment on preference pairs](#effect-of-the-alignment-on-preference-pairs).

### Keeping the more urgent of the two levels

<!-- source: reports/evaluation_results.json -->
This policy keeps, for each case, the more urgent of the two levels, the model's and the rule's.
On the n = 40 urgent cases it misses **5**, against **11** for the model alone, so it catches
**6** more urgent patients without losing any (**p = 0.0312**). Overtriage remains **0.1**.
Raising a level can only add overtriage, and on these sixty cases it adds none, since the only
case the rule overtriages is also overtriaged by the model. Accuracy rises to **0.8167**.

The reply already carries both levels, so the policy needs no new model and works on what the
service returns. The five urgent patients it still misses are those that both systems miss.

### This version against the previous one

<!-- source: reports/before_after.json -->
The model served by version 1.0.0 of this repository, evaluated by the same code on the same
sixty cases, scores **0.7167**, the same accuracy as this version. On the n = 40 urgent cases it
misses **10** and this version **11**. Only this version catches **1** of those urgent patients,
and only the earlier one catches **2**, with **p = 1.0**. The two versions are compared on the
clinical set only, because the internal split was rebuilt along with the dataset.

The retraining did not change the triage decision by any amount sixty cases can detect. Its
changes are upstream of the decision: a corrected dataset, a system prompt that matches the one
the service sends, and an evaluation fixed in advance and repeated on three seeds.

### Spread across the three seeds

<!-- source: reports/seed_study.json -->
Over seeds 42, 43 and 44, the aligned model's accuracy on the n = 60 cases of the clinical set
ranges from **0.7** to **0.7833**, with a mean of **0.7333** and a standard deviation of
**0.0441**. The urgent cases missed, out of n = 40, range from **9** to **13**, with a mean of
**11** and a standard deviation of **2.0**. With the same data and the same setting, the seed
alone moves the model by up to five cases in sixty, and by four urgent misses in forty.

<!-- source: reports/seed_study.json -->
Two systems trained once each can therefore differ by that much through the seed alone. The
shipped seed is not the best of the three. Seed 43 misses **9** of the n = 40 urgent cases and
scores **0.7833**, and shipping it would have reported a better model than the method produces
on average. The seed with the most misses still misses far fewer urgent patients than the rule,
13 against 22.

### The resolution of forty urgent cases

<!-- source: reports/evaluation_results.json -->
A model that missed none of the n = 40 urgent cases would still have an exact upper bound of
**0.0881** on its undertriage. Forty cases cannot distinguish a model that never misses from one
that misses nearly nine urgent patients in a hundred. Bringing that bound under **0.05** takes
**72** urgent cases without a miss. The clinical set can detect a large difference, such as the
one with the rule, and it is too small to certify a low rate.

## Interval estimators

| quantity | estimator | reason |
|---|---|---|
| accuracy | 95% Wilson; exact Clopper-Pearson for a subgroup of thirty cases or fewer, and none under six | Wilson behaves well away from the bounds on sixty cases, and the exact interval keeps its coverage on small subgroups |
| undertriage | 95% exact, Clopper-Pearson | the sample is forty cases and smaller in each subgroup, and the exact interval guarantees the coverage it states |
| difference between two independent sets | 95% Newcombe | the internal and clinical sets are independent, so their difference gets its own interval, built from the two Wilson intervals |
| overtriage | none | it is not the safety measure, and intervals are kept for the quantities the conclusions rest on |

The test suite checks the exact interval against the closed-form beta quantile. The package
finds its bounds by bisection on the binomial distribution function, which avoids a dependency
on a statistics library, and the test confirms that this method loses no precision.

## Errors by kind of presentation and by level

<!-- source: reports/evaluation_results.json -->
![Accuracy of the base, supervised and aligned models by kind of presentation, with a 95% interval where the group is large enough and a hatched bar where it is not, n = 60 cases split by kind](../reports/figures/accuracy_by_case_type.png)

> **How to read it.** The sixty cases are grouped by the difficulty each one was written to
> present, with three bars per group for the base, the supervised and the aligned model. A
> hatched bar marks a group of fewer than six cases, too few for an interval, and the hatching
> warns against ranking the models on it. A difference between models can be claimed only on the
> groups that hold enough cases.

<!-- source: reports/evaluation_results.json -->
![Undertriage of the shipped model by kind of presentation, with its exact interval, n = 40 urgent cases split by kind](../reports/figures/undertriage_by_case_type.png)

> **How to read it.** The same grouping, restricted to the forty urgent cases and to the shipped
> model. Each bar is the share of the group's urgent patients placed below their level, so a
> shorter bar is better. The interval is the exact one in every group, because groups of two or
> three patients are far too small for the usual approximation. A tall bar on a small group
> points to a weakness worth investigating, and it does not measure a rate.

<!-- source: reports/evaluation_results.json -->
On the cases that present directly, n = 32, the model reaches **0.875** accuracy and misses **3**
of the twenty urgent ones. On the falsely reassuring cases, n = 10, all of them urgent, it
reaches **0.4** and misses **6**. This is the model's main failure mode and the one that matters
most, since a serious condition described in reassuring terms is a realistic situation at a
triage desk.

Two subgroups are small, with three cases for discordant vital signs and four for negation. The
accuracy figure hatches every bar drawn on fewer than six cases and gives it no interval, while
the undertriage figure draws the exact interval even on groups of two or three. These subgroups
indicate where to look next, and they do not provide a measurement.

<!-- source: reports/evaluation_results.json -->
![Counted confusion matrices of the base, supervised and aligned models, with an extra off-format column, n = 60 cases](../reports/figures/confusion_matrices.png)

> **How to read it.** There is one grid per model. Rows give the reference level written for
> each case and columns the level the model returned, both ordered from the most to the least
> urgent, and each cell counts the cases in that pair. The diagonal is agreement. Cells to the
> right of the diagonal are patients placed lower than they needed, cells to its left are
> overtriage, and the last column holds answers that no software could parse.

<!-- source: reports/evaluation_results.json -->
The confusion matrices carry a column that the triage scale does not have, **off-format**. An
answer that cannot be parsed predicts no level, and counting it under a level would charge that
level with an error it did not make. For the shipped model the column is empty. For the base
model it holds **32** of the n = 60 answers.

<!-- source: reports/evaluation_results.json -->
By level, with n = 20 cases each, recall on the life-threatening level is **0.45** and precision
on it is **0.8182**. All eleven urgent cases the model misses are life-threatening cases placed
lower, nine of them at the moderate level, and no moderate case is placed below its level. The
model is reluctant to assign the top level, and usually right when it does.

<!-- source: reports/evaluation_results.json -->
The rule fails in a different way. Of the n = 40 urgent cases it misses **22**. In **10** of them
it detects nothing at all, in **8** it detects only a lesser sign, and in **4** it finds a sign
from its life-threatening list and discards it because a negation precedes it. The rule matches
keywords, so a description that uses none of them goes undetected.

## Effect of the alignment on preference pairs

The alignment is measured where it acts, on the likelihood the model gives to the two answers of
a pair, computed once with the alignment adapter and once without it. A pair counts as correctly
ordered when the alignment raised the likelihood of the preferred answer more than that of the
rejected one.

<!-- source: reports/evaluation_results.json -->
On the n = 100 task-format pairs built on the hand-written narratives, the alignment orders
**0.84** correctly: **0.85** of the pairs with an asserted diagnosis and **0.825** of those with
an undertriage. The preference learnt on generated narratives therefore holds on hand-written
narratives as well.

<!-- source: reports/evaluation_results.json -->
On the n = 777 pairs of the external test split it orders **0.3655** correctly, below chance, and
the composition of the split points to a likely cause. Among the **597** pairs outside the group
built to match answer lengths, the preferred answer is the longer of the two in **470**. The
alignment lowered the likelihood of **0.751** of all the answers it read, and settled only
**0.39** of the pairs in favour of the longer answer. On the n = 180 pairs of similar length it
orders **0.5** correctly, exactly chance. A preference for shorter answers would push the score
below chance in this direction, and these aggregate figures cannot separate that effect from
others. The alignment learnt nothing measurable about medicine in general.

On the clinical set, the aligned and the supervised model differ on one urgent case. The
alignment moved the likelihoods it was trained to move, and that change barely reaches the
triage decision.

## Robustness probe

<!-- source: reports/evaluation_results.json -->
Forty-one degraded inputs in nine families are sent to each model: nothing to triage, a request
that is not a triage, an attempt to extract the system prompt, a change of role, an encoding
trick, a third language, a complaint buried in an unrelated story, an over-long input, and
identifying data. An answer is compliant when it keeps the contract: a parseable level, in French,
no leak of the system prompt, a clean stop, no identifying data repeated back, and, when the input
describes a case whose level is known, that level kept. The shipped model is compliant on **35**
of the n = 41. It resists all five extraction attempts, and fails on a request for code, a
greeting, a poem, a description in Spanish, a bleeding mentioned in small talk, and a vital sign
placed at the end of a long input.

Forty-one cases make a probe and are too few for a measurement. They check that the model keeps
its output contract on inputs that are not clinical narratives. A security evaluation of prompt
injection is a separate exercise, and this repository has not carried one out.

## Limits of the protocol

The protocol rests on three training seeds, one GPU and one set of sixty cases. The seeds measure
how much training alone moves the numbers, and the interval on each proportion covers the
sampling of the sixty cases. Nothing here measures performance on a different set of cases
written by someone else.

The evaluation set was written by the same person as the catalogue, so the two share a way of
describing a patient, and a case written by a nurse would read differently. This limitation is
the one most likely to make the numbers look better than they are, and no measurement in this
repository addresses it.

Against the ordinary classifier, nothing is established in either direction, and the clinical
set is too small to certify a low undertriage rate whatever the model does.
