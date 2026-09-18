"""The references the model is compared against.

An accuracy figure means nothing on its own. Four baselines frame the model's result and say
whether it brings anything:

- **the majority class**: what would a system that always answers the same thing give? On an
  unbalanced set this baseline is surprisingly high, and many published results beat it by
  little;
- **maximum caution**: what would a system that files everything as life-threatening give? It
  never undertriages — its undertriage rate is zero — but it sends everyone to resuscitation.
  This baseline is the reminder that safety is not judged without looking at the cost of
  caution;
- **the explicit rule**: what would the system a department can deploy in one afternoon give,
  with no language model at all?
- **an ordinary classifier**: what would a bag of n-grams trained on the same pairs as the
  model give? This is the decisive one. Beating a keyword rule says nothing about the value of
  fine-tuning, since a logistic regression trained in a second and a half beats it too. Without
  this comparison the evaluation cannot answer the question it asks.

The ordinary classifier produces a level and nothing else. The service's output contract asks
for three fields — level, justification, recommendation — and it fabricates neither of the last
two. The comparison therefore covers the one dimension where a reference exists, and the
published protocol says so.
"""

from __future__ import annotations

from collections import Counter

from clinical_triage.data.triage_rules import classify
from clinical_triage.utils import get_logger

logger = get_logger(__name__)


def majority_class(train_levels: list[str], case_count: int) -> list[str]:
    """Always predict the most frequent level of the training set."""
    most_frequent = Counter(train_levels).most_common(1)[0][0]
    return [most_frequent] * case_count


def always_critical(case_count: int) -> list[str]:
    """Always predict the life-threatening level: maximum caution, maximum congestion."""
    return ["URGENCE_VITALE"] * case_count


def explicit_rule(descriptions: list[str]) -> list[str]:
    """Apply the explicit triage rule, vital signs included."""
    return [classify(description) for description in descriptions]


# Settings put in competition. The choice is made on the validation split, never on the
# evaluation set: a baseline tuned on the set that judges it is no longer a baseline, it is an
# after-the-fact adjustment.
CLASSICAL_SETTINGS = (
    ("word_unigrams", "word", (1, 1)),
    ("word_bigrams", "word", (1, 2)),
    ("chars_3_5", "char_wb", (3, 5)),
)


def _build(analyzer: str, ngram: tuple[int, int]):
    """Weighted n-gram vectorisation, then a linear separator."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    return make_pipeline(
        TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2),
        LinearSVC(),
    )


def classical_classifier(
    train_texts: list[str],
    train_levels: list[str],
    validation_texts: list[str],
    validation_levels: list[str],
    descriptions: list[str],
) -> tuple[list[str], dict]:
    """Train an ordinary classifier on the same data as the model.

    The protocol, fixed before looking at any result:

    1. the candidate settings are trained on the **training split**, the very one the model
       sees;
    2. the one with the best accuracy **on the validation split** is kept — the clinical
       evaluation set is never consulted;
    3. it predicts once, on the requested descriptions.

    The baseline is not retrained on training + validation, although that is the usual practice:
    it would see 500 more examples than the model, and the comparison would be between two
    different data volumes.

    Returns the predictions and the trace of the choice, so that the published protocol can give
    the setting kept and its selection score rather than a bare figure.
    """
    results = []
    for name, analyzer, ngram in CLASSICAL_SETTINGS:
        model = _build(analyzer, ngram)
        model.fit(train_texts, train_levels)
        predicted = model.predict(validation_texts)
        accuracy = sum(p == v for p, v in zip(predicted, validation_levels, strict=True)) / len(
            validation_levels
        )
        results.append((accuracy, name, analyzer, ngram))
    results.sort(reverse=True, key=lambda r: r[0])
    selection_accuracy, name, analyzer, ngram = results[0]
    logger.info(
        "  classical baseline: %s kept (validation %.4f) among %s",
        name,
        selection_accuracy,
        ", ".join(f"{n}={a:.4f}" for a, n, _, _ in results),
    )

    kept = _build(analyzer, ngram)
    kept.fit(train_texts, train_levels)
    trace = {
        "configuration": name,
        "selection_accuracy": round(selection_accuracy, 4),
        "candidates": {n: round(a, 4) for a, n, _, _ in results},
        "training_examples": len(train_texts),
    }
    return list(kept.predict(descriptions)), trace
