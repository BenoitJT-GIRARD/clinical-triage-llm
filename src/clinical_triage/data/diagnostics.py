"""What the produced set actually contains, measured rather than asserted.

A dataset is usually described by its volumes. Three other measurements say what those volumes
are worth, and they condition the reading of every result published afterwards:

- **target diversity.** The vignettes derive from a catalogue of presentations, and all those
  coming from one presentation share the same expected answer. Counting the distinct answers
  says how many different decisions the model actually saw;
- **separability.** A random split places rewordings of the same case on either side of the
  boundary: what is then measured on the test set is the recognition of cases already seen. The
  same computation, split by source presentation, measures transfer to new cases. The gap
  between the two numbers is the share of restatement contained in the first;
- **metadata leakage.** The generator draws the onset delay and the vital-sign profile from the
  presentation, hence from the level. A majority vote on those two fields alone measures what
  the level owes to the shape of the template rather than to the clinical content.

These three numbers go into the dataset card and are picked up by the published protocol. They
do not flatter the project; that is precisely why they are computed at every build rather than
left to the reader's judgement.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict


def _expected_answer(example) -> str:
    """The text the model must produce, whatever the input shape.

    Examples travel in two forms in this project: the ``TriageExample`` object, which names this
    field ``assistant_turn``, and the JSONL line read back from disk, which names it
    ``completion``.
    """
    if isinstance(example, dict):
        return example.get("completion") or example.get("assistant_turn", "")
    return getattr(example, "assistant_turn", None) or getattr(example, "completion", "")


def _field(example, name: str, default=""):
    """Read a field, whether the example is an object or a JSONL line."""
    if isinstance(example, dict):
        return example.get(name, default)
    return getattr(example, name, default)


def distinct_completions(examples: list) -> dict:
    """Count the genuinely distinct expected answers, and their repetition."""
    by_source: dict[str, list[str]] = defaultdict(list)
    for example in examples:
        origin = "vignettes" if _field(example, "source") == "clinical_vignette" else "corpus"
        by_source[origin].append(_expected_answer(example))

    detail = {}
    for origin, answers in by_source.items():
        distinct = len(set(answers))
        detail[origin] = {
            "examples": len(answers),
            "distinct_answers": distinct,
            "mean_repetition": round(len(answers) / distinct, 1) if distinct else 0.0,
        }
    return {
        "total": len({_expected_answer(e) for e in examples}),
        "by_origin": detail,
    }


def separability(examples: list, splits: int = 5) -> dict:
    """Accuracy of a simple classifier, under a random split and then a grouped one.

    The classifier is of no interest in itself: it is a probe. If it reaches near-perfection
    under a random split and drops markedly under a split grouped by presentation, then the
    random test set mostly contains rewordings of cases seen during training.

    Returns an empty dictionary when scikit-learn is unavailable: the measurement is a
    diagnostic, not a step the construction of the set depends on.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
    except ImportError:
        return {}

    vignettes = [
        e
        for e in examples
        if _field(e, "source") == "clinical_vignette" and _field(e, "presentation_id")
    ]
    if len(vignettes) < splits * 2:
        return {}
    texts = [_field(e, "user_turn") for e in vignettes]
    levels = [_field(e, "level") for e in vignettes]
    groups = [_field(e, "presentation_id") for e in vignettes]
    if len(set(groups)) < splits:
        return {}

    probe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=2000)
    )
    try:
        random_split = cross_val_score(probe, texts, levels, cv=splits).mean()
        grouped = cross_val_score(
            probe, texts, levels, cv=GroupKFold(n_splits=splits), groups=groups
        ).mean()
    except ValueError:
        # A grouped split can isolate a fold containing a single level; the probe then has
        # nothing to learn. This is a diagnostic: it gives up and says so, it does not fail the
        # construction of the set.
        return {}
    return {
        "vignettes": len(vignettes),
        "presentations": len(set(groups)),
        "random_split": round(float(random_split), 4),
        "split_by_presentation": round(float(grouped), 4),
        "gap": round(float(random_split - grouped), 4),
    }


def metadata_leakage(examples: list, presentations) -> dict:
    """Share of the level predictable from the onset delay and the vital-sign profile alone.

    Both fields are chosen by the generator from the presentation, hence from the level. A
    patient presenting after three weeks with normal vital signs does indeed belong in a
    deferred consultation: the correlation is partly clinical. It is measurable nonetheless, and
    a model can lean on it instead of reading the complaint.
    """
    records = {p.id: p for p in presentations}
    cells: dict[tuple[str, str], Counter] = defaultdict(Counter)
    total = 0
    for example in examples:
        record = records.get(_field(example, "presentation_id"))
        if record is None:
            continue
        cells[(record.onset, record.vitals_profile)][_field(example, "level")] += 1
        total += 1
    if not total:
        return {}

    correct = sum(count.most_common(1)[0][1] for count in cells.values())
    homogeneous = sum(sum(c.values()) for c in cells.values() if len(c) == 1)
    return {
        "vignettes": total,
        "cells": len(cells),
        "majority_vote_accuracy": round(correct / total, 4),
        "share_in_homogeneous_cell": round(homogeneous / total, 4),
    }


# --- What the cleaning of exam statements leaves behind ---

#: The detector is deliberately not the cleaner of `corpus_cases`: measuring the residue with the
#: rules that removed it would find zero by construction. It reads each sentence for what an
#: answer sheet leaves behind: a question mark, a blank, a stem waiting for its option, or a
#: reference to an image the reader does not have. "Normal except for a rash" and "with the
#: following findings: SpO2 60 %" are narrative, and are not counted.
EXAM_RESIDUE = re.compile(
    r"\?\s*$|_{2,}"
    r"|\b(?:which of|most (?:likely|probable|appropriate)|best (?:treatment|next|initial|option)"
    r"|(?:probable |likely )?diagnosis (?:is|would be)|next step"
    r"|(?:drug|investigation|treatment) of choice|true about)\b"
    r"|\bexcept\s*[-:.]?\s*$"
    r"|\bthe (?:following|above|below)(?:\s+\w+){0,2}\s*[.]?\s*$"
    r"|\b(?:is|are|due to|associated with|with|by|of)\s*[.:]?\s*$",
    re.IGNORECASE,
)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def exam_residue(descriptions: list[str]) -> dict:
    """How many delivered descriptions still read as an exam statement, in one sentence at least.

    The 1.0.0 change log announced half a percent; read by hand before the 2.0.0 build, the
    MedMCQA cases carried several times that. The share is now measured at every build, on the
    descriptions actually delivered, and published with its definition.
    """
    flagged = sum(
        any(EXAM_RESIDUE.search(sentence) for sentence in _sentences(text)) for text in descriptions
    )
    total = len(descriptions)
    return {
        "n": total,
        "with_residue": flagged,
        "share": round(flagged / total, 4) if total else 0.0,
    }
