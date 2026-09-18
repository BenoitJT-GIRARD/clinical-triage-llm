"""Tests of the diagnostics of the produced set.

These measurements say what the dataset volumes are worth: how many different answers the model
saw, what share of the test split is restatement, and what the level owes to the generator's
metadata. They are published in the dataset card and picked up by the published protocol, so
they must be right.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from clinical_triage.data.diagnostics import (
    distinct_completions,
    metadata_leakage,
    separability,
)


@dataclass(frozen=True)
class FakeExample:
    """The bare minimum needed to exercise the diagnostics."""

    user_turn: str
    assistant_turn: str
    level: str
    source: str
    presentation_id: str


@dataclass(frozen=True)
class FakeRecord:
    id: str
    onset: str
    vitals_profile: str


def _vignettes(per_presentation: int, presentations: list[str]) -> list[FakeExample]:
    """Fake vignettes, shuffled as the delivered ones are.

    The shuffle is not cosmetic. Left in presentation order, the vignettes make a stratified
    split take contiguous blocks, that is to say a whole presentation per fold: it then becomes
    a grouped split, and the measurement can no longer show the gap it exists to show.
    """
    examples = []
    for rank, presentation in enumerate(presentations):
        for index in range(per_presentation):
            examples.append(
                FakeExample(
                    user_turn=f"Patient {index} avec {presentation}",
                    assistant_turn=f"Réponse pour {presentation}",
                    level="URGENCE_VITALE" if rank % 2 else "URGENCE_MODEREE",
                    source="clinical_vignette",
                    presentation_id=presentation,
                )
            )
    random.Random(0).shuffle(examples)
    return examples


# --- Diversity of the expected answers ---


def test_vignettes_of_one_presentation_share_their_answer():
    """That is the fact the measurement must expose: 60 examples, 3 answers."""
    examples = _vignettes(20, ["abc", "bcd", "cde"])
    measure = distinct_completions(examples)
    assert measure["total"] == 3
    vignettes = measure["by_origin"]["vignettes"]
    assert vignettes["examples"] == 60
    assert vignettes["distinct_answers"] == 3
    assert vignettes["mean_repetition"] == 20.0


def test_corpus_cases_are_counted_separately():
    examples = _vignettes(5, ["abc"]) + [
        FakeExample("Un patient tousse.", "Réponse unique", "URGENCE_MODEREE", "mediqal", "")
    ]
    measure = distinct_completions(examples)
    assert measure["by_origin"]["corpus"]["examples"] == 1
    assert measure["by_origin"]["vignettes"]["examples"] == 5


def test_a_jsonl_row_is_accepted_like_an_object():
    """Examples travel in two shapes; the measurement must read both."""
    row = {
        "user_turn": "Patient",
        "completion": "Réponse",
        "level": "URGENCE_VITALE",
        "source": "clinical_vignette",
        "presentation_id": "abc",
    }
    assert distinct_completions([row])["total"] == 1


# --- Separability: random split against split by presentation ---


def test_the_grouped_split_is_harsher_than_the_random_one():
    """A random split leaves rewordings of the same case on both sides.

    With presentations whose vocabularies do not overlap, a classifier recognises perfectly the
    ones it has seen and does not transfer to the ones it has not. The gap between the two
    numbers is what the measurement publishes.
    """
    presentations = [f"motif{letter}" for letter in "abcdefghij"]
    examples = _vignettes(30, presentations)
    measure = separability(examples)
    assert measure["presentations"] == 10
    assert measure["random_split"] > measure["split_by_presentation"]
    assert measure["gap"] > 0


def test_separability_gives_up_on_too_small_a_set():
    assert separability(_vignettes(1, ["abc"])) == {}


# --- Leakage of the level through the generator metadata ---


def test_a_metadata_field_perfectly_correlated_with_the_level_is_detected():
    """If each vital-sign profile goes with one level only, the vote is perfect."""
    examples = _vignettes(10, ["abc", "bcd"])
    records = [
        FakeRecord(id="abc", onset="minutes", vitals_profile="critical"),
        FakeRecord(id="bcd", onset="jours", vitals_profile="normal"),
    ]
    measure = metadata_leakage(examples, records)
    assert measure["majority_vote_accuracy"] == 1.0
    assert measure["share_in_homogeneous_cell"] == 1.0


def test_a_metadata_field_unrelated_to_the_level_predicts_no_more_than_the_majority():
    """Two levels in the same cell: the majority vote caps at the majority."""
    examples = _vignettes(10, ["abc", "bcd"])
    records = [
        FakeRecord(id="abc", onset="heures", vitals_profile="normal"),
        FakeRecord(id="bcd", onset="heures", vitals_profile="normal"),
    ]
    measure = metadata_leakage(examples, records)
    assert measure["cells"] == 1
    assert measure["majority_vote_accuracy"] == 0.5
    assert measure["share_in_homogeneous_cell"] == 0.0


def test_leakage_gives_up_when_no_presentation_matches():
    assert metadata_leakage(_vignettes(2, ["abc"]), []) == {}
