"""How the delivered set is balanced, and how its card is written.

The corpus contribution is the part of the set nobody chose case by case: it comes from four
public corpora, it concentrates on the urgent levels and on English, and its label is put there
by a keyword rule rather than read from a catalogue. Two bounds keep it from deciding what the
set is, and both are computed here.

The yield table of ``data/README.md`` is written by the same script, from the same counts: the
card published on the Hub next to ``metadata.json`` has to describe the set it travels with.

What is not covered here is the two measurements that need the tokenizer — answer length and
description budget. They load the base model's tokenizer from the Hub, which is the business of
a build, not of a test.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from dataclasses import dataclass

import pytest

from clinical_triage.config import DATA, PATHS, TRIAGE


@pytest.fixture(scope="module")
def build_dataset():
    """Load ``scripts/build_dataset.py``, which is not importable as a module."""
    path = PATHS.root / "scripts" / "build_dataset.py"
    spec = importlib.util.spec_from_file_location("build_dataset_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_dataset_under_test"] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Case:
    """Just enough of an extracted case for the bounding to apply to it."""

    level: str
    lang: str
    identifier: int = 0


def _cases(level: str, lang: str, count: int) -> list[Case]:
    return [Case(level, lang, index) for index in range(count)]


# --- The six cells of the set ------------------------------------------------


def test_the_targets_sum_exactly_to_the_volume_asked_for(build_dataset):
    """The remainder of the division is spread rather than dropped."""
    for size in (4998, 5000, 1, 7):
        grid = build_dataset._grid(size)
        assert sum(target for _, _, target in grid) == size


def test_the_grid_crosses_every_level_with_both_languages(build_dataset):
    grid = build_dataset._grid(6000)
    assert {(level, lang) for level, lang, _ in grid} == {
        (level, lang) for level in TRIAGE.levels for lang in ("fr", "en")
    }
    assert [target for _, _, target in grid] == [1000] * 6


# --- What bounds the corpus contribution -------------------------------------


def test_a_cell_never_receives_more_cases_than_it_has_places(build_dataset):
    """Read in full, the corpora overflow the urgent English cell and unbalance the set."""
    # 6000 asked for, so 1000 places per cell; the corpus offers 1500 in one of them.
    cases = _cases("URGENCE_VITALE", "en", 1500) + _cases("CONSULTATION_DIFFEREE", "fr", 10)
    kept = build_dataset._cap_corpus_cases(cases, 6000, random.Random(42))

    in_cell = [c for c in kept if c.level == "URGENCE_VITALE" and c.lang == "en"]
    assert len(in_cell) == 1000
    assert len([c for c in kept if c.lang == "fr"]) == 10


def test_the_corpus_stays_a_minority_against_the_catalogue(build_dataset):
    """Its label comes from a keyword rule: a majority would make the set a copy of that rule."""
    size = 6000
    cases = [
        case
        for level in TRIAGE.levels
        for lang in ("fr", "en")
        for case in _cases(level, lang, 900)
    ]
    kept = build_dataset._cap_corpus_cases(cases, size, random.Random(42))

    assert len(kept) == int(size * DATA.max_corpus_share)
    assert DATA.max_corpus_share < 0.5


def test_a_corpus_that_offers_little_is_taken_whole(build_dataset):
    cases = _cases("URGENCE_MODEREE", "fr", 12)
    assert len(build_dataset._cap_corpus_cases(cases, 6000, random.Random(42))) == 12


def test_the_selection_is_reproducible_under_the_same_seed(build_dataset):
    """A build replayed six months later must deliver the same set."""
    cases = _cases("URGENCE_VITALE", "en", 300)
    first = build_dataset._cap_corpus_cases(cases, 600, random.Random(42))
    second = build_dataset._cap_corpus_cases(cases, 600, random.Random(42))
    assert [c.identifier for c in first] == [c.identifier for c in second]


# --- What the metadata counts ------------------------------------------------


def test_a_distribution_is_sorted_by_decreasing_frequency(build_dataset):
    """It is read as a ranking in the composition figure; alphabetical order would say nothing."""
    counted = build_dataset._distribution(
        [
            Case("URGENCE_VITALE", "en"),
            Case("CONSULTATION_DIFFEREE", "fr"),
            Case("URGENCE_VITALE", "fr"),
        ],
        lambda case: case.level,
    )
    assert list(counted) == ["URGENCE_VITALE", "CONSULTATION_DIFFEREE"]
    assert counted == {"URGENCE_VITALE": 2, "CONSULTATION_DIFFEREE": 1}


# --- The command the metadata records ----------------------------------------


def test_the_recorded_command_carries_what_was_not_the_default(build_dataset):
    """A run replayed from the metadata must be the run that produced it."""
    arguments = argparse.Namespace(
        sft_size=DATA.target_sft_pairs, dpo_size=1200, skip_upload=True, seed=0
    )
    command = build_dataset._command_run(arguments)

    assert command.startswith("uv run python scripts/build_dataset.py")
    assert "--dpo-size 1200" in command
    assert "--skip-upload" in command
    assert "--sft-size" not in command  # left at its default
    assert "--seed" not in command  # zero is not an option that was passed


# --- The yield table of the data card ----------------------------------------


def _yield_of(read: int, kept: int) -> dict:
    return {
        "entries_read": read,
        "cases_kept": kept,
        "funnel": {
            "no_patient_presentation": 1,
            "outside_length_bounds": 2,
            "no_identified_sign": 3,
            "duplicates": 4,
            "kept": kept,
        },
    }


def test_the_card_publishes_the_counts_of_the_build_that_wrote_it(
    build_dataset, monkeypatch, tmp_path
):
    """Typed by hand, the table describes the previous build and nobody notices."""
    import dataclasses

    opening, closing = build_dataset.YIELD_MARKERS
    card = tmp_path / "README.md"
    card.write_text(
        f"Before.\n\n{opening} — written by the script -->\nstale table\n{closing}\n\nAfter.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        build_dataset, "PATHS", dataclasses.replace(build_dataset.PATHS, data=tmp_path)
    )

    build_dataset._write_yield_into_the_card(
        {
            "mediqal": _yield_of(3075, 313),
            "medquad": _yield_of(16407, 120),
            "medmcqa": _yield_of(4183, 0),
            "frenchmedmcqa": _yield_of(2000, 55),
        }
    )

    written = card.read_text(encoding="utf-8")
    assert "stale table" not in written
    assert written.startswith("Before.")
    assert written.endswith("After.\n")
    assert "| MediQAl | 3,075 |" in written
    assert "**313** | **313 / 3,075**" in written  # the best yield is set in bold
    assert "**0** | **0 / 4,183**" in written  # and so is the corpus that gave nothing
    assert "| 120 | 120 / 16,407 |" in written
    # The caption a reader needs to read the rows, written with the table rather than after.
    assert "n = 25,665 entries read across the four corpora, 488 cases delivered." in written
