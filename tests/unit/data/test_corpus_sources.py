"""Tests of loading the public corpora.

They cover the two behaviours the module promises in the face of a failure: a source unreachable
at opening is skipped, a read interrupted mid-way stops everything. The second was not obvious —
the original ``try`` wrapped the opening only, whereas in streaming mode all network traffic
happens during iteration.
"""

from __future__ import annotations

import pytest

from clinical_triage.data import corpus_sources
from clinical_triage.data.corpus_sources import (
    InterruptedRead,
    _cap_reached,
    _rows,
    load_frenchmedmcqa,
)

# --- An interrupted read must not pass for a complete one ---


def _stream_cut_after(count: int):
    """A stream that delivers ``count`` rows then cuts out, as the network does."""

    def stream():
        for index in range(count):
            yield {"question": f"row {index}"}
        raise ConnectionError("Server disconnected without sending a response.")

    return stream()


def test_a_cut_during_the_read_stops_everything():
    """The published yield is computed on the number of entries read.

    Absorbing the cut would produce a table describing a partial read while presenting it as
    the whole corpus.
    """
    with pytest.raises(InterruptedRead) as error:
        list(_rows(_stream_cut_after(3), "openlifescienceai/medmcqa"))
    message = str(error.value)
    assert "openlifescienceai/medmcqa" in message
    assert "3 entries" in message


def test_a_complete_read_raises_nothing():
    rows = list(_rows(iter([{"a": 1}, {"a": 2}, {"a": 3}]), "trial"))
    assert len(rows) == 3


def test_a_deliberate_stop_is_not_a_cut():
    """The read cap closes the generator; that is not a failure."""
    read = []
    for row in _rows(iter([{"a": 1}, {"a": 2}, {"a": 3}]), "trial"):
        read.append(row)
        if len(read) == 2:
            break
    assert len(read) == 2


# --- The read cap, and its absence ---


@pytest.mark.parametrize(
    ("already_read", "limit", "expected"),
    [
        (5, 0, False),  # zero bounds nothing
        (5, -1, False),  # nor does a negative limit
        (5, 10, False),
        (10, 10, True),
        (11, 10, True),
        (0, 1, False),
        (1, 1, True),
    ],
)
def test_the_read_cap(already_read: int, limit: int, expected: bool):
    assert _cap_reached([None] * already_read, limit) is expected


# --- FrenchMedMCQA: the right answer is an index counted from zero ---


@pytest.mark.parametrize(
    ("index", "expected"), [(0, "option a"), (4, "option e"), (5, ""), (None, "")]
)
def test_the_frenchmedmcqa_answer_index_counts_from_zero(monkeypatch, index, expected):
    """In the mirror used, `correct_answers` is 0 for a and 4 for e.

    Read as counting from one, every answer was shifted by one option and the first option, at
    index zero, was never read. Checked against the authors' archive, which writes letters.
    """
    row = {"question": "Quel est le mécanisme ?", "correct_answers": index}
    row.update({f"answer_{letter}": f"option {letter}" for letter in "abcde"})
    monkeypatch.setattr(corpus_sources, "_stream_all_splits", lambda _repo: iter([row]))

    (entry,) = load_frenchmedmcqa()

    assert entry.answer == expected


# --- External preferences: the test split, every group ---


def _ultramedical_pair(group: str, index: int, rejected: str | None = None) -> dict:
    return {
        "label_type": group,
        "prompt": f"question {index}",
        "chosen": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": f"preferred {index}"},
        ],
        "rejected": [{"role": "assistant", "content": rejected or f"rejected {index}"}],
    }


@pytest.fixture
def test_split(monkeypatch, tmp_path):
    """Replace the download of the test split with a local file."""
    import json

    rows = [
        _ultramedical_pair("easy", 0),
        _ultramedical_pair("length", 1),
        _ultramedical_pair("human", 2, rejected="preferred 2"),  # two identical answers
        _ultramedical_pair("hard", 3),
    ]
    path = tmp_path / "test.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *_, **__: str(path))


def test_the_external_preferences_keep_every_group(test_split):
    """The last message of each answer is read, and a pair whose two answers are identical is
    set aside: it decides nothing."""
    from clinical_triage.data.corpus_sources import load_ultramedical_preferences

    pairs = load_ultramedical_preferences()
    assert [pair.label_type for pair in pairs] == ["easy", "length", "hard"]
    assert (pairs[0].chosen, pairs[0].rejected) == ("preferred 0", "rejected 0")


def test_the_external_preferences_respect_the_cap(test_split):
    from clinical_triage.data.corpus_sources import load_ultramedical_preferences

    assert len(load_ultramedical_preferences(limit=2)) == 2
