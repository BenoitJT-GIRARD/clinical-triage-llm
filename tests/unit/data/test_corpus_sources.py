"""Tests of loading the public corpora.

They cover the two behaviours the module promises in the face of a failure: a source unreachable
at opening is skipped, a read interrupted mid-way stops everything. The second was not obvious —
the original ``try`` wrapped the opening only, whereas in streaming mode all network traffic
happens during iteration.
"""

from __future__ import annotations

import pytest

from clinical_triage.data.corpus_sources import (
    InterruptedRead,
    _cap_reached,
    _rows,
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
