"""Tests of the warm-up step computation.

Too short a warm-up makes the start of training diverge; a warm-up longer than the training
itself stops it before the target learning rate is reached. This computation is the only place
in the repository where a warm-up ratio becomes a number of steps.
"""

from __future__ import annotations

from clinical_triage.training.schedule import warmup_steps


def test_an_imposed_step_count_wins_over_the_corpus_size():
    """``max_steps`` fixes the training: the warm-up is computed on it."""
    assert (
        warmup_steps(example_count=100_000, effective_batch=8, epochs=3, max_steps=200, ratio=0.1)
        == 20
    )


def test_the_warmup_follows_the_corpus_and_the_effective_batch():
    # 4,000 examples in batches of 16 make 250 steps per epoch, so 500 over two epochs; 5% of
    # warm-up makes 25 steps.
    assert (
        warmup_steps(example_count=4000, effective_batch=16, epochs=2, max_steps=0, ratio=0.05)
        == 25
    )


def test_a_partial_batch_still_counts_as_a_whole_step():
    """The last batch, incomplete, is a step like any other."""
    assert warmup_steps(example_count=17, effective_batch=8, epochs=1, max_steps=0, ratio=1.0) == 3


def test_the_warmup_never_drops_to_zero():
    """Zero warm-up steps exposes the first update to the full rate."""
    assert warmup_steps(example_count=10, effective_batch=8, epochs=1, max_steps=0, ratio=0.01) == 1


def test_a_null_batch_or_epoch_count_does_not_divide_by_zero():
    assert (
        warmup_steps(example_count=100, effective_batch=0, epochs=0, max_steps=0, ratio=0.1) == 10
    )
