"""A video-memory failure must be reported where it happens.

When the card is full, the loading library silently spreads part of the layers onto the CPU.
Unsloth memorises that placement, and the error only surfaces at the first generation as an
"Invalid target device: None" that says nothing about its cause. These tests fix the expected
behaviour: fail at loading, with what to do about it.
"""

from __future__ import annotations

import types

import pytest

from clinical_triage.inference import _check_gpu_placement
from clinical_triage.utils import free_gpu_memory


def _model(*locations: str):
    """A model reduced to what the check asks of it: its parameters."""
    parameters = [types.SimpleNamespace(device=types.SimpleNamespace(type=e)) for e in locations]
    return types.SimpleNamespace(parameters=lambda: iter(parameters))


def test_a_model_entirely_on_the_gpu_passes():
    _check_gpu_placement(_model("cuda", "cuda", "cuda"))


def test_one_layer_left_on_the_cpu_is_enough_to_fail():
    """This is the real case: most of the model fits, the end overflows."""
    with pytest.raises(RuntimeError, match="Not enough GPU memory"):
        _check_gpu_placement(_model("cuda", "cuda", "cpu"))


def test_the_message_names_the_offending_locations():
    """Without that, the message is no better than the one it replaces."""
    with pytest.raises(RuntimeError) as failure:
        _check_gpu_placement(_model("cuda", "cpu", "meta"))
    assert "cpu, meta" in str(failure.value)


def test_freeing_memory_does_not_depend_on_a_gpu_being_present():
    """The same code runs on the training machine and in continuous integration."""
    free_gpu_memory()
