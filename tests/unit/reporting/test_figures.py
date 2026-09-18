"""Tests of the figures the report rests on.

A figure is a binary, and a binary is read by nobody. If one breaks, it has to break here. And a
figure is also a claim — these tests check the claims as much as the files: that a subgroup too
small to support a rate does not get drawn as if it were measured, that a value written on a bar
sits above its interval rather than inside it, and that every figure leaves in the manifest the
effective, the estimator and the script it came from.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pytest
from matplotlib.container import ErrorbarContainer

from clinical_triage.reporting import figures

STATISTICS = {
    "sft_total": 4998,
    "by_source": {"clinical_vignette": 4188, "medmcqa": 643, "medquad": 167},
    "by_level": {
        "URGENCE_VITALE": 1666,
        "URGENCE_MODEREE": 1666,
        "CONSULTATION_DIFFEREE": 1666,
    },
    "by_language": {"fr": 2499, "en": 2499},
    "by_confidence": {"high": 4188, "medium": 810},
}


def _system(accuracy, low, high, under, under_low, under_high, misses):
    return {
        "n": 60,
        "accuracy": accuracy,
        "accuracy_ci95": [low, high],
        "undertriage": under,
        "undertriage_ci95": [under_low, under_high],
        "undertriage_detail": {"misses": misses, "urgent_cases": 40},
        "overtriage": 0.05,
        "format_compliance": 1.0,
    }


MEASURES = {
    "explicit_rule": _system(0.58, 0.45, 0.70, 0.42, 0.27, 0.58, 17),
    "dpo-merged": _system(0.80, 0.68, 0.88, 0.15, 0.06, 0.30, 6),
}


@pytest.fixture(autouse=True)
def style():
    figures.apply_style()
    yield
    plt.close("all")


def _manifest(path) -> dict:
    """What the figure recorded about itself, next to the file it wrote."""
    manifest = json.loads((path.parent / "MANIFEST.json").read_text(encoding="utf-8"))
    return manifest["images"][path.name]


def test_the_dataset_composition_is_drawn(tmp_path):
    path = figures.dataset_composition(STATISTICS, tmp_path / "composition.png")
    assert path.read_bytes().startswith(b"\x89PNG")
    assert _manifest(path)["n"] == "n(training examples) = 4998"


def test_the_systems_comparison_is_drawn(tmp_path):
    path = figures.systems_comparison(MEASURES, tmp_path / "comparison.png")
    assert path.stat().st_size > 5000
    recorded = _manifest(path)
    assert "Clopper-Pearson" in recorded["dispersion"]
    assert recorded["source"] == "scripts/build_figures.py"


def test_the_confusion_matrices_are_drawn(tmp_path):
    levels = ["URGENCE_VITALE", "URGENCE_MODEREE", "CONSULTATION_DIFFEREE"]
    confusion = {
        "rows": levels,
        "columns": [*levels, "HORS_FORMAT"],
        "matrix": {
            "URGENCE_VITALE": dict(zip([*levels, "HORS_FORMAT"], [15, 3, 1, 1], strict=True)),
            "URGENCE_MODEREE": dict(zip([*levels, "HORS_FORMAT"], [2, 16, 2, 0], strict=True)),
            "CONSULTATION_DIFFEREE": dict(
                zip([*levels, "HORS_FORMAT"], [1, 2, 17, 0], strict=True)
            ),
        },
    }
    path = figures.confusion_matrices({"SFT + DPO": confusion}, tmp_path / "confusion.png")
    assert path.exists()
    assert _manifest(path)["n"] == "n(cases) = 60"


def test_the_training_curves_are_drawn(tmp_path):
    supervised = [
        {"step": 20, "loss": 1.2},
        {"step": 40, "loss": 0.8, "eval_loss": 0.9},
        {"step": 60, "loss": 0.5, "eval_loss": 0.6},
    ]
    assert figures.sft_training(supervised, tmp_path / "sft.png").exists()

    preference = [
        {"step": 20, "loss": 0.6},
        {"step": 40, "loss": 0.3, "rewards/accuracies": 0.92, "rewards/margins": 2.1},
    ]
    path = figures.dpo_alignment(preference, tmp_path / "dpo.png", pairs=812)
    assert _manifest(path)["n"] == "n(training pairs) = 812"


def test_the_hyperparameter_tuning_is_drawn(tmp_path):
    comparison = {
        "kept": "r16_lr2e-4",
        "protocol": {"evaluated_cases": 60},
        "variants": [
            {"variant": "r16_lr2e-4", "triage_accuracy": 0.82, "eval_loss": 0.41},
            {"variant": "r8_lr2e-4", "triage_accuracy": 0.78, "eval_loss": 0.45},
        ],
    }
    path = figures.hyperparameter_tuning(comparison, tmp_path / "tuning.png")
    recorded = _manifest(path)
    assert recorded["n"] == "n(validation cases) = 60"
    assert "Clopper-Pearson" in recorded["dispersion"]


def test_a_validation_loss_that_was_not_measured_is_not_drawn_as_a_zero(tmp_path):
    """Pins the treatment of a `None` loss: no bar, and the marker in its place.

    The reason is in `figures.hyperparameter_tuning`; what this test holds is that the drawing
    never turns an absence into a zero.
    """
    comparison = {
        "kept": "r16_lr2e-4",
        "protocol": {"evaluated_cases": 60},
        "variants": [
            {"variant": "r16_lr2e-4", "triage_accuracy": 0.82, "eval_loss": 0.41},
            {"variant": "r32_lr2e-4", "triage_accuracy": 0.79, "eval_loss": None},
        ],
    }
    figures.hyperparameter_tuning(comparison, tmp_path / "tuning.png")
    loss_panel = plt.gcf().axes[1]

    assert len(loss_panel.patches) == 1, "only the measured setting carries a bar"
    assert figures.NOT_MEASURED in [text.get_text() for text in loss_panel.texts]
    assert "0.410" in [text.get_text() for text in loss_panel.texts]


def test_the_training_curve_says_on_how_many_examples_it_was_measured(tmp_path):
    """Pins the two effectives into the manifest, where a reader of the image can find them."""
    history = [{"step": 20, "loss": 1.2}, {"step": 40, "loss": 0.8, "eval_loss": 0.9}]
    path = figures.sft_training(
        history,
        tmp_path / "sft.png",
        training_examples=4000,
        validation_examples=500,
    )
    assert _manifest(path)["n"] == "n(training examples) = 4000, n(validation examples) = 500"


def test_the_endpoint_latency_is_drawn(tmp_path):
    bench = {
        "concurrency_1": {"p50_ms": 420.0, "p95_ms": 610.0, "requests_per_s": 2.3, "requests": 40},
        "concurrency_4": {"p50_ms": 700.0, "p95_ms": 980.0, "requests_per_s": 6.1, "requests": 40},
    }
    path = figures.endpoint_latency(bench, tmp_path / "latency.png")
    assert "order statistics" in _manifest(path)["dispersion"]


def test_restatement_and_transfer_are_drawn_side_by_side(tmp_path):
    internal = {"dpo-merged": {"n": 500, "accuracy": 0.95, "accuracy_ci95": [0.93, 0.97]}}
    clinical = {"dpo-merged": {"n": 60, "accuracy": 0.80, "accuracy_ci95": [0.68, 0.88]}}
    path = figures.recall_versus_transfer(internal, clinical, tmp_path / "transfer.png")
    assert "Newcombe" in _manifest(path)["dispersion"]


def test_undertriage_by_subgroup_is_drawn(tmp_path):
    by_subgroup = {
        "direct_presentation": {
            "undertriage": 0.05,
            "undertriage_detail": {"misses": 1, "urgent_cases": 20},
        },
        "falsely_reassuring": {
            "undertriage": 0.40,
            "undertriage_detail": {"misses": 4, "urgent_cases": 10},
        },
    }
    path = figures.subgroup_undertriage(by_subgroup, tmp_path / "subgroup.png")
    assert "drawn at every effective" in _manifest(path)["dispersion"]


# --- What a figure is allowed to claim ---


def test_a_subgroup_too_small_carries_no_whisker(tmp_path):
    """Three cases do not let a rate be estimated.

    Three out of three gives an exact interval of [0.29, 1.00] — most of the scale. Drawn as a
    whisker it would promise a measurement; hatched, it says there is none. The hatch was chosen
    over a colour because it survives a greyscale print.
    """
    by_system = {
        "SFT + DPO": {
            "direct_presentation": {"accuracy": 0.94, "n": 32, "accuracy_ci95": [0.80, 0.99]},
            "discordant_vitals": {"accuracy": 1.0, "n": 3, "accuracy_ci95": None},
        }
    }
    figures.accuracy_by_case_type(by_system, tmp_path / "case-type.png")
    ax = plt.gcf().axes[0]

    whiskers = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    assert len(whiskers) == 1, "only one of the two bars is measurable"
    assert len([bar for bar in ax.patches if bar.get_hatch()]) == 1
    assert any("n = 3" in label.get_text() for label in ax.get_xticklabels())


def test_a_value_is_written_above_its_interval_not_inside_it(tmp_path):
    """Written on the bar, the label lands on the whisker and stops being readable."""
    fig, ax = plt.subplots()
    bars = ax.bar([0], [0.80])
    figures._annotate(ax, bars, [0.80], tops=[0.88])
    written = ax.texts[0]
    assert written.get_text() == "80%"
    assert written.get_position()[1] > 0.88
    plt.close(fig)


def test_an_absent_interval_produces_no_offset():
    """A measurement without an interval must not be drawn with a whisker of length zero.

    Its caps would read as near-certainty, which is the opposite of what the artefact says.
    """
    offsets = figures._bar_interval([0.8, 0.5], [(0.7, 0.9), None])
    assert offsets.tolist() == [[pytest.approx(0.1), 0.0], [pytest.approx(0.1), 0.0]]


def test_a_figure_whose_axes_are_unlabelled_is_refused(tmp_path):
    """The style refuses to write it rather than publishing a chart nobody can read alone."""
    fig, ax = plt.subplots()
    ax.bar([0], [1])
    with pytest.raises(ValueError, match="was not written"):
        figures.save_figure(fig, tmp_path / "nameless.png", n=10, source=figures.SOURCE)
    plt.close(fig)


def test_the_triage_levels_are_shown_in_english():
    """The constants are French because the model contract is; the figures are read in English."""
    assert figures.LEVEL_LABELS["URGENCE_VITALE"] == "Immediate"
    assert figures.SYSTEM_LABELS["dpo-merged"] == "SFT + DPO (merged)"
    assert figures.CASE_TYPE_LABELS["falsely_reassuring"] == "falsely\nreassuring"
