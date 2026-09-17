"""The published figures, drawn from the result artefacts and never from typed-in values.

Every figure here reads ``reports/`` and ``data/processed/``. Re-run the evaluation and the
figures move with it; nothing is copied by hand, so nothing can drift.

Three rules hold across the nine, and they are the reason most of these figures are bars
rather than clouds of points:

- **almost everything plotted here is a binomial proportion on a few dozen cases.** The usual
  "mean ± standard error" does not apply: the normal approximation leaves [0, 1] — two
  undertriages out of forty urgent cases give a negative lower bound — and claims certainty
  the moment a proportion hits 0 or 1. The estimator is chosen once, in
  :func:`clinical_triage.evaluation.metrics.interval_for_proportion`, and read there;
- **an exhaustive count carries no error bar.** The corpus and the cells of a confusion
  matrix are counted, not estimated. Adding a whisker would invent randomness, so the figure
  says in words that the bars are counts;
- **latencies are described by their order statistics**, which are neither a mean nor an
  interval. The benchmark artefact keeps the percentiles, not the individual calls, and the
  figure says which ones it draws.

Sizes, colours, fonts and the sample-size stamp come from the vendored style module, which
also writes every image and refuses one whose axes or effective are missing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from clinical_triage.evaluation.metrics import (
    clopper_pearson,
    difference_newcombe,
    interval_for_proportion,
)
from clinical_triage.figure_style import (
    PALETTE,
    apply_style,
    close,
    reference_line,
    save_figure,
    sequential_cmap,
    series_colours,
)

SOURCE = "scripts/build_figures.py"

#: Triage levels are French constants in the model contract; figures show them in English.
LEVEL_LABELS = {
    "URGENCE_VITALE": "Immediate",
    "URGENCE_MODEREE": "Urgent",
    "CONSULTATION_DIFFEREE": "Deferred",
    "HORS_FORMAT": "Off-format",
}

#: Systems, in the order a reader should meet them: trivial baselines, the deployable rule,
#: the ordinary classifier, then the model at each stage of its training.
SYSTEM_LABELS = {
    "majority_class": "Majority class",
    "always_critical": "Always critical",
    "explicit_rule": "Explicit rule",
    "linear_classifier": "Linear classifier",
    "base": "Qwen3-1.7B-Base",
    "sft": "SFT + LoRA",
    "dpo": "SFT + DPO (adapter)",
    "dpo-merged": "SFT + DPO (merged)",
}

SYSTEM_ORDER = tuple(SYSTEM_LABELS)

#: The four ways the hand-written evaluation set tries to mislead a triage system, plus the
#: cases that present straightforwardly.
CASE_TYPE_LABELS = {
    "direct_presentation": "direct",
    "falsely_reassuring": "falsely\nreassuring",
    "falsely_alarming": "falsely\nalarming",
    "negation": "negation",
    "discordant_vitals": "discordant\nvitals",
}

SOURCE_LABELS = {
    "clinical_vignette": "clinical vignettes",
    "medmcqa": "MedMCQA",
    "mediqal": "MediQAl",
    "medquad": "MedQuAD",
}

CONFIDENCE_LABELS = {"high": "high (catalogue)", "medium": "medium (corpus)"}

#: Said in words on every figure whose bars are counted rather than estimated.
COUNTED = "exhaustive counts, no estimation"

#: Hatch of a bar whose effective forbids publishing a rate. Colour alone would not do: it
#: disappears in a greyscale print.
NOT_MEASURABLE = "///"

#: Written where a bar would be, when the quantity was not measured at all. A bar at zero would
#: be indistinguishable from a measurement that came out at zero.
NOT_MEASURED = "n/a"


def _bar_interval(values: list[float], intervals: list[tuple[float, float] | None]) -> np.ndarray:
    """Turn published confidence intervals into the asymmetric offsets a bar chart wants."""

    low, high = [], []
    for value, interval in zip(values, intervals, strict=True):
        if interval is None:
            low.append(0.0)
            high.append(0.0)
        else:
            low.append(max(0.0, value - interval[0]))
            high.append(max(0.0, interval[1] - value))
    return np.array([low, high])


def _annotate(ax, bars, values, *, tops=None, fmt="{:.0%}", offset=0.02) -> None:
    """Write each value above its bar: colour never carries a number on its own.

    ``tops`` is where the drawing of that bar really ends — the upper bound of its interval when
    there is one. Without it the label lands on the whisker and becomes unreadable, which is the
    one thing a written value exists to avoid.
    """

    heights = tops if tops is not None else [bar.get_height() for bar in bars]
    for bar, value, top in zip(bars, values, heights, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            top + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
            color=PALETTE["ink"],
        )


def _headroom(ax, tops: list[float]) -> None:
    """Leave room above the tallest bar so its label is not clipped by the frame."""

    ceiling = max([t for t in tops if np.isfinite(t)] or [1.0])
    ax.set_ylim(0, min(1.18, ceiling * 1.25) if ceiling <= 1 else ceiling * 1.18)


def dataset_composition(statistics: dict, destination: Path) -> Path:
    """What the training corpus is made of: source, triage level, language, label confidence.

    Four exhaustive counts. Nothing here is a sample of anything, so nothing carries an
    interval — and the figure says so rather than letting the absence read as an omission.
    """

    panels = [
        ("By source", statistics["by_source"], SOURCE_LABELS),
        ("By triage level", statistics["by_level"], LEVEL_LABELS),
        ("By language", statistics["by_language"], {"fr": "French", "en": "English"}),
        ("By label confidence", statistics["by_confidence"], CONFIDENCE_LABELS),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.6))
    for ax, (title, counts, labels) in zip(axes, panels, strict=True):
        names = sorted(counts, key=lambda k: -counts[k])
        heights = [counts[n] for n in names]
        colours = series_colours([labels.get(n, n) for n in names])
        bars = ax.bar(
            range(len(names)),
            heights,
            color=[colours[labels.get(n, n)] for n in names],
            width=0.68,
        )
        ax.set_title(title)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([labels.get(n, n) for n in names], rotation=30, ha="right")
        ax.set_xlabel("category")
        ax.set_ylabel("examples (count)")
        ax.set_ylim(0, max(heights) * 1.2)
        for bar, height in zip(bars, heights, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(heights) * 0.02,
                f"{height:,}".replace(",", " "),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=PALETTE["ink"],
            )
    fig.tight_layout()
    return save_figure(
        fig,
        destination,
        n={"training examples": statistics["sft_total"]},
        source=SOURCE,
        note=COUNTED,
    )


def hyperparameter_tuning(comparison: dict, destination: Path) -> Path:
    """The four LoRA settings put in competition, on the validation split.

    Accuracy is a proportion on a few dozen cases, so it carries its interval; the validation
    loss is a single number per run and carries none. Two panels rather than two vertical
    axes: quantities of different natures do not share a scale.
    """

    variants = list(comparison["variants"])
    cases = int(comparison["protocol"]["evaluated_cases"])
    names = [v["variant"].replace("_", "\n") for v in variants]
    accuracy = [v["triage_accuracy"] for v in variants]
    intervals = [interval_for_proportion(round(v * cases), cases) for v in accuracy]
    losses = [v["eval_loss"] for v in variants]
    kept = comparison["kept"]
    colours = series_colours(
        [v["variant"] for v in variants],
        control=[v["variant"] for v in variants if v["variant"] != kept],
    )
    palette = [colours[v["variant"]] for v in variants]

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    bars = left.bar(range(len(names)), accuracy, color=palette, width=0.62)
    left.errorbar(
        range(len(names)),
        accuracy,
        yerr=_bar_interval(accuracy, intervals),
        fmt="none",
        ecolor=PALETTE["ink"],
        elinewidth=1.1,
        capsize=4,
    )
    _annotate(
        left,
        bars,
        accuracy,
        tops=[i[1] if i else a for a, i in zip(accuracy, intervals, strict=True)],
    )
    left.set_title("Triage accuracy on the validation split")
    left.set_ylabel("accuracy (share of cases)")
    left.set_xlabel("LoRA setting")
    left.set_xticks(range(len(names)))
    left.set_xticklabels(names)
    left.set_ylim(0, 1.15)

    # A missing validation loss is not a loss of zero: the bar is left out and the marker takes
    # its place, as it does for the proportions on the left panel. A bar at zero would read as
    # "this setting has nothing left to learn", which is the opposite of what a missing
    # measurement allows anyone to say.
    measured = [index for index, loss in enumerate(losses) if loss is not None]
    heights = [losses[index] for index in measured]
    right.bar(
        measured,
        heights,
        color=[palette[index] for index in measured],
        width=0.62,
    )
    right.set_title("Validation loss")
    right.set_ylabel("cross-entropy (nats per token)")
    right.set_xlabel("LoRA setting")
    right.set_xticks(range(len(names)))
    right.set_xticklabels(names)
    ceiling = max(heights or [1.0])
    for index, loss in enumerate(losses):
        right.text(
            index,
            (loss + ceiling * 0.03) if loss is not None else 0.0,
            f"{loss:.3f}" if loss is not None else NOT_MEASURED,
            ha="center",
            va="bottom",
            fontsize=9,
            style="normal" if loss is not None else "italic",
            color=PALETTE["ink"] if loss is not None else PALETTE["muted"],
        )
    right.set_ylim(0, ceiling * 1.2)

    fig.tight_layout()
    return save_figure(
        fig,
        destination,
        n={"validation cases": cases},
        dispersion="95% Clopper-Pearson interval on the accuracy; the loss is a single value",
        source=SOURCE,
        note=f"kept: {kept}",
    )


def sft_training(
    history: list[dict],
    destination: Path,
    *,
    training_examples: int = 0,
    validation_examples: int = 0,
) -> Path:
    """Loss against optimiser steps for the supervised run that produced the shipped model.

    The two effectives are those of the sets the losses are computed on. Without them the figure
    shows two curves without saying on how many examples, and a gap between two validation
    points does not read the same way on fifty examples as on five hundred.
    """

    train = [(h["step"], h["loss"]) for h in history if "loss" in h and "step" in h]
    evaluation = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h and "step" in h]
    colours = series_colours(["training", "validation"])

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(*zip(*train, strict=True), color=colours["training"], label="training")
    if evaluation:
        ax.plot(
            *zip(*evaluation, strict=True),
            color=colours["validation"],
            marker="o",
            markersize=4,
            label="validation",
        )
    ax.set_title("Supervised fine-tuning")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("cross-entropy loss (nats per token)")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    fig.tight_layout()
    effective: dict[str, int] = {}
    if training_examples or validation_examples:
        effective = {
            "training examples": training_examples,
            "validation examples": validation_examples,
        }
    return save_figure(
        fig,
        destination,
        n=effective or {"logged steps": len(train)},
        source=SOURCE,
        note="one run, seed 42 — nothing here estimates run-to-run variation",
    )


def dpo_alignment(history: list[dict], destination: Path, *, pairs: int = 0) -> Path:
    """The preference run: loss on the left, the reward margin it opens on the right.

    The margin is the quantity DPO optimises, and it is measured on the project's own
    preference pairs. What it does not say is whether the alignment transfers — that question
    belongs to the external preference set, and its answer is no.
    """

    train = [(h["step"], h["loss"]) for h in history if "loss" in h and "step" in h]
    margin = [(h["step"], h["rewards/margins"]) for h in history if "rewards/margins" in h]
    accuracy = [(h["step"], h["rewards/accuracies"]) for h in history if "rewards/accuracies" in h]
    colours = series_colours(["loss", "reward margin", "pairs ordered correctly"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    left.plot(*zip(*train, strict=True), color=colours["loss"])
    left.set_title("Preference optimisation")
    left.set_xlabel("optimiser step")
    left.set_ylabel("DPO loss (nats)")

    right.plot(*zip(*margin, strict=True), color=colours["reward margin"], label="reward margin")
    right.set_xlabel("optimiser step")
    right.set_ylabel("reward margin (log-probability ratio)")
    right.set_title("Separation opened between the two answers")
    # Both series are dimensionless and of the same order here, so they share one axis. A
    # second vertical scale would let the eye read a crossing that means nothing.
    if accuracy:
        right.plot(
            *zip(*accuracy, strict=True),
            color=colours["pairs ordered correctly"],
            linestyle=":",
            label="pairs ordered correctly (share)",
        )
    right.legend(loc="lower right")

    fig.tight_layout()
    return save_figure(
        fig,
        destination,
        n={"training pairs": pairs} if pairs else {"logged steps": len(train)},
        source=SOURCE,
        note="measured on the project's own preference pairs, not on an external set",
    )


def systems_comparison(measures: dict[str, dict], destination: Path) -> Path:
    """Every system on the same hand-written cases: accuracy, undertriage, overtriage.

    Undertriage gets its own panel because it is the only failure that costs a patient, and
    because reading it off the accuracy panel is impossible: the system that never undertriages
    is also the one that sends everybody to resuscitation.
    """

    names = [n for n in SYSTEM_ORDER if n in measures]
    labels = [SYSTEM_LABELS[n] for n in names]
    cases = int(measures[names[0]]["n"])
    colours = series_colours(
        labels,
        control=[SYSTEM_LABELS[n] for n in names if n in {"majority_class", "always_critical"}],
        reference=[SYSTEM_LABELS["explicit_rule"]] if "explicit_rule" in names else [],
    )
    palette = [colours[label] for label in labels]

    accuracy = [measures[n]["accuracy"] for n in names]
    accuracy_ci = [tuple(measures[n]["accuracy_ci95"]) for n in names]
    under = [measures[n]["undertriage"] for n in names]
    under_ci = [tuple(measures[n]["undertriage_ci95"]) for n in names]
    over = [measures[n]["overtriage"] for n in names]
    urgent = int(measures[names[0]]["undertriage_detail"]["urgent_cases"])

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.4))
    panels = [
        (axes[0], "Accuracy", accuracy, accuracy_ci, "accuracy (share of cases)"),
        (axes[1], "Undertriage — the failure that costs", under, under_ci, "share of urgent cases"),
        (axes[2], "Overtriage — the cost of caution", over, None, "share of all cases"),
    ]
    for ax, title, values, intervals, ylabel in panels:
        bars = ax.bar(range(len(labels)), values, color=palette, width=0.66)
        if intervals is not None:
            ax.errorbar(
                range(len(labels)),
                values,
                yerr=_bar_interval(values, intervals),
                fmt="none",
                ecolor=PALETTE["ink"],
                elinewidth=1.1,
                capsize=3.5,
            )
        tops = [
            (interval[1] if interval else value)
            for value, interval in zip(values, intervals or [None] * len(values), strict=True)
        ]
        _annotate(ax, bars, values, tops=tops)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("system")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        _headroom(ax, [t + 0.10 for t in tops])

    fig.tight_layout()
    return save_figure(
        fig,
        destination,
        n={"cases": cases, "urgent cases": urgent},
        dispersion=(
            "95% interval on each proportion — exact (Clopper-Pearson) on undertriage, "
            "Wilson on accuracy; overtriage is drawn without one, it is not the safety metric"
        ),
        source=SOURCE,
    )


def confusion_matrices(confusions: dict[str, dict], destination: Path) -> Path:
    """Where each model's errors land. Cells are counted, so no cell carries an interval.

    The extra ``Off-format`` column is not a triage class: an unreadable answer is not a
    prediction, and filing it under one would move the error into a class it never named.
    """

    names = list(confusions)
    fig, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 4.0))
    axes = np.atleast_1d(axes)
    cmap = sequential_cmap()
    total = 0
    for ax, name in zip(axes, names, strict=True):
        block = confusions[name]
        rows, columns = block["rows"], block["columns"]
        matrix = np.array([[block["matrix"][r][c] for c in columns] for r in rows], dtype=float)
        total = int(matrix.sum())
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(1.0, matrix.max()))
        ax.set_title(name)
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels([LEVEL_LABELS.get(c, c) for c in columns], rotation=30, ha="right")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([LEVEL_LABELS.get(r, r) for r in rows])
        ax.set_xlabel("predicted level")
        ax.set_ylabel("true level")
        ax.grid(False)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = int(matrix[i, j])
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=PALETTE["paper"] if value > matrix.max() * 0.55 else PALETTE["ink"],
                )
    fig.tight_layout()
    return save_figure(
        fig,
        destination,
        n={"cases": total},
        source=SOURCE,
        note=COUNTED,
    )


def accuracy_by_case_type(by_system: dict[str, dict], destination: Path) -> Path:
    """Accuracy split by the kind of difficulty the case presents.

    The atypical categories hold ten cases or fewer. Their intervals cover half the scale,
    and that is the point of drawing them: they situate a weakness without quantifying it.
    """

    names = list(by_system)
    categories = [c for c in CASE_TYPE_LABELS if c in next(iter(by_system.values()))]
    colours = series_colours(names)
    width = 0.8 / max(1, len(names))

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    sizes: dict[str, int] = {}
    for offset, name in enumerate(names):
        block = by_system[name]
        values = [block[c]["accuracy"] for c in categories]
        intervals = [
            tuple(block[c]["accuracy_ci95"]) if block[c]["accuracy_ci95"] else None
            for c in categories
        ]
        positions = [i + (offset - (len(names) - 1) / 2) * width for i in range(len(categories))]
        bars = ax.bar(positions, values, width=width * 0.92, color=colours[name], label=name)
        # Below six cases no interval is publishable, and a zero-length error bar would draw
        # caps that read as near-certainty — the opposite of what the effective supports. Those
        # bars are hatched instead, and the hatch survives a greyscale print.
        measurable = [i for i, interval in enumerate(intervals) if interval is not None]
        if measurable:
            ax.errorbar(
                [positions[i] for i in measurable],
                [values[i] for i in measurable],
                yerr=_bar_interval(
                    [values[i] for i in measurable], [intervals[i] for i in measurable]
                ),
                fmt="none",
                ecolor=PALETTE["ink"],
                elinewidth=1.0,
                capsize=3,
            )
        for i, interval in enumerate(intervals):
            if interval is None:
                bars[i].set_hatch(NOT_MEASURABLE)
                bars[i].set_edgecolor(PALETTE["paper"])
        for category in categories:
            sizes[category] = int(block[category]["n"])

    ax.set_title("Accuracy by kind of presentation")
    ax.set_ylabel("accuracy (share of cases)")
    ax.set_xlabel("kind of presentation")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([f"{CASE_TYPE_LABELS[c]}\nn = {sizes[c]}" for c in categories], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", ncol=len(names))
    fig.tight_layout()
    return save_figure(
        fig,
        destination,
        n=sizes,
        dispersion=(
            "95% interval on each proportion; below six cases no interval is publishable and "
            "the bar is hatched instead"
        ),
        source=SOURCE,
    )


def recall_versus_transfer(
    internal: dict[str, dict], clinical: dict[str, dict], destination: Path
) -> Path:
    """What the model restates, what it transfers, and the gap between the two.

    The internal test split draws lines, not presentations: every presentation it contains was
    seen during training under another wording. It therefore measures restatement — a ceiling —
    while the hand-written clinical set measures transfer. The right panel prices the gap with
    an interval on the difference, because comparing two marginal intervals by eye is a
    fallacy, and here it gives the wrong answer.
    """

    names = [n for n in SYSTEM_ORDER if n in internal and n in clinical]
    labels = [SYSTEM_LABELS[n] for n in names]
    internal_n = int(next(iter(internal.values()))["n"])
    clinical_n = int(next(iter(clinical.values()))["n"])
    colours = series_colours(["internal test split", "hand-written clinical set"])
    width = 0.38

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for offset, (block, label, size) in enumerate(
        [
            (internal, "internal test split", internal_n),
            (clinical, "hand-written clinical set", clinical_n),
        ]
    ):
        values = [block[n]["accuracy"] for n in names]
        intervals = [tuple(block[n]["accuracy_ci95"]) for n in names]
        positions = [i + (offset - 0.5) * width for i in range(len(names))]
        left.bar(
            positions,
            values,
            width=width * 0.92,
            color=colours[label],
            label=f"{label} (n = {size})",
        )
        left.errorbar(
            positions,
            values,
            yerr=_bar_interval(values, intervals),
            fmt="none",
            ecolor=PALETTE["ink"],
            elinewidth=1.0,
            capsize=3,
        )
    left.set_title("Restatement against transfer")
    left.set_ylabel("accuracy (share of cases)")
    left.set_xlabel("system")
    left.set_xticks(range(len(names)))
    left.set_xticklabels(labels, rotation=20, ha="right")
    left.set_ylim(0, 1.32)
    left.legend(loc="upper left", ncol=1)

    gaps, lows, highs = [], [], []
    for name in names:
        gap, low, high = difference_newcombe(
            round(internal[name]["accuracy"] * internal_n),
            internal_n,
            round(clinical[name]["accuracy"] * clinical_n),
            clinical_n,
        )
        gaps.append(gap)
        lows.append(gap - low)
        highs.append(high - gap)
    right.errorbar(
        gaps,
        range(len(names)),
        xerr=np.array([lows, highs]),
        fmt="o",
        color=PALETTE["primary"],
        ecolor=PALETTE["ink"],
        elinewidth=1.2,
        capsize=4,
    )
    reference_line(right, x=0.0, label="no gap")
    right.set_title("Internal minus clinical, with its interval")
    right.set_xlabel("difference in accuracy (percentage points / 100)")
    right.set_ylabel("system")
    right.set_yticks(range(len(names)))
    right.set_yticklabels(labels)
    right.legend(loc="lower right")

    fig.tight_layout()
    return save_figure(
        fig,
        destination,
        n={"internal cases": internal_n, "clinical cases": clinical_n},
        dispersion=(
            "95% Wilson interval on each accuracy; 95% Newcombe interval on the difference, "
            "the two sets being independent"
        ),
        source=SOURCE,
    )


def endpoint_latency(measures: dict[str, dict], destination: Path) -> Path:
    """What one triage decision costs, and how many the gateway serves at once.

    Latency is an order statistic on a skewed sample: the artefact keeps the median and the
    95th percentile, not the individual calls, so the figure draws those two and names them.
    Neither is a mean, and neither carries a confidence interval.
    """

    names = sorted(measures, key=lambda k: int(k.rsplit("_", 1)[-1]))
    concurrency = [int(n.rsplit("_", 1)[-1]) for n in names]
    p50 = [measures[n]["p50_ms"] for n in names]
    p95 = [measures[n]["p95_ms"] for n in names]
    throughput = [measures[n]["requests_per_s"] for n in names]
    requests = sum(int(measures[n]["requests"]) for n in names)
    colours = series_colours(["median", "95th percentile", "throughput"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.8, 4.2))
    for index, (low, high) in enumerate(zip(p50, p95, strict=True)):
        left.plot(
            [index, index],
            [low, high],
            color=PALETTE["muted"],
            linewidth=2.0,
            solid_capstyle="round",
        )
    left.scatter(range(len(names)), p50, s=48, color=colours["median"], zorder=3, label="median")
    left.scatter(
        range(len(names)),
        p95,
        s=48,
        marker="^",
        color=colours["95th percentile"],
        zorder=3,
        label="95th percentile",
    )
    left.set_title("Perceived latency of POST /triage")
    left.set_xlabel("concurrent callers")
    left.set_ylabel("latency (ms)")
    left.set_xticks(range(len(names)))
    left.set_xticklabels([str(c) for c in concurrency])
    left.set_ylim(0, max(p95) * 1.2)
    left.legend(loc="upper left")

    bars = right.bar(range(len(names)), throughput, color=colours["throughput"], width=0.6)
    for bar, value in zip(bars, throughput, strict=True):
        right.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(throughput) * 0.03,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=PALETTE["ink"],
        )
    right.set_title("Throughput of the gateway")
    right.set_xlabel("concurrent callers")
    right.set_ylabel("requests per second")
    right.set_xticks(range(len(names)))
    right.set_xticklabels([str(c) for c in concurrency])
    right.set_ylim(0, max(throughput) * 1.25)

    fig.tight_layout()
    return save_figure(
        fig,
        destination,
        n={
            "requests per concurrency level": int(measures[names[0]]["requests"]),
            "requests": requests,
        },
        dispersion="order statistics: the bar spans the median to the 95th percentile",
        source=SOURCE,
        note="one machine, one run — nothing here estimates run-to-run variation",
    )


def subgroup_undertriage(by_subgroup: dict[str, dict], destination: Path) -> Path:
    """Undertriage of the shipped model on each subgroup, with its exact interval.

    Exact rather than Wilson, and drawn even below six cases: this is the safety metric, and
    it is by subgroup that its effectives are smallest.
    """

    # Declared order rather than the artefact's: a chart sorted by name would put "discordant
    # vitals" between "direct" and "falsely alarming", and the reader would look for a meaning
    # in a sequence that has none.
    names = [n for n in CASE_TYPE_LABELS if n in by_subgroup]
    values = [by_subgroup[n]["undertriage"] for n in names]
    sizes = [int(by_subgroup[n]["undertriage_detail"]["urgent_cases"]) for n in names]
    faults = [int(by_subgroup[n]["undertriage_detail"]["misses"]) for n in names]
    intervals = [clopper_pearson(f, s) for f, s in zip(faults, sizes, strict=True)]
    colours = series_colours([CASE_TYPE_LABELS.get(n, n) for n in names])

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.bar(
        range(len(names)),
        values,
        color=[colours[CASE_TYPE_LABELS.get(n, n)] for n in names],
        width=0.62,
    )
    ax.errorbar(
        range(len(names)),
        values,
        yerr=_bar_interval(values, intervals),
        fmt="none",
        ecolor=PALETTE["ink"],
        elinewidth=1.1,
        capsize=4,
    )
    _annotate(ax, bars, values, tops=[i[1] for i in intervals])
    ax.set_title("Undertriage by kind of presentation")
    ax.set_ylabel("share of urgent cases missed")
    ax.set_xlabel("kind of presentation")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        [
            f"{CASE_TYPE_LABELS.get(n, n)}\n{f}/{s}"
            for n, f, s in zip(names, faults, sizes, strict=True)
        ],
        fontsize=9,
    )
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    return save_figure(
        fig,
        destination,
        n={
            f"urgent, {CASE_TYPE_LABELS.get(n, n)}".replace("\n", " "): s
            for n, s in zip(names, sizes, strict=True)
        },
        dispersion="95% exact (Clopper-Pearson) interval, drawn at every effective",
        source=SOURCE,
    )


__all__ = [
    "accuracy_by_case_type",
    "apply_style",
    "close",
    "confusion_matrices",
    "dataset_composition",
    "dpo_alignment",
    "endpoint_latency",
    "hyperparameter_tuning",
    "recall_versus_transfer",
    "sft_training",
    "subgroup_undertriage",
    "systems_comparison",
]
