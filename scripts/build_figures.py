"""Draw the published figures from the result artefacts.

Every number in every figure comes from a versioned file under ``reports/`` or
``data/processed/``. Nothing is typed in here, so a re-run of the evaluation moves the figures
with it, and ``reports/figures/MANIFEST.json`` records what was written, from what, and when.

Usage::

    uv run python scripts/build_figures.py
"""

from __future__ import annotations

import argparse

from clinical_triage.config import PATHS
from clinical_triage.reporting import figures
from clinical_triage.reporting.artifacts import read_results
from clinical_triage.utils import get_logger

logger = get_logger("figures")


def _shipped_model(evaluation: dict) -> str:
    """Name of the most advanced model actually evaluated."""
    models = evaluation["clinical_set"]["models"]
    # The merged model comes first: it is the one that is shipped and served.
    for name in ("dpo-merged", "dpo", "sft", "base"):
        if name in models:
            return name
    raise SystemExit("No model evaluated: run scripts/run_evaluation.py.")


def _progression(models: dict) -> dict:
    """Keep only one form of the aligned model for the progression figures.

    The adapter and the merged model carry the same training: showing both would add a fourth
    column indistinguishable from the third, and would shrink the confusion matrices past
    legibility. Their comparison has its own table, where it is the subject rather than noise.
    """
    if "dpo-merged" in models and "dpo" in models:
        return {name: values for name, values in models.items() if name != "dpo"}
    return models


def _labelled(block: dict) -> dict:
    """Index a block of results by the label a reader sees rather than by the artefact key."""
    return {figures.SYSTEM_LABELS.get(name, name): values for name, values in block.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    metadata = read_results(PATHS.data_processed / "metadata.json")
    evaluation = read_results(PATHS.reports / "evaluation_results.json")
    if metadata is None or evaluation is None:
        raise SystemExit("Run scripts/build_dataset.py then scripts/run_evaluation.py first.")

    training = PATHS.reports / "training"
    sft = read_results(training / "sft.json")
    dpo = read_results(training / "dpo.json")
    tuning = read_results(training / "hyperparameter_comparison.json")
    bench = read_results(PATHS.reports / "benchmark_endpoint.json")

    clinical = evaluation["clinical_set"]
    shipped = _shipped_model(evaluation)
    progression = _progression(clinical["models"])
    directory = PATHS.figures

    figures.apply_style()

    figures.dataset_composition(metadata["statistics"], directory / "dataset_composition.png")
    if tuning:
        figures.hyperparameter_tuning(tuning, directory / "hyperparameter_tuning.png")
    if sft:
        figures.sft_training(sft["history"], directory / "sft_training.png")
    if dpo:
        figures.dpo_alignment(
            dpo["history"],
            directory / "dpo_alignment.png",
            pairs=dpo.get("metrics", {}).get("training_pairs", 0),
        )

    figures.systems_comparison(
        {**clinical["baselines"], **progression}, directory / "systems_comparison.png"
    )
    figures.confusion_matrices(
        {label: v["confusion"] for label, v in _labelled(progression).items()},
        directory / "confusion_matrices.png",
    )
    figures.accuracy_by_case_type(
        {label: v["per_case_type"] for label, v in _labelled(progression).items()},
        directory / "accuracy_by_case_type.png",
    )
    figures.subgroup_undertriage(
        clinical["models"][shipped]["per_case_type"], directory / "undertriage_by_case_type.png"
    )
    if evaluation["internal_set"]["models"]:
        figures.recall_versus_transfer(
            _progression(evaluation["internal_set"]["models"]),
            progression,
            directory / "recall_versus_transfer.png",
        )
    if bench:
        figures.endpoint_latency(bench["measures"], directory / "endpoint_latency.png")

    logger.info("Figures written → %s", directory)


if __name__ == "__main__":
    main()
