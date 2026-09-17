"""Experiment tracking with MLflow.

MLflow is used because it runs entirely locally: runs are written under ``var/mlruns/``, no
account and no API key are needed, and no data leaves the machine. For a health project, that
last property is not a detail.

This module does three things and nothing else: open a run, put the hyper-parameters and the
metrics into it, and write at the end a versioned JSON summary under ``reports/``. That summary
is what lets anyone read a training run's results back without installing anything.

Nothing here may ever fail a training run. Training costs hours of GPU; tracking costs a few
kilobytes of JSON. Every way tracking can fail — an absent package, a locked store, an
unreachable path, an unusable environment description — degrades to the JSON summary and lets
the run continue.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from clinical_triage.config import PATHS
from clinical_triage.utils import get_logger, path_for_log

logger = get_logger("tracking")


def tracking_uri() -> str:
    """Address of the MLflow run store.

    Recent MLflow versions store runs in a SQLite database. By default it lives under
    ``var/mlruns/`` at the project root::

        uv run mlflow ui --backend-store-uri sqlite:///var/mlruns/mlflow.db

    **Beware of synchronised folders.** If the repository sits on synchronised storage, the sync
    client locks and copies the file while SQLite writes, which interrupts schema creation with a
    missing-table error. On such a machine the store must go elsewhere, as the Python environment
    and the model caches already do::

        export MLFLOW_TRACKING_URI="sqlite:///$HOME/.local/share/clinical-triage/mlflow.db"

    The variable is MLflow's own: no project-specific configuration.
    """
    configured = os.getenv("MLFLOW_TRACKING_URI")
    if configured:
        return configured
    PATHS.tracking.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(PATHS.tracking / 'mlflow.db').as_posix()}"


@dataclass
class RunSummary:
    """Summary of one training run, written under ``reports/``."""

    name: str
    hyperparameters: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    history: list = field(default_factory=list)


#: Written in place of a version when the library is not installed. The training path imports
#: torch long before reaching here and would fail there with a clear message; this function only
#: describes an environment, and an environment it cannot describe must not raise.
UNAVAILABLE = "unavailable"


def describe_environment() -> dict:
    """Describe the hardware and the library versions used, for reproducibility.

    Degrades rather than raising. The service image and the continuous integration install the
    project without its training group: importing torch there raises ``ModuleNotFoundError``, and
    a summary that cannot name a version is still a summary. Raising here would take down the
    tests that exercise this module's own promise — that tracking never fails a run — in the very
    environment where they run.
    """
    description: dict[str, object] = {}
    for name in ("torch", "transformers", "trl"):
        try:
            description[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - absent, broken or unreadable: same outcome
            description[name] = UNAVAILABLE
    description["gpu"] = "none"
    description["gpu_memory_gb"] = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            description["gpu"] = properties.name
            description["gpu_memory_gb"] = round(properties.total_memory / 1024**3, 1)
    except Exception as error:  # noqa: BLE001 - no torch, no driver, no GPU
        logger.debug("No GPU description available (%s).", error)
    return description


def _open_mlflow(name: str, hyperparameters: dict, environment: dict):
    """Open an MLflow run, or return ``None`` when tracking is unavailable.

    The ways tracking can fail are not limited to the package being absent: a SQLite store locked
    by a sync client, an unreachable tracking path or a malformed environment variable each raise
    something different. In every case the output degrades to the JSON summary alone and training
    continues — training is what costs hours of GPU.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment("clinical-triage")
        mlflow.start_run(run_name=name)
        mlflow.log_params({key: str(value) for key, value in hyperparameters.items()})
        mlflow.log_params({f"env_{key}": str(value) for key, value in environment.items()})
    except ImportError:
        logger.warning("MLflow absent: only the JSON summary will be written.")
        return None
    except Exception as error:  # noqa: BLE001 - deliberate, see the docstring
        logger.warning("MLflow tracking unavailable (%s): only the JSON summary.", error)
        return None
    return mlflow


def _close_mlflow(mlflow, summary: RunSummary, failed: bool) -> None:
    """Put metrics and history into the run, then close it.

    A run closed after an error is marked "FAILED": without that, comparing experiments in the
    MLflow interface would mix completed runs with those that stopped mid-way.
    """
    try:
        for key, value in summary.metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))
        for step, record in enumerate(summary.history):
            for key, value in record.items():
                if isinstance(value, (int, float)) and key != "epoch":
                    mlflow.log_metric(f"train_{key}", float(value), step=step)
        mlflow.end_run(status="FAILED" if failed else "FINISHED")
    except Exception as error:  # noqa: BLE001 - deliberate, see the docstring
        logger.warning(
            "Writing the MLflow run was interrupted (%s); the JSON summary is still written.",
            error,
        )


@contextmanager
def track(name: str, hyperparameters: dict):
    """Open an MLflow run and yield its summary, for the caller to complete.

    The JSON summary is written in every case, including when training fails: what was measured
    before the error remains the first piece of the diagnosis.
    """
    summary = RunSummary(
        name=name, hyperparameters=hyperparameters, environment=describe_environment()
    )
    mlflow = _open_mlflow(name, hyperparameters, summary.environment)

    failed = False
    try:
        yield summary
    except Exception:
        failed = True
        raise
    finally:
        if mlflow is not None:
            _close_mlflow(mlflow, summary, failed)
        write_summary(summary)


def write_summary(summary: RunSummary) -> Path:
    """Write the run summary under ``reports/training/``."""
    directory = PATHS.reports / "training"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{summary.name}.json"
    path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Training summary written → %s", path_for_log(path))
    return path
