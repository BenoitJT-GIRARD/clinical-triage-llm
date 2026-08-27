"""Suivi des expériences d'entraînement avec MLflow.

Le cahier des charges laisse le choix entre MLflow et Weights & Biases. On
retient MLflow parce qu'il fonctionne entièrement en local : les exécutions sont
écrites dans `mlruns/`, aucun compte ni aucune clé d'API n'est nécessaire, et
aucune donnée ne quitte le poste. Pour un projet de santé, cette dernière
propriété n'est pas un détail.

Ce module fait trois choses et rien d'autre : ouvrir une exécution, y déposer
les hyperparamètres et les métriques, et écrire à la fin un résumé JSON versionné
dans `reports/`. Ce résumé est ce qui permet de relire les résultats d'un
entraînement sans installer quoi que ce soit — y compris pour un lecteur du
projet, qui n'aura pas `mlruns/` sous la main.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from clinical_triage.config import PATHS
from clinical_triage.utils import chemin_pour_journal, get_logger

logger = get_logger("tracking")


def tracking_uri() -> str:
    """Adresse du magasin d'exécutions MLflow.

    Les versions récentes de MLflow stockent les exécutions dans une base
    SQLite. Par défaut, elle vit dans `mlruns/` à la racine du projet :

        uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

    **Attention aux dossiers synchronisés.** Si le dépôt est posé sur un espace
    de stockage synchronisé, le client de synchronisation verrouille et recopie
    le fichier pendant que SQLite écrit, ce qui interrompt la création du schéma
    avec une erreur de table manquante. Sur un tel poste, il faut placer le
    magasin ailleurs, comme on le fait déjà pour l'environnement Python et les
    caches de modèles :

        export MLFLOW_TRACKING_URI="sqlite:///$HOME/.local/share/clinical-triage/mlflow.db"

    La variable est celle de MLflow : aucune configuration propre au projet.
    """
    configure = os.getenv("MLFLOW_TRACKING_URI")
    if configure:
        return configure
    PATHS.tracking.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(PATHS.tracking / 'mlflow.db').as_posix()}"


@dataclass
class RunSummary:
    """Résumé d'une exécution d'entraînement, écrit dans `reports/`."""

    nom: str
    hyperparametres: dict = field(default_factory=dict)
    metriques: dict = field(default_factory=dict)
    environnement: dict = field(default_factory=dict)
    historique: list = field(default_factory=list)


def describe_environment() -> dict:
    """Décrit le matériel et les versions utilisés, pour la reproductibilité."""
    import torch
    import transformers
    import trl

    description = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "gpu": "aucun",
        "memoire_gpu_go": 0.0,
    }
    if torch.cuda.is_available():
        proprietes = torch.cuda.get_device_properties(0)
        description["gpu"] = proprietes.name
        description["memoire_gpu_go"] = round(proprietes.total_memory / 1024**3, 1)
    return description


def _ouvrir_mlflow(nom: str, hyperparametres: dict, environnement: dict):
    """Ouvre une exécution MLflow, ou renvoie `None` si le suivi est indisponible.

    Le suivi ne doit jamais faire échouer un entraînement, et les façons dont il
    peut lui-même échouer ne se limitent pas à l'absence du paquet : un magasin
    SQLite verrouillé par un client de synchronisation, un chemin de suivi
    inaccessible ou une variable d'environnement mal formée lèvent tout autre
    chose. Dans tous ces cas, la sortie se dégrade au seul résumé JSON et
    l'entraînement continue — c'est lui qui coûte des heures de GPU.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment("clinical-triage")
        mlflow.start_run(run_name=nom)
        mlflow.log_params({cle: str(valeur) for cle, valeur in hyperparametres.items()})
        mlflow.log_params({f"env_{cle}": str(valeur) for cle, valeur in environnement.items()})
    except ImportError:
        logger.warning("MLflow absent : seul le résumé JSON sera écrit.")
        return None
    except Exception as erreur:  # noqa: BLE001 — capture volontaire, voir la docstring
        logger.warning("Suivi MLflow indisponible (%s) : seul le résumé JSON sera écrit.", erreur)
        return None
    return mlflow


def _refermer_mlflow(mlflow, resume: RunSummary, echoue: bool) -> None:
    """Dépose métriques et historique dans l'exécution, puis la referme.

    Une exécution refermée après une erreur est marquée « FAILED » : sans cela,
    une comparaison d'expériences dans l'interface MLflow mélangerait les
    entraînements aboutis et ceux qui se sont interrompus en route.
    """
    try:
        for cle, valeur in resume.metriques.items():
            if isinstance(valeur, (int, float)):
                mlflow.log_metric(cle, float(valeur))
        for etape, enregistrement in enumerate(resume.historique):
            for cle, valeur in enregistrement.items():
                if isinstance(valeur, (int, float)) and cle != "epoch":
                    mlflow.log_metric(f"train_{cle}", float(valeur), step=etape)
        mlflow.end_run(status="FAILED" if echoue else "FINISHED")
    except Exception as erreur:  # noqa: BLE001 — capture volontaire, voir la docstring
        logger.warning(
            "Écriture du suivi MLflow interrompue (%s) ; le résumé JSON reste écrit.", erreur
        )


@contextmanager
def track(nom: str, hyperparametres: dict):
    """Ouvre une exécution MLflow et renvoie son résumé, à compléter par l'appelant.

    Le résumé JSON est écrit dans tous les cas, y compris si l'entraînement
    échoue : ce qui a été mesuré avant l'erreur reste la première pièce du
    diagnostic.
    """
    resume = RunSummary(
        nom=nom, hyperparametres=hyperparametres, environnement=describe_environment()
    )
    mlflow = _ouvrir_mlflow(nom, hyperparametres, resume.environnement)

    echoue = False
    try:
        yield resume
    except Exception:
        echoue = True
        raise
    finally:
        if mlflow is not None:
            _refermer_mlflow(mlflow, resume, echoue)
        write_summary(resume)


def write_summary(resume: RunSummary) -> Path:
    """Écrit le résumé d'exécution dans `reports/training/`."""
    dossier = PATHS.reports / "training"
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"{resume.nom}.json"
    chemin.write_text(
        json.dumps(asdict(resume), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Résumé d'entraînement écrit → %s", chemin_pour_journal(chemin))
    return chemin
