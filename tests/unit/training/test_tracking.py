"""Le suivi d'expériences ne doit jamais faire échouer un entraînement.

Un entraînement coûte des heures de GPU ; le suivi coûte quelques kilo-octets de
JSON. Si le second tombe, le premier continue. Ces tests forcent les pannes que
`tracking` prétend absorber, parce qu'une promesse écrite dans une docstring et
jamais éprouvée n'est qu'une intention.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import types

import pytest

from clinical_triage.training import tracking


@pytest.fixture
def rapports(tmp_path, monkeypatch):
    """Redirige les résumés JSON vers un dossier jetable.

    `PATHS` est un dataclass gelé : on remplace l'objet entier plutôt qu'un champ.
    """
    monkeypatch.setattr(tracking, "PATHS", dataclasses.replace(tracking.PATHS, reports=tmp_path))
    return tmp_path


def _mlflow_en_panne(message: str) -> types.ModuleType:
    """Un faux module MLflow qui échoue dès l'ouverture, comme un magasin verrouillé."""
    faux = types.ModuleType("mlflow")

    def echouer(*_args, **_kwargs):
        raise RuntimeError(message)

    faux.set_tracking_uri = echouer
    faux.set_experiment = echouer
    faux.start_run = echouer
    faux.log_params = echouer
    faux.log_metric = echouer
    faux.end_run = echouer
    return faux


def test_un_magasin_verrouille_laisse_l_entrainement_continuer(rapports, monkeypatch):
    """C'est la panne observée quand le dépôt est posé sur un dossier synchronisé."""
    monkeypatch.setitem(sys.modules, "mlflow", _mlflow_en_panne("database is locked"))

    with tracking.track("essai", {"lr": 2e-4}) as resume:
        resume.metriques["exactitude"] = 0.9

    ecrit = json.loads((rapports / "training" / "essai.json").read_text(encoding="utf-8"))
    assert ecrit["metriques"]["exactitude"] == 0.9


def test_mlflow_absent_laisse_l_entrainement_continuer(rapports, monkeypatch):
    """Une entrée à `None` dans `sys.modules` fait lever `ImportError` à l'import."""
    monkeypatch.setitem(sys.modules, "mlflow", None)

    with tracking.track("sans_mlflow", {}) as resume:
        resume.metriques["exactitude"] = 0.5

    ecrit = json.loads((rapports / "training" / "sans_mlflow.json").read_text(encoding="utf-8"))
    assert ecrit["metriques"]["exactitude"] == 0.5


def test_un_entrainement_qui_echoue_laisse_quand_meme_son_resume(rapports, monkeypatch):
    """Ce qui a été mesuré avant l'erreur est la première pièce du diagnostic."""
    monkeypatch.setitem(sys.modules, "mlflow", _mlflow_en_panne("database is locked"))

    with (
        pytest.raises(ValueError, match="perte divergente"),
        tracking.track("interrompu", {}) as resume,
    ):
        resume.metriques["derniere_perte"] = 42.0
        raise ValueError("perte divergente")

    ecrit = json.loads((rapports / "training" / "interrompu.json").read_text(encoding="utf-8"))
    assert ecrit["metriques"]["derniere_perte"] == 42.0


def test_une_execution_interrompue_est_marquee_en_echec(rapports, monkeypatch):
    """Sinon l'interface MLflow mélangerait entraînements aboutis et interrompus."""
    statuts = []
    faux = types.ModuleType("mlflow")
    faux.set_tracking_uri = lambda *_a, **_k: None
    faux.set_experiment = lambda *_a, **_k: None
    faux.start_run = lambda *_a, **_k: None
    faux.log_params = lambda *_a, **_k: None
    faux.log_metric = lambda *_a, **_k: None
    faux.end_run = lambda status="FINISHED": statuts.append(status)
    monkeypatch.setitem(sys.modules, "mlflow", faux)

    with tracking.track("abouti", {}):
        pass
    with pytest.raises(ValueError, match="boum"), tracking.track("interrompu", {}):
        raise ValueError("boum")

    assert statuts == ["FINISHED", "FAILED"]


def test_l_adresse_du_magasin_suit_la_variable_de_mlflow(monkeypatch):
    """Sur un poste dont le dépôt est synchronisé, c'est la seule échappatoire."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///ailleurs/mlflow.db")
    assert tracking.tracking_uri() == "sqlite:///ailleurs/mlflow.db"
