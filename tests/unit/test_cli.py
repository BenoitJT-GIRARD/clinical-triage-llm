"""Le résumé de projet doit décrire ce qui existe, pas ce qui était prévu.

`uv run clinical-triage` est la première commande lancée sur un poste neuf. Si elle
annonce la configuration LoRA par défaut alors qu'un adaptateur entraîné avec une
autre est sur le disque, elle induit en erreur dès la première seconde.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from clinical_triage import cli


@pytest.fixture
def adaptateur(tmp_path, monkeypatch):
    """Redirige `PATHS` vers une racine jetable et renvoie le dossier d'adaptateur."""
    monkeypatch.setattr(cli, "PATHS", dataclasses.replace(cli.PATHS, models=tmp_path))
    return cli.PATHS.sft_adapter


def test_sans_adaptateur_la_configuration_est_annoncee_comme_un_defaut(adaptateur):
    resume = cli._lora()
    assert "défaut" in resume
    assert f"r={cli.TRAINING.lora_r}" in resume


def test_avec_un_adaptateur_c_est_son_rang_qui_est_affiche(adaptateur):
    """Le réglage des hyperparamètres peut retenir un rang autre que celui par défaut."""
    adaptateur.mkdir(parents=True)
    (adaptateur / "adapter_config.json").write_text(
        json.dumps({"r": 32, "lora_alpha": 64}), encoding="utf-8"
    )

    resume = cli._lora()
    assert "r=32" in resume
    assert "alpha=64" in resume
    assert "défaut" not in resume


def test_le_resume_s_affiche_sans_dataset_ni_modele(adaptateur, capsys):
    """Sur un poste neuf, rien n'est encore produit : la commande doit tenir."""
    cli.main()
    sortie = capsys.readouterr().out
    assert "Agent IA de triage médical" in sortie
    assert "Adaptateur SFT" in sortie
