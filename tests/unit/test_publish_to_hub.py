"""Cohérence de ce qui est publié sur le Hugging Face Hub.

Un adaptateur LoRA ne contient pas de modèle : il décrit une correction à
appliquer à des poids précis, qu'il désigne par leur dépôt. Publier l'adaptateur
sans ces poids produit un artefact que personne ne peut ouvrir — et l'erreur ne
se voit qu'au premier téléchargement, c'est-à-dire trop tard.

Ces tests lisent la configuration de publication et les cartes de modèle : ils ne
touchent pas au réseau.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys

import pytest

from clinical_triage.config import MODEL, PATHS
from clinical_triage.data import dataset_io


def _script_de_publication():
    """Charge `scripts/publish_to_hub.py`, dont le nom n'est pas importable tel quel."""
    chemin = PATHS.root / "scripts" / "08_publish_hf.py"
    spec = importlib.util.spec_from_file_location("publication_hf", chemin)
    module = importlib.util.module_from_spec(spec)
    sys.modules["publication_hf"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def publication():
    return _script_de_publication()


def _base_declaree(carte) -> str | None:
    """Lit `base_model` dans l'en-tête YAML d'une carte de modèle."""
    trouve = re.search(r"^base_model:\s*(\S+)\s*$", carte.read_text(encoding="utf-8"), re.MULTILINE)
    return trouve.group(1) if trouve else None


def test_chaque_modele_publie_a_sa_carte(publication):
    manquantes = [
        quoi
        for quoi in publication.TOUT
        if quoi != "dataset" and not publication._carte_du_modele(quoi).exists()
    ]
    assert manquantes == []


def test_le_modele_de_base_de_chaque_adaptateur_est_publie(publication):
    """Sans lui, `PeftModel.from_pretrained` irait chercher un dépôt inexistant."""
    depots_publies = {
        publication._depot_du_modele(quoi) for quoi in publication.TOUT if quoi != "dataset"
    }
    depots_publies.add(MODEL.base_model)  # le modèle de base vient de son éditeur

    for quoi in publication.TOUT:
        if not quoi.startswith("adaptateur"):
            continue
        base = _base_declaree(publication._carte_du_modele(quoi))
        assert base is not None, f"{quoi} : la carte ne déclare pas son modèle de base"
        assert base in depots_publies, f"{quoi} : son modèle de base {base} n'est pas publié"


def test_l_adaptateur_dpo_sur_le_disque_designe_le_depot_public(publication):
    """Le chemin local écrit par la bibliothèque rendrait les poids inchargeables ailleurs."""
    configuration = PATHS.dpo_adapter / "adapter_config.json"
    if not configuration.exists():
        pytest.skip("Adaptateur DPO absent : l'alignement n'a pas encore été joué.")
    contenu = json.loads(configuration.read_text(encoding="utf-8"))
    assert contenu["base_model_name_or_path"] == MODEL.hub_sft_merged_model_id


def test_les_identifiants_de_depot_sont_tous_distincts():
    """Deux artefacts qui partagent un dépôt s'écraseraient l'un l'autre."""
    identifiants = [
        MODEL.hub_dataset_id,
        MODEL.hub_sft_model_id,
        MODEL.hub_sft_merged_model_id,
        MODEL.hub_dpo_model_id,
        MODEL.hub_merged_model_id,
    ]
    assert len(set(identifiants)) == len(identifiants)


def test_un_dataset_incomplet_ne_part_pas_sur_le_hub(publication, tmp_path, monkeypatch):
    """Publier depuis un dépôt cloné téléverserait deux fichiers sur six.

    Les quatre jeux d'entraînement ne sont pas versionnés : ils se reconstruisent
    par `scripts/build_dataset.py`. Sans ce garde-fou, la publication réussissait et
    l'annonçait, en laissant sur le Hub un dataset amputé.
    """
    import dataclasses

    (tmp_path / "clinical_eval.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        publication, "PATHS", dataclasses.replace(publication.PATHS, data_processed=tmp_path)
    )

    with pytest.raises(SystemExit) as refus:
        publication.publier_dataset(api=None)
    assert "sft_train.jsonl" in str(refus.value)
    assert "clinical_eval.jsonl" not in str(refus.value)


def test_les_six_fichiers_attendus_sont_ceux_que_produit_la_preparation(publication):
    """La liste ne doit pas dériver de ce que `scripts/build_dataset.py` écrit réellement.

    Elle est écrite une seule fois, dans `dataset_io`, et le script de
    préparation comme la publication la lisent de là. Le jeu d'entraînement
    n'étant pas versionné — quatre fichiers sur six —, la comparer au contenu du
    disque ferait passer le test sur la machine qui vient de construire le jeu et
    échouer partout ailleurs : la liste attendue est donc écrite en clair ici.
    """
    assert publication.FICHIERS_DU_DATASET is dataset_io.FICHIERS_DU_DATASET
    assert set(publication.FICHIERS_DU_DATASET) == {
        "sft_train.jsonl",
        "sft_validation.jsonl",
        "sft_test.jsonl",
        "dpo_train.jsonl",
        "clinical_eval.jsonl",
        "metadata.json",
    }


def test_chaque_carte_porte_le_marqueur_de_ses_chiffres(publication):
    """Une carte de modèle sans performance mesurable n'est pas une carte.

    C'est la première page que voit quiconque ouvre le dépôt sur le Hub ;
    elle renvoyait vers un rapport hébergé ailleurs.
    """
    for quoi in publication.MODELES:
        carte = publication._carte_du_modele(quoi)
        assert "{{EVALUATION}}" in carte.read_text(encoding="utf-8"), quoi


def test_chaque_carte_sait_de_quel_modele_evalue_elle_parle(publication):
    assert set(publication.EVALUATION_DE_LA_CARTE) == set(publication.MODELES)


def test_une_carte_sans_evaluation_n_est_pas_publiable(publication, tmp_path, monkeypatch):
    import dataclasses

    monkeypatch.setattr(
        publication, "PATHS", dataclasses.replace(publication.PATHS, reports=tmp_path)
    )
    with pytest.raises(SystemExit, match="Évaluation introuvable"):
        publication._tableau_d_evaluation("modele-final")


def test_le_tableau_de_la_carte_est_ecrit_a_la_francaise(publication, tmp_path, monkeypatch):
    import dataclasses
    import json as json_

    (tmp_path / "evaluation_results.json").write_text(
        json_.dumps(
            {
                "jeu_clinique": {
                    "n": 60,
                    "modeles": {
                        "dpo-fusionne": {
                            "exactitude": 0.917,
                            "exactitude_ic95": [0.82, 0.96],
                            "sous_triage": 0.048,
                            "surclassement": 0.117,
                            "respect_format": 1.0,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        publication, "PATHS", dataclasses.replace(publication.PATHS, reports=tmp_path)
    )
    tableau = publication._tableau_d_evaluation("modele-final")
    assert "0,917 [0,82 – 0,96]" in tableau
    assert "4,8 %" in tableau
    assert "100 %" in tableau
    assert "." not in tableau.replace("|", "")


def test_aucune_carte_ne_fige_une_revision_dans_sa_commande_de_service(publication):
    """La commande de service doit épingler la version publiée, pas la première.

    Écrite en dur, elle continuait de désigner `modele-v1.0.0` après chaque
    nouvelle publication : la carte décrivait des poids et en faisait servir
    d'autres.
    """
    for quoi in publication.MODELES:
        texte = publication._carte_du_modele(quoi).read_text(encoding="utf-8")
        assert "--revision modele-v" not in texte, quoi
        if "--revision" in texte:
            assert "{{REVISION}}" in texte, quoi


def test_la_carte_ecrite_par_la_bibliotheque_n_est_pas_publiee(publication):
    """Elle porte le chemin local du modèle de base, que le Hub rejette.

    L'envoi du dossier entier échouait dessus, avant même d'avoir commencé.
    """
    assert "README.md" in publication.EXCLUSIONS

    dossier = PATHS.dpo_adapter
    carte = dossier / "README.md"
    if carte.exists():
        assert "base_model: " in carte.read_text(encoding="utf-8")
