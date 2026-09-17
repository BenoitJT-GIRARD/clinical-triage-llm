"""Le descripteur de déploiement Modal doit se charger, et lire sa révision.

`modal deploy` commence par importer `infra/modal_app.py` : une erreur de
décorateur, un mot-clé disparu de l'API ou un import manquant y échoue avant même
d'atteindre le réseau. Ce fichier n'étant exécuté nulle part ailleurs, rien ne le
vérifierait sans ces tests.

Le déploiement continu passe la révision du modèle par l'environnement, au moment
du `modal deploy`. Elle est ensuite gravée dans l'image du moteur, car le
conteneur réimporte ce fichier dans un environnement qui ne contient rien de
celui du déploiement.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from clinical_triage.config import MODEL, PATHS

CHEMIN = PATHS.root / "deploy" / "modal_app.py"


def _charger():
    """Importe le descripteur comme le ferait `modal deploy`."""
    spec = importlib.util.spec_from_file_location("modal_app_teste", CHEMIN)
    module = importlib.util.module_from_spec(spec)
    sys.modules["modal_app_teste"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def descripteur(monkeypatch):
    monkeypatch.delenv("TRIAGE_MODEL_ID", raising=False)
    monkeypatch.delenv("TRIAGE_MODEL_REVISION", raising=False)
    return _charger()


def test_le_descripteur_se_charge(descripteur):
    """Première étape de `modal deploy` : si elle passe ici, elle passera là-bas."""
    assert descripteur.app.name == "clinical-triage"
    assert descripteur.Moteur is not None
    assert descripteur.passerelle is not None


def test_le_moteur_sert_le_modele_final_du_projet(descripteur):
    assert MODEL.hub_merged_model_id == descripteur.MODELE


def test_les_references_du_modele_sont_gravees_dans_l_image(descripteur):
    """Le conteneur réimporte ce fichier sans l'environnement du déploiement.

    Ce qui n'est pas inscrit dans l'image au moment du déploiement est perdu.
    """
    variables = descripteur.ENVIRONNEMENT_MOTEUR
    assert variables["TRIAGE_MODEL_ID"] == descripteur.MODELE
    assert variables["TRIAGE_MODEL_REVISION"] == descripteur.REVISION


def test_la_revision_suit_la_variable_du_deploiement_continu(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL_ID", "un-autre-compte/un-modele")
    monkeypatch.setenv("TRIAGE_MODEL_REVISION", "modele-v9.9.9")
    descripteur = _charger()
    assert descripteur.MODELE == "un-autre-compte/un-modele"
    assert descripteur.REVISION == "modele-v9.9.9"


def test_une_variable_vide_retombe_sur_la_version_epinglee(monkeypatch):
    """Une variable de dépôt GitHub non définie arrive vide, pas absente."""
    monkeypatch.setenv("TRIAGE_MODEL_ID", "")
    monkeypatch.setenv("TRIAGE_MODEL_REVISION", "")
    descripteur = _charger()
    assert MODEL.hub_merged_model_id == descripteur.MODELE
    assert descripteur.REVISION.startswith("modele-v")


def test_un_compte_vide_ne_produit_pas_un_identifiant_a_barre_oblique(monkeypatch):
    """La chaîne de déploiement ne passe que le compte : vide, il doit se replier.

    Composé par interpolation, un compte non défini donnait
    « /qwen3-1.7b-clinical-triage », que le Hub refuse.
    """
    monkeypatch.delenv("TRIAGE_MODEL_ID", raising=False)
    monkeypatch.setenv("HF_NAMESPACE", "")
    descripteur = _charger()

    assert not descripteur.MODELE.startswith("/")
    assert descripteur.MODELE == "BenoitJT-GIRARD/qwen3-1.7b-clinical-triage"


def test_le_compte_du_deploiement_continu_est_repris(monkeypatch):
    """Un autre compte publie sous son nom sans que le code change."""
    monkeypatch.delenv("TRIAGE_MODEL_ID", raising=False)
    monkeypatch.setenv("HF_NAMESPACE", "un-autre-compte")
    descripteur = _charger()

    assert descripteur.MODELE == "un-autre-compte/qwen3-1.7b-clinical-triage"
