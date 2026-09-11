"""Cohérence des dépendances de l'image de service.

`requirements-api.txt` est généré depuis `requirements-api.in` : le premier porte
la fermeture complète épinglée, le second les contraintes. Les deux fichiers se
modifient séparément, et rien n'empêche d'éditer le généré à la main — auquel cas
l'image installerait autre chose que ce qui a été résolu et audité.

Ces tests sont statiques : ils comparent les deux fichiers, sans résoudre quoi
que ce soit ni toucher au réseau.
"""

from __future__ import annotations

import re

import pytest

from clinical_triage.config import PATHS

CONTRAINTES = PATHS.root / "deploy" / "requirements-api.in"
FERMETURE = PATHS.root / "deploy" / "requirements-api.txt"


def _epingles(chemin) -> dict[str, str]:
    """Paquets épinglés d'un fichier de dépendances, nom normalisé."""
    trouves = {}
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        correspondance = re.match(r"^([a-zA-Z0-9._-]+)(?:\[[^\]]+\])?==(\S+)", ligne)
        if correspondance:
            trouves[correspondance.group(1).lower().replace("_", "-")] = correspondance.group(2)
    return trouves


@pytest.fixture(scope="module")
def contraintes() -> dict[str, str]:
    return _epingles(CONTRAINTES)


@pytest.fixture(scope="module")
def fermeture() -> dict[str, str]:
    return _epingles(FERMETURE)


def test_la_fermeture_est_bien_plus_large_que_les_contraintes(contraintes, fermeture):
    """Garde-fou du test : une lecture cassée le rendrait toujours vert."""
    assert len(contraintes) >= 7
    assert len(fermeture) > 3 * len(contraintes)


def test_chaque_contrainte_se_retrouve_a_la_meme_version(contraintes, fermeture):
    """Le fichier généré ne doit pas dériver de ce qui a été demandé."""
    ecarts = {
        nom: (version, fermeture.get(nom))
        for nom, version in contraintes.items()
        if fermeture.get(nom) != version
    }
    assert ecarts == {}


def test_les_deux_modeles_spacy_sont_dans_la_fermeture(fermeture):
    """Presidio ne masque rien sans eux, et l'anonymisation du journal tombe."""
    texte = FERMETURE.read_text(encoding="utf-8")
    assert "fr_core_news_md-3.8.0" in texte
    assert "en_core_web_sm-3.8.0" in texte


def test_l_image_n_embarque_ni_torch_ni_transformers(fermeture):
    """La génération est déléguée à vLLM : l'image n'a pas à peser trois gigaoctets."""
    assert "torch" not in fermeture
    assert "transformers" not in fermeture
    assert "datasets" not in fermeture


def test_le_fichier_genere_se_declare_comme_tel():
    """Sans cet avertissement, la prochaine correction se ferait dans le mauvais fichier."""
    assert "FICHIER GENERE" in FERMETURE.read_text(encoding="utf-8")
