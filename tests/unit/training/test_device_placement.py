"""Une panne de mémoire vidéo doit se dire à l'endroit où elle se produit.

Quand la carte est pleine, la bibliothèque de chargement répartit une partie des
couches sur le processeur sans rien signaler. Unsloth mémorise cet emplacement,
et l'erreur ne remonte qu'à la première génération sous la forme d'un
« Invalid target device: None » qui ne dit rien de sa cause. Ces tests fixent le
comportement attendu : échouer au chargement, avec la conduite à tenir.
"""

from __future__ import annotations

import types

import pytest

from clinical_triage.inference import _verifier_le_placement_sur_gpu
from clinical_triage.utils import free_gpu_memory


def _modele(*emplacements: str):
    """Un modèle réduit à ce que la vérification lui demande : ses paramètres."""
    parametres = [types.SimpleNamespace(device=types.SimpleNamespace(type=e)) for e in emplacements]
    return types.SimpleNamespace(parameters=lambda: iter(parametres))


def test_un_modele_entierement_sur_gpu_passe():
    _verifier_le_placement_sur_gpu(_modele("cuda", "cuda", "cuda"))


def test_une_seule_couche_restee_sur_le_processeur_suffit_a_echouer():
    """C'est le cas réel : l'essentiel du modèle tient, la fin déborde."""
    with pytest.raises(RuntimeError, match="Mémoire GPU insuffisante"):
        _verifier_le_placement_sur_gpu(_modele("cuda", "cuda", "cpu"))


def test_le_message_nomme_les_emplacements_fautifs():
    """Sans cela, le message ne vaut pas mieux que celui qu'il remplace."""
    with pytest.raises(RuntimeError) as echec:
        _verifier_le_placement_sur_gpu(_modele("cuda", "cpu", "meta"))
    assert "cpu, meta" in str(echec.value)


def test_liberer_la_memoire_ne_depend_pas_de_la_presence_d_un_gpu():
    """Le même code tourne sur la machine d'entraînement et dans l'intégration continue."""
    free_gpu_memory()
