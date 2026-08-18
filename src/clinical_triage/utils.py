"""Fonctions utilitaires transverses : graines, revision git et journalisation.

On isole ici ce qui est partage par toutes les etapes afin d'eviter les
duplications et de garantir une reproductibilite homogene.
"""

from __future__ import annotations

import gc
import logging
import random
import subprocess  # nosec B404
from pathlib import Path

from clinical_triage.config import PATHS, SEED


def set_seed(seed: int = SEED, include_torch: bool = True) -> None:
    """Fixe la graine de `random`, `numpy` et (optionnellement) `torch`.

    On importe numpy/torch paresseusement pour que ce module reste utilisable
    sans dependance lourde. `include_torch=False` est utile pour les etapes de
    preparation des donnees : sous Windows, importer torch avant pyarrow peut
    provoquer un conflit de DLL ; on garde donc le pipeline de donnees sans torch.

    Poser ici `PYTHONHASHSEED` n'apporterait rien : CPython ne lit cette variable
    qu'au demarrage du processus, la graine serait donc annoncee sans etre fixee.
    Le seul endroit du pipeline dont le resultat depend de l'ordre d'un ensemble
    se rend deterministe par lui-meme (`triage_rules._compile`), ce qui vaut mieux
    qu'une variable d'environnement a ne pas oublier.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy toujours present en pratique
        pass
    if not include_torch:
        return
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def liberer_la_memoire_gpu() -> None:
    """Rend au GPU la mémoire d'un modèle dont on n'a plus besoin.

    `del modele` ne suffit pas, pour deux raisons qui se cumulent : les modules
    PyTorch forment des cycles de références que seul le ramasse-miettes casse,
    et l'allocateur CUDA conserve ensuite les blocs libérés dans son cache. Un
    script qui enchaîne plusieurs modèles dans le même processus finit donc par
    saturer la carte, et la bibliothèque de chargement bascule silencieusement
    une partie des couches sur le processeur — l'erreur ne remonte qu'à la
    première génération, loin de sa cause.
    """
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - torch toujours present a l'entrainement
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_logger(name: str = "clinical_triage") -> logging.Logger:
    """Renvoie un logger configure simplement (format horodate, niveau INFO)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def chemin_pour_journal(chemin: str | Path) -> str:
    """Rend un chemin tel qu'il doit apparaître dans une trace du projet.

    Les journaux des modules finissent dans les sorties des carnets, qui sont
    versionnées et livrées : un chemin absolu y inscrirait l'arborescence de la
    machine qui a exécuté le code, et le lecteur n'en tire rien. Ce qui est sous
    la racine du dépôt est donc écrit relativement à elle, avec des barres
    obliques quelle que soit la plateforme. Le reste est rendu tel quel — un
    identifiant de dépôt Hugging Face, un modèle monté ailleurs dans un
    conteneur : le raccourcir le rendrait faux.
    """
    texte = str(chemin)
    candidat = Path(texte)
    if not candidat.is_absolute():
        return texte
    try:
        return candidat.relative_to(PATHS.root).as_posix()
    except ValueError:
        return texte


def revision_git() -> str:
    """Revision courte du depot, ou « inconnue » hors depot git.

    La carte du dataset et le manifeste des figures inscrivent tous deux cette
    revision : c'est ce qui permet de retrouver le code exact qui a produit les
    chiffres livres. Les deux scripts la lisent ici, et nulle part ailleurs :
    deux copies, sous deux noms et avec deux valeurs de repli, laisseraient les
    deux tracabilites diverger sans que cela se voie.
    """
    try:
        # Liste d'arguments constante, sans shell ni entree exterieure. Les
        # trois regles de bandit sur `subprocess` sont ecartees en ligne, ici
        # pour l'appel et a l'import du module pour B404, et nulle part ailleurs
        # dans le paquet. `git` est cherche dans le PATH a dessein, son
        # emplacement variant d'une machine a l'autre.
        sortie = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PATHS.root,
            capture_output=True,
            text=True,
            check=True,
        )
        return sortie.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "inconnue"
