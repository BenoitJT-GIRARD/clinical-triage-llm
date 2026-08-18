"""Agent IA de triage médical du service des urgences.

Le paquet regroupe toute la logique réutilisable du POC : configuration,
préparation des données, entraînement (SFT puis DPO), évaluation clinique et
service d'inférence. Les scripts de `scripts/` orchestrent ces briques dans
l'ordre du pipeline, et les notebooks de `notebooks/` les commentent.
"""

from __future__ import annotations

import linecache
import os
import warnings

__version__ = "1.0.0"

# MLflow écrit à son premier import un message d'accueil qui cite ses chemins
# d'installation. Il n'a rien à faire dans les sorties d'un carnet, qui sont
# versionnées et livrées. La variable est posée ici, à l'import du paquet : dans
# un carnet, MLflow arrive en cascade derrière Unsloth, bien avant que le module
# de suivi ne soit sollicité.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# Un avertissement de bibliothèque s'affiche précédé du fichier qui l'émet,
# chemin d'installation compris : les sorties de carnet, versionnées et livrées,
# portent alors l'arborescence du poste qui les a produites. Le préfixe jusqu'au
# répertoire d'installation est retiré à l'affichage, et rien d'autre : la
# catégorie, le texte et la ligne de code en cause restent tels quels, et aucun
# avertissement n'est supprimé. Le formateur est remplacé pour tout le processus,
# ce qui est le seul niveau où un carnet en profite sans avoir à s'en occuper.
_MARQUEUR_D_INSTALLATION = "/site-packages/"
_format_d_origine = warnings.formatwarning


def _avertissement_sans_chemin_d_installation(message, category, filename, lineno, line=None):
    """Formate un avertissement en nommant le module, sans son chemin d'installation."""
    emplacement = filename.replace("\\", "/")
    if _MARQUEUR_D_INSTALLATION in emplacement:
        # La ligne en cause est lue tant que le chemin complet est connu : le
        # formateur ne saurait plus la retrouver à partir du chemin raccourci.
        line = line or linecache.getline(filename, lineno)
        emplacement = emplacement.split(_MARQUEUR_D_INSTALLATION)[-1]
    return _format_d_origine(message, category, emplacement, lineno, line)


warnings.formatwarning = _avertissement_sans_chemin_d_installation
