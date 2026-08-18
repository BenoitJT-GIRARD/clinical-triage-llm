"""Réglages à appliquer au tout début d'un script, avant les imports lourds.

Deux contraintes de la machine de développement, réunies ici pour que les
scripts n'aient pas à les répéter :

- **ordre des imports.** Sous Windows, importer `torch` avant `pyarrow` (que
  `datasets` charge) provoque un conflit de bibliothèques dynamiques qui termine
  le processus par une faute de segmentation, sans message. Importer ce module
  en premier charge `datasets` dans le bon ordre.
- **encodage de la console.** La sortie standard de Windows est en cp1252 ;
  toute trace contenant un caractère accentué ou un symbole interrompt le script.

Un script d'entraînement commence donc par :

    from clinical_triage.bootstrap import use_utf8_console

    use_utf8_console()
"""

from __future__ import annotations

import sys

import datasets  # noqa: F401  (import volontaire : il doit précéder celui de torch)


def use_utf8_console() -> None:
    """Passe la sortie standard en UTF-8, en remplaçant ce qui n'est pas encodable."""
    for flux in (sys.stdout, sys.stderr):
        if hasattr(flux, "reconfigure"):
            flux.reconfigure(encoding="utf-8", errors="replace")
