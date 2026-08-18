"""Point d'entrée en ligne de commande : état du projet en un coup d'œil.

`uv run clinical-triage` vérifie que le paquet est correctement installé et affiche
ce qui est disponible localement : configuration, dataset, modèles entraînés.
C'est la première commande à lancer sur un poste neuf pour savoir où l'on en est.
"""

from __future__ import annotations

import json

from clinical_triage import __version__
from clinical_triage.config import DATA, MODEL, PATHS, TRAINING, TRIAGE


def _etat(chemin) -> str:
    """Dit si un élément du pipeline est présent sur le disque."""
    return "présent" if chemin.exists() else "absent"


def _lora() -> str:
    """Décrit la configuration LoRA, celle de l'adaptateur s'il existe.

    Les valeurs de `config.py` ne sont qu'un point de départ : le fine-tuning
    suit la variante retenue par le réglage des hyperparamètres. Annoncer ici la
    configuration par défaut alors qu'un adaptateur est sur le disque
    décrirait quelque chose qui n'a pas été entraîné.
    """
    defaut = f"r={TRAINING.lora_r}, alpha={TRAINING.lora_alpha}, lr={TRAINING.sft_lr} (défaut)"
    configuration = PATHS.sft_adapter / "adapter_config.json"
    if not configuration.exists():
        return defaut
    adaptateur = json.loads(configuration.read_text(encoding="utf-8"))
    return (
        f"r={adaptateur['r']}, alpha={adaptateur['lora_alpha']} "
        f"(adaptateur supervisé sur le disque)"
    )


def main() -> None:
    """Affiche un résumé de la configuration et de l'état du pipeline."""
    lignes = [
        ("Version du projet", __version__),
        ("Modèle de base", MODEL.base_model),
        ("Niveaux de triage", " · ".join(TRIAGE.levels)),
        (
            "Cible du jeu supervisé",
            f"{DATA.target_sft_pairs} paires, {DATA.french_share:.0%} en français",
        ),
        ("Cible du jeu de préférences", f"{DATA.target_dpo_pairs} paires"),
        ("LoRA", _lora()),
        ("Racine du projet", str(PATHS.root)),
    ]
    largeur = max(len(intitule) for intitule, _ in lignes)
    print(f"Agent IA de triage médical (POC) v{__version__}\n")
    for intitule, valeur in lignes:
        print(f"  {intitule.ljust(largeur)} : {valeur}")

    print("\n  État du pipeline")
    metadonnees = PATHS.data_processed / "metadata.json"
    etapes = [
        ("Dataset", metadonnees),
        ("Adaptateur SFT", PATHS.sft_adapter),
        ("Modèle SFT fusionné", PATHS.sft_merged),
        ("Adaptateur DPO", PATHS.dpo_adapter),
        ("Modèle final fusionné", PATHS.dpo_merged),
    ]
    largeur_etapes = max(len(intitule) for intitule, _ in etapes)
    for intitule, chemin in etapes:
        print(f"    {intitule.ljust(largeur_etapes)} : {_etat(chemin)}")

    if metadonnees.exists():
        statistiques = json.loads(metadonnees.read_text(encoding="utf-8"))["statistiques"]
        print(
            f"\n  Dataset : {statistiques['sft_total']} exemples supervisés, "
            f"{statistiques['dpo_total']} paires de préférence, "
            f"{statistiques['evaluation_clinique']} cas d'évaluation clinique."
        )


if __name__ == "__main__":
    main()
