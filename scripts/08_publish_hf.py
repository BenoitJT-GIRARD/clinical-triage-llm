"""Publication du dataset et des modèles sur le Hugging Face Hub.

Le dépôt git porte déjà le dataset et sa carte ; le Hub apporte la consultation
en ligne, l'aperçu des données et le téléchargement des poids, trop volumineux
pour un dépôt git.

Nécessite un jeton d'écriture dans `HF_TOKEN`. Le compte cible se règle par
`HF_NAMESPACE`, sans modifier le code :

    $env:HF_TOKEN = "hf_..."
    uv run python scripts/08_publish_hf.py --what dataset
    uv run python scripts/08_publish_hf.py --what modele-final

`--etiquette` pose en plus une étiquette sur les dépôts touchés. C'est ce qui
rend une version de modèle citable : un serveur d'inférence peut alors demander
`revision=modele-v1.0.0` et obtenir toujours les mêmes poids, là où `main` bouge
à chaque publication. Le déploiement continu s'en sert pour épingler la version
qu'il met en ligne.

    uv run python scripts/08_publish_hf.py --what cartes --etiquette modele-v1.0.0

Les dossiers de modèle contiennent des points de reprise intermédiaires, utiles
localement mais sans valeur sur le Hub : ils sont exclus de la publication.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path

from clinical_triage.config import MODEL, PATHS
from clinical_triage.data.dataset_io import FICHIERS_DU_DATASET
from clinical_triage.utils import get_logger

logger = get_logger("publication")

# Fichiers exclus de toute publication : points de reprise intermédiaires et
# états d'optimiseur, qui pèsent lourd et ne servent qu'à reprendre un
# entraînement interrompu.
EXCLUSIONS = [
    "checkpoint-*/*",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    # La bibliothèque d'entraînement dépose sa propre carte dans le dossier de
    # l'adaptateur, et y inscrit le **chemin local** du modèle de base. Le Hub
    # refuse l'envoi entier pour cette seule ligne, et la carte du projet arrive
    # de toute façon juste après, sous le même nom.
    "README.md",
]

# Les quatre dépôts de modèles, avec le dossier local qui les alimente.
MODELES = {
    "adaptateur-sft": "sft_adapter",
    "modele-sft-fusionne": "sft_merged",
    "adaptateur-dpo": "dpo_adapter",
    "modele-final": "dpo_merged",
}

# Ce que `--what tout` publie : les quatre modèles et le dataset.
#
# Le modèle supervisé fusionné en fait partie, alors qu'il pèse quatre gigaoctets
# et que personne ne le sert. Il y a une raison, et elle n'est pas négociable :
# **l'adaptateur DPO est entraîné par-dessus lui**, et sa configuration le
# désigne comme modèle de base. Ne pas le publier ferait de l'adaptateur DPO un
# artefact inchargeable — `PeftModel.from_pretrained` irait chercher un dépôt
# inexistant. Quatre gigaoctets de stockage valent mieux qu'un artefact public
# que personne ne peut ouvrir.
TOUT = [
    "dataset",
    "adaptateur-sft",
    "modele-sft-fusionne",
    "adaptateur-dpo",
    "modele-final",
]


def _jeton() -> str:
    jeton = os.getenv("HF_TOKEN")
    if not jeton:
        raise SystemExit(
            "HF_TOKEN manquant. Créez un jeton d'écriture sur "
            "https://huggingface.co/settings/tokens puis exportez-le."
        )
    return jeton


def _depot_du_modele(quoi: str) -> str:
    identifiants = {
        "adaptateur-sft": MODEL.hub_sft_model_id,
        "modele-sft-fusionne": MODEL.hub_sft_merged_model_id,
        "adaptateur-dpo": MODEL.hub_dpo_model_id,
        "modele-final": MODEL.hub_merged_model_id,
    }
    return identifiants[quoi]


def _carte_du_modele(quoi: str) -> Path:
    return PATHS.reports / "cartes_modeles" / f"{quoi}.md"


def publier_dataset(api) -> str:
    """Publie les fichiers JSONL, la carte de données et les métadonnées."""
    # Les quatre jeux d'entraînement ne sont pas versionnés : ils se
    # reconstruisent par `scripts/01`. Sans ce refus, une publication lancée
    # depuis un dépôt fraîchement cloné téléverserait deux fichiers sur six et
    # l'annoncerait comme un succès.
    absents = [n for n in FICHIERS_DU_DATASET if not (PATHS.data_processed / n).exists()]
    if absents:
        raise SystemExit(
            "Dataset incomplet, publication refusée. Manquent : "
            + ", ".join(absents)
            + ".\nLancez `python scripts/01_build_dataset.py` sur la machine qui publie."
        )
    depot = MODEL.hub_dataset_id
    # `private=False` est explicite à dessein : le cahier des charges demande un
    # jeu public, et publier des données ne se défait pas d'un clic. On ne laisse
    # pas ce choix-là à une valeur par défaut de bibliothèque.
    api.create_repo(depot, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(PATHS.data_processed),
        repo_id=depot,
        repo_type="dataset",
        allow_patterns=["*.jsonl", "*.json"],
    )
    api.upload_file(
        path_or_fileobj=str(PATHS.data / "README.md"),
        path_in_repo="README.md",
        repo_id=depot,
        repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{depot}"


def publier_modele(api, quoi: str, etiquette: str | None = None) -> str:
    """Publie un adaptateur LoRA ou le modèle final fusionné, avec sa carte."""
    dossier = getattr(PATHS, MODELES[quoi])
    depot = _depot_du_modele(quoi)
    if not dossier.exists():
        raise SystemExit(f"Dossier introuvable : {dossier}")
    api.create_repo(depot, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(dossier),
        repo_id=depot,
        repo_type="model",
        ignore_patterns=EXCLUSIONS,
    )
    publier_carte(api, quoi, etiquette)
    return f"https://huggingface.co/{depot}"


def _pourcentage(valeur: float) -> str:
    """Pourcentage à la française, sans décimale inutile."""
    return f"{valeur * 100:.1f}".replace(".", ",").removesuffix(",0") + " %"


# Quel modèle évalué correspond à quelle carte. L'adaptateur supervisé et le
# modèle supervisé fusionné portent le même apprentissage : ils partagent leurs
# chiffres, comme ils partagent leurs poids à l'arrondi près.
EVALUATION_DE_LA_CARTE = {
    "adaptateur-sft": "sft",
    "modele-sft-fusionne": "sft",
    "adaptateur-dpo": "dpo",
    "modele-final": "dpo-fusionne",
}


def _tableau_d_evaluation(quoi: str) -> str:
    """Rend les chiffres du modèle, tels qu'ils seront lus sur le Hub.

    Une carte de modèle sans performance mesurable n'est pas une carte de
    modèle : c'est la première page que voit quiconque ouvre le dépôt, et un
    renvoi vers un rapport hébergé ailleurs n'en tient pas lieu.
    """
    resultats = PATHS.reports / "evaluation_results.json"
    if not resultats.exists():
        raise SystemExit(
            f"Évaluation introuvable ({resultats}). Lancez `python scripts/06_evaluate.py` "
            "avant de publier : une carte de modèle sans chiffres n'a pas d'objet."
        )
    jeu = json.loads(resultats.read_text(encoding="utf-8"))["jeu_clinique"]
    nom = EVALUATION_DE_LA_CARTE[quoi]
    mesures = jeu["modeles"].get(nom)
    if mesures is None:
        raise SystemExit(f"Le modèle `{nom}` n'a pas été évalué : carte `{quoi}` non publiable.")

    basse, haute = mesures["exactitude_ic95"]
    pour_cent = lambda valeur: f"{valeur * 100:.1f} %".replace(".", ",").replace(",0 %", " %")
    return "\n".join(
        [
            f"| Mesure | Valeur sur {jeu['n']} cas |",
            "|---|---|",
            f"| Exactitude du niveau de triage | {mesures['exactitude']:.3f}".replace(".", ",")
            + f" [{basse:.2f} – {haute:.2f}]".replace(".", ",")
            + " |",
            f"| **Sous-triage des cas urgents** | **{pour_cent(mesures['sous_triage'])}** |",
            f"| Surclassement, tous cas | {pour_cent(mesures['surclassement'])} |",
            f"| Réponses exploitables par le système d'information | {pour_cent(mesures['respect_format'])} |",
        ]
    )


def publier_carte(api, quoi: str, etiquette: str | None = None) -> str:
    """Met à jour la seule carte d'un modèle, sans retoucher aux poids.

    C'est ce dont l'intégration continue a besoin : les cartes sont versionnées
    dans le dépôt git et évoluent avec la documentation, alors que les poids
    sont téléversés depuis la machine d'entraînement. Les chiffres, eux, sont
    injectés ici depuis le fichier de résultats : la carte publiée porte donc les
    mesures de la version qu'elle décrit, sans recopie à la main.

    La commande de service qu'elle donne épingle l'étiquette posée par la même
    exécution. Écrite en dur, elle désignerait la première publication après
    chaque nouvelle : la carte décrirait des poids et en ferait servir d'autres.
    """
    depot = _depot_du_modele(quoi)
    carte = _carte_du_modele(quoi)
    if not carte.exists():
        raise SystemExit(f"Carte introuvable : {carte}")
    texte = carte.read_text(encoding="utf-8").replace("{{EVALUATION}}", _tableau_d_evaluation(quoi))
    texte = texte.replace("{{REVISION}}", etiquette or "main")
    api.create_repo(depot, repo_type="model", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=texte.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=depot,
        repo_type="model",
    )
    return f"https://huggingface.co/{depot}"


def etiqueter(api, depot: str, type_depot: str, nom: str) -> None:
    """Pose une étiquette sur un dépôt du Hub, en remplaçant la précédente.

    Réétiqueter est utile quand une publication est reprise après correction :
    `create_tag` sait ignorer une étiquette existante, mais pas la déplacer sur
    le nouveau contenu. On la retire donc d'abord, si elle est là.
    """
    from huggingface_hub.errors import RevisionNotFoundError

    with contextlib.suppress(RevisionNotFoundError):
        api.delete_tag(depot, tag=nom, repo_type=type_depot)
    api.create_tag(depot, tag=nom, repo_type=type_depot)
    logger.info("Étiquette %s posée sur %s", nom, depot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--what",
        choices=["dataset", *MODELES, "cartes", "tout"],
        required=True,
        help="dataset, un modèle précis, « cartes » pour les seules cartes, ou « tout »",
    )
    parser.add_argument(
        "--etiquette",
        help="nom d'étiquette à poser sur les dépôts touchés, par exemple modele-v1.0.0",
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi(token=_jeton())

    if args.what == "tout":
        demandes = list(TOUT)
    elif args.what == "cartes":
        # Seulement les modèles réellement publiés : rafraîchir la carte d'un
        # dépôt jamais créé le créerait, vide, avec un README pour tout contenu.
        demandes = [nom for nom in TOUT if nom != "dataset"]
    else:
        demandes = [args.what]

    seulement_les_cartes = args.what == "cartes"
    touches: list[tuple[str, str]] = []
    for demande in demandes:
        if demande == "dataset":
            adresse = publier_dataset(api)
            depot, type_depot = MODEL.hub_dataset_id, "dataset"
        elif seulement_les_cartes:
            adresse = publier_carte(api, demande, args.etiquette)
            depot, type_depot = _depot_du_modele(demande), "model"
        else:
            adresse = publier_modele(api, demande, args.etiquette)
            depot, type_depot = _depot_du_modele(demande), "model"
        touches.append((depot, type_depot))
        logger.info("Publié : %s", adresse)

    # Avec `--what cartes`, le dataset n'est pas republié — l'exécuteur
    # d'intégration continue n'en a pas les fichiers — mais son dépôt doit
    # recevoir l'étiquette comme les autres : c'est elle qui rend la version
    # citable, et une révision où le dataset n'est pas étiqueté est incomplète.
    if seulement_les_cartes and api.repo_exists(MODEL.hub_dataset_id, repo_type="dataset"):
        touches.append((MODEL.hub_dataset_id, "dataset"))

    if args.etiquette:
        for depot, type_depot in touches:
            etiqueter(api, depot, type_depot, args.etiquette)


if __name__ == "__main__":
    main()
