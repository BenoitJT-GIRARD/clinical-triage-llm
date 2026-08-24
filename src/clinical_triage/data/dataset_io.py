"""Sérialisation, découpage et documentation du dataset.

Le format retenu est celui attendu par les bibliothèques d'entraînement, et il
est le même du SFT au service :

- jeu supervisé : `prompt` (l'échange ChatML jusqu'à l'ouverture du tour
  assistant) et `completion` (la réponse attendue). Ce couple permet à
  l'entraînement de ne calculer la perte que sur la réponse, sans collateur
  particulier ;
- jeu de préférences : `prompt`, `chosen`, `rejected`, avec exactement la même
  invite que ci-dessus.

Les deux jeux partagent donc mot pour mot le format utilisé à l'inférence. Les
métadonnées cliniques voyagent dans des colonnes supplémentaires, que
l'entraînement ignore mais qui rendent le dataset auditable.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from clinical_triage.data.dpo_builder import PreferencePairRecord
from clinical_triage.data.sft_builder import TriageExample, empreinte_de_cas
from clinical_triage.prompts import SYSTEM_PROMPT, build_messages, format_chatml
from clinical_triage.utils import chemin_pour_journal, get_logger

logger = get_logger(__name__)


def _prompt_for(user_turn: str) -> str:
    """Invite ChatML complète, ouvrant le tour assistant."""
    return format_chatml(build_messages(user_turn), add_generation_prompt=True)


def sft_record(example: TriageExample) -> dict:
    """Sérialise un exemple de triage pour l'entraînement supervisé."""
    return {
        "prompt": _prompt_for(example.user_turn),
        "completion": example.assistant_turn,
        "user_turn": example.user_turn,
        "level": example.level,
        "lang": example.lang,
        "source": example.source,
        "confiance": example.confiance,
        "symptomes": list(example.symptomes),
        "antecedents": list(example.antecedents),
        "constantes": example.constantes,
        "presentation_id": example.presentation_id,
    }


def dpo_record(pair: PreferencePairRecord) -> dict:
    """Sérialise une paire de préférence pour l'alignement DPO."""
    return {
        "prompt": _prompt_for(pair.user_turn),
        "chosen": pair.chosen,
        "rejected": pair.rejected,
        "user_turn": pair.user_turn,
        "level": pair.level,
        "lang": pair.lang,
        "strategie": pair.strategie,
        "source": pair.source,
    }


def eval_record(case) -> dict:
    """Sérialise un cas du jeu d'évaluation clinique."""
    return {
        "prompt": _prompt_for(case.user_turn),
        "completion": "",
        "user_turn": case.user_turn,
        "id": case.id,
        "level": case.level,
        "lang": case.lang,
        "piege": case.piege,
        "description": case.description,
        "note_clinique": case.note,
    }


# Les six fichiers que produit la préparation. La liste vit ici parce que deux
# endroits en dépendent et doivent rester d'accord : le script qui les écrit, et
# la publication qui refuse de partir s'il en manque un.
FICHIERS_DU_DATASET = (
    "sft_train.jsonl",
    "sft_validation.jsonl",
    "sft_test.jsonl",
    "dpo_train.jsonl",
    "clinical_eval.jsonl",
    "metadata.json",
)


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Écrit une liste d'enregistrements en JSONL UTF-8, un objet par ligne."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Écrit %d lignes → %s", len(rows), chemin_pour_journal(path))


def read_jsonl(path: Path) -> list[dict]:
    """Relit un fichier JSONL."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_train_val_test(
    examples: list[TriageExample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[TriageExample]]:
    """Découpe en train / validation / test.

    Les exemples ont déjà été dédupliqués sur le tour utilisateur complet : un
    même énoncé ne peut donc pas se retrouver des deux côtés du découpage.
    `check_no_leakage` le vérifie explicitement après coup.
    """
    rng = random.Random(seed)
    melanges = list(examples)
    rng.shuffle(melanges)
    n = len(melanges)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    return {
        "test": melanges[:n_test],
        "validation": melanges[n_test : n_test + n_val],
        "train": melanges[n_test + n_val :],
    }


def check_no_leakage(splits: dict[str, list[TriageExample]]) -> dict[str, int]:
    """Compte les tours utilisateur partagés entre deux découpages.

    Toutes les valeurs doivent être nulles. Cette vérification est appelée par le
    script de préparation, qui échoue si elle ne l'est pas : le cahier des charges
    interdit de mélanger données d'entraînement et données d'évaluation, et une
    affirmation de ce type doit être contrôlée par le code, pas par la
    documentation.

    La comparaison porte sur la forme normalisée : un contrôle qui n'attrape que
    les doublons exacts donne une assurance qu'il ne mérite pas.
    """
    ensembles = {
        nom: {empreinte_de_cas(e.user_turn) for e in items} for nom, items in splits.items()
    }
    noms = sorted(ensembles)
    chevauchements: dict[str, int] = {}
    for i, premier in enumerate(noms):
        for second in noms[i + 1 :]:
            chevauchements[f"{premier}∩{second}"] = len(ensembles[premier] & ensembles[second])
    return chevauchements


# --- Carte du dataset ---

# Licence et provenance de chaque corpus, vérifiées sur la fiche du dépôt.
SOURCES_DOCUMENTEES = {
    "vignette_clinique": {
        "origine": "Catalogue de présentations cliniques rédigé pour ce projet (src/clinical_triage/data/clinical_catalogue.py)",
        "langue": "fr + en",
        "licence": "MIT",
        "role": "Vérité terrain du triage : le niveau vient de la présentation, pas du texte.",
    },
    "mediqal": {
        "origine": "https://huggingface.co/datasets/ANR-MALADES/MediQAl",
        "langue": "fr",
        "licence": "CC BY 4.0",
        "role": "Corpus du cahier des charges, et le seul à décrire des patients. Socle des vignettes cliniques françaises authentiques.",
    },
    "medquad": {
        "origine": "https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset",
        "langue": "en",
        # Le miroir Hub ne déclare aucune licence. Celle-ci est lue sur le dépôt
        # d'origine des auteurs, github.com/abachaa/MedQuAD, dont le `LICENSE.txt`
        # est le texte CC BY 4.0 et dont le `readme.txt` le répète en toutes
        # lettres. Reprendre une licence sans la vérifier serait l'inventer, d'où
        # le champ qui dit où elle a été lue.
        "licence": "CC BY 4.0",
        "licence_verifiee_sur": "dépôt d'origine abachaa/MedQuAD",
        "role": "Corpus du cahier des charges. Descriptions de symptômes, étiquetées par la règle (confiance moyenne).",
    },
    "medmcqa": {
        "origine": "https://huggingface.co/datasets/openlifescienceai/medmcqa",
        "langue": "en",
        "licence": "Apache-2.0",
        "role": "Hors cahier des charges, ajouté faute de vignettes cliniques anglophones dans les corpus imposés.",
    },
    "frenchmedmcqa": {
        "origine": "https://huggingface.co/datasets/nthngdy/frenchmedmcqa",
        "langue": "fr",
        # Même remarque : le miroir Parquet ne déclare rien, le dépôt des auteurs
        # si. On passe par le miroir parce que le dépôt d'origine n'expose ses
        # données que par un script de chargement, que `datasets` n'exécute plus.
        "licence": "Apache-2.0",
        "licence_verifiee_sur": "dépôt d'origine qanastek/frenchmedmcqa",
        "role": "Corpus du cahier des charges. Questions de pharmacie, dont aucune n'est étiquetable en triage.",
    },
    "ultramedical_preference": {
        "origine": "https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference",
        "langue": "en",
        "licence": "MIT",
        "role": "Jeu de préférences externe, tenu à l'écart de l'entraînement : mesure indépendante de l'alignement.",
    },
}


def metadata_schema() -> dict:
    """Schéma des métadonnées du dataset, champ par champ."""
    return {
        "champs_sft": {
            "prompt": "Invite ChatML complète (consigne système + tour patient), ouvrant le tour assistant.",
            "completion": "Réponse de triage attendue : niveau, justification, recommandation.",
            "user_turn": "Tour patient seul, utile pour la déduplication et l'audit.",
            "level": "Niveau de triage : URGENCE_VITALE | URGENCE_MODEREE | CONSULTATION_DIFFEREE.",
            "lang": "Langue de la description du patient (fr | en).",
            "source": "Origine de l'exemple (vignette_clinique, mediqal, medquad, frenchmedmcqa, medmcqa).",
            "confiance": "Niveau de confiance de l'étiquette : haute (catalogue clinique) | moyenne (règle appliquée à un corpus).",
            "symptomes": "Signes cliniques présents dans la description.",
            "antecedents": "Antécédents du patient mentionnés dans la description.",
            "constantes": "Relevé de constantes vitales, vide si le triage se fait sans mesure.",
            "presentation_id": "Identifiant de la présentation type du catalogue, vide pour les cas issus des corpus.",
        },
        # Les trois listes donnent les colonnes **exactes** de chaque fichier, et
        # pas seulement ce que l'un ajoute à l'autre : le jeu de préférences n'a
        # ni `confiance` ni `constantes`, et le jeu d'évaluation expose une
        # colonne `description` que rien ne documentait. Un consommateur du
        # dataset publié qui filtre sur une colonne absente n'obtient rien, sans
        # comprendre pourquoi.
        "champs_dpo": {
            "prompt": "Invite ChatML identique à celle du jeu supervisé.",
            "chosen": "Réponse préférée : bon niveau, format respecté, conduite à tenir sûre.",
            "rejected": "Réponse rejetée, de même format et de longueur comparable.",
            "user_turn": "Tour patient seul, utile pour la déduplication et l'audit.",
            "level": "Niveau de triage de référence du cas.",
            "lang": "Langue de la description du patient.",
            "strategie": "Nature du défaut introduit : sous_triage | recommandation_dangereuse | diagnostic_affirme | reponse_en_anglais.",
            "source": "preference_securite.",
        },
        "champs_evaluation": {
            "prompt": "Invite ChatML identique à celle du jeu supervisé.",
            "completion": "Toujours vide : la réponse attendue n'est pas donnée, seul le niveau l'est.",
            "user_turn": "Tour patient seul, tel qu'il est soumis au modèle.",
            "id": "Identifiant du cas d'évaluation.",
            "level": "Niveau de triage de référence, écrit à la main.",
            "lang": "Langue de la description du patient.",
            "piege": "Nature de la difficulté : vide (présentation directe), faux_rassurant, faux_alarmant, negation, constantes_discordantes.",
            "description": "Description du patient seule, sans la consigne qui l'encadre.",
            "note_clinique": "Raison clinique de l'étiquette, pour l'auditabilité et l'analyse d'erreurs.",
        },
        "taxonomie_triage": {
            "URGENCE_VITALE": "Prise en charge immédiate (tris 1 et 2 de l'échelle FRENCH).",
            "URGENCE_MODEREE": "Prise en charge sous quelques heures (tris 3 et 4).",
            "CONSULTATION_DIFFEREE": "Pas de critère de gravité immédiat (tri 5).",
        },
        "consigne_systeme": SYSTEM_PROMPT,
        "sources": SOURCES_DOCUMENTEES,
    }


def write_metadata(meta: dict, path: Path) -> None:
    """Enregistre la carte du dataset en JSON indenté."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    logger.info("Métadonnées écrites → %s", chemin_pour_journal(path))
