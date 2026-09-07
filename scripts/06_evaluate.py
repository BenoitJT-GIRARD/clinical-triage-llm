"""Évaluation clinique comparée : références, modèle de base, SFT et DPO.

L'évaluation porte sur deux jeux, et c'est volontaire :

- le **jeu clinique indépendant**, écrit à la main, qui n'a jamais servi à
  l'entraînement et dont les étiquettes ne viennent pas de la règle. C'est le
  seul jeu sur lequel les chiffres publiés ont un sens ;
- le **jeu de test interne**, issu du même générateur que l'entraînement. Il
  mesure la capacité du modèle à reproduire ce qu'on lui a montré. L'écart entre
  les deux est précisément ce qu'il faut regarder : un modèle excellent sur le
  second et médiocre sur le premier a appris des gabarits, pas du triage.

Quatre références encadrent les résultats : toujours la classe majoritaire,
toujours l'urgence vitale, la règle explicite, et un classifieur classique
entraîné sur les mêmes paires que le modèle. Les deux dernières sont celles à
battre, et la quatrième est la plus exigeante : elle dit ce que le fine-tuning
apporte par-dessus un apprentissage ordinaire sur les mêmes données.

Usage :
    uv run python scripts/06_evaluate.py
    uv run python scripts/06_evaluate.py --models sft dpo --interne 0
"""

from __future__ import annotations

from clinical_triage.bootstrap import use_utf8_console

use_utf8_console()

import argparse
import json
import random
from pathlib import Path

from clinical_triage.config import MODEL, PATHS, SEED, TRIAGE
from clinical_triage.data.corpus_sources import load_ultramedical_preferences
from clinical_triage.data.dataset_io import read_jsonl
from clinical_triage.evaluation import preference, robustness
from clinical_triage.evaluation.baselines import (
    always_critical,
    classical_classifier,
    explicit_rule,
    majority_class,
)
from clinical_triage.evaluation.metrics import mcnemar_exact
from clinical_triage.evaluation.runner import (
    error_table,
    evaluate_agent,
    evaluate_predictions,
)
from clinical_triage.inference import TriageAgent
from clinical_triage.utils import get_logger, liberer_la_memoire_gpu, set_seed

logger = get_logger("evaluation")


# Les modèles évaluables, dans l'ordre de la progression. `argparse` refuse un
# nom absent de cette liste : sans ce contrôle, une faute de frappe échouerait sur
# un `KeyError` une fois les autres modèles évalués, et le fichier de résultats
# ne serait jamais écrit.
MODELES_EVALUABLES = ("base", "sft", "dpo", "dpo-fusionne")


def _specification(nom: str) -> tuple[str, str | None]:
    """Renvoie (modèle de base, adaptateur) pour chaque modèle à évaluer.

    Le DPO ayant été entraîné au-dessus du modèle SFT fusionné, il doit être
    évalué avec ce modèle comme base : appliquer l'adaptateur DPO sur le modèle
    de base produirait une composition de poids incohérente.

    `dpo-fusionne` désigne les **mêmes poids** que `dpo`, mais fusionnés en un
    seul modèle au lieu d'être appliqués à chaud. Les deux sont évalués côte à
    côte parce que ce sont deux façons de livrer le modèle, et que le choix entre
    elles se tranche sur des mesures, pas sur un principe.
    """
    return {
        "base": (MODEL.base_model, None),
        "sft": (MODEL.base_model, str(PATHS.sft_adapter)),
        "dpo": (str(PATHS.sft_merged), str(PATHS.dpo_adapter)),
        "dpo-fusionne": (str(PATHS.dpo_merged), None),
    }[nom]


def _poids_sur_disque(chemin: Path) -> float:
    """Taille des fichiers de poids d'un dossier de modèle, en mégaoctets."""
    if not chemin.exists():
        return 0.0
    motifs = ("*.safetensors", "*.bin")
    octets = sum(f.stat().st_size for motif in motifs for f in chemin.glob(motif))
    return round(octets / 1024**2, 1)


def _comparer_livraisons(resultats: dict) -> dict:
    """Compare la livraison par adaptateur et la livraison fusionnée.

    Les deux portent les mêmes poids à l'arrondi bf16 près. Ce qui les sépare est
    opérationnel : ce qu'il faut télécharger, ce que le moteur d'inférence sait
    charger, et ce que coûte l'application de l'adaptateur à chaque jeton.
    """
    modeles = resultats["jeu_clinique"]["modeles"]
    if "dpo" not in modeles or "dpo-fusionne" not in modeles:
        return {}

    adaptateur, fusionne = modeles["dpo"], modeles["dpo-fusionne"]
    predictions_a = resultats["_predictions"].get("dpo", [])
    predictions_f = resultats["_predictions"].get("dpo-fusionne", [])
    accord = (
        round(
            sum(a == f for a, f in zip(predictions_a, predictions_f, strict=True))
            / len(predictions_a),
            4,
        )
        if predictions_a and len(predictions_a) == len(predictions_f)
        else None
    )
    return {
        "accord_des_predictions": accord,
        "adaptateur": {
            "exactitude": adaptateur["exactitude"],
            "exactitude_ic95": adaptateur["exactitude_ic95"],
            "latence_mediane_ms": adaptateur.get("latence", {}).get("p50_ms"),
            "poids_a_telecharger_mo": _poids_sur_disque(PATHS.dpo_adapter),
            "modele_de_base_requis": str(PATHS.sft_merged.name),
        },
        "fusionne": {
            "exactitude": fusionne["exactitude"],
            "exactitude_ic95": fusionne["exactitude_ic95"],
            "latence_mediane_ms": fusionne.get("latence", {}).get("p50_ms"),
            "poids_a_telecharger_mo": _poids_sur_disque(PATHS.dpo_merged),
            "modele_de_base_requis": None,
        },
    }


def _cas_internes(nombre: int) -> list[dict]:
    """Échantillon aléatoire du jeu de test interne, au format du jeu clinique."""
    lignes = read_jsonl(PATHS.data_processed / "sft_test.jsonl")
    tirage = random.Random(SEED)
    tirage.shuffle(lignes)
    return [
        {
            "id": f"interne_{index}",
            "user_turn": ligne["user_turn"],
            "description": ligne["user_turn"],
            "level": ligne["level"],
            "lang": ligne["lang"],
            "piege": "",
            "note_clinique": "",
        }
        for index, ligne in enumerate(lignes[:nombre])
    ]


def _references(
    cas: list[dict],
    niveaux_entrainement: list[str],
    entrainement: list[dict],
    validation: list[dict],
    avec_regle: bool = True,
) -> dict:
    """Évalue les références sur un jeu de cas.

    `avec_regle` existe parce que la règle explicite n'est pas une référence
    légitime partout. La préparation des données s'en sert pour écarter du corpus
    les cas dont elle contredit l'étiquette : sur la part du jeu interne qui vient
    de ce corpus, elle retrouve l'étiquette à 100 % par construction. La comparer
    au modèle là-bas mesurerait ce filtre, pas la règle. Le jeu clinique, lui, est
    écrit à la main sans jamais passer par `classify` : la comparaison y est
    honnête, et c'est celle que le rapport publie.

    La référence classique, elle, s'applique partout : elle apprend des mêmes
    paires que le modèle, et aucune des lignes qu'elle juge n'était dans son
    entraînement. Sur le jeu interne, elle en a néanmoins vu des reformulations —
    exactement comme le modèle, et c'est ce qui rend la comparaison juste là-bas.
    """
    references = {
        "classe_majoritaire": evaluate_predictions(
            cas, majority_class(niveaux_entrainement, len(cas))
        ),
        "toujours_urgence_vitale": evaluate_predictions(cas, always_critical(len(cas))),
    }
    if avec_regle:
        references["regle_explicite"] = evaluate_predictions(
            cas, explicit_rule([c["description"] for c in cas])
        )
    predictions, trace = classical_classifier(
        [e["user_turn"] for e in entrainement],
        [e["level"] for e in entrainement],
        [e["user_turn"] for e in validation],
        [e["level"] for e in validation],
        [c["description"] for c in cas],
    )
    references["classifieur_classique"] = {
        **evaluate_predictions(cas, predictions),
        "selection": trace,
        # Conservées le temps de la comparaison appariée, puis retirées avant
        # écriture : les prédictions brutes n'ont pas leur place dans le fichier
        # de résultats.
        "_predictions": predictions,
    }
    return references


def _comparer_a_une_reference(
    cas: list[dict],
    predictions: list[str | None],
    nom_du_modele: str,
    reference: list[str | None],
    nom_de_la_reference: str,
) -> dict | None:
    """Compare le modèle à une référence sur les **mêmes** cas.

    L'affirmation centrale du rapport — « le modèle fait mieux que ce qu'il doit
    remplacer » — se tranche sur un test apparié, et non sur le recouvrement de deux
    intervalles de confiance : le non-recouvrement prouve une différence, le
    recouvrement ne prouve rien. Sur données **appariées**, ce recouvrement est en
    plus systématiquement trop prudent, parce que l'information est portée par les
    seuls cas où les deux systèmes divergent — deux intervalles quasi superposés
    sont parfaitement compatibles avec un écart réel.

    Les deux systèmes voient ici exactement les mêmes cas. Le test qui leur
    convient est celui de McNemar, exact, également employé pour départager le
    modèle supervisé et le modèle aligné.

    Deux références sont comparées de cette façon, et il faut les deux. La règle
    explicite dit ce qu'un service peut déployer en un après-midi. Le classifieur
    classique dit ce qu'apporte le fine-tuning par-dessus un apprentissage
    ordinaire sur les mêmes données : battre la règle sans battre le classifieur
    n'établit pas qu'il fallait un modèle de langage.
    """
    if not predictions:
        return None
    attendus = [c["level"] for c in cas]

    modele_seul = sum(
        p == a and r != a for p, r, a in zip(predictions, reference, attendus, strict=True)
    )
    reference_seule = sum(
        r == a and p != a for p, r, a in zip(predictions, reference, attendus, strict=True)
    )
    return {
        "modele": nom_du_modele,
        "reference": nom_de_la_reference,
        "exactitude": {
            "n": len(cas),
            "modele_seul": modele_seul,
            "reference_seule": reference_seule,
            "accords": len(cas) - modele_seul - reference_seule,
            "p_mcnemar": round(mcnemar_exact(modele_seul, reference_seule), 4),
        },
        # L'exactitude globale n'est pas la mesure qui décide. Un système peut se
        # tromper souvent sans danger — surclasser un rhume coûte une place
        # d'attente — et rarement avec danger. Le même test apparié, restreint
        # aux cas urgents et à la seule faute qui tue, dit ce que l'exactitude
        # dilue : c'est sur le sous-triage que se juge un outil de triage.
        "sous_triage": _comparer_le_sous_triage(cas, predictions, reference),
    }


def _comparer_le_sous_triage(
    cas: list[dict], predictions: list[str | None], reference: list[str | None]
) -> dict:
    """Compare modèle et référence sur la seule faute dangereuse, cas urgents seulement."""

    def sous_trie(attendu: str, predit: str | None) -> bool:
        rang = TRIAGE.severity.get(predit, -1) if predit is not None else -1
        return rang < TRIAGE.severity[attendu]

    urgents = [
        (c["level"], p, r)
        for c, p, r in zip(cas, predictions, reference, strict=True)
        if TRIAGE.severity[c["level"]] >= 1
    ]
    modele_seul = sum(not sous_trie(a, p) and sous_trie(a, r) for a, p, r in urgents)
    reference_seule = sum(sous_trie(a, p) and not sous_trie(a, r) for a, p, r in urgents)
    return {
        "cas_urgents": len(urgents),
        "sous_triages_modele": sum(sous_trie(a, p) for a, p, _ in urgents),
        "sous_triages_reference": sum(sous_trie(a, r) for a, _, r in urgents),
        "modele_seul": modele_seul,
        "reference_seule": reference_seule,
        "p_mcnemar": round(mcnemar_exact(modele_seul, reference_seule), 4),
    }


def _comparer_les_preferences(resultats: dict) -> dict | None:
    """Compare le modèle supervisé et le modèle aligné sur les **mêmes** paires.

    Les deux ont été mesurés sur le même jeu externe : ce qui les départage, ce
    sont les seules paires où l'un réussit et l'autre échoue. Comparer leurs deux
    proportions reviendrait à jeter cette information, et à déclarer un gain sur
    un écart de deux paires sur cent cinquante.
    """
    preferences = resultats.get("preferences_externes", {})
    supervise = preferences.get("sft", {}).get("bien_ordonnees")
    aligne = preferences.get("dpo", {}).get("bien_ordonnees")
    if not supervise or not aligne or len(supervise) != len(aligne):
        return None

    aligne_seul = sum(a and not s for s, a in zip(supervise, aligne, strict=True))
    supervise_seul = sum(s and not a for s, a in zip(supervise, aligne, strict=True))
    return {
        "n": len(supervise),
        "aligne_seul": aligne_seul,
        "supervise_seul": supervise_seul,
        "accords": len(supervise) - aligne_seul - supervise_seul,
        "p_mcnemar": round(mcnemar_exact(aligne_seul, supervise_seul), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELES_EVALUABLES,
        default=list(MODELES_EVALUABLES),
        help="modèles à évaluer, dans cet ordre",
    )
    parser.add_argument("--interne", type=int, default=120, help="cas du jeu de test interne")
    parser.add_argument("--taille-lot", type=int, default=8)
    parser.add_argument(
        "--preferences",
        type=int,
        default=150,
        help="paires du jeu de préférences externe (0 pour ignorer)",
    )
    args = parser.parse_args()

    set_seed(SEED)
    cas_cliniques = read_jsonl(PATHS.data_processed / "clinical_eval.jsonl")
    cas_internes = _cas_internes(args.interne) if args.interne else []
    # La reference classique apprend des memes paires que le modele : le jeu
    # d'entrainement pour apprendre, le jeu de validation pour choisir sa
    # configuration. Elle ne voit jamais les cas sur lesquels elle est jugee.
    entrainement = read_jsonl(PATHS.data_processed / "sft_train.jsonl")
    validation = read_jsonl(PATHS.data_processed / "sft_validation.jsonl")
    niveaux_entrainement = [ligne["level"] for ligne in entrainement]
    paires_externes = (
        load_ultramedical_preferences(limit=args.preferences) if args.preferences else []
    )
    logger.info(
        "Jeu clinique indépendant : %d cas. Jeu de test interne : %d cas. "
        "Préférences externes : %d paires.",
        len(cas_cliniques),
        len(cas_internes),
        len(paires_externes),
    )

    resultats: dict = {
        "jeu_clinique": {
            "n": len(cas_cliniques),
            "references": _references(
                cas_cliniques, niveaux_entrainement, entrainement, validation
            ),
            "modeles": {},
        },
        "jeu_interne": {"n": len(cas_internes), "references": {}, "modeles": {}},
        "robustesse": {},
        "preferences_externes": {},
        "erreurs": {},
        # Gardées le temps de comparer les deux livraisons, puis retirées : les
        # prédictions brutes n'ont pas leur place dans le fichier de résultats.
        "_predictions": {},
    }
    if cas_internes:
        resultats["jeu_interne"]["references"] = _references(
            cas_internes, niveaux_entrainement, entrainement, validation, avec_regle=False
        )

    for nom in args.models:
        base, adaptateur = _specification(nom)
        if adaptateur is not None and not Path(adaptateur).exists():
            logger.warning("Adaptateur %s introuvable (%s) — modèle ignoré.", nom, adaptateur)
            continue
        # `base` est soit l'identifiant du modèle de base sur le Hub — qu'on ne
        # peut pas chercher sur le disque — soit l'un des deux dossiers produits
        # par la fusion, qui manque si cette étape n'a pas été jouée.
        bases_locales = {str(PATHS.sft_merged), str(PATHS.dpo_merged)}
        if base in bases_locales and not Path(base).exists():
            logger.warning("Modèle %s introuvable (%s) — modèle ignoré.", nom, base)
            continue
        logger.info("=== Évaluation du modèle : %s ===", nom)
        liberer_la_memoire_gpu()
        agent = TriageAgent(adapter_dir=adaptateur, base_model=base)

        clinique = evaluate_agent(agent, cas_cliniques, taille_lot=args.taille_lot)
        resultats["jeu_clinique"]["modeles"][nom] = {
            **clinique.metriques,
            "securite": clinique.securite,
            "par_langue": clinique.par_langue,
            "par_piege": clinique.par_piege,
        }
        resultats["erreurs"][nom] = error_table(
            cas_cliniques, clinique.predictions, clinique.reponses
        )
        resultats["_predictions"][nom] = list(clinique.predictions)

        if cas_internes:
            interne = evaluate_agent(agent, cas_internes, taille_lot=args.taille_lot)
            resultats["jeu_interne"]["modeles"][nom] = {
                **interne.metriques,
                "securite": interne.securite,
            }

        # Entrées dégradées : saisie de trois lettres, copier-coller de deux
        # pages, question hors domaine, consigne détournée. Aucune n'a de bonne
        # réponse de triage ; ce qui est vérifié, c'est que l'agent tient son
        # contrat de sortie.
        controles = robustness.run(agent)
        resultats["robustesse"][nom] = robustness.summarize(controles)

        # Mesure indépendante de l'alignement, sur un corpus de préférences
        # humaines tenu hors de l'entraînement.
        if paires_externes:
            scores = preference.score_pairs(
                agent.model, agent.tokenizer, paires_externes, agent.device
            )
            resultats["preferences_externes"][nom] = preference.summarize(scores)

        del agent
        liberer_la_memoire_gpu()

    final = next(
        (n for n in ("dpo-fusionne", "dpo", "sft", "base") if n in resultats["_predictions"]),
        None,
    )
    if final:
        predictions_finales = resultats["_predictions"][final]
        resultats["comparaison_modele_regle"] = _comparer_a_une_reference(
            cas_cliniques,
            predictions_finales,
            final,
            explicit_rule([c["description"] for c in cas_cliniques]),
            "regle_explicite",
        )
        # La même comparaison appariée contre le classifieur classique. Les deux
        # prédictions viennent d'être calculées sur ces mêmes cas : on relit celle
        # de la référence plutôt que de réentraîner le classifieur.
        predictions_classiques = resultats["jeu_clinique"]["references"]["classifieur_classique"][
            "_predictions"
        ]
        resultats["comparaison_modele_classifieur"] = _comparer_a_une_reference(
            cas_cliniques,
            predictions_finales,
            final,
            predictions_classiques,
            "classifieur_classique",
        )
    resultats["comparaison_livraisons"] = _comparer_livraisons(resultats)
    resultats["preferences_appariees"] = _comparer_les_preferences(resultats)
    del resultats["_predictions"]
    for jeu in ("jeu_clinique", "jeu_interne"):
        classique = resultats[jeu]["references"].get("classifieur_classique")
        if classique:
            classique.pop("_predictions", None)
    # Le détail paire par paire sert à la comparaison appariée ; il n'a pas sa
    # place dans le fichier de résultats, où il ajouterait des centaines de
    # booléens que personne ne relira.
    for mesures in resultats["preferences_externes"].values():
        mesures.pop("bien_ordonnees", None)

    PATHS.reports.mkdir(parents=True, exist_ok=True)
    destination = PATHS.reports / "evaluation_results.json"
    destination.write_text(
        json.dumps(resultats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Résultats écrits → %s", destination)

    print("\n=== Jeu clinique indépendant ===")
    entete = f"{'système':24s} | {'exactitude':>18s} | {'sous-triage':>11s} | {'format':>6s} | {'surclass.':>9s}"
    print(entete)
    print("-" * len(entete))
    lignes = [
        *resultats["jeu_clinique"]["references"].items(),
        *resultats["jeu_clinique"]["modeles"].items(),
    ]
    for nom, mesures in lignes:
        basse, haute = mesures["exactitude_ic95"]
        print(
            f"{nom:24s} | {mesures['exactitude']:.3f} [{basse:.2f}-{haute:.2f}] | "
            f"{mesures['sous_triage']:>11.3f} | {mesures['respect_format']:>6.3f} | "
            f"{mesures['surclassement']:>9.3f}"
        )

    if resultats["robustesse"]:
        print("\n=== Robustesse aux entrées dégradées ===")
        for nom, mesures in resultats["robustesse"].items():
            echecs = ", ".join(mesures["cas_non_conformes"]) or "aucun"
            print(f"{nom:24s} | conforme {mesures['part_conforme']:.3f} | en échec : {echecs}")

    if resultats["preferences_externes"]:
        print("\n=== Préférences externes (UltraMedical, hors entraînement) ===")
        for nom, mesures in resultats["preferences_externes"].items():
            print(
                f"{nom:24s} | bien ordonnées {mesures['part_bien_ordonnees']:.3f} "
                f"sur {mesures['n']} paires | marge {mesures['marge_moyenne']:+.4f}"
            )


if __name__ == "__main__":
    main()
