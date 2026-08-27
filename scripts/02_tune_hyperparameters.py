"""Réglage des hyperparamètres du fine-tuning supervisé.

Avant de lancer l'entraînement complet, on compare quelques configurations sur un
sous-ensemble du corpus. Chaque variante est entraînée dans les mêmes conditions
— même graine, même sous-ensemble, même nombre de pas — puis jugée sur deux
critères complémentaires :

- la **perte de validation**, qui dit si le modèle apprend la distribution ;
- l'**exactitude de triage** mesurée sur des cas du jeu de validation, qui dit si
  cet apprentissage se traduit par la bonne décision clinique. Les deux ne vont
  pas toujours ensemble, et c'est la seconde qui compte pour le service.

Les variantes portent sur le rang de l'adaptation LoRA et sur le taux
d'apprentissage : ce sont les deux réglages qui pèsent le plus sur le compromis
entre capacité d'adaptation et sur-apprentissage.

Le tableau produit est écrit dans `reports/training/comparaison_hyperparametres.json`
et alimente la figure de réglage.

Usage :
    uv run python scripts/02_tune_hyperparameters.py
    uv run python scripts/02_tune_hyperparameters.py --steps 40 --eval-cases 40
"""

from __future__ import annotations

from clinical_triage.bootstrap import use_utf8_console

use_utf8_console()

import argparse
import json
import shutil

from clinical_triage.config import MODEL, PATHS, SEED
from clinical_triage.data.dataset_io import read_jsonl, write_jsonl
from clinical_triage.evaluation.metrics import accuracy, wilson_interval
from clinical_triage.inference import TriageAgent
from clinical_triage.training.sft import train_sft
from clinical_triage.utils import get_logger, liberer_la_memoire_gpu, set_seed

logger = get_logger("tune")

# Variantes comparées. On fait varier un réglage à la fois autour de la
# configuration de référence, pour que chaque écart observé soit attribuable.
VARIANTES = (
    {"nom": "r8_lr2e-4", "lora_r": 8, "lora_alpha": 16, "learning_rate": 2e-4},
    {"nom": "r16_lr2e-4", "lora_r": 16, "lora_alpha": 32, "learning_rate": 2e-4},
    {"nom": "r32_lr2e-4", "lora_r": 32, "lora_alpha": 64, "learning_rate": 2e-4},
    {"nom": "r16_lr1e-4", "lora_r": 16, "lora_alpha": 32, "learning_rate": 1e-4},
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-subset", type=int, default=1200)
    parser.add_argument("--steps", type=int, default=75)
    parser.add_argument("--eval-cases", type=int, default=60)
    args = parser.parse_args()

    set_seed(SEED)
    processed = PATHS.data_processed
    travail = PATHS.models / "reglage"
    travail.mkdir(parents=True, exist_ok=True)

    # Sous-ensemble figé, identique pour toutes les variantes.
    sous_ensemble = read_jsonl(processed / "sft_train.jsonl")[: args.train_subset]
    chemin_sous_ensemble = travail / "sft_train_subset.jsonl"
    write_jsonl(sous_ensemble, chemin_sous_ensemble)

    cas_validation = read_jsonl(processed / "sft_validation.jsonl")[: args.eval_cases]
    symptomes = [ligne["user_turn"] for ligne in cas_validation]
    attendus = [ligne["level"] for ligne in cas_validation]

    resultats = []
    for variante in VARIANTES:
        logger.info("=== Variante %s ===", variante["nom"])
        sortie = travail / variante["nom"]
        entrainement = train_sft(
            train_path=chemin_sous_ensemble,
            eval_path=processed / "sft_validation.jsonl",
            output_dir=sortie,
            run_name=f"reglage_{variante['nom']}",
            epochs=1,
            max_steps=args.steps,
            lora_r=variante["lora_r"],
            lora_alpha=variante["lora_alpha"],
            learning_rate=variante["learning_rate"],
        )

        # La carte doit être rendue avant de charger la variante suivante : quatre
        # cycles entraînement + évaluation dans le même processus saturent
        # autrement les 16 Go, et le chargement bascule sur le processeur.
        liberer_la_memoire_gpu()
        agent = TriageAgent(adapter_dir=sortie, base_model=MODEL.base_model)
        reponses = agent.generate_batch(symptomes)
        exactitude = accuracy(attendus, [r.level for r in reponses])
        arrets_propres = sum(r.arret_propre for r in reponses) / len(reponses)
        del agent
        liberer_la_memoire_gpu()

        resultats.append(
            {
                "variante": variante["nom"],
                "lora_r": variante["lora_r"],
                "lora_alpha": variante["lora_alpha"],
                "learning_rate": variante["learning_rate"],
                "parametres_entrainables": entrainement.trainable_params,
                "train_loss": entrainement.train_loss,
                "eval_loss": entrainement.eval_loss,
                "exactitude_triage": round(exactitude, 3),
                "part_arrets_propres": round(arrets_propres, 3),
                "duree_s": entrainement.duree_s,
                "memoire_gpu_max_go": entrainement.memoire_gpu_max_go,
            }
        )
        # Les arrêts nets sont journalisés à chaque variante, pas seulement dans
        # le tableau final : c'est la métrique qui a révélé que le modèle ne
        # pouvait pas émettre son jeton de fin, et un zéro doit sauter aux yeux
        # pendant l'exécution plutôt qu'une heure plus tard.
        logger.info(
            "  exactitude de triage : %.3f | arrêts nets : %.3f", exactitude, arrets_propres
        )
        # Les adaptateurs de réglage n'ont pas vocation à être conservés.
        shutil.rmtree(sortie, ignore_errors=True)

    resultats.sort(key=lambda r: (-r["exactitude_triage"], r["eval_loss"] or 9.9))
    destination = PATHS.reports / "training" / "comparaison_hyperparametres.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "protocole": {
                    "exemples_entrainement": len(sous_ensemble),
                    "pas": args.steps,
                    "cas_evalues": len(cas_validation),
                    "graine": SEED,
                },
                "variantes": resultats,
                "retenue": resultats[0]["variante"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(travail, ignore_errors=True)

    # L'intervalle de confiance est affiché avec l'exactitude, jamais sans : sur
    # soixante cas de validation, trois points d'écart font deux cas, et lire une
    # colonne d'exactitudes nues donne l'illusion d'un classement.
    print("\n=== Comparaison des variantes ===")
    entete = (
        f"{'variante':12s} | {'eval_loss':9s} | {'exactitude [IC 95 %]':22s} | "
        f"{'arrêts nets':11s} | {'mém. GPU':8s}"
    )
    print(entete)
    print("-" * len(entete))
    for r in resultats:
        perte = f"{r['eval_loss']:.4f}" if r["eval_loss"] is not None else "n/a"
        basse, haute = wilson_interval(
            round(r["exactitude_triage"] * len(cas_validation)), len(cas_validation)
        )
        exactitude = f"{r['exactitude_triage']:.3f} [{basse:.2f} – {haute:.2f}]"
        print(
            f"{r['variante']:12s} | {perte:9s} | {exactitude:22s} | "
            f"{r['part_arrets_propres']:<11.3f} | {r['memoire_gpu_max_go']:.1f} Go"
        )
    print(f"\nConfiguration retenue : {resultats[0]['variante']}")
    print("Les intervalles se recouvrent : la comparaison classe, elle ne tranche pas.")


if __name__ == "__main__":
    main()
