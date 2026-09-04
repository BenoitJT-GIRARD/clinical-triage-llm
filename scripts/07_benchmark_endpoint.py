"""Banc de performance de l'endpoint de triage.

Mesure la latence perçue et le débit sous charge de `POST /triage`, c'est-à-dire
de ce que l'infirmière d'accueil appelle réellement : la passerelle FastAPI, qui
interroge le moteur puis applique la règle explicite, anonymise les deux textes
et écrit au journal d'audit avant de répondre. C'est ce chiffre-là que le critère
de passage en production retient.

Avec `--url-moteur`, une seconde série interroge vLLM directement. Elle ne
remplace pas la première : elle la décompose, en donnant le coût de la seule
génération. Le rapport publie les deux sous leurs vrais noms.

À lancer contre la pile locale (`docker compose -f deploy/docker-compose.yml up`)
ou contre l'endpoint déployé dans le cloud : la commande est la même, seule
l'URL change.

Usage :
    uv run python scripts/07_benchmark_endpoint.py --api-key "$TRIAGE_API_KEY"
    uv run python scripts/07_benchmark_endpoint.py --url https://mon-endpoint --api-key "$CLE"
"""

from __future__ import annotations

import argparse
import json

from clinical_triage.config import PATHS, SERVING
from clinical_triage.data.dataset_io import read_jsonl
from clinical_triage.evaluation.latency import benchmark
from clinical_triage.utils import get_logger

logger = get_logger("banc")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="http://localhost:8080", help="passerelle de triage (POST /triage)"
    )
    parser.add_argument(
        "--url-moteur",
        default=None,
        help="moteur vLLM, pour mesurer en plus le coût de la seule génération",
    )
    parser.add_argument("--model", default=SERVING.served_model_name)
    parser.add_argument("--api-key", default=None, help="clé de la passerelle (X-API-Key)")
    parser.add_argument(
        "--vllm-key", default=None, help="clé du moteur, si vLLM a été lancé avec --api-key"
    )
    parser.add_argument("--cas", type=int, default=20, help="cas distincts utilisés")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument(
        "--concurrences", type=int, nargs="+", default=[1, 4, 8], help="niveaux de charge testés"
    )
    args = parser.parse_args()

    cas = [
        ligne["description"]
        for ligne in read_jsonl(PATHS.data_processed / "clinical_eval.jsonl")[: args.cas]
    ]
    if not cas:
        raise SystemExit("Aucun cas d'évaluation : lancez d'abord scripts/01_build_dataset.py.")

    logger.info(
        "Banc de performance sur la passerelle %s (%d cas, %d répétitions)",
        args.url,
        len(cas),
        args.repetitions,
    )
    resultats = benchmark(
        url=args.url,
        modele=args.model,
        cas=cas,
        api_key=args.api_key,
        concurrences=tuple(args.concurrences),
        repetitions=args.repetitions,
        cible="passerelle",
    )

    mesures_moteur = None
    version_moteur = None
    if args.url_moteur:
        version_moteur = _version_du_moteur(args.url_moteur, args.vllm_key)
        logger.info("Moteur vLLM %s", version_moteur or "de version inconnue")
        logger.info("Décomposition : coût du moteur seul sur %s", args.url_moteur)
        mesures_moteur = benchmark(
            url=args.url_moteur,
            modele=args.model,
            cas=cas,
            api_key=args.vllm_key,
            concurrences=tuple(args.concurrences),
            repetitions=args.repetitions,
            cible="moteur",
        )

    destination = PATHS.reports / "benchmark_endpoint.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "endpoint": args.url,
                "cible": "passerelle (POST /triage)",
                "modele": args.model,
                "mesures": resultats,
                "endpoint_moteur": args.url_moteur,
                "version_moteur": version_moteur,
                "mesures_moteur": mesures_moteur,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Mesures écrites → %s", destination)

    _afficher("Latence et débit de la passerelle (POST /triage)", resultats)
    if mesures_moteur:
        _afficher("Décomposition : coût du moteur seul (vLLM)", mesures_moteur)


def _version_du_moteur(url: str, api_key: str | None) -> str | None:
    """Demande au moteur la version qu'il fait tourner.

    Une latence ne vaut que pour un moteur donné. La pile épingle son image par
    empreinte, mais cette image est surchargeable — le moteur V1 de vLLM ne
    démarre pas sous WSL 2 — et le rapport doit publier la version réellement
    mesurée, pas celle qu'annonce le fichier de composition.
    """
    import httpx

    entetes = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    reponse = httpx.get(f"{url}/version", headers=entetes, timeout=10)
    reponse.raise_for_status()
    return reponse.json().get("version")


def _afficher(titre: str, resultats: dict) -> None:
    """Imprime un tableau de mesures à la console."""
    print(f"\n=== {titre} ===")
    entete = (
        f"{'charge':12s} | {'p50 (ms)':>9s} | {'p95 (ms)':>9s} | "
        f"{'req/s':>6s} | {'passerelle':>10s}"
    )
    print(entete)
    print("-" * len(entete))
    for nom, mesures in resultats.items():
        surcout = mesures.get("surcout_passerelle_ms")
        print(
            f"{nom:12s} | {mesures['p50_ms']:>9.0f} | {mesures['p95_ms']:>9.0f} | "
            f"{mesures['debit_req_par_s']:>6.2f} | "
            f"{(f'{surcout:.0f} ms' if surcout is not None else '—'):>10s}"
        )


if __name__ == "__main__":
    main()
