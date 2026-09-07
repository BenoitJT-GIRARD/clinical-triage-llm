"""Banc de performance de l'endpoint de triage.

L'évaluation clinique mesure la qualité des décisions ; ce module mesure ce que
coûte une décision. Les deux chiffres qui comptent pour un service d'urgences ne
sont pas les mêmes :

- la **latence perçue**, requête par requête, sans concurrence : c'est le temps
  que l'infirmière d'accueil attend devant son écran ;
- le **débit sous charge**, plusieurs requêtes en parallèle : c'est ce qui dit
  combien de postes d'accueil un serveur peut alimenter aux heures de pointe.

Ces deux chiffres se mesurent sur **la passerelle**, `POST /triage`, parce que
c'est elle que l'infirmière appelle. Les mesurer sur le moteur vLLM seul
laisserait dehors tout ce que la passerelle fait à chaque requête : la règle
explicite, puis l'anonymisation Presidio des deux textes avant l'écriture au
journal d'audit, le tout de façon synchrone. Le critère de passage en production
porte sur ce que le service rend, pas sur ce que le moteur produit.

Le coût du moteur seul reste mesurable (`cible="moteur"`), et le rapport le
présente comme ce qu'il est : une décomposition. La passerelle donne d'ailleurs
sa propre durée d'inférence dans chaque réponse, ce qui isole son surcoût sans
avoir à relancer quoi que ce soit.

Dans les deux cas, on mesure l'endpoint réel servi par vLLM, jamais le moteur de
développement : un chiffre obtenu avec `transformers` en génération séquentielle
n'a rien à voir, vLLM regroupant les requêtes et réutilisant le cache d'attention.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from clinical_triage.evaluation.metrics import latency_summary
from clinical_triage.prompts import completion_payload, extract_level
from clinical_triage.utils import get_logger

logger = get_logger("latence")


@dataclass
class Measure:
    """Une requête chronométrée.

    `inference_ms` n'est renseigné que pour la passerelle : elle publie la durée
    de sa propre génération, et la différence avec la latence totale donne son
    surcoût — règle explicite, anonymisation, écriture au journal.
    """

    latence_ms: float
    tokens: int
    niveau: str | None
    inference_ms: float | None = None


def _appel_moteur(client, url: str, symptomes: str, modele: str) -> Measure:
    """Interroge le moteur vLLM directement et chronomètre la réponse complète."""
    debut = time.perf_counter()
    reponse = client.post(f"{url}/v1/completions", json=completion_payload(symptomes, modele))
    reponse.raise_for_status()
    latence = (time.perf_counter() - debut) * 1000
    corps = reponse.json()
    texte = corps["choices"][0]["text"]
    tokens = corps.get("usage", {}).get("completion_tokens", 0)
    return Measure(latence_ms=latence, tokens=tokens, niveau=extract_level(texte))


def _appel_passerelle(client, url: str, symptomes: str, modele: str) -> Measure:
    """Interroge `POST /triage` et chronomètre la réponse complète.

    `modele` n'est pas utilisé : la passerelle sert le modèle qu'elle a chargé.
    Le paramètre est là pour que les deux fonctions d'appel aient la même
    signature, et que `benchmark` puisse choisir l'une ou l'autre sans détour.
    """
    debut = time.perf_counter()
    reponse = client.post(f"{url}/triage", json={"symptoms": symptomes})
    if reponse.status_code == 429:
        # Le quota est une politique de service, pas une limite du moteur : une
        # mesure de débit le heurte nécessairement, puisqu'elle tient une seule
        # clé. Le message brut du client HTTP ne dit pas quoi en faire.
        raise SystemExit(
            "La passerelle applique son quota par appelant (429). Le banc n'utilise "
            "qu'une clé : relevez TRIAGE_RATE_LIMIT le temps de la mesure, ou baissez "
            "--concurrences et --repetitions."
        )
    reponse.raise_for_status()
    latence = (time.perf_counter() - debut) * 1000
    corps = reponse.json()
    return Measure(
        latence_ms=latence,
        # La passerelle ne compte pas les jetons produits : ce chiffre-là se lit
        # sur le moteur, et la mesure « moteur » est là pour ça.
        tokens=0,
        niveau=corps.get("level"),
        inference_ms=corps.get("latency_ms"),
    )


APPELS = {"passerelle": _appel_passerelle, "moteur": _appel_moteur}


def benchmark(
    url: str,
    modele: str,
    cas: list[str],
    api_key: str | None = None,
    concurrences: tuple[int, ...] = (1, 4, 8),
    repetitions: int = 2,
    cible: str = "passerelle",
) -> dict:
    """Mesure latence et débit d'un endpoint à plusieurs niveaux de concurrence.

    `cible` vaut « passerelle » — `POST /triage`, ce que le service rend, et ce
    sur quoi porte le critère de passage en production — ou « moteur », qui
    interroge vLLM directement et donne le coût de la seule génération.
    """
    import httpx

    if cible not in APPELS:
        raise ValueError(f"Cible de mesure inconnue : {cible!r} (attendu : {set(APPELS)})")
    appeler = APPELS[cible]

    # Les deux cibles n'ont pas le même schéma d'authentification : la passerelle
    # attend sa propre clé dans `X-API-Key`, le moteur celle que vLLM reçoit par
    # `--api-key` et vérifie comme un jeton porteur.
    if not api_key:
        entetes: dict[str, str] = {}
    elif cible == "passerelle":
        entetes = {"X-API-Key": api_key}
    else:
        entetes = {"Authorization": f"Bearer {api_key}"}
    requetes = [cas[i % len(cas)] for i in range(len(cas) * repetitions)]
    resultats: dict[str, dict] = {}

    with httpx.Client(timeout=120, headers=entetes) as client:
        # Chauffe : la première requête paie le chargement des poids en mémoire GPU.
        appeler(client, url, cas[0], modele)

        for concurrence in concurrences:
            debut = time.perf_counter()
            if concurrence == 1:
                mesures = [appeler(client, url, cas_patient, modele) for cas_patient in requetes]
            else:
                with ThreadPoolExecutor(max_workers=concurrence) as pool:
                    mesures = list(pool.map(lambda c: appeler(client, url, c, modele), requetes))
            duree_totale = time.perf_counter() - debut

            resume = latency_summary([m.latence_ms for m in mesures])
            resume["requetes"] = len(mesures)
            resume["debit_req_par_s"] = round(len(mesures) / duree_totale, 2)
            resume["tokens_par_s"] = round(sum(m.tokens for m in mesures) / duree_totale, 1)
            resume["part_niveau_extrait"] = round(
                sum(m.niveau is not None for m in mesures) / len(mesures), 3
            )
            surcouts = [
                m.latence_ms - m.inference_ms for m in mesures if m.inference_ms is not None
            ]
            if surcouts:
                # Ce que la passerelle ajoute à la génération : règle explicite,
                # anonymisation des deux textes, écriture au journal d'audit.
                resume["surcout_passerelle_ms"] = latency_summary(surcouts)["p50_ms"]
            resultats[f"concurrence_{concurrence}"] = resume
            logger.info(
                "  concurrence %d : p50 %.0f ms, p95 %.0f ms, %.2f req/s",
                concurrence,
                resume["p50_ms"],
                resume["p95_ms"],
                resume["debit_req_par_s"],
            )

    return resultats
