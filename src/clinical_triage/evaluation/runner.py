"""Exécution de l'évaluation : génération sur le jeu clinique puis métriques.

Le jeu d'évaluation est celui, écrit à la main, que `clinical_eval_set` produit.
Il n'a jamais servi à l'entraînement, et les réponses attendues n'ont pas été
fabriquées par la règle à laquelle on compare le modèle.

La génération se fait par lots, pour que passer soixante cas prenne des minutes
et non des heures. La latence rapportée à ce titre est donc une latence par cas
en traitement par lots : elle sert à comparer les modèles entre eux, pas à
décrire ce que ressent un utilisateur. La latence perçue est mesurée séparément,
requête par requête, par le banc de performance de `latency.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

from clinical_triage.evaluation import safety
from clinical_triage.evaluation.metrics import subgroup_accuracy, summarize
from clinical_triage.utils import get_logger

logger = get_logger("evaluation")


@dataclass
class EvaluationOutcome:
    """Résultat d'une évaluation : prédictions, métriques et contrôles de sécurité."""

    predictions: list[str | None]
    reponses: list[str]
    metriques: dict
    securite: dict
    par_langue: dict
    par_piege: dict


def evaluate_predictions(cas: list[dict], predictions: list[str | None]) -> dict:
    """Calcule les métriques d'une liste de prédictions, sans génération.

    C'est le chemin utilisé pour les références : elles produisent un niveau sans
    produire de texte, donc sans latence ni contrôle de sécurité.
    """
    attendus = [c["level"] for c in cas]
    resume = summarize(attendus, predictions)
    resume["par_langue"] = subgroup_accuracy(attendus, predictions, [c["lang"] for c in cas])
    resume["par_piege"] = subgroup_accuracy(attendus, predictions, [c["piege"] for c in cas])
    return resume


def evaluate_agent(agent, cas: list[dict], taille_lot: int = 8) -> EvaluationOutcome:
    """Fait générer l'agent sur tous les cas, puis calcule métriques et sécurité."""
    descriptions = [c["description"] for c in cas]
    tours = [c["user_turn"] for c in cas]
    attendus = [c["level"] for c in cas]

    reponses = []
    for debut in range(0, len(tours), taille_lot):
        lot = tours[debut : debut + taille_lot]
        reponses.extend(agent.generate_batch(lot))
        logger.info("  %d/%d cas évalués", min(debut + taille_lot, len(tours)), len(tours))

    predictions = [r.level for r in reponses]
    textes = [r.text for r in reponses]
    metriques = summarize(
        attendus,
        predictions,
        latences_ms=[r.latency_ms for r in reponses],
        arrets_propres=[r.arret_propre for r in reponses],
    )
    metriques["tokens_generes_moyen"] = round(
        sum(r.tokens_generes for r in reponses) / max(1, len(reponses)), 1
    )

    rapports = [
        safety.check(description, texte, prediction)
        for description, texte, prediction in zip(descriptions, textes, predictions, strict=True)
    ]

    return EvaluationOutcome(
        predictions=predictions,
        reponses=textes,
        metriques=metriques,
        securite=safety.summarize(rapports),
        par_langue=subgroup_accuracy(attendus, predictions, [c["lang"] for c in cas]),
        par_piege=subgroup_accuracy(attendus, predictions, [c["piege"] for c in cas]),
    )


def error_table(cas: list[dict], predictions: list[str | None], reponses: list[str]) -> list[dict]:
    """Détaille les cas mal classés, pour l'analyse d'erreurs du rapport."""
    erreurs = []
    for c, prediction, reponse in zip(cas, predictions, reponses, strict=True):
        if prediction == c["level"]:
            continue
        erreurs.append(
            {
                "id": c["id"],
                "langue": c["lang"],
                "piege": c["piege"],
                "attendu": c["level"],
                "predit": prediction,
                "description": c["description"],
                "note_clinique": c["note_clinique"],
                "reponse": reponse,
            }
        )
    return erreurs
