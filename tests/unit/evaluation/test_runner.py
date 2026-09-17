"""Tests de l'exécution d'une évaluation et du banc de performance.

Aucun modèle n'est chargé : un agent factice produit des réponses contrôlées.
Ce qui est vérifié, c'est le chaînage — génération par lots, calcul des
métriques, contrôles de sécurité, détail par sous-groupe et table d'erreurs.
"""

from __future__ import annotations

from dataclasses import dataclass

from clinical_triage.evaluation.latency import Measure
from clinical_triage.evaluation.runner import error_table, evaluate_agent, evaluate_predictions
from clinical_triage.prompts import completion_payload

CAS = [
    {
        "id": "ev01",
        "user_turn": "Situation clinique.\nDouleur thoracique et sueurs.\nNiveau ?",
        "description": "Homme de 67 ans, douleur thoracique et sueurs depuis 20 minutes.",
        "level": "URGENCE_VITALE",
        "lang": "fr",
        "piege": "",
        "note_clinique": "Syndrome coronarien aigu probable.",
    },
    {
        "id": "ev02",
        "user_turn": "Situation clinique.\nRhume banal.\nNiveau ?",
        "description": "Femme de 34 ans, nez bouché depuis deux jours, pas de fièvre.",
        "level": "CONSULTATION_DIFFEREE",
        "lang": "fr",
        "piege": "faux_alarmant",
        "note_clinique": "Rhinopharyngite virale bénigne.",
    },
]

REPONSE_JUSTE = (
    "Niveau de priorité : URGENCE_VITALE\n"
    "Justification : Douleur thoracique avec sueurs chez un patient à risque.\n"
    "Recommandation : Prise en charge immédiate et appel du 15 (SAMU)."
)
REPONSE_FAUSSE = (
    "Niveau de priorité : URGENCE_VITALE\n"
    "Justification : Symptômes inquiétants.\n"
    "Recommandation : Prise en charge immédiate et appel du 15 (SAMU)."
)


@dataclass
class ReponseFictive:
    text: str
    level: str | None
    latency_ms: float
    tokens_generes: int
    arret_propre: bool


class AgentFictif:
    """Agent qui répond toujours « urgence vitale » : juste une fois sur deux."""

    def generate_batch(self, symptomes: list[str]) -> list[ReponseFictive]:
        return [
            ReponseFictive(
                text=REPONSE_JUSTE if "thoracique" in s else REPONSE_FAUSSE,
                level="URGENCE_VITALE",
                latency_ms=120.0,
                tokens_generes=64,
                arret_propre=True,
            )
            for s in symptomes
        ]


def test_evaluation_d_un_agent():
    resultat = evaluate_agent(AgentFictif(), CAS, taille_lot=1)
    assert resultat.predictions == ["URGENCE_VITALE", "URGENCE_VITALE"]
    assert resultat.metriques["exactitude"] == 0.5
    assert resultat.metriques["respect_format"] == 1.0
    assert resultat.metriques["arrets_propres"] == 1.0
    assert resultat.metriques["surclassement"] == 0.5
    assert resultat.metriques["tokens_generes_moyen"] == 64.0


def test_l_evaluation_produit_les_controles_de_securite():
    resultat = evaluate_agent(AgentFictif(), CAS, taille_lot=2)
    assert resultat.securite["n"] == 2
    assert 0.0 <= resultat.securite["part_sans_defaut"] <= 1.0


def test_l_evaluation_detaille_les_sous_groupes():
    resultat = evaluate_agent(AgentFictif(), CAS, taille_lot=2)
    assert resultat.par_langue["fr"]["n"] == 2
    assert "faux_alarmant" in resultat.par_piege
    assert "presentation_directe" in resultat.par_piege


def test_evaluation_de_predictions_sans_generation():
    """Chemin utilisé par les références : un niveau, sans texte ni latence."""
    resume = evaluate_predictions(CAS, ["URGENCE_VITALE", "CONSULTATION_DIFFEREE"])
    assert resume["exactitude"] == 1.0
    assert "latence" not in resume
    assert resume["par_langue"]["fr"]["exactitude"] == 1.0


def test_la_table_d_erreurs_ne_retient_que_les_cas_mal_classes():
    erreurs = error_table(
        CAS, ["URGENCE_VITALE", "URGENCE_VITALE"], [REPONSE_JUSTE, REPONSE_FAUSSE]
    )
    assert len(erreurs) == 1
    erreur = erreurs[0]
    assert erreur["id"] == "ev02"
    assert erreur["attendu"] == "CONSULTATION_DIFFEREE"
    assert erreur["predit"] == "URGENCE_VITALE"
    assert erreur["note_clinique"]


# --- Banc de performance ---


def test_le_corps_de_requete_impose_l_arret_sur_le_jeton_de_fin():
    """Le banc et la passerelle assemblent désormais la même requête.

    Ils en avaient chacun leur version : identiques champ pour champ, mais rien
    ne les tenait ensemble, et les latences publiées auraient fini par décrire
    une requête que le service n'émet plus.
    """
    charge = completion_payload("Douleur thoracique.", "qwen3-1.7b-clinical-triage")
    assert charge["model"] == "qwen3-1.7b-clinical-triage"
    assert charge["prompt"].endswith("<|im_start|>assistant\n")
    assert charge["stop"] == ["<|im_end|>"]
    assert charge["temperature"] == 0.0


def test_une_mesure_porte_le_niveau_extrait():
    mesure = Measure(latence_ms=100.0, tokens=50, niveau="URGENCE_VITALE")
    assert mesure.niveau == "URGENCE_VITALE"
