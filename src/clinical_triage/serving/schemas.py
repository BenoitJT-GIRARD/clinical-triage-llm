"""Contrats d'API — le point d'intégration au système d'information hospitalier.

Ces schémas définissent un format d'échange stable et documenté. Le SIH n'a pas
à connaître le modèle, ni le moteur d'inférence : il envoie une description de
patient et reçoit un niveau de priorité, sa justification, la conduite à tenir et
l'identifiant d'audit correspondant.

La réponse expose aussi le niveau qu'aurait retenu la **règle explicite**. Ce
n'est pas un détail d'implémentation qui fuit : c'est un garde-fou. Quand les
deux divergent, l'interface d'accueil peut le signaler à l'infirmière, qui
tranche. Un agent d'aide à la décision doit rendre son désaccord visible.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from clinical_triage.config import TRIAGE


class TriageRequest(BaseModel):
    """Demande de triage à partir d'une description de patient."""

    symptoms: str = Field(
        ...,
        min_length=3,
        max_length=4000,
        description="Description libre des symptômes, antécédents et constantes du patient.",
        examples=[
            "Homme de 62 ans, douleur thoracique et sueurs depuis 20 minutes. TA 148/92, FC 102."
        ],
    )
    patient_age: int | None = Field(
        None, ge=0, le=120, description="Âge du patient, s'il n'est pas dans la description."
    )


class TriageReply(BaseModel):
    """Réponse de triage structurée."""

    level: str | None = Field(..., description=f"Niveau prédit parmi {', '.join(TRIAGE.levels)}.")
    level_label: str | None = Field(None, description="Libellé lisible du niveau prédit.")
    justification: str | None = Field(None, description="Explication clinique de la décision.")
    recommendation: str | None = Field(None, description="Conduite à tenir proposée.")
    rule_level: str | None = Field(
        None, description="Niveau retenu par la règle explicite, pour comparaison."
    )
    rule_reasons: list[str] = Field(
        default_factory=list, description="Signes ayant motivé la décision de la règle."
    )
    agreement: bool = Field(
        ..., description="Vrai si le modèle et la règle explicite retiennent le même niveau."
    )
    raw_response: str = Field(..., description="Réponse complète du modèle, après troncature.")
    description_truncated: bool = Field(
        False,
        description=(
            "Vrai si la description reçue dépassait la fenêtre du modèle et a été bornée. "
            "Le triage n'a alors pas lu la totalité du récit : seul un soignant peut juger "
            "si ce qui manque comptait."
        ),
    )
    latency_ms: float = Field(..., description="Durée de l'inférence, en millisecondes.")
    request_id: str = Field(..., description="Identifiant de traçabilité de l'interaction.")
    model_version: str = Field(..., description="Modèle ayant réellement produit la réponse.")


class QuestionnaireRequest(BaseModel):
    """État courant du questionnaire adaptatif."""

    chief_complaint: str = Field(
        ..., min_length=2, max_length=500, description="Motif principal de consultation."
    )
    # Ce champ est borné comme les autres champs de texte. Sans ces bornes, un
    # dictionnaire de taille arbitraire à valeurs arbitraires serait désérialisé
    # entier en mémoire, puis concaténé en une seule chaîne par la synthèse, qui
    # la donne à lire à la règle de triage : une requête suffirait à saturer le
    # conteneur. Trente-deux entrées couvrent largement le plan le plus long,
    # qui en compte six.
    answers: dict[
        Annotated[str, StringConstraints(max_length=64)],
        Annotated[str, StringConstraints(max_length=1000)],
    ] = Field(
        default_factory=dict,
        max_length=32,
        description="Réponses déjà collectées, par identifiant de question.",
    )


class QuestionnaireReply(BaseModel):
    """Prochaine question, ou signal de fin de collecte."""

    next_question_id: str | None = Field(None, description="Identifiant de la prochaine question.")
    next_question: str | None = Field(None, description="Texte de la prochaine question.")
    theme: str = Field(..., description="Thème clinique détecté à partir du motif.")
    finished: bool = Field(..., description="Vrai si la collecte est terminée.")
    compiled_symptoms: str = Field(
        ..., description="Synthèse des informations collectées, prête pour le triage."
    )


class HealthReply(BaseModel):
    """État de santé du service."""

    status: str = Field(..., description="ok si le service peut répondre, degraded sinon.")
    backend: str = Field(..., description="Moteur d'inférence : transformers ou vllm.")
    model_loaded: bool = Field(..., description="Vrai si le moteur d'inférence répond.")
    model_version: str = Field(..., description="Modèle servi.")
    detail: str | None = Field(None, description="Cause de l'indisponibilité, le cas échéant.")
