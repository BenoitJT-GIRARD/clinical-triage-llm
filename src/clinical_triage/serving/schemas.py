"""API contracts — the integration point with the hospital information system.

These schemas define a stable, documented exchange format. The information system needs to know
neither the model nor the inference engine: it sends a patient description and receives a
priority level, its justification, what to do next, and the matching audit id.

The reply also exposes the level the **explicit rule** would have chosen. That is not an
implementation detail leaking out: it is a guard rail. When the two diverge, the reception
screen can flag it to the nurse, who decides. A decision-support agent must make its
disagreement visible.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from clinical_triage.config import TRIAGE


class TriageRequest(BaseModel):
    """A triage request built from a patient description."""

    symptoms: str = Field(
        ...,
        min_length=3,
        max_length=4000,
        description="Free-text description of the patient's symptoms, history and vital signs.",
        examples=[
            "Homme de 62 ans, douleur thoracique et sueurs depuis 20 minutes. TA 148/92, FC 102."
        ],
    )
    patient_age: int | None = Field(
        None, ge=0, le=120, description="Patient age, when it is not in the description."
    )


class TriageReply(BaseModel):
    """A structured triage answer."""

    level: str | None = Field(
        ..., description=f"Predicted level, one of {', '.join(TRIAGE.levels)}."
    )
    level_label: str | None = Field(None, description="Human-readable label of the level.")
    justification: str | None = Field(None, description="Clinical explanation of the decision.")
    recommendation: str | None = Field(None, description="Suggested course of action.")
    rule_level: str | None = Field(
        None, description="Level chosen by the explicit rule, for comparison."
    )
    rule_reasons: list[str] = Field(
        default_factory=list, description="Signs that drove the rule's decision."
    )
    agreement: bool = Field(
        ..., description="True when the model and the explicit rule choose the same level."
    )
    raw_response: str = Field(..., description="The model's full answer, after truncation.")
    description_truncated: bool = Field(
        False,
        description=(
            "True when the description received exceeded the model window and was bounded. "
            "The triage then did not read the whole narrative: only a clinician can judge "
            "whether what is missing mattered."
        ),
    )
    latency_ms: float = Field(..., description="Inference duration, in milliseconds.")
    request_id: str = Field(..., description="Traceability id of the interaction.")
    model_version: str = Field(..., description="The model that actually produced the answer.")


class QuestionnaireRequest(BaseModel):
    """Current state of the adaptive questionnaire."""

    chief_complaint: str = Field(
        ..., min_length=2, max_length=500, description="Main reason for the visit."
    )
    # This field is bounded like every other text field. Without those bounds an arbitrarily
    # large dictionary with arbitrary values would be deserialised whole into memory, then
    # concatenated into a single string by the summary, which hands it to the triage rule: one
    # request would be enough to saturate the container. Thirty-two entries cover the longest
    # plan by a wide margin — it has six.
    answers: dict[
        Annotated[str, StringConstraints(max_length=64)],
        Annotated[str, StringConstraints(max_length=1000)],
    ] = Field(
        default_factory=dict,
        max_length=32,
        description="Answers already collected, by question id.",
    )


class QuestionnaireReply(BaseModel):
    """The next question, or the signal that collection is over."""

    next_question_id: str | None = Field(None, description="Id of the next question.")
    next_question: str | None = Field(None, description="Text of the next question.")
    theme: str = Field(..., description="Clinical theme detected from the complaint.")
    finished: bool = Field(..., description="True when collection is over.")
    compiled_symptoms: str = Field(
        ..., description="Summary of what was collected, ready for triage."
    )


class HealthReply(BaseModel):
    """Health of the service."""

    status: str = Field(..., description="ok when the service can answer, degraded otherwise.")
    backend: str = Field(..., description="Inference engine: transformers or vllm.")
    model_loaded: bool = Field(..., description="True when the inference engine responds.")
    model_version: str = Field(..., description="The model being served.")
    detail: str | None = Field(None, description="Cause of the unavailability, when there is one.")
