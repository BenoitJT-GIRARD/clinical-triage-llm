"""Central configuration of the triage project.

Every path, seed and hyper-parameter lives here, in frozen dataclasses. One place is
authoritative: notebooks, scripts and tests import these constants instead of redefining them,
which is what makes a training run relaunched six months later start from the same values.

Hugging Face repository ids can be overridden by environment variable (``HF_NAMESPACE``), so
that another account can republish the dataset and the weights without touching the code.

This module is imported by everything else, so this is where the ``.env`` file is loaded,
once, and scripts, tests and the service all see the same settings without each having to care.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

# Project root = two levels above this file (src/clinical_triage/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Local settings, if any. ``override=False`` is the point: a variable already present in the
# environment always wins over the file. That is what lets the same code run here with a
# ``.env`` and inside a container where the host injects its own secrets — without the file,
# absent from the image, being able to overwrite them.
load_dotenv(PROJECT_ROOT / ".env", override=False)

# Global seed for random / numpy / torch and for every draw of the data pipeline. It is the
# only value to change in order to replay the project under another seed.
SEED = 42


@dataclass(frozen=True)
class Paths:
    """Project layout (created on demand by :meth:`ensure`).

    Everything a run produces and nothing tracks — weights, logs, the experiment store — sits
    under ``var/``. Keeping them at the root would put four directories in a listing that the
    repository never carries, and a reader would have to open each one to find out.
    """

    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    models: Path = PROJECT_ROOT / "var" / "models"
    reports: Path = PROJECT_ROOT / "reports"
    figures: Path = PROJECT_ROOT / "reports" / "figures"
    images: Path = PROJECT_ROOT / "docs" / "images"
    logs: Path = PROJECT_ROOT / "var" / "logs"
    tracking: Path = PROJECT_ROOT / "var" / "mlruns"

    def ensure(self) -> None:
        """Make the directories a run writes into, on first use."""
        for directory in (self.data_processed, self.models, self.figures, self.logs):
            directory.mkdir(parents=True, exist_ok=True)

    def for_seed(self, seed: int) -> Paths:
        """The weights of one training seed. The shipped seed keeps the canonical folders.

        The other seeds measure the spread of the method and are never served: they live apart,
        so that nothing reading the canonical folders can pick one up by mistake.
        """
        return self if seed == SEED else replace(self, models=self.models / f"seed-{seed}")

    def capped(self) -> Paths:
        """Where a run capped to a few steps writes: never over a delivered artefact."""
        return replace(self, models=self.models / "capped")

    def training_summaries(self, seed: int = SEED, capped: bool = False) -> Path:
        """Where a training run writes its summary.

        The summaries of the shipped seed are published results. Those of the other seeds are
        published apart, as the spread of the method; those of a capped run check a chain and
        are published nowhere.
        """
        if capped:
            return self.root / "var" / "runs"
        if seed == SEED:
            return self.reports / "training"
        return self.reports / "training" / "seeds" / f"seed-{seed}"

    # Adapters and merged models produced by the pipeline.
    @property
    def sft_adapter(self) -> Path:
        return self.models / "qwen3-1.7b-triage-sft"

    @property
    def sft_merged(self) -> Path:
        return self.models / "qwen3-1.7b-triage-sft-merged"

    @property
    def dpo_adapter(self) -> Path:
        return self.models / "qwen3-1.7b-triage-dpo"

    @property
    def dpo_merged(self) -> Path:
        return self.models / "qwen3-1.7b-triage-dpo-merged"


@dataclass(frozen=True)
class ModelConfig:
    """Base model and publication targets."""

    base_model: str = "Qwen/Qwen3-1.7B-Base"
    max_seq_length: int = 768

    @property
    def hub_namespace(self) -> str:
        """Target Hugging Face account (overridable by ``HF_NAMESPACE``).

        The ``or`` is not a shortcut: an undefined GitHub repository variable arrives in the
        environment as an empty string, not as a missing variable. Without it, the published id
        would start with a slash.
        """
        return os.getenv("HF_NAMESPACE") or "BenoitJT-GIRARD"

    @property
    def hub_dataset_id(self) -> str:
        return f"{self.hub_namespace}/clinical-triage-bilingual"

    @property
    def hub_sft_model_id(self) -> str:
        return f"{self.hub_namespace}/qwen3-1.7b-clinical-triage-sft"

    @property
    def hub_dpo_model_id(self) -> str:
        return f"{self.hub_namespace}/qwen3-1.7b-clinical-triage-dpo"

    @property
    def hub_sft_merged_model_id(self) -> str:
        return f"{self.hub_namespace}/qwen3-1.7b-clinical-triage-sft-merged"

    @property
    def hub_merged_model_id(self) -> str:
        return f"{self.hub_namespace}/qwen3-1.7b-clinical-triage"


@dataclass(frozen=True)
class DataConfig:
    """Target composition of the bilingual triage dataset.

    The corpus is built from two complementary contributions:

    - **generated clinical vignettes**, derived from a catalogue of typical presentations
      (``clinical_catalogue``). The triage label comes from the presentation itself, not from
      reading the produced text: that is what keeps the evaluation from measuring a rule the
      model learnt by heart;
    - **cases extracted from public medical corpora**, kept only when they genuinely describe
      clinical signs.

    The Hugging Face ids of those corpora are the single source of truth: ``corpus_sources``
    reads them here.
    """

    corpora: dict[str, str] = field(
        default_factory=lambda: {
            # The four source repositories, at their exact addresses on the Hub.
            "mediqal": "ANR-MALADES/MediQAl",
            "medquad": "keivalya/MedQuad-MedicalQnADataset",
            "frenchmedmcqa": "nthngdy/frenchmedmcqa",
            "ultramedical_pref": "TsinghuaC3I/UltraMedical-Preference",
            # English complement: MediQAl and FrenchMedMCQA are French-language, and MedQuAD
            # does not describe a patient. MedMCQA brings the volume of English vignettes.
            "medmcqa": "openlifescienceai/medmcqa",
        }
    )
    # Target size of the supervised set.
    target_sft_pairs: int = 5000
    # Maximum share of the supervised set coming from the public corpora. Beyond it, the
    # medium-confidence labels — those the triage rule assigns to extracted cases — would
    # dominate the catalogue's validated labels. The remainder comes from generated vignettes,
    # hence the name of the cap: the preparation script applies this one, and only this one.
    max_corpus_share: float = 0.35
    # The set must be genuinely bilingual: half French, half English.
    french_share: float = 0.50
    target_dpo_pairs: int = 2400
    val_ratio: float = 0.10
    test_ratio: float = 0.10


@dataclass(frozen=True)
class TriageTaxonomy:
    """Priority levels expected out of the model.

    The taxonomy follows the operational need (immediate / urgent / deferred) and maps onto the
    French hospital triage scales: ``URGENCE_VITALE`` covers sorts 1 and 2 of the FRENCH scale,
    ``URGENCE_MODEREE`` sorts 3 and 4, ``CONSULTATION_DIFFEREE`` sort 5.

    The level names stay in French: they are the strings the model emits, the strings written
    into the dataset and into every published artefact. Renaming them would retrain the model.
    """

    levels: tuple[str, ...] = ("URGENCE_VITALE", "URGENCE_MODEREE", "CONSULTATION_DIFFEREE")
    labels_fr: dict[str, str] = field(
        default_factory=lambda: {
            "URGENCE_VITALE": "Urgence maximale — prise en charge immédiate",
            "URGENCE_MODEREE": "Urgence modérée — prise en charge sous quelques heures",
            "CONSULTATION_DIFFEREE": "Consultation différée — pas de critère de gravité immédiat",
        }
    )
    # Severity rank, used by the safety metrics and for ordering.
    severity: dict[str, int] = field(
        default_factory=lambda: {
            "CONSULTATION_DIFFEREE": 0,
            "URGENCE_MODEREE": 1,
            "URGENCE_VITALE": 2,
        }
    )


@dataclass(frozen=True)
class TrainingConfig:
    """Hyper-parameters of the SFT (LoRA) phase and then of DPO.

    Values calibrated for a 16 GB card (RTX 4060 Ti) in bf16 + LoRA. The training scripts
    accept overrides on the command line, and always log the values actually used.
    """

    # LoRA. The rank, the scaling and the learning rate of the supervised pass have no default
    # here: they are the setting `scripts/tune_hyperparameters.py` keeps, read back by
    # `scripts/train_sft.py`. A default in this file once announced a rank the shipped model did
    # not have.
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    # SFT — effective batch = batch_size x grad_accum = 16.
    sft_epochs: int = 2
    sft_batch_size: int = 4
    sft_grad_accum: int = 4
    sft_warmup_ratio: float = 0.03
    # DPO — conservative settings, to stay close to the reference SFT model. ``rpo_alpha`` adds
    # the supervised loss on the preferred answer, which keeps the alignment from unlearning
    # the output format acquired during SFT.
    dpo_epochs: int = 1
    dpo_batch_size: int = 2
    dpo_grad_accum: int = 8
    dpo_lr: float = 5e-6
    # The alignment adapter's own rank and scaling, set once and not tuned: measuring an effect
    # this small does not support a search over it.
    dpo_lora_r: int = 16
    dpo_lora_alpha: int = 32
    dpo_beta: float = 0.1
    dpo_rpo_alpha: float = 1.0


@dataclass(frozen=True)
class ServingConfig:
    """Generation settings of the service and of the evaluation."""

    max_new_tokens: int = 220
    temperature: float = 0.0
    # The longest patient narrative, in characters. It is the bound the dataset build places on
    # the corpus cases, so no training narrative exceeds it, and the service refuses a longer
    # one rather than serving the model a text unlike anything it learnt on.
    max_description_characters: int = 800
    # Name of the model served by vLLM and written into the audit log.
    served_model_name: str = "qwen3-1.7b-clinical-triage"


# Ready-to-use instances, importable everywhere.
PATHS = Paths()
MODEL = ModelConfig()
DATA = DataConfig()
TRIAGE = TriageTaxonomy()
TRAINING = TrainingConfig()
SERVING = ServingConfig()
