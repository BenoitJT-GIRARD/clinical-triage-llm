"""Configuration centralisée du POC de triage médical.

Tous les chemins, graines et hyperparamètres du projet sont regroupés ici, dans
des dataclasses figées. Un seul endroit fait foi : les notebooks, les scripts et
les tests importent ces constantes au lieu de les redéfinir, ce qui garantit
qu'un entraînement relancé six mois plus tard part des mêmes valeurs.

Les identifiants de dépôt Hugging Face se surchargent par variable
d'environnement (`HF_NAMESPACE`), pour qu'un autre compte puisse republier le
dataset et les poids sans modifier le code.

Ce module est importé par tout le reste : c'est donc ici que le fichier `.env`
est chargé, une fois, pour que les scripts, les tests et le service voient les
mêmes réglages sans que chacun ait à s'en occuper.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Racine du projet = deux niveaux au-dessus de ce fichier (src/clinical_triage/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Réglages locaux, s'il y en a. `override=False` est le point important : une
# variable déjà posée dans l'environnement l'emporte toujours sur le fichier.
# C'est ce qui permet au même code de tourner ici avec un `.env`, et dans un
# conteneur où l'hébergeur injecte ses propres secrets — sans que le fichier,
# absent de l'image, ne puisse les écraser.
load_dotenv(PROJECT_ROOT / ".env", override=False)

# Graine globale pour random / numpy / torch et pour tous les tirages du
# pipeline de données. C'est la seule valeur à changer pour rejouer le projet
# sous une autre graine.
SEED = 42


@dataclass(frozen=True)
class Paths:
    """Arborescence du projet (créée à la demande par `ensure()`)."""

    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    models: Path = PROJECT_ROOT / "models"
    reports: Path = PROJECT_ROOT / "reports"
    figures: Path = PROJECT_ROOT / "reports" / "figures"
    logs: Path = PROJECT_ROOT / "logs"
    tracking: Path = PROJECT_ROOT / "mlruns"

    def ensure(self) -> None:
        """Crée les répertoires de travail s'ils n'existent pas encore."""
        for directory in (self.data_processed, self.models, self.figures, self.logs):
            directory.mkdir(parents=True, exist_ok=True)

    # Adaptateurs et modèles fusionnés produits par le pipeline.
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
    """Modèle de base et destinations de publication."""

    base_model: str = "Qwen/Qwen3-1.7B-Base"
    max_seq_length: int = 768

    @property
    def hub_namespace(self) -> str:
        """Compte Hugging Face cible (surchargeable par `HF_NAMESPACE`).

        Le `or` n'est pas un raccourci : une variable de dépôt GitHub non
        définie arrive dans l'environnement comme chaîne vide, et non comme
        variable absente. Sans lui, l'identifiant publié commencerait par une
        barre oblique.
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
    """Composition cible du dataset bilingue de triage.

    Le corpus se construit à partir de deux apports complémentaires :

    - des **vignettes cliniques générées** à partir d'un catalogue de
      présentations types (`clinical_catalogue`). L'étiquette de triage vient de
      la présentation elle-même, pas d'une lecture du texte produit : c'est ce
      qui évite de n'évaluer qu'une règle apprise par cœur ;
    - des **cas extraits des corpus médicaux publics**, retenus uniquement
      lorsqu'ils décrivent réellement des signes cliniques.

    Les identifiants Hugging Face des corpus sont la seule source de vérité :
    `corpus_sources` les lit ici.
    """

    corpora: dict[str, str] = field(
        default_factory=lambda: {
            # Les quatre dépôts désignés par le cahier des charges, à l'adresse
            # exacte qu'il donne.
            "mediqal": "ANR-MALADES/MediQAl",
            "medquad": "keivalya/MedQuad-MedicalQnADataset",
            "frenchmedmcqa": "nthngdy/frenchmedmcqa",
            "ultramedical_pref": "TsinghuaC3I/UltraMedical-Preference",
            # Complément anglophone : MediQAl et FrenchMedMCQA sont francophones,
            # et MedQuAD ne décrit pas de patient. MedMCQA apporte le volume de
            # vignettes cliniques en anglais.
            "medmcqa": "openlifescienceai/medmcqa",
        }
    )
    # Volume cible du jeu SFT : environ 5 000 paires.
    target_sft_pairs: int = 5000
    # Part maximale du jeu SFT provenant des corpus publics. Au-delà, les
    # étiquettes de confiance moyenne — celles que la règle de triage attribue
    # aux cas extraits — domineraient les étiquettes validées du catalogue. Le
    # complément vient des vignettes générées, d'où le nom du plafond : c'est
    # lui qu'applique le script de préparation, et lui seul.
    max_corpus_share: float = 0.35
    # Le jeu doit être réellement bilingue : moitié français, moitié anglais.
    french_share: float = 0.50
    target_dpo_pairs: int = 2400
    val_ratio: float = 0.10
    test_ratio: float = 0.10


@dataclass(frozen=True)
class TriageTaxonomy:
    """Niveaux de priorité attendus en sortie du modèle.

    La taxonomie suit la demande métier (urgence maximale / modérée / différée) et
    se rattache aux échelles de triage hospitalières françaises : `URGENCE_VITALE`
    couvre les tris 1 et 2 de l'échelle FRENCH, `URGENCE_MODEREE` les tris 3 et 4,
    `CONSULTATION_DIFFEREE` le tri 5.
    """

    levels: tuple[str, ...] = ("URGENCE_VITALE", "URGENCE_MODEREE", "CONSULTATION_DIFFEREE")
    labels_fr: dict[str, str] = field(
        default_factory=lambda: {
            "URGENCE_VITALE": "Urgence maximale — prise en charge immédiate",
            "URGENCE_MODEREE": "Urgence modérée — prise en charge sous quelques heures",
            "CONSULTATION_DIFFEREE": "Consultation différée — pas de critère de gravité immédiat",
        }
    )
    # Rang de gravité, utilisé par les métriques de sécurité et par le tri.
    severity: dict[str, int] = field(
        default_factory=lambda: {
            "CONSULTATION_DIFFEREE": 0,
            "URGENCE_MODEREE": 1,
            "URGENCE_VITALE": 2,
        }
    )


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparamètres des phases SFT (LoRA) puis DPO.

    Valeurs calibrées pour une carte 16 Go (RTX 4060 Ti) en bf16 + LoRA. Les
    scripts d'entraînement les acceptent en surcharge sur la ligne de commande,
    et journalisent systématiquement les valeurs effectivement utilisées.
    """

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
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
    # SFT — lot effectif = batch_size x grad_accum = 16.
    sft_epochs: int = 2
    sft_batch_size: int = 4
    sft_grad_accum: int = 4
    sft_lr: float = 2e-4
    sft_warmup_ratio: float = 0.03
    # DPO — réglages prudents pour rester proche du modèle SFT de référence.
    # `rpo_alpha` ajoute la perte supervisée sur la réponse préférée, ce qui
    # empêche l'alignement de désapprendre le format acquis au SFT.
    dpo_epochs: int = 1
    dpo_batch_size: int = 2
    dpo_grad_accum: int = 8
    dpo_lr: float = 5e-6
    dpo_beta: float = 0.1
    dpo_rpo_alpha: float = 1.0


@dataclass(frozen=True)
class ServingConfig:
    """Paramètres de génération du service et de l'évaluation."""

    max_new_tokens: int = 220
    temperature: float = 0.0
    # Nom du modèle servi par vLLM et inscrit au journal d'audit.
    served_model_name: str = "qwen3-1.7b-clinical-triage"


# Instances prêtes à l'emploi, importables partout.
PATHS = Paths()
MODEL = ModelConfig()
DATA = DataConfig()
TRIAGE = TriageTaxonomy()
TRAINING = TrainingConfig()
SERVING = ServingConfig()
