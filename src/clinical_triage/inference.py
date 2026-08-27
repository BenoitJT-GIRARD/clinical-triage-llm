"""Inférence de l'agent de triage : chargement du modèle et génération.

Brique partagée par l'évaluation et par le service en mode démonstration. Le
déploiement de production passe par vLLM (voir `deploy/`), mais les deux chemins
produisent exactement la même invite et la même réponse : c'est le même gabarit
ChatML, le même jeton de fin de séquence et la même troncature de sécurité.

La troncature mérite un mot. Un modèle de 1,7 milliard de paramètres finit
parfois par oublier d'émettre son jeton de fin et continue à écrire. Sans filet,
le texte renvoyé au personnel soignant et écrit au journal d'audit contient alors
la suite de la génération, consigne système comprise. `truncate_to_answer` coupe
après la ligne de recommandation, quoi qu'il arrive.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from clinical_triage.config import MODEL, SERVING
from clinical_triage.prompts import (
    borner_la_description,
    budget_de_description,
    build_messages,
    extract_level,
    format_chatml,
    prepare_tokenizer,
    truncate_to_answer,
)
from clinical_triage.utils import chemin_pour_journal, get_logger

logger = get_logger("inference")


@dataclass
class TriageResponse:
    """Résultat d'une inférence de triage."""

    text: str
    level: str | None
    latency_ms: float
    tokens_generes: int
    arret_propre: bool  # le modèle a émis son jeton de fin avant la limite
    # La description dépassait la fenêtre du modèle et a été coupée. Ce drapeau
    # remonte jusqu'à la réponse rendue : le soignant est seul à savoir si ce qui
    # manque comptait, et une troncature silencieuse d'un récit clinique est
    # exactement ce qu'un système d'aide à la décision ne doit pas faire.
    description_tronquee: bool = False


def _verifier_le_placement_sur_gpu(model) -> None:
    """Vérifie qu'Unsloth a bien placé tout le modèle sur le GPU.

    Quand la mémoire vidéo manque, le chargement répartit silencieusement une
    partie des couches sur le processeur. Unsloth mémorise l'emplacement de
    chaque couche à cet instant, et une couche restée sur le processeur reçoit
    un indice de périphérique vide. L'erreur ne remonte qu'à la première
    génération, sous la forme d'un « Invalid target device: None » qui ne dit
    rien de sa cause. On préfère la dire ici, avant d'avoir perdu le temps de
    chargement, et avec la conduite à tenir.
    """
    emplacements = {parametre.device.type for parametre in model.parameters()}
    if emplacements <= {"cuda"}:
        return
    raise RuntimeError(
        "Mémoire GPU insuffisante : le modèle a été chargé en partie sur "
        f"{', '.join(sorted(emplacements - {'cuda'}))}. Libérez la carte — un seul "
        "travail GPU à la fois — puis relancez."
    )


class TriageAgent:
    """Agent de triage : encapsule le modèle, le tokenizer et la génération."""

    def __init__(
        self,
        adapter_dir: str | Path | None = None,
        base_model: str = MODEL.base_model,
        device: str | None = None,
    ):
        # Unsloth remplace le `forward` des couches d'attention de Qwen3 dès son
        # import, et sa version attend des attributs qu'il ne pose que sur les
        # modèles qu'il a lui-même chargés. Un modèle chargé par `transformers`
        # nu dans le même processus tomberait donc sur un `forward` qui ne sait
        # pas le manipuler. On charge avec Unsloth dès qu'il est présent, et on
        # retombe sur `transformers` là où il ne l'est pas — l'image de service,
        # qui n'embarque ni torch ni Unsloth.
        import importlib.util

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        avec_unsloth = importlib.util.find_spec("unsloth") is not None and torch.cuda.is_available()

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        logger.info(
            "Chargement du modèle %s sur %s (%s)",
            chemin_pour_journal(base_model),
            self.device,
            "unsloth" if avec_unsloth else "transformers",
        )

        if avec_unsloth:
            from unsloth import FastLanguageModel

            model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(base_model),
                max_seq_length=MODEL.max_seq_length,
                dtype=dtype,
                load_in_4bit=False,
            )
            _verifier_le_placement_sur_gpu(model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(str(base_model))
            model = AutoModelForCausalLM.from_pretrained(str(base_model), dtype=dtype)

        self.end_token_id = prepare_tokenizer(self.tokenizer)
        # Le remplissage se fait à gauche : en génération par lots, les réponses
        # doivent toutes commencer juste après le dernier jeton de leur invite.
        self.tokenizer.padding_side = "left"

        if adapter_dir is not None:
            from peft import PeftModel

            logger.info("Application de l'adaptateur LoRA : %s", chemin_pour_journal(adapter_dir))
            model = PeftModel.from_pretrained(model, str(adapter_dir))

        if avec_unsloth:
            from unsloth import FastLanguageModel

            FastLanguageModel.for_inference(model)

        self.model = model.to(self.device).eval()
        self.description = (
            f"{Path(str(base_model)).name}+{Path(str(adapter_dir)).name}"
            if adapter_dir
            else Path(str(base_model)).name
        )

    def _prompts(self, symptoms: list[str]) -> tuple[list[str], list[bool]]:
        """Construit les invites ChatML, en bornant les descriptions trop longues.

        Une description qui déborde la fenêtre du modèle n'y produit pas une
        réponse dégradée : elle interrompt la génération sur une erreur de
        dimension. On la borne donc ici, et on dit pour chaque cas si elle l'a été.
        """
        budget = budget_de_description(self.tokenizer, MODEL.max_seq_length, SERVING.max_new_tokens)
        bornees = [borner_la_description(s, self.tokenizer, budget) for s in symptoms]
        invites = [
            format_chatml(build_messages(texte), add_generation_prompt=True) for texte, _ in bornees
        ]
        return invites, [coupee for _, coupee in bornees]

    def generate_batch(
        self,
        symptoms: list[str],
        max_new_tokens: int = SERVING.max_new_tokens,
        temperature: float = SERVING.temperature,
    ) -> list[TriageResponse]:
        """Génère les réponses de triage d'un lot de descriptions.

        Le traitement par lots n'existe que pour l'évaluation : il divise par
        plusieurs le temps nécessaire pour passer le jeu clinique complet. La
        latence renvoyée est alors la latence moyenne par cas du lot, ce que
        `latency.py` signale explicitement pour éviter toute confusion avec la
        latence perçue par un utilisateur.
        """
        invites, coupees = self._prompts(symptoms)
        if any(coupees):
            logger.warning(
                "%d description(s) sur %d dépassaient la fenêtre du modèle et ont été bornées.",
                sum(coupees),
                len(coupees),
            )
        entrees = self.tokenizer(invites, return_tensors="pt", padding=True).to(self.device)
        longueur_invite = entrees["input_ids"].shape[1]

        debut = time.perf_counter()
        with self._torch.no_grad():
            sortie = self.model.generate(
                **entrees,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.end_token_id,
            )
        latence_totale = (time.perf_counter() - debut) * 1000
        latence_par_cas = latence_totale / max(1, len(symptoms))

        reponses: list[TriageResponse] = []
        for ligne, coupee in zip(sortie, coupees, strict=True):
            nouveaux = ligne[longueur_invite:].tolist()
            arret_propre = self.end_token_id in nouveaux
            if arret_propre:
                nouveaux = nouveaux[: nouveaux.index(self.end_token_id)]
            texte = truncate_to_answer(self.tokenizer.decode(nouveaux, skip_special_tokens=True))
            reponses.append(
                TriageResponse(
                    text=texte,
                    level=extract_level(texte),
                    latency_ms=latence_par_cas,
                    tokens_generes=len(nouveaux),
                    arret_propre=arret_propre,
                    description_tronquee=coupee,
                )
            )
        return reponses

    def generate(
        self,
        symptoms: str,
        max_new_tokens: int = SERVING.max_new_tokens,
        temperature: float = SERVING.temperature,
    ) -> TriageResponse:
        """Génère la réponse de triage d'une description unique."""
        return self.generate_batch([symptoms], max_new_tokens, temperature)[0]
