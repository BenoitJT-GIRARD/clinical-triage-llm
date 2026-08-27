"""Fusion d'un adaptateur LoRA dans son modèle de base.

Deux usages :

- `--adapter sft` fusionne l'adaptateur SFT dans Qwen3-1.7B-Base et produit
  `models/qwen3-1.7b-triage-sft-merged`. Ce modèle sert de base **et** de
  référence à l'alignement DPO ;
- `--adapter dpo` fusionne l'adaptateur DPO dans le modèle SFT fusionné et
  produit `models/qwen3-1.7b-triage-dpo-merged`, le modèle final servi par vLLM.

Le tokenizer exporté avec le modèle porte le gabarit de dialogue du projet et
`<|im_end|>` comme jeton de fin de séquence. C'est ce qui fait que le serveur
d'inférence s'arrête au bon endroit sans réglage supplémentaire : la
configuration de génération est écrite dans le modèle, pas dans l'appelant.

Usage :
    uv run python scripts/04_merge_and_export.py --adapter sft
    uv run python scripts/04_merge_and_export.py --adapter dpo
"""

from __future__ import annotations

from clinical_triage.bootstrap import use_utf8_console

use_utf8_console()

import argparse

from clinical_triage.config import MODEL, PATHS
from clinical_triage.prompts import prepare_tokenizer
from clinical_triage.utils import get_logger

logger = get_logger("merge")


def _verifier_export(destination) -> None:
    """Refuse un export dont la tête de sortie n'a pas survécu au découplage.

    Deux façons dont l'export peut être faux sans lever la moindre erreur : la
    configuration réactive le lien entre embeddings et tête de sortie, ou le
    fichier de poids ne contient pas de tenseur `lm_head.weight` distinct. Dans
    les deux cas le modèle servi n'est pas celui qui a été entraîné.
    """
    import json

    configuration = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    if configuration.get("tie_word_embeddings"):
        raise SystemExit(
            f"{destination} : les poids sont encore liés, la tête de sortie fusionnée est perdue."
        )

    index = destination / "model.safetensors.index.json"
    if index.exists():
        cles = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    else:
        from safetensors import safe_open

        with safe_open(str(destination / "model.safetensors"), framework="pt") as fichier:
            cles = set(fichier.keys())
    if "lm_head.weight" not in cles:
        raise SystemExit(f"{destination} : `lm_head.weight` absent des poids exportés.")
    logger.info("Export vérifié : tête de sortie découplée et présente dans les poids.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=["sft", "dpo"], required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.adapter == "sft":
        base = MODEL.base_model
        adaptateur = PATHS.sft_adapter
        destination = PATHS.sft_merged
    else:
        base = str(PATHS.sft_merged)
        adaptateur = PATHS.dpo_adapter
        destination = PATHS.dpo_merged

    if not adaptateur.exists():
        raise SystemExit(f"Adaptateur introuvable : {adaptateur}")

    destination.mkdir(parents=True, exist_ok=True)

    # Fusion en float32, pas en bfloat16. Un delta LoRA est petit devant les
    # poids qu'il corrige ; additionné sur huit bits de mantisse, une partie
    # disparaît purement et simplement dans l'arrondi. On fusionne donc en
    # simple précision, et on ne repasse en bfloat16 qu'au moment d'écrire.
    logger.info("Chargement du modèle de base en float32 : %s", base)
    modele = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)

    if getattr(modele.config, "tie_word_embeddings", False):
        # Le modèle de base partage un seul tenseur entre sa matrice
        # d'embeddings et sa tête de sortie. Notre adaptateur corrige la tête :
        # fusionner sans délier écrirait le delta dans le tenseur partagé, donc
        # dans l'entrée du modèle autant que dans sa sortie, et le corromprait
        # silencieusement. On donne sa propre copie à la tête avant la fusion.
        logger.info("Poids liés détectés : découplage de la tête de sortie avant fusion.")
        modele.lm_head.weight = torch.nn.Parameter(modele.lm_head.weight.detach().clone())
        modele.config.tie_word_embeddings = False

    logger.info("Fusion de l'adaptateur %s", adaptateur)
    fusionne = PeftModel.from_pretrained(modele, str(adaptateur)).merge_and_unload()
    fusionne = fusionne.to(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(str(adaptateur))
    jeton_de_fin = prepare_tokenizer(tokenizer)
    # La configuration de génération voyage avec le modèle : tout moteur
    # d'inférence s'arrêtera sur le bon jeton sans réglage de l'appelant.
    fusionne.generation_config.eos_token_id = jeton_de_fin
    fusionne.generation_config.pad_token_id = jeton_de_fin
    fusionne.config.eos_token_id = jeton_de_fin

    fusionne.save_pretrained(str(destination), safe_serialization=True)
    tokenizer.save_pretrained(str(destination))
    _verifier_export(destination)
    logger.info("Modèle fusionné sauvegardé dans %s", destination)


if __name__ == "__main__":
    main()
