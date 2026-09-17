---
license: mit
language:
  - fr
  - en
base_model: Qwen/Qwen3-1.7B-Base
library_name: peft
tags:
  - medical
  - triage
  - emergency-department
  - lora
---

# Agent de triage médical the emergency department — adaptateur LoRA (fine-tuning supervisé)

Adaptateur LoRA issu du **fine-tuning supervisé** de `Qwen3-1.7B-Base` sur le
corpus de triage du service. Il ne contient pas le modèle : il s'applique sur le
modèle de base.

Pour un usage direct, préférer le
[modèle final fusionné](https://huggingface.co/BenoitJT-GIRARD/qwen3-1.7b-clinical-triage),
qui inclut en plus l'alignement par préférences.

> **Prototype pédagogique.** Aide à la décision sous supervision humaine
> obligatoire. Le catalogue clinique ayant servi à construire les données **n'a
> pas été validé par un médecin urgentiste** : ne pas utiliser en situation
> réelle. Devant tout signe vital engagé, appeler le 15 (SAMU).

## Utilisation

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B-Base", device_map="auto")
modele = PeftModel.from_pretrained(base, "BenoitJT-GIRARD/qwen3-1.7b-clinical-triage-sft")
tokenizer = AutoTokenizer.from_pretrained("BenoitJT-GIRARD/qwen3-1.7b-clinical-triage-sft")
```

Le tokenizer publié ici porte le **gabarit de dialogue du projet** et déclare
`<|im_end|>` comme jeton de fin de séquence. Charger le tokenizer du modèle de
base à la place ferait générer sans jamais s'arrêter : le format appris n'est pas
celui d'origine.

## Format de sortie

```
Niveau de priorité : URGENCE_VITALE | URGENCE_MODEREE | CONSULTATION_DIFFEREE
Justification : <explication clinique courte>
Recommandation : <conduite à tenir>
```

## Évaluation

Mesuré sur un jeu de cas **écrits à la main**, jamais vus à l'entraînement, dont près de
la moitié sont des présentations atypiques.

{{EVALUATION}}

Comparaison aux références, analyse d'erreurs et détail par langue : rapport
technique du dépôt.

## Données, évaluation et limites

Voir la carte du [modèle final](https://huggingface.co/BenoitJT-GIRARD/qwen3-1.7b-clinical-triage)
et le dépôt du projet : <https://github.com/BenoitJT-GIRARD/clinical-triage>

## Licence

MIT.
