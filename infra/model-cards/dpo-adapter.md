---
license: mit
language:
  - fr
  - en
base_model: BenoitJT-GIRARD/qwen3-1.7b-clinical-triage-sft-merged
library_name: peft
tags:
  - medical
  - triage
  - emergency-department
  - lora
  - dpo
---

# Agent de triage médical the emergency department — adaptateur LoRA (alignement par préférences)

Adaptateur LoRA issu de l'**alignement par préférences (DPO)** du modèle de
triage du service. Il s'applique sur le **modèle supervisé fusionné**, pas sur
`Qwen3-1.7B-Base` : l'alignement a été entraîné au-dessus du premier, et
l'appliquer sur le second produirait une composition de poids incohérente.

Pour un usage direct, préférer le
[modèle final fusionné](https://huggingface.co/BenoitJT-GIRARD/qwen3-1.7b-clinical-triage),
qui combine déjà les deux étapes.

> **Prototype pédagogique.** Aide à la décision sous supervision humaine
> obligatoire. Le catalogue clinique ayant servi à construire les données **n'a
> pas été validé par un médecin urgentiste** : ne pas utiliser en situation
> réelle. Devant tout signe vital engagé, appeler le 15 (SAMU).

## Ce que l'alignement apprend

Entre deux réponses au même cas, préférer celle qui :

- ne **sous-évalue** pas le niveau d'urgence ;
- ne propose pas une conduite à tenir qui **retarde** la prise en charge ;
- n'**affirme** pas de diagnostic, la consigne système l'interdisant ;
- répond dans la **langue** attendue.

Les paires de préférence ont le même format et une longueur comparable : sans
cette précaution, l'alignement apprend la longueur plutôt que le fond, et le
modèle cesse d'émettre son jeton de fin.

## Utilisation

Cet adaptateur s'applique sur le **modèle supervisé fusionné**, pas sur le modèle
de base : il a été entraîné au-dessus, et l'y appliquer directement produirait une
composition de poids incohérente. Ce modèle intermédiaire est publié pour cette
raison précise, et la configuration de l'adaptateur le désigne : rien n'est donc
à reconstruire.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "BenoitJT-GIRARD/qwen3-1.7b-clinical-triage-sft-merged", device_map="auto"
)
modele = PeftModel.from_pretrained(base, "BenoitJT-GIRARD/qwen3-1.7b-clinical-triage-dpo")
tokenizer = AutoTokenizer.from_pretrained("BenoitJT-GIRARD/qwen3-1.7b-clinical-triage-dpo")
```

**Pour servir le modèle, préférez le [modèle final
fusionné](https://huggingface.co/BenoitJT-GIRARD/qwen3-1.7b-clinical-triage)** : il ne
demande aucune reconstruction, et c'est une seule référence à épingler, là où
cet adaptateur demande au serveur d'héberger aussi le modèle supervisé fusionné
et de les apparier.

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
