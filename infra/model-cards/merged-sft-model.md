---
license: mit
language:
  - fr
  - en
base_model: Qwen/Qwen3-1.7B-Base
library_name: transformers
pipeline_tag: text-generation
tags:
  - medical
  - triage
  - emergency-department
---

# Agent de triage médical the emergency department — modèle supervisé fusionné

`Qwen3-1.7B-Base` après **fine-tuning supervisé** sur le corpus de triage du
the emergency department, adaptateur LoRA fusionné dans les poids. C'est le modèle **intermédiaire**
du projet : il sait trier et respecte le format de sortie, mais il n'a pas encore
reçu l'alignement par préférences.

Pour un usage direct, préférer le
[modèle final](https://huggingface.co/BenoitJT-GIRARD/qwen3-1.7b-clinical-triage),
qui inclut cet alignement.

Ce dépôt existe pour deux raisons précises :

1. **c'est le modèle de base de l'adaptateur DPO.** L'alignement a été entraîné
   par-dessus ces poids-ci ; sans eux,
   [l'adaptateur DPO](https://huggingface.co/BenoitJT-GIRARD/qwen3-1.7b-clinical-triage-dpo)
   ne peut pas être chargé ;
2. **il permet de mesurer ce que l'alignement apporte**, en comparant les deux
   modèles sur le même jeu d'évaluation.

> **Prototype pédagogique.** Aide à la décision sous supervision humaine
> obligatoire. Le catalogue clinique ayant servi à construire les données **n'a
> pas été validé par un médecin urgentiste** : ne pas utiliser en situation
> réelle. Devant tout signe vital engagé, appeler le 15 (SAMU).

## Utilisation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

depot = "BenoitJT-GIRARD/qwen3-1.7b-clinical-triage-sft-merged"
modele = AutoModelForCausalLM.from_pretrained(depot, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(depot)
```

Le tokenizer publié ici porte le **gabarit de dialogue du projet** et déclare
`<|im_end|>` comme jeton de fin de séquence.

La tête de sortie a été **découplée** de la matrice d'embeddings avant la fusion.
Le modèle de base partage un seul tenseur entre les deux ; comme le fine-tuning
adapte la tête, fusionner sans délier aurait écrit la correction dans l'entrée du
modèle autant que dans sa sortie. Ce dépôt porte donc un `lm_head.weight` propre,
et `tie_word_embeddings` à `false`.

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
