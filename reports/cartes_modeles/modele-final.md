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
  - lora
  - dpo
---

# Agent de triage médical — modèle final

`Qwen3-1.7B-Base` spécialisé au triage des urgences par fine-tuning supervisé
avec LoRA, puis aligné par préférences (DPO). Les adaptateurs sont **fusionnés** :
le modèle se charge comme n'importe quel modèle causal, sans bibliothèque
d'adaptation.

> **Prototype pédagogique.** Aide à la décision destinée au personnel soignant,
> sous supervision humaine obligatoire. Le catalogue de présentations cliniques
> ayant servi à construire les données d'entraînement **n'a pas été validé par un
> médecin urgentiste** : ce modèle ne doit pas être utilisé en situation réelle.
> Devant tout signe vital engagé, appeler le 15 (SAMU).

## Ce que fait le modèle

À partir d'une description de patient — motif, symptômes, antécédents,
constantes relevées à l'accueil, en français ou en anglais — il produit une
réponse en français, toujours structurée ainsi :

```
Niveau de priorité : URGENCE_VITALE | URGENCE_MODEREE | CONSULTATION_DIFFEREE
Justification : <explication clinique courte>
Recommandation : <conduite à tenir>
```

| Niveau | Délai de prise en charge | Échelle FRENCH |
|---|---|---|
| `URGENCE_VITALE` | immédiate | tris 1 et 2 |
| `URGENCE_MODEREE` | quelques heures | tris 3 et 4 |
| `CONSULTATION_DIFFEREE` | consultation programmée | tri 5 |

## Utilisation

Le gabarit de dialogue et le jeton de fin de séquence sont inscrits dans le
tokenizer exporté : aucun réglage n'est nécessaire côté appelant.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

nom = "BenoitJT-GIRARD/qwen3-1.7b-clinical-triage"
tokenizer = AutoTokenizer.from_pretrained(nom)
modele = AutoModelForCausalLM.from_pretrained(nom, device_map="auto")

messages = [
    {"role": "system", "content": CONSIGNE_SYSTEME},
    {"role": "user", "content": "Homme de 67 ans, douleur thoracique et sueurs depuis 20 minutes."},
]
invite = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

La consigne système exacte est publiée dans la carte du dataset et dans son
fichier `metadata.json`. Servi par vLLM :

```bash
vllm serve BenoitJT-GIRARD/qwen3-1.7b-clinical-triage \
  --revision {{REVISION}} \
  --served-model-name qwen3-1.7b-clinical-triage --max-model-len 1024
```

`--revision` épingle une version citable : la branche par défaut bouge à chaque
publication, l'étiquette non.

### Une particularité à connaître

Le modèle de base porte les jetons ChatML comme un **vecteur unique jamais
entraîné**, et lie sa tête de sortie à ses embeddings. Un fine-tuning LoRA qui
n'adapte que les projections produit donc un modèle incapable d'émettre son jeton
de fin de tour : il génère jusqu'à la limite. Ce modèle-ci a été entraîné **avec
sa tête de sortie**, et ses poids sont exportés avec le lien défait
(`tie_word_embeddings: false`). Il s'arrête donc de lui-même.

Quiconque repart du modèle de base pour refaire l'exercice rencontrera ce piège :
il est décrit en détail dans la documentation du projet.

## Données d'entraînement

Dataset [`BenoitJT-GIRARD/clinical-triage-bilingual`](https://huggingface.co/datasets/BenoitJT-GIRARD/clinical-triage-bilingual) :
corpus équilibré sur trois niveaux de triage et deux langues, construit à partir
d'un catalogue de présentations cliniques rédigé pour le projet, complété de cas
filtrés issus de **MediQAl** (vignettes cliniques françaises), MedQuAD et MedMCQA.
**Aucune donnée patient réelle.**

## Évaluation

Les chiffres ci-dessous portent sur un jeu de cas **écrits à la main**, jamais vus
à l'entraînement, dont près de la moitié sont des présentations atypiques. Le modèle est
comparé à quatre références, dont une règle de triage explicite et un
classifieur classique entraîné sur les mêmes paires.

{{EVALUATION}}

Analyse d'erreurs, détail par langue et par type de présentation : rapport
technique du dépôt.

## Limites

- **Catalogue clinique non validé par un urgentiste** — limite principale.
- Données d'entraînement en majorité synthétiques : elles n'ont pas le désordre
  du langage réel, et les vignettes générées partagent un petit nombre de
  réponses attendues.
- Jeu d'évaluation de quelques dizaines de cas : à cet effectif, aucun écart
  d'exactitude entre systèmes n'est concluant.
- Aucune vérité terrain externe : le catalogue, la règle de triage et le jeu
  d'évaluation ont la même source.
- 1,7 milliard de paramètres : un format et trois classes, pas un raisonnement clinique.

## Reproduire

Code, hyperparamètres, graine et pipeline complet :
<https://github.com/BenoitJT-GIRARD/clinical-triage>

## Licence

MIT, comme le modèle de base `Qwen/Qwen3-1.7B-Base`.
