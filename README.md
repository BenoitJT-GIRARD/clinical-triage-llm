# Agent IA de triage médical

Prototype d'un agent d'aide au triage des urgences.

À partir d'une description de patient, l'agent doit proposer un niveau de
priorité, le justifier, et formuler une conduite à tenir.

## Feuille de route

1. Préparation et structuration des données bilingues.
2. Fine-tuning supervisé du Qwen3-1.7B-Base avec LoRA.
3. Alignement par préférences (DPO).
4. Déploiement d'un endpoint vLLM et évaluation.

## Installation

```bash
uv sync
```
