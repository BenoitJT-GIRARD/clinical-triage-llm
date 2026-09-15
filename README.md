# Agent IA de triage médical

[![CI](https://github.com/BenoitJT-GIRARD/clinical-triage/actions/workflows/ci.yml/badge.svg)](https://github.com/BenoitJT-GIRARD/clinical-triage/actions/workflows/ci.yml)
[![Licence MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

Prototype d'un **agent d'aide au triage des urgences**. À partir d'une description de patient — motif, symptômes,
antécédents, constantes relevées à l'accueil, en français ou en anglais — l'agent
propose un **niveau de priorité**, le **justifie** et formule une **conduite à
tenir**, en traçant chaque interaction pour les audits médicaux.

Le modèle est un `Qwen3-1.7B-Base` spécialisé par **fine-tuning supervisé avec
LoRA** — noyaux **Unsloth** — puis **aligné par préférences (DPO)**, servi par
**vLLM** derrière une passerelle **FastAPI**, avec un pipeline **GitHub Actions**
de bout en bout.

> **Aide à la décision, sous supervision humaine obligatoire.** L'agent ne pose
> pas de diagnostic et ne remplace pas une évaluation clinique. Le catalogue de
> présentations ayant servi à construire les données n'a pas été validé par un
> médecin urgentiste : ce prototype ne doit pas être utilisé en situation réelle.
> Devant tout signe vital engagé, appeler le 15 (SAMU).

## Ce que fait l'agent

| Point d'entrée | Rôle |
|---|---|
| `POST /questionnaire/next` | questionnaire adaptatif : les questions dépendent du motif, la collecte s'arrête dès qu'un signe vital apparaît |
| `POST /triage` | niveau de priorité, justification, conduite à tenir, **et le niveau qu'aurait retenu la règle explicite** |
| `GET /health` | état du service et du moteur d'inférence |

La réponse expose délibérément l'avis de la règle explicite et l'accord entre les
deux : un agent d'aide à la décision doit rendre son désaccord visible.

## Ce que ce dépôt contient

```
clinical-triage/
├── src/clinical_triage/
│   ├── config.py            # chemins, graines, hyperparamètres — source unique
│   ├── prompts.py           # format de dialogue, partagé par toutes les étapes
│   ├── inference.py         # chargement du modèle et génération
│   ├── data/                # catalogue clinique, générateur, corpus, RGPD, jeux
│   ├── training/            # fine-tuning supervisé, alignement DPO, suivi MLflow
│   ├── evaluation/          # métriques, contrôles de sécurité, références, latence
│   ├── serving/             # API, questionnaire adaptatif, journal d'audit
│   └── reporting/           # figures
├── scripts/                 # le pipeline, dans l'ordre (01 → 08)
├── tests/                   # tests unitaires et tests de contrat de l'API
├── notebooks/               # la démarche, commentée pas à pas
├── deploy/                  # Dockerfile, compose, guide de déploiement
├── data/processed/          # jeu d'évaluation, fiche d'audit, jeux reconstruits
└── reports/                 # figures, résultats d'évaluation
```

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/), Python 3.12. Un GPU NVIDIA est
nécessaire pour l'entraînement — Unsloth en exige un dès l'import — et Docker pour
servir le modèle avec vLLM.

```bash
uv sync
uv run clinical-triage        # état du projet et de la configuration

# Noyau Jupyter du projet, utilisé par les notebooks.
uv run python -m ipykernel install --user --name clinical-triage --display-name "Python (clinical-triage)"
```

Sur un poste dont le répertoire personnel est synchronisé dans le nuage, placer
l'environnement et les caches ailleurs : les téléchargements se comptent en
gigaoctets.

```bash
export UV_PROJECT_ENVIRONMENT=~/.local/share/clinical-triage/venv
export HF_HOME=~/.local/share/clinical-triage/hf
# Le suivi d'expériences écrit dans une base SQLite : un client de
# synchronisation qui la recopie pendant l'écriture interrompt la création
# du schéma.
export MLFLOW_TRACKING_URI="sqlite:///$HOME/.local/share/clinical-triage/mlflow.db"
```

## Le pipeline, dans l'ordre

```bash
uv run python scripts/01_build_dataset.py         # dataset bilingue et jeu d'évaluation
uv run python scripts/02_tune_hyperparameters.py  # comparaison des configurations
uv run python scripts/03_train_sft.py             # fine-tuning supervisé
uv run python scripts/04_merge_and_export.py --adapter sft
uv run python scripts/05_train_dpo.py             # alignement par préférences
uv run python scripts/04_merge_and_export.py --adapter dpo
uv run python scripts/06_evaluate.py              # évaluation comparée aux références
docker compose -f deploy/docker-compose.yml up    # endpoint vLLM + passerelle
uv run python scripts/07_benchmark_endpoint.py --api-key "$TRIAGE_API_KEY"  # latence et débit
uv run python scripts/08_publish_hf.py --what tout
```

Les notebooks `01` à `05` commentent ces mêmes étapes et se rejouent une fois le
pipeline déroulé :

```bash
uv run jupyter nbconvert --to notebook --inplace --execute notebooks/*.ipynb
```

Suivi des entraînements, en local et sans compte — sur la même base que celle
écrite par les scripts :

```bash
uv run mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI"
```

## Trois décisions qui expliquent le reste

**Les données ne viennent pas d'où on croit.** Aucun des corpus imposés n'est
annoté en niveaux de triage, et FrenchMedMCQA ne compte que 1 080 questions dont
six décrivent un patient. La vérité terrain vient donc d'un **catalogue de
présentations cliniques** rédigé pour le projet, dont un générateur tire des
vignettes ; les corpus publics apportent des cas authentiques, filtrés et
étiquetés avec une confiance explicitement moindre. Détail dans
[`data/README.md`](data/README.md).

**L'évaluation ne mesure pas une règle contre elle-même.** Les chiffres publiés
portent sur un jeu de **cas écrits à la main**, jamais vus à l'entraînement, dont
près de la moitié sont des présentations atypiques. Le modèle y est comparé à
quatre références : deux triviales, la règle explicite qu'il doit remplacer, et
un classifieur classique entraîné sur les mêmes paires — celle-ci dit ce que le
fine-tuning apporte par-dessus un apprentissage ordinaire.

**Un seul format de dialogue, et un jeton de fin que le modèle peut produire.**
Le projet installe son propre gabarit ChatML sur le tokenizer et déclare
`<|im_end|>` comme fin de séquence. Ça ne suffit pas : dans `Qwen3-1.7B-Base`, les
vingt-cinq jetons ChatML sont un seul et même vecteur jamais entraîné — `im_start`
et `im_end` ont une similarité cosinus de 1,000 — et la tête de sortie est liée aux
embeddings, que LoRA gèle. Le modèle ne peut donc pas émettre le jeton de fin :
0 % d'arrêts nets, mesuré. **La tête de sortie est donc entraînée avec les
projections** ; les arrêts nets passent à 100 % et la réponse tombe de 220 à 97
jetons. Détail et mesures dans `reports/training/`.

## Qualité

| Outil | Rôle | Commande |
|---|---|---|
| Ruff | style et format | `uv run ruff check .` · `uv run ruff format .` |
| Pytest | tests et couverture | `uv run pytest` |
| Bandit | analyse statique de sécurité | `uv run bandit -c pyproject.toml -r src/clinical_triage` |
| pip-audit | vulnérabilités des dépendances | `uv run pip-audit` |
| pre-commit | vérifications avant commit | `uv run pre-commit install` |

L'intégration continue exécute tout cela, puis **construit l'image de service et
vérifie qu'elle démarre, répond, et refuse un appel sans clé**.

## Contenu

| # | Élément | Où |
|---|---|---|
| 1 | Dataset médical bilingue | [`data/processed/`](data/processed) et le Hugging Face Hub |
| 2 | Modèle spécialisé (SFT + LoRA, puis DPO) | Hugging Face Hub, cartes dans `reports/` |
| 3 | Endpoint de démonstration | [`deploy/`](deploy/README.md) |
| 4 | Pipeline CI/CD | [`.github/workflows/`](.github/workflows) |

## Licence

MIT — voir [LICENSE](LICENSE).
