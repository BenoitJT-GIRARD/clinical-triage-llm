# Déploiement de l'endpoint de triage

L'agent est servi par **vLLM** derrière une passerelle **FastAPI**. Ce document
couvre les trois situations : la démonstration locale, la pile complète en
conteneurs, et le déploiement dans le cloud.

## Architecture

```
poste d'accueil / SIH  ──HTTPS──▶  passerelle FastAPI  ──HTTP──▶  serveur vLLM (GPU)
                                   authentification              modèle fusionné
                                   questionnaire adaptatif       traitement par lots
                                   règle de contrôle             cache d'attention
                                   anonymisation et audit
```

La séparation est délibérée. La passerelle est légère — ni torch, ni poids — et
se redéploie en quelques secondes ; le serveur d'inférence se met à jour sans
toucher à la logique métier. Le contrat OpenAPI rend l'intégration au système
d'information indépendante des deux.

## 1. Préparer le modèle

vLLM sert un modèle complet : il faut fusionner les adaptateurs.

```bash
uv run python scripts/04_merge_and_export.py --adapter sft
uv run python scripts/05_train_dpo.py
uv run python scripts/04_merge_and_export.py --adapter dpo
# produit models/qwen3-1.7b-triage-dpo-merged
```

Le tokenizer exporté porte le gabarit de dialogue du projet et `<|im_end|>` comme
jeton de fin de séquence. La configuration de génération voyage donc avec le
modèle : tout moteur d'inférence s'arrête au bon endroit sans réglage côté appelant.

## 2. Pile complète en conteneurs

Prérequis : un hôte avec GPU NVIDIA et `nvidia-container-toolkit`. Sous Windows,
Docker Desktop avec le backend WSL 2 convient.

```bash
export TRIAGE_API_KEY="une-cle-choisie"
docker compose -f deploy/docker-compose.yml up --build
```

- passerelle : <http://localhost:8080/docs>
- serveur vLLM : <http://localhost:8000/v1>

8000 et 8080 sont des ports courants : si l'un est déjà pris sur la machine, la
pile refuse de démarrer. Les deux se choisissent sans toucher au fichier de
composition, et sans rien arrêter d'autre :

```bash
TRIAGE_VLLM_PORT=8001 TRIAGE_API_PORT=8081 \
  docker compose -f deploy/docker-compose.yml up --build
```

Seule la publication côté hôte change ; à l'intérieur de la pile, la passerelle
joint le moteur sur son port interne, qui ne bouge pas.

**Si le dépôt est sur un disque synchronisé dans le nuage**, les poids doivent en
sortir. Docker Desktop ne monte pas un disque virtuel de synchronisation : le
conteneur démarre, ne trouve rien sous `/models`, prend le chemin pour un
identifiant de dépôt Hugging Face et s'arrête sur `Repo id must be in the form
'repo_name' or 'namespace/repo_name'`. La sonde de santé échoue alors sans dire
pourquoi.

```bash
cp -r models/qwen3-1.7b-triage-dpo-merged ~/.local/share/clinical-triage/models/
TRIAGE_MODELS_DIR=~/.local/share/clinical-triage/models \
  docker compose -f deploy/docker-compose.yml up --build
```

**Si l'hôte est un poste Windows**, le moteur épinglé ne démarre pas. Le moteur
V1 de vLLM exige l'adressage virtuel unifié de CUDA ; WSL 2 ne l'expose pas, et
le serveur s'arrête sur `RuntimeError: UVA is not available` avant d'avoir chargé
les poids. La bascule `VLLM_USE_V1=0` ne sert à rien : le moteur V0 a été retiré.
Une version antérieure passe, au prix d'un moteur différent de celui du
déploiement :

```bash
TRIAGE_VLLM_IMAGE=vllm/vllm-openai:v0.11.0 \
  docker compose -f deploy/docker-compose.yml up --build
```

C'est la seule raison de surcharger cette variable. Sur un hôte Linux avec
`nvidia-container-toolkit`, la version épinglée par empreinte fonctionne, et
c'est elle qui doit servir : elle fige le moteur d'inférence sous un modèle et
un contrat d'API donnés.

```bash
curl -X POST http://localhost:8080/triage \
  -H 'Content-Type: application/json' -H "X-API-Key: $TRIAGE_API_KEY" \
  -d '{"symptoms": "Homme de 62 ans, douleur thoracique et sueurs depuis 20 minutes. TA 148/92, FC 102."}'
```

La passerelle attend que vLLM se déclare en bonne santé avant de démarrer. Sans
cette condition, les premières requêtes échouent pendant le chargement des poids
et la sonde de la passerelle serait verte alors que l'inférence ne répond pas.

Mesurer la performance de la pile, de bout en bout sur la passerelle — c'est ce
que le service rend, anonymisation et journal d'audit compris :

```bash
uv run python scripts/07_benchmark_endpoint.py --api-key "$TRIAGE_API_KEY"
```

Ajouter `--url-moteur http://localhost:8000` pour mesurer en plus le coût du
moteur seul, que le rapport présente comme une décomposition et non comme la
performance de l'endpoint.

Le banc n'utilise qu'une clé d'API, donc un seul seau de quota : au-delà du
premier palier de concurrence, la passerelle lui répond 429. Relever le quota le
temps de la mesure, sans toucher à la valeur de service :

```bash
TRIAGE_RATE_LIMIT=6000 docker compose -f deploy/docker-compose.yml up -d
```

## 3. Démonstration sans GPU ni Docker

vLLM ne fonctionne pas nativement sous Windows. Pour une démonstration locale, la
passerelle sait charger le modèle elle-même avec `transformers` — plus lent, mais
suffisant pour montrer le comportement.

```powershell
$env:TRIAGE_BACKEND = "transformers"
$env:TRIAGE_BASE_MODEL = "models/qwen3-1.7b-triage-sft-merged"
$env:TRIAGE_ADAPTER_DIR = "models/qwen3-1.7b-triage-dpo"
$env:TRIAGE_API_KEY = "cle-de-demonstration"
uv run uvicorn clinical_triage.serving.api:app --port 8080
```

Les latences obtenues dans ce mode ne sont pas comparables à celles de
l'endpoint vLLM et ne doivent pas être rapportées comme telles.

## 4. Déploiement cloud

### Option retenue — Modal

[Modal](https://modal.com) alloue un GPU à la demande, facture à la seconde et
éteint le conteneur quand il n'est plus sollicité. Le compte gratuit reçoit 30 $
de crédits renouvelés chaque mois, sans carte bancaire : une démonstration
consomme moins d'une heure de L4, soit environ 0,80 $.

`modal_app.py` monte les deux mêmes briques que la pile Docker de ce dossier — un
conteneur GPU servant le modèle fusionné sous vLLM, et la passerelle FastAPI du
projet exposée telle quelle. C'est la séparation du projet, pas une adaptation à
l'hébergeur : le code de service n'est pas modifié d'une ligne.

```bash
uv run modal setup                      # authentification, ouvre le navigateur
uv run modal secret create clinical-triage \
    TRIAGE_API_KEY="une-cle-choisie" \
    HF_TOKEN="hf_..."
uv run modal deploy deploy/modal_app.py
```

Le déploiement affiche deux adresses. Reporter celle du moteur dans le secret,
puis redéployer pour que la passerelle sache où s'adresser :

```bash
uv run modal secret create clinical-triage --force \
    TRIAGE_API_KEY="une-cle-choisie" \
    HF_TOKEN="hf_..." \
    TRIAGE_VLLM_URL="https://<compte>--clinical-triage-moteur.modal.run"
uv run modal deploy deploy/modal_app.py
```

`modal setup` écrit les jetons du compte dans `~/.modal.toml`. En intégration
continue, où il n'y a ni navigateur ni fichier de configuration, `MODAL_TOKEN_ID`
et `MODAL_TOKEN_SECRET` les remplacent dans l'environnement du job.

Le moteur charge les poids **depuis le Hub, à la révision étiquetée** par le
pipeline de déploiement continu : la démonstration reste rejouable à l'identique
même si la branche par défaut bouge ensuite. Le cache Hugging Face est monté sur
un volume persistant, faute de quoi chaque démarrage à froid retéléchargerait
trois gigaoctets.

Le dépôt et la révision se choisissent par `HF_NAMESPACE`, `TRIAGE_MODEL_ID` et
`TRIAGE_MODEL_REVISION`, posées dans le shell au moment du `modal deploy` :
`deploy/modal_app.py` s'exécute sur le poste et lit son environnement dès
l'import, avant que le paquet du projet — et donc `.env` — ne soit chargé. Sans elles, le
déploiement sert le modèle et la révision épinglés dans le fichier.

**Le jour d'une démonstration**, réveiller l'endpoint cinq minutes avant de
passer : un démarrage à froid tient en une minute quand le cache est chaud, mais
une minute de silence devant un auditoire est longue. `GET /health` interroge
réellement le moteur et sert exactement à ça.

### Option A — conteneur vLLM managé

1. Publier le modèle fusionné sur le Hub :
   ```bash
   export HF_TOKEN="hf_..."      # jeton d'écriture
   export HF_NAMESPACE="votre-compte"
   uv run python scripts/08_publish_hf.py --what modele-final
   ```
2. Créer un endpoint d'inférence chez l'hébergeur, en choisissant le dépôt du
   modèle, un GPU de la classe L4 ou équivalent, et le conteneur vLLM.
3. Déployer la passerelle — l'image publiée par le pipeline sur GHCR — sur un
   service de conteneurs sans GPU, avec `TRIAGE_VLLM_URL` pointant vers
   l'endpoint d'inférence et `TRIAGE_API_KEY` en secret.

### Option B — machine virtuelle avec GPU

1. Provisionner une machine avec GPU, Docker et `nvidia-container-toolkit`.
2. Copier `models/qwen3-1.7b-triage-dpo-merged` sur la machine, ou laisser vLLM
   le télécharger depuis le Hub.
3. `docker compose -f deploy/docker-compose.yml up -d`
4. Exposer le port 8080 derrière un proxy inverse avec certificat TLS.

### Livrer une nouvelle version du modèle

Les poids sont téléversés depuis la machine d'entraînement — ils ne transitent
pas par l'intégration continue. C'est l'**étiquette de version** qui fait le lien.

```bash
uv run python scripts/08_publish_hf.py --what tout     # poids, dataset, cartes
git tag modele-v1.0.0 && git push origin modele-v1.0.0 # déclenche le pipeline
```

L'étiquette déclenche le job `modele` de `cd.yml` : republication du dataset et
des cartes, puis pose de l'étiquette `modele-v1.0.0` sur les dépôts du Hub. Le
job de déploiement relance ensuite `modal deploy` avec cette révision, et le
serveur d'inférence redémarre sur ces poids-là :

```bash
vllm serve <compte>/qwen3-1.7b-clinical-triage --revision modele-v1.0.0 \
  --served-model-name qwen3-1.7b-clinical-triage --max-model-len 1024
```

Épingler la révision plutôt que la branche par défaut est ce qui rend un
déploiement reproductible : `main` désigne un contenu qui change à chaque
publication, `modele-v1.0.0` désigne toujours les mêmes poids. C'est aussi ce qui
rend le retour arrière trivial — redéployer l'étiquette précédente.

### Ce que le dépôt GitHub doit connaître

Le déploiement automatique est **désactivé tant qu'on ne l'a pas armé**. Il ne
s'exécute que si la variable `DEPLOY_ENABLED` vaut `true` : sans elle, la chaîne
construit, teste et publie, mais ne touche à rien de vivant.

| Nom | Type | Rôle |
|---|---|---|
| `MODAL_TOKEN_ID` | secret | jeton Modal, obtenu par `modal token new` |
| `MODAL_TOKEN_SECRET` | secret | sa moitié secrète |
| `HF_TOKEN` | secret | republication du dataset et des cartes sur le Hub |
| `DEPLOY_ENABLED` | variable | `true` pour armer le déploiement |
| `DEPLOY_ENDPOINT_URL` | variable | adresse de la passerelle, sondée après déploiement |
| `HF_NAMESPACE` | variable | compte Hugging Face cible, si ce n'est pas celui par défaut |

La clé d'API du service et le jeton Hugging Face du conteneur ne passent pas par
GitHub : ils vivent dans le secret Modal `clinical-triage`, créé plus haut. GitHub
n'a besoin que de savoir déployer, pas de connaître ce que le service manipule.

## Configuration

Variables lues par la passerelle elle-même, dans `.env` à la racine ou dans
l'environnement du conteneur :

| Variable | Rôle | Défaut |
|---|---|---|
| `TRIAGE_API_KEY` | clé d'API attendue en en-tête `X-API-Key` | **aucun — le service refuse de démarrer sans** |
| `TRIAGE_ALLOW_ANONYMOUS` | autorise explicitement le mode ouvert, pour la démonstration | `false` |
| `TRIAGE_BACKEND` | `vllm` ou `transformers` | `vllm` |
| `TRIAGE_VLLM_URL` | adresse du serveur d'inférence | `http://localhost:8000` |
| `TRIAGE_VLLM_MODEL` | nom du modèle servi par vLLM | `qwen3-1.7b-clinical-triage` |
| `TRIAGE_VLLM_API_KEY` | clé présentée au moteur d'inférence | à défaut, `TRIAGE_API_KEY` |
| `TRIAGE_BASE_MODEL` | modèle chargé en mode `transformers` | modèle SFT fusionné |
| `TRIAGE_ADAPTER_DIR` | adaptateur appliqué en mode `transformers` | adaptateur DPO |
| `TRIAGE_MODEL_VERSION` | version inscrite au journal d'audit | `qwen3-1.7b-clinical-triage` |
| `TRIAGE_AUDIT_LOG` | chemin du journal d'audit | `logs/audit_triage.jsonl` |
| `TRIAGE_RATE_LIMIT` | requêtes par minute et par appelant | `60` |

Variables lues par les outils qui montent la pile, et non par le service. Elles
ne passent pas par le `.env` de la racine — `HF_NAMESPACE` exceptée, que
`config.py` lit aussi pour la publication sur le Hub : `docker compose -f
deploy/docker-compose.yml` cherche son fichier d'environnement dans `deploy/`,
et la commande `modal` charge `deploy/modal_app.py` hors du processus qui lit
`.env`. Les unes comme les autres se posent dans le shell, en préfixe de la
commande ou par `export`, comme dans les exemples plus haut.

| Variable | Lue par | Rôle | Défaut |
|---|---|---|---|
| `TRIAGE_VLLM_PORT` | `docker compose` | port du moteur publié sur l'hôte | `8000` |
| `TRIAGE_API_PORT` | `docker compose` | port de la passerelle publié sur l'hôte | `8080` |
| `TRIAGE_MODELS_DIR` | `docker compose` | dossier des poids, côté hôte, monté sur `/models` | `../models` |
| `TRIAGE_VLLM_IMAGE` | `docker compose` | image du moteur d'inférence | version épinglée par empreinte |
| `HF_NAMESPACE` | `modal deploy`, et `config.py` pour la publication | compte Hugging Face d'où le moteur tire les poids | `BenoitJT-GIRARD` |
| `TRIAGE_MODEL_ID` | `modal deploy` | dépôt du modèle servi | `<HF_NAMESPACE>/qwen3-1.7b-clinical-triage` |
| `TRIAGE_MODEL_REVISION` | `modal deploy` | révision épinglée des poids | `modele-v1.0.0` |

`docker compose` interpole aussi `TRIAGE_API_KEY` et `TRIAGE_RATE_LIMIT` avant
de les passer au conteneur : sur la pile en conteneurs, ces deux-là viennent
donc du shell elles aussi. Sans `TRIAGE_API_KEY`, la pile s'arrête avant de
démarrer, en le disant.

`.env.example` à la racine liste les variables du premier tableau, ainsi que le
jeton et le compte Hugging Face des scripts de publication et les emplacements
de cache et de suivi. Copié en `.env`, il est lu par `config.py` — sans jamais
écraser une variable déjà posée dans l'environnement, pour qu'un hébergeur garde
la main sur ses propres secrets. Le fichier est exclu de git et du contexte de
construction Docker.

## Sécurité

- **Clé d'API obligatoire.** Le service refuse de démarrer sans elle. Le mode
  ouvert existe pour la démonstration locale et doit être demandé explicitement.
- **Comparaison à temps constant**, pour ne pas laisser reconstituer la clé par la
  durée de réponse.
- **Quota par appelant** sur fenêtre glissante : chaque requête mobilise un GPU,
  et un endpoint sans quota est trivialement saturable.
- **Secrets** hors du dépôt, en variables d'environnement chiffrées côté hébergeur.
- **Conteneur** exécuté sans privilège, image de base épinglée par empreinte,
  dépendances épinglées jusqu'à la dernière transitive et auditées en intégration
  continue, sans dérogation.
- **Moteur d'inférence** lié à la boucle locale dans la pile en conteneurs, et
  protégé par une clé dès qu'il est joignable autrement : il n'a ni quota ni
  journal d'audit, c'est la passerelle qui les porte.
- **TLS** à la charge du proxy inverse ou de l'hébergeur.

## Exploitation

| Indicateur | Seuil d'alerte | Réaction |
|---|---|---|
| Sonde `/health` | deux échecs consécutifs | redémarrage du conteneur, alerte |
| Latence 95ᵉ centile | dépassement du double de la référence | vérification de la charge GPU |
| Réponses hors format | plus de 2 % sur une heure | gel de la version, retour à la précédente |
| Désaccord modèle / règle | plus de 25 % sur une journée | revue clinique de l'échantillon |
| Taux d'erreur HTTP | plus de 1 % | alerte d'exploitation |

Le **journal d'audit** est monté sur un volume persistant nommé `audit` : sans
cela, la traçabilité exigée pour les audits médicaux disparaîtrait à chaque
redéploiement. Un dossier de l'hôte ne convient pas — Docker le monte au compte
de root, et le service, qui tourne sans privilège, ne peut pas y écrire ; il
refuse alors de démarrer, en le disant. Le volume survit à `docker compose
down` ; `docker compose down -v` l'efface.

```bash
docker compose -f deploy/docker-compose.yml cp api:/app/logs/audit_triage.jsonl .
```

Prévoir sa rotation et sa centralisation, et respecter la durée de conservation
inscrite dans chaque ligne.

Le **taux de désaccord entre le modèle et la règle explicite** est l'indicateur de
production le plus utile : il ne nécessite aucune étiquette, se calcule en
continu, et détecte une dérive avant qu'un patient n'en pâtisse.

## Limites d'usage

Aide à la décision destinée au personnel soignant, sous supervision humaine
obligatoire. L'agent ne pose pas de diagnostic et ne remplace pas une évaluation
clinique. Le catalogue de présentations qui a servi à construire les données n'a
pas été validé par un médecin urgentiste : ce service ne doit pas être utilisé en
situation réelle avant que ses critères de mise en service soient
satisfaits.
