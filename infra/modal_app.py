"""Déploiement de l'endpoint de démonstration sur Modal.

Deux applications, exactement comme la pile locale de `docker-compose.yml` :

- **`moteur`** — un conteneur GPU qui sert le modèle fusionné avec vLLM, par son
  API compatible OpenAI. Il s'éteint de lui-même après quelques minutes sans
  requête, et se rallume à la suivante ;
- **`passerelle`** — la même application FastAPI que celle de l'image Docker,
  exposée telle quelle. Pas de GPU, pas de torch, pas de poids.

La séparation n'est pas un choix propre à Modal : c'est celle du projet. On ne
réécrit donc rien, on rebranche.

## Pourquoi charger les poids depuis le Hub plutôt que de les embarquer

Le modèle vit déjà sur le Hugging Face Hub, étiqueté par le pipeline de
déploiement continu. Le conteneur le lit à une **révision précise** : la
démonstration reste reproductible même si la branche par défaut bouge ensuite.
Le cache est monté sur un volume persistant, sinon chaque démarrage à froid
retélécharge trois gigaoctets — la différence entre quarante secondes et quatre
minutes.

## Déploiement

    modal secret create clinical-triage TRIAGE_API_KEY=... HF_TOKEN=...
    modal deploy infra/modal_app.py

La commande affiche deux adresses. Reporter celle du moteur dans le secret, puis
redéployer la passerelle :

    modal secret create clinical-triage --force \
        TRIAGE_API_KEY=... HF_TOKEN=... TRIAGE_VLLM_URL=https://...moteur.modal.run
    modal deploy infra/modal_app.py

Le détour par le secret est volontaire : la passerelle reçoit l'adresse du moteur
par configuration, comme chez n'importe quel hébergeur, et non par une API propre
à Modal. Le même code tourne ainsi en local, sur Modal, ou ailleurs.
"""

import os
from pathlib import Path

import modal

# Révision publiée par le job `modele` de `.github/workflows/cd.yml`, qui la passe
# dans l'environnement au moment du déploiement. Sans elle, on épingle la version
# publiée : la démonstration reste rejouable à l'identique.
# Le `or` n'est pas un raccourci, et il y en a deux : une variable de dépôt
# GitHub non définie arrive dans l'environnement comme chaîne vide, et non comme
# variable absente. Sans eux, l'identifiant demandé au Hub commencerait par une
# barre oblique. C'est la même règle que `config.hub_namespace`.
COMPTE = os.environ.get("HF_NAMESPACE") or "BenoitJT-GIRARD"
MODELE = os.environ.get("TRIAGE_MODEL_ID") or f"{COMPTE}/qwen3-1.7b-clinical-triage"
REVISION = os.environ.get("TRIAGE_MODEL_REVISION") or "modele-v1.0.0"

PORT_VLLM = 8000
MINUTE = 60

app = modal.App("clinical-triage")

# Le secret porte la clé d'API du service, le jeton Hugging Face — indispensable
# même pour un dépôt public, sans quoi le téléchargement tombe dans le quota
# anonyme partagé par adresse IP — et l'adresse du moteur.
secret = modal.Secret.from_name("clinical-triage")

cache_hf = modal.Volume.from_name("clinical-triage-cache-hf", create_if_missing=True)
cache_vllm = modal.Volume.from_name("clinical-triage-cache-vllm", create_if_missing=True)

# Environnement gravé dans l'image du moteur. Les deux références du modèle y
# figurent, plutôt que d'être lues par le conteneur : celui-ci réimporte ce
# fichier au démarrage, dans un environnement qui ne contient rien de celui du
# déploiement. Ce qui n'est pas gravé ici serait perdu.
ENVIRONNEMENT_MOTEUR = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_HOME": "/cache/hf",
    "TRIAGE_MODEL_ID": MODELE,
    "TRIAGE_MODEL_REVISION": REVISION,
}

image_moteur = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.29.0", "huggingface_hub[hf_transfer]>=1.0")
    .env(ENVIRONNEMENT_MOTEUR)
)

image_passerelle = (
    modal.Image.debian_slim(python_version="3.12")
    # Chemin ancré sur ce fichier : `modal deploy` se lance depuis la racine
    # du dépôt, pas depuis `infra/`.
    .pip_install_from_requirements(str(Path(__file__).parent / "requirements-api.txt"))
    .add_local_python_source("clinical_triage")
)


@app.server(
    image=image_moteur,
    gpu="L4",
    # Quinze minutes sans requête et le conteneur s'éteint : sur un compte à
    # crédits, une démonstration oubliée en marche coûte le reste du mois.
    scaledown_window=15 * MINUTE,
    startup_timeout=10 * MINUTE,
    volumes={"/cache/hf": cache_hf, "/root/.cache/vllm": cache_vllm},
    secrets=[secret],
    port=PORT_VLLM,
    # Le moteur reçoit une adresse publique, comme la passerelle : Modal expose
    # tout serveur ainsi. Ce n'est pas pour autant une porte ouverte — vLLM
    # réclame lui-même la clé du service, que la passerelle lui présente. Sans
    # cela, l'adresse suffirait à obtenir une inférence GPU gratuite, sans quota,
    # sans anonymisation et sans ligne au journal d'audit, aux frais du compte.
    unauthenticated=True,
)
class Moteur:
    """Serveur vLLM servant le modèle de triage fusionné."""

    @modal.enter()
    def demarrer(self):
        import subprocess

        commande = [
            "vllm",
            "serve",
            os.environ["TRIAGE_MODEL_ID"],
            "--revision",
            os.environ["TRIAGE_MODEL_REVISION"],
            "--served-model-name",
            "qwen3-1.7b-clinical-triage",
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT_VLLM),
            "--dtype",
            "bfloat16",
            # L'invite tient en 550 jetons et la réponse en 220 : 1024 suffit, et
            # une fenêtre plus large réserverait du cache d'attention pour rien.
            "--max-model-len",
            "1024",
            "--gpu-memory-utilization",
            "0.85",
            # La clé que la passerelle présentera. Elle vient du secret Modal,
            # comme celle de la passerelle : c'est la même.
            "--api-key",
            os.environ["TRIAGE_API_KEY"],
        ]
        # La clé ne va pas au journal : on imprime la commande sans sa valeur.
        print(" ".join(commande[:-1] + ["<clé du secret>"]), flush=True)
        self.processus = subprocess.Popen(commande)

    @modal.exit()
    def arreter(self):
        self.processus.terminate()


@app.function(image=image_passerelle, secrets=[secret], scaledown_window=5 * MINUTE)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def passerelle():
    """Expose l'API de triage du projet, sans la modifier."""
    from clinical_triage.serving.api import app as api

    return api
