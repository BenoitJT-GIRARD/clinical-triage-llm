# Passerelle API de triage : FastAPI devant un serveur vLLM.
#
# L'image ne contient ni torch ni poids de modèle. Elle valide les requêtes,
# conduit le questionnaire adaptatif, applique la règle explicite de contrôle,
# anonymise et journalise, puis délègue la génération à vLLM. Cette séparation
# permet de redéployer la passerelle sans redéployer le modèle, et inversement.
#
# L'image de base est épinglée par empreinte et les dépendances Python le sont
# jusqu'à la dernière transitive : deux constructions de la même révision
# produisent la même image.
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

# Sorties non tamponnées : les traces du conteneur arrivent en temps réel dans
# le collecteur de journaux.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dépendances épinglées, installées avant le code : la couche est réutilisée
# tant que le fichier ne change pas. Ce fichier est généré depuis
# `requirements-api.in`, qui porte les contraintes et la procédure.
COPY infra/requirements-api.txt /app/infra/requirements-api.txt
RUN pip install --no-cache-dir -r /app/infra/requirements-api.txt

# Code du paquet.
COPY src/clinical_triage /app/src/clinical_triage
ENV PYTHONPATH=/app/src

# Utilisateur sans privilège, propriétaire du répertoire des journaux d'audit.
RUN useradd --create-home --uid 10001 triage \
    && mkdir -p /app/logs \
    && chown -R triage:triage /app
USER triage

ENV TRIAGE_BACKEND=vllm \
    TRIAGE_VLLM_URL=http://vllm:8000 \
    TRIAGE_AUDIT_LOG=/app/logs/audit_triage.jsonl

EXPOSE 8080

# La sonde laisse au service le temps de charger les modèles spaCy avant de
# compter les échecs.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=60s \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8080/health', timeout=4).json()['status']=='ok' else 1)"

CMD ["uvicorn", "clinical_triage.serving.api:app", "--host", "0.0.0.0", "--port", "8080"]
