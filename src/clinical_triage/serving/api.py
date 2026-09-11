"""API de l'agent de triage.

Trois points d'entrée : la sonde de santé, le questionnaire adaptatif et le
triage. Deux moteurs d'inférence interchangeables derrière le même contrat :

- `vllm` — le mode de déploiement. L'API est une passerelle légère, sans torch,
  qui délègue la génération à un serveur vLLM ;
- `transformers` — le mode démonstration. Le modèle est chargé dans le processus,
  ce qui permet de faire tourner la démonstration sur un poste sans vLLM.

Les précautions d'exploitation sont regroupées au démarrage plutôt que dispersées
dans les points d'entrée :

- **la clé d'API est obligatoire.** Le service refuse de démarrer sans elle, sauf
  si l'exploitant demande explicitement le mode ouvert. Une authentification qui
  se désactive toute seule quand une variable manque n'est pas une authentification ;
- **la comparaison des clés est à temps constant**, pour ne pas laisser fuiter la
  clé octet par octet par la durée de la réponse ;
- **le débit est limité par appelant**, parce que chaque requête mobilise un GPU ;
- **la sonde de santé interroge réellement le moteur.** Une sonde qui répond
  toujours « ok » ne déclenche aucun redémarrage le jour où l'inférence tombe.
"""

from __future__ import annotations

import os
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from clinical_triage.config import PATHS, SERVING, TRIAGE
from clinical_triage.data.triage_rules import classify, explain
from clinical_triage.prompts import (
    completion_payload,
    extract_level,
    parse_response,
    truncate_to_answer,
)
from clinical_triage.serving.audit import (
    new_request_id,
    record_interaction,
    verifier_que_le_journal_est_ecrivable,
)
from clinical_triage.serving.questionnaire import compile_symptoms, next_question
from clinical_triage.serving.schemas import (
    HealthReply,
    QuestionnaireReply,
    QuestionnaireRequest,
    TriageReply,
    TriageRequest,
)
from clinical_triage.utils import chemin_pour_journal, get_logger

logger = get_logger("api")


@dataclass(frozen=True)
class Settings:
    """Configuration du service, lue une fois au démarrage."""

    backend: str
    base_model: str
    adapter_dir: str
    model_version: str
    api_key: str
    allow_anonymous: bool
    vllm_url: str
    vllm_model: str
    vllm_api_key: str
    requetes_par_minute: int

    @property
    def entetes_vllm(self) -> dict[str, str]:
        """En-têtes envoyés au moteur d'inférence.

        Quand le moteur est joignable autrement que par la boucle locale — c'est
        le cas sur un hébergeur à la demande — il doit demander une clé, sinon il
        offre une inférence GPU gratuite, sans quota et sans trace d'audit à qui
        trouve son adresse. vLLM sait le faire avec `--api-key` ; la passerelle
        la lui présente ici.
        """
        return {"Authorization": f"Bearer {self.vllm_api_key}"} if self.vllm_api_key else {}

    @staticmethod
    def from_env() -> Settings:
        """Construit la configuration à partir des variables d'environnement."""
        return Settings(
            backend=os.getenv("TRIAGE_BACKEND", "vllm"),
            base_model=os.getenv("TRIAGE_BASE_MODEL", str(PATHS.sft_merged)),
            adapter_dir=os.getenv("TRIAGE_ADAPTER_DIR", str(PATHS.dpo_adapter)),
            model_version=os.getenv("TRIAGE_MODEL_VERSION", SERVING.served_model_name),
            api_key=os.getenv("TRIAGE_API_KEY", ""),
            allow_anonymous=os.getenv("TRIAGE_ALLOW_ANONYMOUS", "").lower() == "true",
            vllm_url=os.getenv("TRIAGE_VLLM_URL", "http://localhost:8000"),
            vllm_model=os.getenv("TRIAGE_VLLM_MODEL", SERVING.served_model_name),
            # À défaut de clé propre au moteur, on lui présente celle du service :
            # sur la pile locale le moteur n'en demande pas et l'en-tête est
            # ignoré, chez un hébergeur il la réclame et la passerelle l'a.
            vllm_api_key=os.getenv("TRIAGE_VLLM_API_KEY") or os.getenv("TRIAGE_API_KEY", ""),
            requetes_par_minute=int(os.getenv("TRIAGE_RATE_LIMIT", "60")),
        )


@dataclass
class RateLimiter:
    """Limitation de débit par appelant, sur une fenêtre glissante d'une minute."""

    requetes_par_minute: int
    _historique: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))
    _dernier_oubli: float = field(default_factory=time.monotonic)

    def _oublier_les_appelants_silencieux(self, maintenant: float) -> None:
        """Retire du suivi les appelants qui n'ont rien demandé depuis une minute.

        Sans cela, le dictionnaire conserve une entrée par appelant vu depuis le
        démarrage. En mode ouvert, où la clé est l'adresse de l'appelant, il
        grossit indéfiniment : un compteur censé protéger le service finirait
        par le mettre en difficulté lui-même. Le balayage ne coûte qu'une fois
        par minute.
        """
        if maintenant - self._dernier_oubli < 60:
            return
        self._dernier_oubli = maintenant
        silencieux = [
            appelant
            for appelant, recentes in self._historique.items()
            if not recentes or maintenant - recentes[-1] > 60
        ]
        for appelant in silencieux:
            del self._historique[appelant]

    def autorise(self, appelant: str) -> bool:
        """Enregistre une requête et dit si elle reste dans le quota."""
        maintenant = time.monotonic()
        self._oublier_les_appelants_silencieux(maintenant)
        recentes = self._historique[appelant]
        while recentes and maintenant - recentes[0] > 60:
            recentes.popleft()
        if len(recentes) >= self.requetes_par_minute:
            return False
        recentes.append(maintenant)
        return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lit la configuration, vérifie la sécurité et charge le moteur d'inférence."""
    settings = Settings.from_env()
    if not settings.api_key and not settings.allow_anonymous:
        raise RuntimeError(
            "TRIAGE_API_KEY n'est pas définie. Renseignez une clé, ou demandez explicitement "
            "le mode ouvert avec TRIAGE_ALLOW_ANONYMOUS=true (démonstration locale uniquement)."
        )
    if not settings.api_key:
        logger.warning("Service démarré sans clé d'API : mode ouvert, réservé à la démonstration.")

    # Le journal d'audit est ouvert ici, à vide, pour que son indisponibilité
    # arrête le démarrage plutôt que la première requête.
    logger.info(
        "Journal d'audit : %s", chemin_pour_journal(verifier_que_le_journal_est_ecrivable())
    )

    app.state.settings = settings
    app.state.limiteur = RateLimiter(settings.requetes_par_minute)
    app.state.agent = None
    app.state.modele_charge = settings.model_version

    if settings.backend == "transformers":
        from clinical_triage.inference import TriageAgent

        base = settings.base_model
        if not os.path.isdir(base):
            raise RuntimeError(
                f"Modèle introuvable : {base}. Lancez scripts/04_merge_and_export.py, ou "
                "pointez TRIAGE_BASE_MODEL vers un modèle disponible."
            )
        adaptateur = settings.adapter_dir if os.path.isdir(settings.adapter_dir) else None
        if adaptateur is None:
            # Le service reste utilisable, mais sert alors le modèle supervisé
            # seul : le journal d'audit doit en porter la trace, sans quoi les
            # réponses seraient attribuées à un modèle qui ne les a pas produites.
            logger.warning(
                "Adaptateur introuvable (%s) : le modèle est servi sans alignement.",
                chemin_pour_journal(settings.adapter_dir),
            )
        app.state.agent = TriageAgent(adapter_dir=adaptateur, base_model=base)
        app.state.modele_charge = app.state.agent.description
        logger.info("Moteur transformers prêt : %s", app.state.modele_charge)
    else:
        logger.info("Moteur vLLM : génération déléguée à %s", settings.vllm_url)

    yield
    app.state.agent = None


app = FastAPI(
    title="Agent IA de triage médical",
    description=(
        "Aide à la décision pour le triage des urgences. Le niveau proposé ne remplace "
        "pas l'évaluation d'un soignant ; en cas de signe vital engagé, appeler le 15 (SAMU)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Vérifie la clé d'API, en temps constant, puis applique la limite de débit."""
    settings: Settings = request.app.state.settings
    if settings.api_key:
        fournie = x_api_key or ""
        if not secrets.compare_digest(fournie, settings.api_key):
            raise HTTPException(
                status_code=401,
                detail="Clé d'API absente ou invalide. Renseignez l'en-tête X-API-Key.",
            )
    # L'en-tête ne sert d'identité de comptage que s'il vient d'être vérifié. En
    # mode ouvert, il n'est vérifié par personne : un appelant qui en change à
    # chaque requête obtiendrait un seau neuf à chaque fois, et le quota — seul
    # garde-fou restant dans ce mode — ne compterait plus rien.
    adresse = request.client.host if request.client else "inconnu"
    appelant = x_api_key if (settings.api_key and x_api_key) else adresse
    if not request.app.state.limiteur.autorise(appelant):
        # Annoncer le quota évite à l'intégrateur de le découvrir par tâtonnement.
        raise HTTPException(
            status_code=429,
            detail=(
                f"Quota de {settings.requetes_par_minute} requêtes par minute dépassé. "
                "Réessayez dans une minute."
            ),
        )


# Nombre de caractères au-delà duquel la passerelle borne la description.
#
# Le moteur vLLM sert une fenêtre de 1 024 jetons, dont 220 sont réservés à la
# réponse et environ 224 à la consigne système : il reste 580 jetons pour le
# récit du patient. Sur du français clinique, un jeton vaut environ 2,7
# caractères ; on retient 1 500, soit une marge confortable sous la borne.
#
# La passerelle n'embarque pas de tokenizer — c'est tout l'intérêt d'une
# passerelle légère — d'où cette approximation par les caractères. Le moteur
# local, lui, borne en jetons parce qu'il en a un.
CARACTERES_MAXIMUM = 1500


def borner_en_caracteres(texte: str) -> tuple[str, bool]:
    """Borne une description trop longue, et dit si elle l'a été."""
    if len(texte) <= CARACTERES_MAXIMUM:
        return texte, False
    return texte[:CARACTERES_MAXIMUM], True


def _generer_transformers(request: Request, symptomes: str) -> tuple[str, str | None, float, bool]:
    """Génère la réponse avec le moteur local."""
    reponse = request.app.state.agent.generate(symptomes)
    return reponse.text, reponse.level, reponse.latency_ms, reponse.description_tronquee


def _generer_vllm(request: Request, symptomes: str) -> tuple[str, str | None, float, bool]:
    """Génère la réponse via le serveur vLLM, par son API compatible OpenAI."""
    import httpx

    settings: Settings = request.app.state.settings
    symptomes, tronquee = borner_en_caracteres(symptomes)
    charge_utile = completion_payload(symptomes, settings.vllm_model)
    debut = time.perf_counter()
    try:
        with httpx.Client(timeout=60) as client:
            reponse = client.post(
                f"{settings.vllm_url}/v1/completions",
                json=charge_utile,
                headers=settings.entetes_vllm,
            )
            reponse.raise_for_status()
            texte = reponse.json()["choices"][0]["text"]
    except httpx.HTTPError as erreur:
        # La cause technique va au journal, pas à l'appelant : elle contient
        # l'adresse interne du serveur d'inférence, que rien ne justifie
        # d'exposer. L'appelant reçoit ce qu'il peut en faire — réessayer.
        logger.error("Appel au serveur vLLM en échec : %s", erreur)
        raise HTTPException(
            status_code=503,
            detail=(
                "Le moteur d'inférence ne répond pas. Le triage est momentanément "
                "indisponible ; réessayez dans quelques instants."
            ),
        ) from erreur
    latence = (time.perf_counter() - debut) * 1000
    texte = truncate_to_answer(texte)
    return texte, extract_level(texte), latence, tronquee


@app.get("/health", response_model=HealthReply)
def health(request: Request) -> HealthReply:
    """Sonde de disponibilité : interroge réellement le moteur d'inférence."""
    settings: Settings = request.app.state.settings
    if settings.backend == "transformers":
        disponible = request.app.state.agent is not None
        detail = None if disponible else "Le modèle n'est pas chargé."
    else:
        import httpx

        try:
            with httpx.Client(timeout=3) as client:
                client.get(
                    f"{settings.vllm_url}/v1/models", headers=settings.entetes_vllm
                ).raise_for_status()
            disponible, detail = True, None
        except Exception as erreur:  # noqa: BLE001 - toute panne du moteur doit être signalée
            # Le message d'httpx contient l'adresse interrogée — sur Modal, c'est
            # l'adresse du moteur. La sonde n'exige ni clé ni quota : elle doit
            # donc dire qu'il y a panne, jamais où. La cause technique part au
            # journal, comme pour la génération.
            logger.error("Sonde vLLM en échec : %s", erreur)
            disponible, detail = False, "Le moteur d'inférence ne répond pas."
    return HealthReply(
        status="ok" if disponible else "degraded",
        backend=settings.backend,
        model_loaded=disponible,
        model_version=request.app.state.modele_charge,
        detail=detail,
    )


@app.post(
    "/questionnaire/next",
    response_model=QuestionnaireReply,
    dependencies=[Depends(require_api_key)],
)
def questionnaire_next(req: QuestionnaireRequest) -> QuestionnaireReply:
    """Renvoie la prochaine question du questionnaire adaptatif, ou la fin de collecte."""
    etape = next_question(req.chief_complaint, req.answers)
    return QuestionnaireReply(
        next_question_id=etape.identifiant,
        next_question=etape.texte,
        theme=etape.theme,
        finished=etape.termine,
        compiled_symptoms=compile_symptoms(req.chief_complaint, req.answers),
    )


@app.post("/triage", response_model=TriageReply, dependencies=[Depends(require_api_key)])
def triage(req: TriageRequest, request: Request) -> TriageReply:
    """Évalue le niveau de priorité, le compare à la règle explicite et trace l'interaction."""
    symptomes = req.symptoms.strip()
    if req.patient_age is not None:
        symptomes = f"Patient de {req.patient_age} ans. {symptomes}"

    generer = (
        _generer_transformers
        if request.app.state.settings.backend == "transformers"
        else _generer_vllm
    )
    texte, niveau, latence, description_tronquee = generer(request, symptomes)

    niveau_regle = classify(symptomes)
    raisons_regle = explain(symptomes)
    parties = parse_response(texte)

    identifiant = new_request_id()
    record_interaction(
        request_id=identifiant,
        symptomes=symptomes,
        niveau=niveau,
        reponse=texte,
        latence_ms=latence,
        modele=request.app.state.modele_charge,
        moteur=request.app.state.settings.backend,
        niveau_regle=niveau_regle,
        raisons_regle=raisons_regle,
        description_tronquee=description_tronquee,
    )

    return TriageReply(
        level=niveau,
        level_label=TRIAGE.labels_fr.get(niveau) if niveau else None,
        justification=parties["justification"],
        recommendation=parties["recommendation"],
        rule_level=niveau_regle,
        rule_reasons=raisons_regle,
        agreement=niveau == niveau_regle,
        raw_response=texte,
        description_truncated=description_tronquee,
        latency_ms=round(latence, 1),
        request_id=identifiant,
        model_version=request.app.state.modele_charge,
    )
