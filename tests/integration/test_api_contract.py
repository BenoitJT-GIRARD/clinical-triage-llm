"""Tests de l'API de triage.

Les tests n'ont besoin d'aucun modèle : le moteur vLLM est remplacé par une
fonction de génération contrôlée. Ce qui est vérifié ici, ce n'est pas la
qualité des réponses — c'est le contrat de service : authentification, limitation
de débit, sonde de santé, traçabilité et forme de la réponse.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from clinical_triage.serving import api as module_api
from clinical_triage.serving.api import RateLimiter

CLE = "cle-de-test"

REPONSE_MODELE = (
    "Niveau de priorité : URGENCE_VITALE\n"
    "Justification : Douleur thoracique avec sueurs chez un patient à risque.\n"
    "Recommandation : Prise en charge immédiate et appel du 15 (SAMU)."
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Service configuré en mode vLLM, avec une génération simulée."""
    monkeypatch.setenv("TRIAGE_BACKEND", "vllm")
    monkeypatch.setenv("TRIAGE_API_KEY", CLE)
    monkeypatch.setenv("TRIAGE_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("TRIAGE_RATE_LIMIT", "5")
    monkeypatch.setattr(
        module_api,
        "_generer_vllm",
        lambda request, symptomes: (REPONSE_MODELE, "URGENCE_VITALE", 42.0, False),
    )
    with TestClient(module_api.app) as testeur:
        yield testeur


def test_le_service_refuse_de_demarrer_sans_cle(monkeypatch):
    """Une authentification qui se désactive quand une variable manque n'en est pas une."""
    monkeypatch.setenv("TRIAGE_BACKEND", "vllm")
    monkeypatch.delenv("TRIAGE_API_KEY", raising=False)
    monkeypatch.delenv("TRIAGE_ALLOW_ANONYMOUS", raising=False)
    with pytest.raises(RuntimeError, match="TRIAGE_API_KEY"), TestClient(module_api.app):
        pass


def test_le_mode_ouvert_doit_etre_demande_explicitement(monkeypatch, tmp_path):
    monkeypatch.setenv("TRIAGE_BACKEND", "vllm")
    monkeypatch.delenv("TRIAGE_API_KEY", raising=False)
    monkeypatch.setenv("TRIAGE_ALLOW_ANONYMOUS", "true")
    monkeypatch.setenv("TRIAGE_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    with TestClient(module_api.app) as testeur:
        assert (
            testeur.post("/questionnaire/next", json={"chief_complaint": "toux"}).status_code == 200
        )


def test_la_sonde_signale_un_moteur_injoignable(client):
    """Une sonde toujours verte ne déclenche aucun redémarrage le jour de la panne."""
    reponse = client.get("/health")
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["backend"] == "vllm"
    assert corps["status"] == "degraded"
    assert corps["model_loaded"] is False
    assert corps["detail"]


def test_la_sonde_ne_publie_pas_l_adresse_du_moteur(client):
    """La sonde n'exige ni clé ni quota : elle dit qu'il y a panne, jamais où.

    Le message d'httpx porte l'URL interrogée. Sur Modal, cette URL est
    précisément ce qui protège le moteur, et la sonde la rendait publique.
    """
    detail = client.get("/health").json()["detail"]
    adresse = module_api.app.state.settings.vllm_url
    assert adresse not in detail
    assert "http" not in detail


def test_le_triage_exige_une_cle(client):
    assert client.post("/triage", json={"symptoms": "douleur thoracique"}).status_code == 401
    assert (
        client.post(
            "/triage", json={"symptoms": "douleur thoracique"}, headers={"X-API-Key": "mauvaise"}
        ).status_code
        == 401
    )


def test_le_triage_renvoie_une_reponse_structuree(client):
    reponse = client.post(
        "/triage",
        json={"symptoms": "Douleur thoracique et sueurs depuis 20 minutes.", "patient_age": 62},
        headers={"X-API-Key": CLE},
    )
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["level"] == "URGENCE_VITALE"
    assert corps["level_label"].startswith("Urgence maximale")
    assert corps["justification"]
    assert corps["recommendation"]
    assert corps["request_id"]
    assert corps["latency_ms"] == 42.0


def test_la_reponse_expose_l_avis_de_la_regle(client):
    """Quand le modèle et la règle divergent, l'accueil doit pouvoir le voir."""
    reponse = client.post(
        "/triage",
        json={"symptoms": "Douleur thoracique et sueurs depuis 20 minutes."},
        headers={"X-API-Key": CLE},
    ).json()
    assert reponse["rule_level"] == "URGENCE_VITALE"
    assert reponse["rule_reasons"]
    assert reponse["agreement"] is True


def test_un_desaccord_entre_le_modele_et_la_regle_est_signale(client):
    reponse = client.post(
        "/triage",
        json={"symptoms": "Rhume banal depuis deux jours, pas de fièvre."},
        headers={"X-API-Key": CLE},
    ).json()
    assert reponse["level"] == "URGENCE_VITALE"  # réponse simulée
    assert reponse["rule_level"] == "CONSULTATION_DIFFEREE"
    assert reponse["agreement"] is False


def test_chaque_triage_est_trace(client, tmp_path):
    client.post("/triage", json={"symptoms": "Douleur thoracique."}, headers={"X-API-Key": CLE})
    lignes = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lignes) == 1
    trace = json.loads(lignes[0])
    assert trace["niveau"] == "URGENCE_VITALE"
    assert trace["moteur"] == "vllm"
    assert trace["symptomes_anonymises"]


def test_une_description_vide_est_refusee(client):
    reponse = client.post("/triage", json={"symptoms": "  "}, headers={"X-API-Key": CLE})
    assert reponse.status_code == 422


def test_la_limitation_de_debit_protege_le_gpu(client):
    """Chaque appel mobilise un GPU : un endpoint sans quota est saturable."""
    codes = [
        client.post(
            "/triage", json={"symptoms": "Douleur thoracique."}, headers={"X-API-Key": CLE}
        ).status_code
        for _ in range(7)
    ]
    assert codes[:5] == [200] * 5
    assert codes[5] == 429


def test_le_limiteur_oublie_les_appelants_silencieux():
    """Un compteur qui protège le service ne doit pas grossir sans fin.

    En mode ouvert la clé est l'adresse de l'appelant : sans oubli, le service
    garderait une entrée par adresse vue depuis son démarrage.
    """
    limiteur = RateLimiter(requetes_par_minute=10)
    for numero in range(50):
        limiteur.autorise(f"10.0.0.{numero}")
    assert len(limiteur._historique) == 50

    # On se place une minute et une seconde plus tard, sans attendre : le
    # limiteur lit l'horloge monotone, il suffit de la décaler.
    plus_tard = time.monotonic() + 61
    limiteur._oublier_les_appelants_silencieux(plus_tard)
    assert limiteur._historique == {}


def test_le_questionnaire_adapte_ses_questions_au_motif(client):
    reponse = client.post(
        "/questionnaire/next",
        json={
            "chief_complaint": "gêne dans la poitrine à l'effort",
            "answers": {"conscience": "oui", "respiration": "non", "saignement": "non"},
        },
        headers={"X-API-Key": CLE},
    ).json()
    assert reponse["theme"] == "douleur_thoracique"
    assert reponse["next_question_id"] == "irradiation"
    assert reponse["finished"] is False


def test_le_questionnaire_s_arrete_sur_un_signe_vital(client):
    reponse = client.post(
        "/questionnaire/next",
        json={"chief_complaint": "douleur thoracique violente"},
        headers={"X-API-Key": CLE},
    ).json()
    assert reponse["finished"] is True
    assert reponse["next_question_id"] is None


def test_le_contrat_openapi_est_publie(client):
    schema = client.get("/openapi.json").json()
    assert "/triage" in schema["paths"]
    assert "/questionnaire/next" in schema["paths"]
    assert "/health" in schema["paths"]


def test_la_passerelle_presente_une_cle_au_moteur(monkeypatch):
    """Sans elle, l'adresse du moteur suffit à obtenir une inférence GPU.

    Gratuite, sans quota, sans anonymisation et sans ligne au journal d'audit.
    En local le moteur n'en demande pas et l'en-tête est ignoré ; chez un
    hébergeur il la réclame, et la passerelle doit l'avoir.
    """
    monkeypatch.setenv("TRIAGE_API_KEY", "cle-du-service")
    monkeypatch.delenv("TRIAGE_VLLM_API_KEY", raising=False)
    settings = module_api.Settings.from_env()
    assert settings.entetes_vllm == {"Authorization": "Bearer cle-du-service"}


def test_une_cle_propre_au_moteur_prime_sur_celle_du_service(monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEY", "cle-du-service")
    monkeypatch.setenv("TRIAGE_VLLM_API_KEY", "cle-du-moteur")
    assert module_api.Settings.from_env().entetes_vllm["Authorization"] == "Bearer cle-du-moteur"


def test_sans_aucune_cle_aucun_entete_n_est_envoye(monkeypatch):
    """Le mode ouvert de démonstration ne doit pas envoyer un « Bearer » vide."""
    monkeypatch.delenv("TRIAGE_API_KEY", raising=False)
    monkeypatch.delenv("TRIAGE_VLLM_API_KEY", raising=False)
    assert module_api.Settings.from_env().entetes_vllm == {}


def test_en_mode_ouvert_le_quota_ne_se_contourne_pas_en_changeant_d_entete(monkeypatch, tmp_path):
    """L'en-tête ne compte comme identité que lorsqu'il a été vérifié.

    En mode ouvert, personne ne le vérifie : il servait quand même de clé au
    seau de comptage, si bien qu'un appelant qui en changeait à chaque requête
    obtenait un seau neuf à chaque fois. Le quota est le dernier garde-fou de ce
    mode, et il ne comptait plus rien.
    """
    monkeypatch.setenv("TRIAGE_BACKEND", "vllm")
    monkeypatch.delenv("TRIAGE_API_KEY", raising=False)
    monkeypatch.setenv("TRIAGE_ALLOW_ANONYMOUS", "true")
    monkeypatch.setenv("TRIAGE_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("TRIAGE_RATE_LIMIT", "3")
    with TestClient(module_api.app) as testeur:
        codes = [
            testeur.post(
                "/questionnaire/next",
                json={"chief_complaint": "toux"},
                headers={"X-API-Key": f"forge-{numero}"},
            ).status_code
            for numero in range(6)
        ]
    assert codes[:3] == [200, 200, 200]
    assert 429 in codes[3:]


def test_le_questionnaire_refuse_un_corps_demesure(client):
    """Le champ de réponses était le seul texte de l'API sans borne.

    Tout y est concaténé en une chaîne unique, relue par la règle de triage :
    une requête portant quelques centaines de mégaoctets de JSON suffisait à
    saturer le conteneur, qui n'a pas non plus de limite mémoire.
    """
    demesure = {"chief_complaint": "toux", "answers": {f"q{i}": "oui" for i in range(64)}}
    assert (
        client.post("/questionnaire/next", json=demesure, headers={"X-API-Key": CLE}).status_code
        == 422
    )
    trop_long = {"chief_complaint": "toux", "answers": {"q1": "o" * 5000}}
    assert (
        client.post("/questionnaire/next", json=trop_long, headers={"X-API-Key": CLE}).status_code
        == 422
    )


def test_une_description_trop_longue_est_bornee_et_annoncee(client, monkeypatch):
    """Une description qui déborde la fenêtre du modèle interrompait la génération.

    Le contrat acceptait quatre mille caractères, la fenêtre en tient bien
    moins, et la sortie était une erreur de dimension de tenseur au moment
    précis où un soignant attend une réponse. Elle est désormais bornée — et la
    réponse le dit, parce qu'un récit clinique tronqué en silence est exactement
    ce qu'un système d'aide à la décision ne doit pas produire.
    """
    recues: list[str] = []

    def _generer(request, symptomes):
        borne, tronquee = module_api.borner_en_caracteres(symptomes)
        recues.append(borne)
        return REPONSE_MODELE, "URGENCE_VITALE", 42.0, tronquee

    monkeypatch.setattr(module_api, "_generer_vllm", _generer)
    longue = "Le patient décrit une gêne diffuse et variable depuis plusieurs jours. " * 40
    reponse = client.post("/triage", json={"symptoms": longue}, headers={"X-API-Key": CLE})
    assert reponse.status_code == 200
    assert reponse.json()["description_truncated"] is True
    assert len(recues[0]) == module_api.CARACTERES_MAXIMUM


def test_une_description_normale_n_est_pas_signalee_comme_bornee(client):
    reponse = client.post(
        "/triage",
        json={"symptoms": "Homme de 62 ans, douleur thoracique depuis 20 minutes."},
        headers={"X-API-Key": CLE},
    )
    assert reponse.json()["description_truncated"] is False
