"""Tests du journal d'audit.

Le journal porte l'exigence métier de traçabilité. Deux propriétés comptent
autant que son contenu : il anonymise ce qu'il écrit, et il enregistre le modèle
réellement utilisé plutôt qu'une valeur de configuration.
"""

from __future__ import annotations

import json

import pytest

from clinical_triage.serving.audit import (
    RETENTION_JOURS,
    new_request_id,
    read_entries,
    record_interaction,
    check_log_is_writable,
)


def test_les_identifiants_sont_uniques():
    identifiants = {new_request_id() for _ in range(500)}
    assert len(identifiants) == 500


def test_une_interaction_est_ecrite_avec_tous_ses_champs(tmp_path):
    journal = tmp_path / "audit.jsonl"
    entree = record_interaction(
        request_id="abc123",
        symptomes="Homme de 62 ans, douleur thoracique.",
        niveau="URGENCE_VITALE",
        reponse="Niveau de priorité : URGENCE_VITALE",
        latence_ms=421.37,
        modele="qwen3-1.7b-clinical-triage",
        moteur="vllm",
        niveau_regle="URGENCE_VITALE",
        raisons_regle=["douleur thoracique"],
        log_path=journal,
    )
    ligne = json.loads(journal.read_text(encoding="utf-8").strip())
    assert ligne["request_id"] == "abc123"
    assert ligne["niveau"] == "URGENCE_VITALE"
    assert ligne["niveau_regle"] == "URGENCE_VITALE"
    assert ligne["raisons_regle"] == ["douleur thoracique"]
    assert ligne["latence_ms"] == 421.4
    assert ligne["modele"] == "qwen3-1.7b-clinical-triage"
    assert ligne["moteur"] == "vllm"
    assert ligne["retention_jours"] == RETENTION_JOURS
    assert ligne["timestamp"].endswith("+00:00")
    assert entree.request_id == "abc123"


def test_les_interactions_s_ajoutent_sans_ecraser(tmp_path):
    journal = tmp_path / "audit.jsonl"
    for index in range(3):
        record_interaction(
            request_id=f"id{index}",
            symptomes="Rhume banal.",
            niveau="CONSULTATION_DIFFEREE",
            reponse="Niveau de priorité : CONSULTATION_DIFFEREE",
            latence_ms=10.0,
            modele="modele",
            moteur="vllm",
            log_path=journal,
        )
    lignes = read_entries(journal)
    assert [ligne["request_id"] for ligne in lignes] == ["id0", "id1", "id2"]
    assert read_entries(journal, limit=1)[0]["request_id"] == "id2"


def test_un_journal_absent_se_lit_comme_vide(tmp_path):
    assert read_entries(tmp_path / "inexistant.jsonl") == []


def test_le_nom_du_patient_est_masque_avant_ecriture(tmp_path):
    """Le module qui prétend écrire des données anonymisées doit les anonymiser
    lui-même, sans faire confiance à son appelant."""
    journal = tmp_path / "audit.jsonl"
    record_interaction(
        request_id="pii1",
        symptomes="Madame Dupont, 72 ans, joignable au 06 12 34 56 78, douleur thoracique.",
        niveau="URGENCE_VITALE",
        reponse="Niveau de priorité : URGENCE_VITALE",
        latence_ms=12.0,
        modele="modele",
        moteur="vllm",
        log_path=journal,
    )
    ecrit = json.loads(journal.read_text(encoding="utf-8").strip())["symptomes_anonymises"]
    assert "Dupont" not in ecrit
    assert "06 12 34 56 78" not in ecrit
    # L'information clinique, elle, doit survivre au masquage.
    assert "douleur thoracique" in ecrit


def test_la_reponse_du_modele_est_masquee_elle_aussi(tmp_path):
    """La justification reprend le récit du soignant, nom compris.

    Masquer la description et laisser passer la réponse reviendrait à faire
    rentrer par la porte ce qu'on a chassé par la fenêtre.
    """
    journal = tmp_path / "audit.jsonl"
    record_interaction(
        request_id="pii2",
        symptomes="Madame Dupont, 72 ans, douleur thoracique.",
        niveau="URGENCE_VITALE",
        reponse=(
            "Niveau de priorité : URGENCE_VITALE\n"
            "Justification : Madame Dupont présente une douleur thoracique.\n"
            "Recommandation : orientation immédiate."
        ),
        latence_ms=12.0,
        modele="modele",
        moteur="vllm",
        log_path=journal,
    )
    ligne = json.loads(journal.read_text(encoding="utf-8").strip())
    assert "Dupont" not in ligne["reponse_anonymisee"]
    assert "URGENCE_VITALE" in ligne["reponse_anonymisee"]
    assert "douleur thoracique" in ligne["reponse_anonymisee"]


def test_un_texte_en_anglais_est_masque_comme_un_texte_en_francais(tmp_path):
    """Le service reçoit du texte libre, et le modèle est bilingue.

    Le masquage partait toujours du moteur français, qui ne repère pas un nom
    dans une syntaxe anglaise : « Mr Jenkins » restait en clair dans le seul
    fichier du système qui conserve des données personnelles, et pour un an.
    """
    journal = tmp_path / "audit.jsonl"
    record_interaction(
        request_id="pii3",
        symptomes="Mr Jenkins, 72, reachable at +33 6 12 34 56 78, reports chest pain.",
        niveau="URGENCE_VITALE",
        reponse="Priority level: URGENCE_VITALE — Mr Jenkins needs immediate care.",
        latence_ms=12.0,
        modele="modele",
        moteur="vllm",
        log_path=journal,
    )
    ligne = json.loads(journal.read_text(encoding="utf-8").strip())
    assert "Jenkins" not in ligne["symptomes_anonymises"]
    assert "Jenkins" not in ligne["reponse_anonymisee"]
    assert "chest pain" in ligne["symptomes_anonymises"]


def test_une_decision_prise_sur_un_recit_tronque_laisse_une_trace(tmp_path):
    """C'est le sens même de ce journal.

    Une description plus longue que la fenêtre du modèle est bornée avant
    l'inférence — sans quoi la génération s'interrompt. Le triage est alors rendu
    sur un récit incomplet, et l'audit doit pouvoir le retrouver des mois plus
    tard.
    """
    journal = tmp_path / "audit.jsonl"
    record_interaction(
        request_id="trq1",
        symptomes="Description très longue " * 200,
        niveau="URGENCE_MODEREE",
        reponse="Niveau de priorité : URGENCE_MODEREE",
        latence_ms=12.0,
        modele="modele",
        moteur="vllm",
        description_tronquee=True,
        log_path=journal,
    )
    ligne = json.loads(journal.read_text(encoding="utf-8").strip())
    assert ligne["description_tronquee"] is True


def test_une_interaction_normale_n_est_pas_marquee_tronquee(tmp_path):
    journal = tmp_path / "audit.jsonl"
    record_interaction(
        request_id="trq2",
        symptomes="Homme de 62 ans, douleur thoracique.",
        niveau="URGENCE_VITALE",
        reponse="Niveau de priorité : URGENCE_VITALE",
        latence_ms=12.0,
        modele="modele",
        moteur="vllm",
        log_path=journal,
    )
    assert json.loads(journal.read_text(encoding="utf-8").strip())["description_tronquee"] is False


def test_un_journal_ecrivable_est_cree_a_vide(tmp_path):
    """Le contrôle de démarrage prépare le fichier sans y écrire de ligne."""
    journal = tmp_path / "traces" / "audit_triage.jsonl"

    assert check_log_is_writable(journal) == journal
    assert journal.exists()
    assert journal.read_text(encoding="utf-8") == ""


def test_un_journal_inaccessible_arrete_le_demarrage(tmp_path):
    """Un chemin impossible échoue au démarrage, pas à la première requête.

    Le dossier parent est ici un fichier : le cas se produit en conteneur quand
    le volume monté appartient à un autre utilisateur, avec la même conséquence
    — impossible d'écrire — et c'est le démarrage qui doit s'arrêter.
    """
    obstacle = tmp_path / "logs"
    obstacle.write_text("ceci n'est pas un dossier", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Journal d'audit inaccessible"):
        check_log_is_writable(obstacle / "audit_triage.jsonl")
