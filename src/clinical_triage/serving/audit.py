"""Journal d'audit des interactions de triage.

Le cahier des charges demande de « garantir la traçabilité de chaque interaction
pour les audits médicaux ». Concrètement : pouvoir, des mois plus tard, retrouver
ce qui a été demandé, ce que l'agent a répondu, avec quel modèle, en combien de
temps, et ce que la règle explicite en aurait dit.

Deux points de conformité, qui ne vont pas de soi :

- **les deux textes sont anonymisés avant d'être écrits.** Le journal est le seul
  endroit du système où des données patient réelles se déposeraient durablement.
  La description reçue passe donc par le masquage RGPD avant d'atteindre le
  disque — **et la réponse aussi**, parce que la justification produite par le
  modèle reprend le récit du soignant et y ramènerait un nom masqué d'un côté
  par la porte de l'autre. Le module qui prétend écrire des données anonymisées
  doit les anonymiser lui-même, pas faire confiance à son appelant ;
- **la version du modèle est celle réellement chargée**, transmise par le service
  au moment de l'appel. Une constante de configuration pourrait décrire un modèle
  qui n'est pas celui qui a répondu, et une traçabilité falsifiable ne trace rien.

Le fichier est un JSONL : une ligne par interaction, lisible sans outil, et dont
l'ajout est atomique tant que les écritures restent d'un seul processus.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from clinical_triage.config import PATHS
from clinical_triage.utils import get_logger

logger = get_logger("audit")

# Durée de conservation annoncée aux utilisateurs du service. Elle est inscrite
# dans chaque ligne pour que l'obligation soit portée par la donnée elle-même.
RETENTION_JOURS = 365


def audit_log_path() -> Path:
    """Chemin du journal d'audit, surchargeable pour le déploiement en conteneur."""
    configure = os.getenv("TRIAGE_AUDIT_LOG")
    return Path(configure) if configure else PATHS.logs / "audit_triage.jsonl"


def verifier_que_le_journal_est_ecrivable(log_path: Path | None = None) -> Path:
    """Ouvre le journal en écriture au démarrage, et refuse de servir sinon.

    La traçabilité est une exigence de conformité : un service qui répond sans
    pouvoir consigner ce qu'il a répondu ne remplit pas le contrat. Sans ce
    contrôle, l'indisponibilité du journal ne se voit qu'à la première requête,
    après que la décision a été produite, sous la forme d'une erreur 500 dont la
    cause ne figure que dans les traces du conteneur.

    Le cas se produit dès qu'un volume monté appartient à un autre utilisateur
    que celui qui fait tourner le service.
    """
    chemin = log_path or audit_log_path()
    try:
        chemin.parent.mkdir(parents=True, exist_ok=True)
        with chemin.open("a", encoding="utf-8"):
            pass
    except OSError as erreur:
        raise RuntimeError(
            f"Journal d'audit inaccessible en écriture ({chemin}) : {erreur}. "
            "La traçabilité de chaque interaction est une exigence du service : "
            "il ne démarre pas sans elle."
        ) from erreur
    return chemin


def new_request_id() -> str:
    """Génère un identifiant d'interaction unique."""
    return uuid.uuid4().hex[:16]


@dataclass(frozen=True)
class AuditEntry:
    """Ligne du journal d'audit."""

    request_id: str
    timestamp: str
    symptomes_anonymises: str
    niveau: str | None
    niveau_regle: str | None
    raisons_regle: list[str]
    reponse_anonymisee: str
    # Vrai lorsque la description dépasse la fenêtre du modèle et se trouve bornée
    # avant l'appel : le modèle ne lit alors qu'une partie du récit. Une décision de
    # triage prise sur un récit incomplet doit laisser une trace, c'est le sens même
    # de ce journal.
    description_tronquee: bool
    latence_ms: float
    modele: str
    moteur: str
    retention_jours: int


def masquer(texte: str) -> str:
    """Masque les données personnelles d'un texte, sans savoir en quelle langue il est.

    L'API accepte du texte libre et le modèle est bilingue : une description
    arrive aussi bien en anglais qu'en français. Anonymiser avec le seul moteur
    français laisserait « Mr Jenkins » en clair dans le journal — le seul endroit
    du système où une donnée personnelle est conservée, et pour un an.

    Plutôt que de deviner la langue, on enchaîne les deux passes. Les deux
    moteurs spaCy sont déjà chargés dans l'image de service, et la seconde passe
    ne voit du texte de la première que ce qu'elle n'a pas masqué : les balises
    posées par l'une ne sont pas des noms propres pour l'autre.
    """
    from clinical_triage.data.anonymize import anonymize_text

    return anonymize_text(anonymize_text(texte, "fr"), "en")


def record_interaction(
    request_id: str,
    symptomes: str,
    niveau: str | None,
    reponse: str,
    latence_ms: float,
    modele: str,
    moteur: str,
    niveau_regle: str | None = None,
    raisons_regle: list[str] | None = None,
    description_tronquee: bool = False,
    log_path: Path | None = None,
) -> AuditEntry:
    """Anonymise puis consigne une interaction, et renvoie la ligne écrite."""
    entree = AuditEntry(
        request_id=request_id,
        timestamp=datetime.now(UTC).isoformat(),
        symptomes_anonymises=masquer(symptomes),
        niveau=niveau,
        niveau_regle=niveau_regle,
        raisons_regle=raisons_regle or [],
        reponse_anonymisee=masquer(reponse),
        description_tronquee=description_tronquee,
        latence_ms=round(latence_ms, 1),
        modele=modele,
        moteur=moteur,
        retention_jours=RETENTION_JOURS,
    )
    chemin = log_path or audit_log_path()
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("a", encoding="utf-8") as fichier:
        fichier.write(json.dumps(asdict(entree), ensure_ascii=False) + "\n")
    return entree


def read_entries(log_path: Path | None = None, limit: int | None = None) -> list[dict]:
    """Relit le journal d'audit, pour vérification ou export."""
    chemin = log_path or audit_log_path()
    if not chemin.exists():
        return []
    with chemin.open(encoding="utf-8") as fichier:
        lignes = [json.loads(ligne) for ligne in fichier if ligne.strip()]
    return lignes[-limit:] if limit else lignes
