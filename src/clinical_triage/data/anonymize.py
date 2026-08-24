"""Anonymisation RGPD des textes cliniques, avec Presidio.

Deux usages distincts dans le projet :

- à la **préparation des données**, sur les textes issus des corpus publics ;
- au **service**, avant d'écrire quoi que ce soit dans le journal d'audit. C'est
  là que le besoin est réel : les corpus publics ne contiennent pas de données
  patient, mais l'API, elle, en recevra le jour où elle sera branchée au système
  d'information hospitalier.

Le réglage par défaut de Presidio est inutilisable tel quel sur du texte
médical. Avec les entités `DATE_TIME`, `LOCATION` et `NRP` actives, il masque
« depuis trois jours », les noms d'organes et les noms de molécules :
« inhibiteurs de recapture de la sérotonine » devient « inhibiteurs de
recapture de la <PERSON> ». Masquer un mot qui n'identifie personne ne protège
personne et détruit l'information utile au soin.

On retient donc trois principes :

1. **seules les entités réellement identifiantes sont masquées** (nom, téléphone,
   e-mail, identifiants bancaires, adresse IP, numéro de sécurité sociale,
   date de naissance) ;
2. **le vocabulaire clinique du projet est protégé** : aucun terme du catalogue
   de présentations ni du lexique de triage ne peut être masqué ;
3. **le contrôle qualité est indépendant** : on ne revérifie pas le masquage avec
   le détecteur qui l'a produit — il ne trouverait évidemment rien. Un jeu
   d'expressions régulières distinct cherche ce qui aurait pu passer au travers.
"""

from __future__ import annotations

import re
from functools import lru_cache

from clinical_triage.utils import get_logger

logger = get_logger(__name__)

# Entités masquées : uniquement ce qui identifie une personne.
PII_ENTITIES = (
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "URL",
    "US_SSN",
)

# Entités volontairement écartées, et la raison de leur écartement.
ENTITES_ECARTEES = {
    "DATE_TIME": "masque les délais d'évolution, qui sont un critère de triage",
    "LOCATION": "masque les noms d'organes et de services hospitaliers",
    "NRP": "masque des noms de pathologies et de molécules",
    "MEDICAL_LICENSE": "déclenche sur des références bibliographiques des corpus",
}

# Seuil de confiance : en dessous, la détection est trop incertaine pour
# justifier la destruction d'un mot du récit clinique.
SCORE_THRESHOLD = 0.6


@lru_cache(maxsize=1)
def _protected_vocabulary() -> frozenset[str]:
    """Vocabulaire clinique du projet, qui ne doit jamais être masqué."""
    from clinical_triage.data.clinical_catalogue import PRESENTATIONS
    from clinical_triage.data.triage_rules import RED_FLAGS, WARNING_FLAGS

    mots: set[str] = set()
    for terme in RED_FLAGS + WARNING_FLAGS:
        mots.update(terme.lower().split())
    for presentation in PRESENTATIONS:
        textes = (
            presentation.motif_fr,
            presentation.motif_en,
            *presentation.signes_fr,
            *presentation.signes_en,
            *presentation.antecedents_fr,
            *presentation.antecedents_en,
        )
        for texte in textes:
            mots.update(re.findall(r"[\w'-]+", texte.lower()))
    return frozenset(mots)


@lru_cache(maxsize=1)
def _engines():
    """Initialise, une seule fois, les moteurs d'analyse et d'anonymisation.

    Le moteur linguistique est bilingue (spaCy français et anglais), comme le
    recommande l'énoncé. On y ajoute quatre reconnaisseurs absents de Presidio :
    le numéro de sécurité sociale, le téléphone au format national comme
    international, et la date de naissance en JJ/MM/AAAA comme en AAAA-MM-JJ.
    Chacun est enregistré dans les deux langues.
    """
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "fr", "model_name": "fr_core_news_md"},
            {"lang_code": "en", "model_name": "en_core_web_sm"},
        ],
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["fr", "en"])

    # Reconnaisseurs propres au contexte français, absents de Presidio :
    # numéro de sécurité sociale, téléphone au format national, date de naissance.
    motifs_francais = {
        "FR_NIR": Pattern(
            "NIR", r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2,3}\s?\d{2,3}\s?\d{3}\s?\d{2}\b", 0.85
        ),
        # Le numéro national (06 11 22 33 44) et sa forme internationale
        # (+33 6 11 22 33 44) désignent le même abonné : le motif couvre les deux
        # écritures, faute de quoi l'une d'elles resterait en clair.
        "FR_TELEPHONE": Pattern(
            "téléphone",
            r"\b0[1-9](?:[ .-]?\d{2}){4}\b|\+33[ .-]?[1-9](?:[ .-]?\d{2}){4}\b",
            0.85,
        ),
        # Deux écritures d'une date de naissance : JJ/MM/AAAA, et la forme ISO
        # AAAA-MM-JJ qu'emploient les corpus anglophones. Elle identifie une
        # personne aussi sûrement dans un format que dans l'autre.
        "DATE_NAISSANCE": Pattern(
            "JJ/MM/AAAA",
            r"\b(?:0?[1-9]|[12]\d|3[01])[/.-](?:0?[1-9]|1[0-2])[/.-](?:19|20)\d{2}\b",
            0.8,
        ),
        "DATE_NAISSANCE_ISO": Pattern(
            "AAAA-MM-JJ",
            r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b",
            0.8,
        ),
    }
    for entite, motif in motifs_francais.items():
        for langue in ("fr", "en"):
            analyzer.registry.add_recognizer(
                PatternRecognizer(
                    supported_entity=entite, patterns=[motif], supported_language=langue
                )
            )

    anonymizer = AnonymizerEngine()
    logger.info("Moteurs Presidio initialisés (français et anglais).")
    return analyzer, anonymizer


ENTITES_MASQUEES = PII_ENTITIES + (
    "FR_NIR",
    "FR_TELEPHONE",
    "DATE_NAISSANCE",
    "DATE_NAISSANCE_ISO",
)


def _operators():
    """Chaque entité est remplacée par son type, entre chevrons."""
    from presidio_anonymizer.entities import OperatorConfig

    return {
        "DEFAULT": OperatorConfig("replace", {"new_value": "<DONNEE_PERSONNELLE>"}),
        **{ent: OperatorConfig("replace", {"new_value": f"<{ent}>"}) for ent in ENTITES_MASQUEES},
    }


def analyze_and_anonymize(text: str, lang: str = "fr") -> tuple[str, int]:
    """Masque les données personnelles et renvoie (texte masqué, nombre d'entités)."""
    if not text:
        return text, 0
    analyzer, anonymizer = _engines()
    langue = lang if lang in ("fr", "en") else "en"
    # Toutes les entités ne sont pas reconnues dans les deux langues ; demander
    # une entité sans reconnaisseur produit un avertissement à chaque appel.
    disponibles = set(analyzer.get_supported_entities(language=langue))
    entites = [entite for entite in ENTITES_MASQUEES if entite in disponibles]
    resultats = analyzer.analyze(
        text=text, language=langue, entities=entites, score_threshold=SCORE_THRESHOLD
    )
    protege = _protected_vocabulary()
    retenus = [r for r in resultats if text[r.start : r.end].lower() not in protege]
    if not retenus:
        return text, 0
    masque = anonymizer.anonymize(text=text, analyzer_results=retenus, operators=_operators()).text
    return masque, len(retenus)


def anonymize_text(text: str, lang: str = "fr") -> str:
    """Masque les données personnelles d'un texte et renvoie le texte masqué."""
    return analyze_and_anonymize(text, lang)[0]


# --- Contrôle qualité indépendant du détecteur ---

# Motifs cherchés APRÈS masquage. Ils sont volontairement écrits à la main et
# n'utilisent pas Presidio : un contrôle qui réutiliserait le détecteur de départ
# ne pourrait, par construction, jamais rien trouver.
RESIDUAL_PII_PATTERNS = {
    "adresse e-mail": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    "téléphone français": re.compile(r"\b0[1-9](?:[ .-]?\d{2}){4}\b"),
    "téléphone international": re.compile(r"\+\d{1,3}[ .-]?\d[\d .-]{7,}\d\b"),
    "numéro de sécurité sociale": re.compile(
        r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b"
    ),
    "date de naissance": re.compile(
        r"\b(?:0?[1-9]|[12]\d|3[01])[/.-](?:0?[1-9]|1[0-2])[/.-](?:19|20)\d{2}\b"
    ),
    "date de naissance ISO": re.compile(
        r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b"
    ),
    "civilité suivie d'un nom": re.compile(
        r"\b(?:M\.|Mme|Mlle|Dr|Docteur|Monsieur|Madame)\s+[A-ZÀ-Ý][\w'-]+"
    ),
    "carte bancaire": re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
}


def residual_pii(text: str) -> dict[str, int]:
    """Cherche, dans un texte déjà masqué, ce qui ressemble encore à une donnée personnelle."""
    trouvailles: dict[str, int] = {}
    for nom, motif in RESIDUAL_PII_PATTERNS.items():
        occurrences = len(motif.findall(text))
        if occurrences:
            trouvailles[nom] = occurrences
    return trouvailles


def audit_corpus(textes: list[str]) -> dict[str, int]:
    """Agrège le contrôle qualité sur un ensemble de textes masqués."""
    total: dict[str, int] = {}
    for texte in textes:
        for nom, occurrences in residual_pii(texte).items():
            total[nom] = total.get(nom, 0) + occurrences
    return total
