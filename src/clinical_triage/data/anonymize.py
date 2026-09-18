"""GDPR anonymisation of clinical texts, with Presidio.

Two distinct uses in this project:

- at **data preparation**, on the texts coming from the public corpora;
- at **serving time**, before writing anything into the audit log. That is where the need is
  real: the public corpora hold no patient data, but the API will, the day it is plugged into a
  hospital information system.

Presidio's default setting is unusable as is on medical text. With the ``DATE_TIME``,
``LOCATION`` and ``NRP`` entities active it masks "depuis trois jours", the names of organs and
the names of molecules: "inhibiteurs de recapture de la sérotonine" becomes "inhibiteurs de
recapture de la <PERSON>". Masking a word that identifies nobody protects nobody and destroys
information the care depends on.

Three principles follow:

1. **only genuinely identifying entities are masked** (name, phone, email, banking identifiers,
   IP address, social security number, date of birth);
2. **the project's clinical vocabulary is protected**: no term of the presentation catalogue or
   of the triage lexicon can be masked;
3. **the quality check is independent**: the masking is not re-checked with the detector that
   produced it — it would obviously find nothing. A separate set of regular expressions looks
   for what might have slipped through.
"""

from __future__ import annotations

import re
from functools import lru_cache

from clinical_triage.utils import get_logger

logger = get_logger(__name__)

# Entities masked: only what identifies a person.
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

# Entities deliberately left out, and the reason for it.
ENTITIES_LEFT_OUT = {
    "DATE_TIME": "masks onset delays, which are a triage criterion",
    "LOCATION": "masks the names of organs and of hospital departments",
    "NRP": "masks the names of conditions and of molecules",
    "MEDICAL_LICENSE": "fires on bibliographic references in the corpora",
}

# Confidence threshold: below it, the detection is too uncertain to justify destroying a word of
# the clinical narrative.
SCORE_THRESHOLD = 0.6


@lru_cache(maxsize=1)
def _protected_vocabulary() -> frozenset[str]:
    """The project's clinical vocabulary, which must never be masked."""
    from clinical_triage.data.clinical_catalogue import PRESENTATIONS
    from clinical_triage.data.triage_rules import RED_FLAGS, WARNING_FLAGS

    words: set[str] = set()
    for term in RED_FLAGS + WARNING_FLAGS:
        words.update(term.lower().split())
    for presentation in PRESENTATIONS:
        texts = (
            presentation.complaint_fr,
            presentation.complaint_en,
            *presentation.signs_fr,
            *presentation.signs_en,
            *presentation.history_fr,
            *presentation.history_en,
        )
        for text in texts:
            words.update(re.findall(r"[\w'-]+", text.lower()))
    return frozenset(words)


@lru_cache(maxsize=1)
def _engines():
    """Initialise, once, the analysis and anonymisation engines.

    The language engine is bilingual (French and English spaCy). Four recognisers absent from
    Presidio are added: the French social security number, the phone number in national and in
    international form, and the date of birth in DD/MM/YYYY as well as YYYY-MM-DD. Each is
    registered in both languages.
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

    # Recognisers specific to the French context, absent from Presidio: social security number,
    # national phone format, date of birth.
    french_patterns = {
        "FR_NIR": Pattern(
            "NIR", r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2,3}\s?\d{2,3}\s?\d{3}\s?\d{2}\b", 0.85
        ),
        # The national number (06 11 22 33 44) and its international form (+33 6 11 22 33 44)
        # designate the same subscriber: the pattern covers both spellings, failing which one of
        # them would stay in the clear.
        "FR_TELEPHONE": Pattern(
            "phone",
            r"\b0[1-9](?:[ .-]?\d{2}){4}\b|\+33[ .-]?[1-9](?:[ .-]?\d{2}){4}\b",
            0.85,
        ),
        # Two spellings of a date of birth: DD/MM/YYYY, and the ISO form YYYY-MM-DD the
        # English-language corpora use. It identifies a person as surely in one format as in the
        # other.
        "DATE_NAISSANCE": Pattern(
            "DD/MM/YYYY",
            r"\b(?:0?[1-9]|[12]\d|3[01])[/.-](?:0?[1-9]|1[0-2])[/.-](?:19|20)\d{2}\b",
            0.8,
        ),
        "DATE_NAISSANCE_ISO": Pattern(
            "YYYY-MM-DD",
            r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b",
            0.8,
        ),
    }
    for entity, pattern in french_patterns.items():
        for language in ("fr", "en"):
            analyzer.registry.add_recognizer(
                PatternRecognizer(
                    supported_entity=entity, patterns=[pattern], supported_language=language
                )
            )

    anonymizer = AnonymizerEngine()
    logger.info("Presidio engines initialised (French and English).")
    return analyzer, anonymizer


MASKED_ENTITIES = PII_ENTITIES + (
    "FR_NIR",
    "FR_TELEPHONE",
    "DATE_NAISSANCE",
    "DATE_NAISSANCE_ISO",
)


def _operators():
    """Each entity is replaced by its type, in angle brackets."""
    from presidio_anonymizer.entities import OperatorConfig

    return {
        "DEFAULT": OperatorConfig("replace", {"new_value": "<DONNEE_PERSONNELLE>"}),
        **{ent: OperatorConfig("replace", {"new_value": f"<{ent}>"}) for ent in MASKED_ENTITIES},
    }


def analyze_and_anonymize(text: str, lang: str = "fr") -> tuple[str, int]:
    """Mask personal data and return (masked text, number of entities)."""
    if not text:
        return text, 0
    analyzer, anonymizer = _engines()
    language = lang if lang in ("fr", "en") else "en"
    # Not every entity is recognised in both languages; asking for an entity with no recogniser
    # produces a warning on every call.
    available = set(analyzer.get_supported_entities(language=language))
    entities = [entity for entity in MASKED_ENTITIES if entity in available]
    results = analyzer.analyze(
        text=text, language=language, entities=entities, score_threshold=SCORE_THRESHOLD
    )
    protected = _protected_vocabulary()
    kept = [r for r in results if text[r.start : r.end].lower() not in protected]
    if not kept:
        return text, 0
    masked = anonymizer.anonymize(text=text, analyzer_results=kept, operators=_operators()).text
    return masked, len(kept)


def anonymize_text(text: str, lang: str = "fr") -> str:
    """Mask the personal data of a text and return the masked text."""
    return analyze_and_anonymize(text, lang)[0]


# --- A quality check independent of the detector ---

# Patterns looked for AFTER masking. They are deliberately hand-written and do not use Presidio:
# a check that reused the original detector could, by construction, never find anything.
RESIDUAL_PII_PATTERNS = {
    "email address": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    "French phone number": re.compile(r"\b0[1-9](?:[ .-]?\d{2}){4}\b"),
    "international phone number": re.compile(r"\+\d{1,3}[ .-]?\d[\d .-]{7,}\d\b"),
    "social security number": re.compile(
        r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b"
    ),
    "date of birth": re.compile(
        r"\b(?:0?[1-9]|[12]\d|3[01])[/.-](?:0?[1-9]|1[0-2])[/.-](?:19|20)\d{2}\b"
    ),
    "ISO date of birth": re.compile(
        r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b"
    ),
    "title followed by a name": re.compile(
        r"\b(?:M\.|Mme|Mlle|Dr|Docteur|Monsieur|Madame)\s+[A-ZÀ-Ý][\w'-]+"
    ),
    "payment card": re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
}


def residual_pii(text: str) -> dict[str, int]:
    """Look, in an already masked text, for what still resembles personal data."""
    findings: dict[str, int] = {}
    for name, pattern in RESIDUAL_PII_PATTERNS.items():
        occurrences = len(pattern.findall(text))
        if occurrences:
            findings[name] = occurrences
    return findings


def audit_corpus(texts: list[str]) -> dict[str, int]:
    """Aggregate the quality check over a set of masked texts."""
    total: dict[str, int] = {}
    for text in texts:
        for name, occurrences in residual_pii(text).items():
            total[name] = total.get(name, 0) + occurrences
    return total
