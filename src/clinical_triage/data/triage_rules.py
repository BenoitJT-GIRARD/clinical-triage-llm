"""The explicit triage rule: the baseline to beat, and the questionnaire's safety net.

This rule has four uses, and one non-use that is the key to the whole evaluation.

Uses:

1. **Evaluation baseline.** A readable rule of a few dozen lines is the first thing a
   department can deploy. Any language model has to show it does better: the rule is therefore
   evaluated next to the model, on the same test set (see ``evaluation/baselines.py``).
2. **Early stop of the questionnaire.** As soon as a life-threatening sign appears in the
   patient's answers, questioning stops and triage fires.
3. **Second opinion at call time.** The ``/triage`` reply carries the level the rule would have
   given next to the model's: a disagreement is shown to the nurse.
4. **Bounded labelling of cases extracted from the public corpora.** Those texts describe a
   patient without announcing a triage level; the rule assigns it (``corpus_cases``), then
   checks after anonymisation that it still gives the same one (``scripts/build_dataset.py``).
   Those examples carry ``confidence: medium`` for that reason and never exceed
   ``DATA.max_corpus_share`` of the training set — 35%, the rest coming from the catalogue.

Non-use: the rule **never** labels the catalogue vignettes, nor the clinical evaluation set,
which is written by hand. Labelling by keyword and then evaluating with the same keywords
measures only the model's ability to relearn the rule; it is on the clinical set, out of its
reach, that the published rule-versus-model comparison is honest. On the internal set, where
the corpus share was filtered by the rule itself, it would recover the label 100% of the time
by construction, so ``scripts/run_evaluation.py`` drops it from the baselines there.

Three precautions make the rule defensible:

- **inflected forms are recognised**: « convulsions », « douleurs thoraciques »,
  « hémorragies » fire as readily as the singular;
- **negation is taken into account**: « sans perte de connaissance » fires nothing;
- **vital signs count**: a saturation of 88% is enough to classify as life-threatening, even
  when the narrative sounds unremarkable.

The patterns are French and English clinical vocabulary: they are the data the rule matches
against, not prose about the project.
"""

from __future__ import annotations

import re
import unicodedata

from clinical_triage.data.vital_signs import (
    VitalSigns,
    critical_findings,
    parse_age,
    warning_findings,
)
from clinical_triage.data.vital_signs import parse as parse_vitals

VITAL = "URGENCE_VITALE"
MODERATE = "URGENCE_MODEREE"
DEFERRED = "CONSULTATION_DIFFEREE"

# --- Life-threatening signs: immediate care ---
# The terms are chosen for their specificity: too general a word (« choc », « infection »)
# would fire wrongly on unremarkable narratives.
RED_FLAGS: tuple[str, ...] = (
    # French
    "douleur thoracique",
    "douleur dans la poitrine",
    "detresse respiratoire",
    "difficulte a respirer",
    "gene respiratoire severe",
    # French clinical vocabulary the first version ignored. These are the words a triage nurse
    # actually writes — « dyspnée » rather than « difficulté à respirer », « marbrures » rather
    # than « signes de choc ». The French-language corpora use them, and a rule that does not
    # know them lets through the presentations it is supposed to catch.
    "dyspnee au repos",
    "polypnee",
    "tirage",
    "stridor",
    "desaturation",
    "marbrure",
    "raideur de nuque",
    "defense abdominale",
    "contracture abdominale",
    "obnubilation",
    "prostration",
    "anurie",
    "hemoptysie abondante",
    "perte de connaissance",
    "perdu connaissance",
    "inconscient",
    # « Ne répond pas » on its own is a common clinical turn of phrase — « ne répond pas au
    # traitement antibiotique depuis trois jours » — which is in no way life-threatening. Only
    # the forms that describe an unresponsive patient count.
    "ne repond pas aux stimulations",
    "ne repond plus",
    "ne reagit pas",
    "ne reagit plus",
    "ne se reveille pas",
    # The words of a relative, not of a clinician. The questionnaire collects their sentences
    # verbatim: a rule that knows only clinical vocabulary does not read what it is actually
    # given.
    "du mal a respirer",
    "n'arrive pas a respirer",
    "n'arrive plus a respirer",
    "saigne beaucoup",
    "saigne en abondance",
    "convulsion",
    "etat de mal epileptique",
    # « Hémorragie » alone also covers subconjunctival haemorrhage, benign and spectacular:
    # exactly the false positive a reception desk meets every day. The forms that carry
    # severity are kept, and « saigne beaucoup » above covers the relative's account.
    "hemorragie active",
    "hemorragie massive",
    "hemorragie abondante",
    "hemorragie exteriorisee",
    "hemorragie digestive",
    "hemorragie de la delivrance",
    "hemorragie du post partum",
    "saignement abondant",
    "saignement actif et important",
    "vomissement de sang",
    "hematemese",
    "accident vasculaire",
    "deficit moteur",
    "trouble de la parole",
    "hemiplegie",
    "hemiparesie",
    "aphasie",
    "ne peut plus parler",
    "ne parle plus",
    "paralysie",
    "infarctus",
    "anaphylaxie",
    "choc anaphylactique",
    "choc hemorragique",
    "etat de choc",
    "arret cardiaque",
    "arret respiratoire",
    "cyanose",
    "levre bleue",
    "raideur de la nuque",
    "syndrome meninge",
    "purpura",
    "septicemie",
    "overdose",
    "intoxication volontaire",
    "tentative de suicide",
    "idee suicidaire",
    "ideation suicidaire",
    "projet suicidaire",
    "envie d'en finir",
    "trouble de la conscience",
    "trouble de la vigilance",
    "somnolence inhabituelle",
    "pause respiratoire",
    "masse abdominale battante",
    "plaie penetrante",
    "brulure etendue",
    # English
    "chest pain",
    "crushing chest pain",
    "difficulty breathing",
    "struggling to breathe",
    "cannot breathe",
    "bleeding a lot",
    "severe shortness of breath",
    "unconscious",
    "unresponsive",
    "loss of consciousness",
    "seizure",
    # Same remark as in French: « subconjunctival haemorrhage » is benign.
    "active haemorrhage",
    "active hemorrhage",
    "massive haemorrhage",
    "massive hemorrhage",
    "gastrointestinal haemorrhage",
    "gastrointestinal hemorrhage",
    "postpartum haemorrhage",
    "postpartum hemorrhage",
    "heavy bleeding",
    "vomiting blood",
    "stroke",
    "sudden weakness",
    "slurred speech",
    "paralysis",
    "heart attack",
    "anaphylaxis",
    "anaphylactic shock",
    "cardiac arrest",
    "respiratory arrest",
    "cyanosis",
    "blue lip",
    "neck stiffness",
    "stiff neck",
    "purple skin blotch",
    "sepsis",
    "suicidal thought",
    "suicide attempt",
    "altered consciousness",
    "pause in breathing",
    "pulsatile abdominal mass",
    "stab wound",
    "extensive burn",
)

# --- Warning signs: medical assessment within a few hours ---
WARNING_FLAGS: tuple[str, ...] = (
    # French
    "fievre elevee",
    "fievre persistante",
    "frisson",
    # The same additions as on the red side, at the degree that belongs to them: dyspnoea on
    # exertion is not dyspnoea at rest, and a syncope calls for a work-up without belonging in
    # resuscitation.
    "dyspnee",
    "tachypnee",
    "bradypnee",
    "syncope",
    "lipothymie",
    "malaise",
    "agitation",
    "oligurie",
    "hemoptysie",
    "melena",
    "rectorragie",
    "sueur profuse",
    "vomissement repete",
    "deshydratation",
    "douleur abdominale",
    "fracture",
    "entorse",
    "deformation",
    "impotence fonctionnelle",
    "migraine severe",
    "cephalee intense",
    "vertige",
    "palpitation",
    "douleur intense",
    # « Brûlure » alone caught heartburn and burning on urination, which are sensations and not
    # lesions. The word is therefore qualified: a burn injury announces itself by its mechanism
    # or its degree. Urinary burning keeps its own entry a few lines below, because it does
    # warrant an opinion.
    "brulure thermique",
    "brulure chimique",
    "brulure electrique",
    "brulure par",
    "brulure du deuxieme degre",
    "brulure du troisieme degre",
    "plaie profonde",
    "crise d'asthme",
    "sifflement",
    "tachycardie",
    "erysipele",
    "abces",
    "colique nephretique",
    "brulure urinaire",
    "corps etranger",
    "urticaire",
    "diarrhee",
    "toux grasse",
    "essoufflement a l'effort",
    # English. A few words are spelt the same in both languages — « fracture »,
    # « palpitation » — and appear once, above: the list is a set of patterns, not a dictionary
    # per language.
    "high fever",
    "persistent fever",
    "shivering",
    "repeated vomiting",
    "dehydration",
    "abdominal pain",
    "sprain",
    "deformity",
    "severe migraine",
    "severe headache",
    "dizziness",
    "severe pain",
    "burn",
    "deep wound",
    "asthma attack",
    "wheezing",
    "tachycardia",
    "abscess",
    "foreign body",
    "hives",
    "diarrhoea",
    "diarrhea",
    "productive cough",
    "breathless on exertion",
)

# Negation markers looked for just before a detected sign.
NEGATIONS: tuple[str, ...] = (
    "pas de",
    "pas d'",
    "sans",
    "aucun",
    "aucune",
    "absence de",
    "absence d'",
    "ni",
    "n'a pas",
    "ne presente pas",
    "no",
    "not",
    "without",
    "denies",
    "absence of",
    "free of",
)
NEGATION_PATTERN = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in NEGATIONS) + r")\b")

# Window of text inspected before a sign, to look for a negation there.
NEGATION_WINDOW = 28

# Boundaries beyond which a negation no longer carries. A reception note enumerates: « pas de
# fièvre, douleur thoracique constrictive ». Without these boundaries the twenty-eight-character
# window reaches back over the comma, finds « pas de », and cancels the chest pain — a heart
# attack filed as a deferred consultation, with nothing shown to explain it.
CLAUSE_BOUNDARY = re.compile(r"[.;:,\n]|\b(?:mais|toutefois|cependant|but|however)\b")


def normalize(text: str) -> str:
    """Lower-case and strip accents, for a robust search."""
    lowered = text.lower()
    decomposed = unicodedata.normalize("NFKD", lowered)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _flexible_pattern(term: str) -> str:
    """Make a term tolerant to the common inflected forms.

    Each word of the term accepts an optional plural or feminine ending: « convulsion » matches
    « convulsions », « douleur thoracique » matches « douleurs thoraciques », « inconscient »
    matches « inconsciente », « seizure » matches « seizures ».
    """
    words = normalize(term).split()
    return r"\s+".join(re.escape(word) + r"(?:e|s|es|x|ux)?" for word in words)


def _compile(terms: tuple[str, ...]) -> re.Pattern[str]:
    """Compile the whole set of terms into a single regular expression."""
    # Long terms come first so that « douleur thoracique » is preferred over « douleur » when
    # both could match. At equal length, alphabetical order decides: a `set` has no stable
    # order from one run to the next, and it is that order which decides which term `explain`
    # shows the nurse when two match at the same place.
    ordered = sorted(dict.fromkeys(terms), key=lambda term: (-len(term), term))
    return re.compile(r"\b(?:" + "|".join(_flexible_pattern(t) for t in ordered) + r")\b")


RED_PATTERN = _compile(RED_FLAGS)
WARNING_PATTERN = _compile(WARNING_FLAGS)


def _is_negated(normalized_text: str, start: int) -> bool:
    """Say whether the sign starting at ``start`` is preceded by a negation.

    The search stops at the current clause: a negation placed before a comma, a full stop or a
    « mais » no longer carries over what follows.
    """
    begin = max(0, start - NEGATION_WINDOW)
    upstream = normalized_text[begin:start]
    boundaries = [m.end() for m in CLAUSE_BOUNDARY.finditer(upstream)]
    if boundaries:
        upstream = upstream[boundaries[-1] :]
    return NEGATION_PATTERN.search(upstream) is not None


def matched_flags(text: str, level: str) -> list[str]:
    """Return the signs of the requested level that the text actually asserts."""
    normalized = normalize(text)
    pattern = RED_PATTERN if level == VITAL else WARNING_PATTERN
    found = [
        m.group(0) for m in pattern.finditer(normalized) if not _is_negated(normalized, m.start())
    ]
    return sorted(set(found))


def _reading(text: str, vitals: VitalSigns | None, age: int | None) -> tuple[VitalSigns, int]:
    """Fill in the reading and the age from the text when they are not provided."""
    return (
        vitals if vitals is not None else parse_vitals(text),
        age if age is not None else parse_age(text),
    )


def classify(
    text: str,
    vitals: VitalSigns | None = None,
    age: int | None = None,
) -> str:
    """Assign a triage level to a description of symptoms.

    When the vital signs and the age are not provided separately, they are read from the text:
    readings are often written into the reception narrative, and a rule that ignored them would
    be compared to the model on unequal terms.

    Escalation wins: a single life-threatening sign, in the narrative or in the vital signs, is
    enough to classify as life-threatening.
    """
    reading, patient_age = _reading(text, vitals, age)
    if critical_findings(reading, patient_age) or matched_flags(text, VITAL):
        return VITAL
    if warning_findings(reading, patient_age) or matched_flags(text, MODERATE):
        return MODERATE
    return DEFERRED


def explain(text: str, vitals: VitalSigns | None = None, age: int | None = None) -> list[str]:
    """List what drove the decision, for auditability."""
    reading, patient_age = _reading(text, vitals, age)
    reasons = critical_findings(reading, patient_age) + matched_flags(text, VITAL)
    if not reasons:
        reasons = warning_findings(reading, patient_age) + matched_flags(text, MODERATE)
    return reasons


# Default courses of action, used when no specific recommendation is available (cases derived
# from the public corpora).
RECOMMENDATIONS: dict[str, str] = {
    VITAL: (
        "Prise en charge immédiate. Alerter sans délai le médecin urgentiste et, "
        "en cas de signe vital engagé, contacter le 15 (SAMU)."
    ),
    MODERATE: (
        "Orienter vers une évaluation médicale sous quelques heures. Surveiller "
        "l'évolution des symptômes et réévaluer en cas d'aggravation."
    ),
    DEFERRED: (
        "Pas de critère de gravité immédiat. Proposer une consultation différée "
        "auprès du médecin traitant et donner des consignes de surveillance."
    ),
}


def recommendation_for(level: str) -> str:
    """Return the standard course of action attached to a level."""
    return RECOMMENDATIONS[level]
