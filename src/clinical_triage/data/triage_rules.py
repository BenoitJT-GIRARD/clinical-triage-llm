"""Règle de triage explicite : la référence à battre, et le filet du questionnaire.

Cette règle a quatre usages, et un non-usage qui est la clé de l'évaluation.

Usages :

1. **Référence d'évaluation.** Une règle lisible de quelques dizaines de lignes
   est la première chose qu'un service hospitalier peut déployer. Tout modèle de
   langage doit démontrer qu'il fait mieux : la règle est donc évaluée à côté du
   modèle, sur le même jeu de test (voir `evaluation/baselines.py`).
2. **Arrêt anticipé du questionnaire.** Dès qu'un signe vital apparaît dans les
   réponses du patient, on cesse de poser des questions et on déclenche le triage.
3. **Second avis à l'appel.** La réponse de `/triage` porte le niveau que la règle
   aurait donné à côté de celui du modèle : un désaccord est affiché au soignant.
4. **Étiquetage borné des cas extraits des corpus publics.** Ces textes-là
   décrivent un patient sans annoncer son niveau de triage ; c'est la règle qui le
   leur attribue (`corpus_cases`), puis qui vérifie après anonymisation qu'elle
   donne toujours le même (`scripts/01`). Ces exemples portent pour cette raison
   `confiance: moyenne` et ne dépassent jamais `DATA.max_corpus_share` du jeu
   d'entraînement — 35 %, le reste venant du catalogue.

Non-usage : la règle n'étiquette **jamais** les vignettes du catalogue, ni le jeu
d'évaluation clinique, qui est écrit à la main. Étiqueter par mots-clés puis
évaluer avec les mêmes mots-clés ne mesure que la capacité du modèle à réapprendre
la règle ; c'est sur le jeu clinique, hors de sa portée, que la comparaison
règle/modèle publiée dans le rapport est honnête. Sur le jeu interne, où la
part corpus a été filtrée par la règle elle-même, elle retrouverait l'étiquette à
100 % par construction : `scripts/06` l'y écarte donc des références.

Trois précautions rendent la règle défendable :

- **les formes fléchies sont reconnues** : « convulsions », « douleurs
  thoraciques », « hémorragies » déclenchent autant que le singulier ;
- **la négation est prise en compte** : « sans perte de connaissance » ne
  déclenche rien ;
- **les constantes vitales comptent** : une saturation à 88 % suffit à classer en
  urgence vitale, même si le récit paraît anodin.
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

# --- Signes de détresse vitale : prise en charge immédiate ---
# Les termes sont choisis pour leur spécificité : un mot trop général
# (« choc », « infection ») déclencherait à tort sur des récits anodins.
RED_FLAGS: tuple[str, ...] = (
    # Français
    "douleur thoracique",
    "douleur dans la poitrine",
    "detresse respiratoire",
    "difficulte a respirer",
    "gene respiratoire severe",
    # Vocabulaire clinique français que la première version ignorait. Ce sont
    # les mots qu'une infirmière d'accueil écrit réellement — « dyspnée » plutôt
    # que « difficulté à respirer », « marbrures » plutôt que « signes de choc ».
    # Les corpus francophones les emploient, et une règle qui ne les connaît pas
    # laisse passer les présentations qu'elle est censée attraper.
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
    # « Ne répond pas » tout court est une tournure clinique courante — « ne
    # répond pas au traitement antibiotique depuis trois jours » — qui n'a rien
    # d'une urgence vitale. Seules comptent les formes qui décrivent un patient
    # aréactif.
    "ne repond pas aux stimulations",
    "ne repond plus",
    "ne reagit pas",
    "ne reagit plus",
    "ne se reveille pas",
    # Les mots d'un accompagnant, pas ceux d'un soignant. Le questionnaire
    # recueille ses phrases telles quelles : une règle qui ne connaît que le
    # vocabulaire clinique ne lit pas ce qu'on lui donne réellement.
    "du mal a respirer",
    "n'arrive pas a respirer",
    "n'arrive plus a respirer",
    "saigne beaucoup",
    "saigne en abondance",
    "convulsion",
    "etat de mal epileptique",
    # « Hémorragie » nu couvre aussi l'hémorragie sous-conjonctivale, bénigne et
    # spectaculaire : c'est exactement le faux positif qu'un service d'accueil
    # rencontre tous les jours. On garde les formes qui portent la gravité, et
    # « saigne beaucoup » ci-dessus couvre le récit de l'accompagnant.
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
    # Anglais
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
    # Même remarque qu'en français : « subconjunctival haemorrhage » est bénin.
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

# --- Signes d'alerte : évaluation médicale sous quelques heures ---
WARNING_FLAGS: tuple[str, ...] = (
    # Français
    "fievre elevee",
    "fievre persistante",
    "frisson",
    # Mêmes ajouts que côté rouge, au degré qui leur revient : une dyspnée à
    # l'effort n'est pas une dyspnée au repos, et une syncope demande un bilan
    # sans relever du déchocage.
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
    # « Brûlure » tout court attrapait les brûlures d'estomac et les brûlures
    # mictionnelles, qui sont des sensations et non des lésions. Le mot est donc
    # qualifié : une brûlure lésionnelle s'annonce par son mécanisme ou son
    # degré. La brûlure urinaire garde son entrée propre, quelques lignes plus
    # bas, parce qu'elle relève bien d'un avis.
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
    # Anglais. Quelques mots s'écrivent de la même façon dans les deux langues
    # — « fracture », « palpitation » — et figurent une seule fois, plus haut :
    # la liste est un jeu de motifs, pas un dictionnaire par langue.
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

# Marqueurs de négation cherchés juste avant un signe détecté.
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

# Fenêtre de texte inspectée avant un signe pour y chercher une négation.
NEGATION_WINDOW = 28

# Frontières au-delà desquelles une négation ne porte plus. Une note d'accueil
# énumère : « pas de fièvre, douleur thoracique constrictive ». Sans ces
# frontières, la fenêtre de vingt-huit caractères remonte par-dessus la virgule,
# trouve « pas de », et annule la douleur thoracique — un infarctus classé en
# consultation différée, sans rien afficher qui l'explique.
FIN_DE_PROPOSITION = re.compile(r"[.;:,\n]|\b(?:mais|toutefois|cependant|but|however)\b")


def normalize(text: str) -> str:
    """Passe en minuscules et retire les accents, pour une recherche robuste."""
    lowered = text.lower()
    decomposed = unicodedata.normalize("NFKD", lowered)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _flexible_pattern(term: str) -> str:
    """Rend un terme tolérant aux formes fléchies courantes.

    Chaque mot du terme accepte une terminaison optionnelle de pluriel ou de
    féminin : « convulsion » reconnaît « convulsions », « douleur thoracique »
    reconnaît « douleurs thoraciques », « inconscient » reconnaît
    « inconsciente », « seizure » reconnaît « seizures ».
    """
    mots = normalize(term).split()
    return r"\s+".join(re.escape(mot) + r"(?:e|s|es|x|ux)?" for mot in mots)


def _compile(terms: tuple[str, ...]) -> re.Pattern[str]:
    """Compile l'ensemble des termes en une seule expression régulière."""
    # Les termes longs passent en premier pour que « douleur thoracique » soit
    # préféré à « douleur » lorsque les deux pourraient correspondre. À longueur
    # égale, l'ordre alphabétique tranche : un `set` n'a pas d'ordre stable d'une
    # exécution à l'autre, et c'est cet ordre qui décide quel terme `explain`
    # affiche au soignant quand deux correspondent au même endroit.
    ordonnes = sorted(dict.fromkeys(terms), key=lambda terme: (-len(terme), terme))
    return re.compile(r"\b(?:" + "|".join(_flexible_pattern(t) for t in ordonnes) + r")\b")


RED_PATTERN = _compile(RED_FLAGS)
WARNING_PATTERN = _compile(WARNING_FLAGS)


def _is_negated(normalized_text: str, start: int) -> bool:
    """Dit si le signe qui commence à `start` est précédé d'une négation.

    La recherche s'arrête à la proposition courante : une négation située avant
    une virgule, un point ou un « mais » ne porte plus sur ce qui suit.
    """
    debut = max(0, start - NEGATION_WINDOW)
    amont = normalized_text[debut:start]
    frontieres = [m.end() for m in FIN_DE_PROPOSITION.finditer(amont)]
    if frontieres:
        amont = amont[frontieres[-1] :]
    return NEGATION_PATTERN.search(amont) is not None


def matched_flags(text: str, level: str) -> list[str]:
    """Renvoie les signes du niveau demandé réellement affirmés dans le texte."""
    normalized = normalize(text)
    pattern = RED_PATTERN if level == VITAL else WARNING_PATTERN
    trouves = [
        m.group(0) for m in pattern.finditer(normalized) if not _is_negated(normalized, m.start())
    ]
    return sorted(set(trouves))


def _reading(text: str, vitals: VitalSigns | None, age: int | None) -> tuple[VitalSigns, int]:
    """Complète le relevé et l'âge à partir du texte quand ils ne sont pas fournis."""
    return (
        vitals if vitals is not None else parse_vitals(text),
        age if age is not None else parse_age(text),
    )


def classify(
    text: str,
    vitals: VitalSigns | None = None,
    age: int | None = None,
) -> str:
    """Attribue un niveau de triage à une description de symptômes.

    Quand les constantes et l'âge ne sont pas fournis séparément, on les lit dans
    le texte : les relevés sont souvent écrits dans le récit d'accueil, et une
    règle qui les ignorerait serait comparée au modèle à armes inégales.

    Le surclassement prime : un seul signe de détresse vitale, dans le récit ou
    dans les constantes, suffit à classer en urgence vitale.
    """
    releve, age_patient = _reading(text, vitals, age)
    if critical_findings(releve, age_patient) or matched_flags(text, VITAL):
        return VITAL
    if warning_findings(releve, age_patient) or matched_flags(text, MODERATE):
        return MODERATE
    return DEFERRED


def explain(text: str, vitals: VitalSigns | None = None, age: int | None = None) -> list[str]:
    """Énumère les éléments qui ont motivé la décision, pour l'auditabilité."""
    releve, age_patient = _reading(text, vitals, age)
    raisons = critical_findings(releve, age_patient) + matched_flags(text, VITAL)
    if not raisons:
        raisons = warning_findings(releve, age_patient) + matched_flags(text, MODERATE)
    return raisons


# Conduites à tenir par défaut, utilisées quand aucune recommandation
# spécifique n'est disponible (cas dérivés des corpus publics).
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
    """Renvoie la conduite à tenir standard associée à un niveau."""
    return RECOMMENDATIONS[level]
