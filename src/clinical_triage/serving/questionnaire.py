"""Questionnaire de symptômes adaptatif.

L'agent doit collecter les symptômes « via un questionnaire intelligent
adaptatif ». Adaptatif veut dire deux choses ici :

- **les questions dépendent du motif.** Un patient qui arrive pour une douleur
  thoracique et un patient qui arrive pour une plaie ne doivent pas se voir poser
  les mêmes questions. Le motif est rattaché à un thème clinique, et chaque thème
  a sa propre liste de questions ciblées ;
- **la collecte s'arrête dès qu'elle n'est plus utile.** Si un signe de détresse
  vitale apparaît dans une réponse, on cesse d'interroger et on déclenche le
  triage immédiatement : faire patienter un infarctus le temps de cinq questions
  serait absurde.

La logique est déterministe et lisible : chaque question posée est justifiable
devant une équipe soignante, ce qui n'est pas accessoire en contexte clinique.

Un point qui paraît mineur et ne l'est pas : la synthèse transmise au modèle ne
recopie **jamais** la question devant sa réponse. Chaque réponse devient une
phrase déclarative écrite d'avance (table `RAPPORTS`), parce que le texte des
questions porte tout le vocabulaire des signes de gravité, et que cette synthèse
est ensuite relue par `classify`. Recopiée telle quelle, « Y a-t-il une
difficulté à respirer ? non » s'y lit comme le signe lui-même : un rhume dont
tout est nié ressortirait en urgence vitale. Concaténer les seules réponses ne
marche pas davantage — « oui. non. depuis hier » est une suite de mots sans
référent. La phrase déclarative garde le sens et ne transporte pas la question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from clinical_triage.data.triage_rules import VITAL, classify, normalize

# Questions de dépistage des signes de gravité, posées quel que soit le motif.
QUESTIONS_DE_GRAVITE: tuple[tuple[str, str], ...] = (
    ("conscience", "Le patient est-il pleinement conscient et orienté ?"),
    ("respiration", "Y a-t-il une difficulté à respirer ou un essoufflement au repos ?"),
    ("saignement", "Existe-t-il un saignement actif ou important ?"),
)

# Questions propres à chaque thème clinique.
QUESTIONS_PAR_THEME: dict[str, tuple[tuple[str, str], ...]] = {
    "douleur_thoracique": (
        ("irradiation", "La douleur irradie-t-elle vers le bras, la mâchoire ou le dos ?"),
        ("sueurs", "Y a-t-il des sueurs, des nausées ou un malaise associés ?"),
        ("debut", "La douleur a-t-elle débuté brutalement, et depuis combien de temps ?"),
    ),
    "respiratoire": (
        ("effort", "L'essoufflement survient-il au repos ou seulement à l'effort ?"),
        ("parole", "Le patient peut-il terminer ses phrases sans reprendre son souffle ?"),
        ("antecedent_pulmonaire", "Existe-t-il un asthme ou une maladie pulmonaire connue ?"),
    ),
    "neurologique": (
        ("deficit", "Y a-t-il une faiblesse, un engourdissement ou une difficulté à parler ?"),
        ("heure_debut", "À quelle heure précise les symptômes ont-ils commencé ?"),
        ("cephalee", "La céphalée est-elle apparue brutalement, comme un coup de tonnerre ?"),
    ),
    "digestif": (
        ("localisation", "Où la douleur abdominale est-elle localisée ?"),
        ("vomissements", "Y a-t-il des vomissements, et contiennent-ils du sang ?"),
        ("transit", "Le transit est-il normal : gaz, selles, présence de sang ?"),
    ),
    "traumatologie": (
        ("mecanisme", "Quel est le mécanisme du traumatisme et sa violence ?"),
        ("appui", "L'appui ou la mobilisation du membre sont-ils possibles ?"),
        ("deformation", "Existe-t-il une déformation, une plaie ou une perte de sensibilité ?"),
    ),
    "fievre": (
        ("temperature", "Quelle est la température mesurée, et depuis combien de temps ?"),
        ("frissons", "Y a-t-il des frissons intenses ou une éruption cutanée ?"),
        ("nuque", "La nuque est-elle souple, ou douloureuse à la flexion ?"),
    ),
    "psychiatrique": (
        ("intention", "Le patient exprime-t-il des idées suicidaires, et avec quel scénario ?"),
        ("moyens", "Dispose-t-il de moyens pour passer à l'acte ?"),
        ("entourage", "Le patient est-il accompagné, et par qui ?"),
    ),
}

# Thème retenu quand le motif ne correspond à aucune famille identifiée.
QUESTIONS_GENERALES: tuple[tuple[str, str], ...] = (
    ("anciennete", "Depuis combien de temps les symptômes sont-ils présents ?"),
    ("evolution", "Les symptômes s'aggravent-ils, stagnent-ils ou s'améliorent-ils ?"),
    ("antecedents", "Le patient a-t-il des antécédents ou des traitements en cours ?"),
)

# Comment la réponse à chaque question est reportée dans la description transmise
# au modèle : un intitulé court pour les réponses libres, et — pour les questions
# qui attendent oui ou non — la phrase à écrire dans chacun des deux cas.
#
# Recopier la question serait plus simple, et serait faux. Elle contient par
# construction le vocabulaire des signes de gravité, et la description compilée
# est ensuite relue par la règle de triage : « Y a-t-il une difficulté à
# respirer ? non » se classe urgence vitale, avec « difficulté à respirer » en
# justification, pour un patient qui vient de la nier. La règle sait en revanche
# traiter une négation antéposée — « Pas de difficulté à respirer » — d'où ces
# formulations écrites une par une.
RAPPORTS: dict[str, tuple[str, str, str]] = {
    # identifiant : (intitulé si réponse libre, phrase si oui, phrase si non)
    "conscience": (
        "Conscience",
        "Patient conscient et orienté.",
        "Patient non conscient ou désorienté.",
    ),
    "respiration": (
        "Respiration",
        "Difficulté à respirer, essoufflement au repos.",
        "Pas de difficulté à respirer ni d'essoufflement au repos.",
    ),
    "saignement": (
        "Saignement",
        "Saignement actif et important.",
        "Aucun saignement actif.",
    ),
    "irradiation": (
        "Irradiation",
        "La douleur irradie vers le bras, la mâchoire ou le dos.",
        "Aucune irradiation vers le bras, la mâchoire ou le dos.",
    ),
    "sueurs": (
        "Signes associés",
        "Sueurs, nausées ou malaise associés.",
        "Ni sueurs, ni nausées, ni malaise associés.",
    ),
    "debut": ("Début de la douleur", "", ""),
    "effort": ("Circonstances de l'essoufflement", "", ""),
    "parole": (
        "Parole",
        "Le patient termine ses phrases sans reprendre son souffle.",
        "Le patient ne termine pas ses phrases sans reprendre son souffle.",
    ),
    "antecedent_pulmonaire": (
        "Antécédent pulmonaire",
        "Asthme ou maladie pulmonaire connue.",
        "Ni asthme ni maladie pulmonaire connue.",
    ),
    "deficit": (
        "Déficit neurologique",
        "Faiblesse, engourdissement ou difficulté à parler.",
        "Ni faiblesse, ni engourdissement, ni difficulté à parler.",
    ),
    "heure_debut": ("Heure de début des symptômes", "", ""),
    "cephalee": (
        "Céphalée",
        "Céphalée apparue brutalement, en coup de tonnerre.",
        "Aucune céphalée brutale.",
    ),
    "localisation": ("Localisation", "", ""),
    "vomissements": (
        "Vomissements",
        "Vomissements, contenant du sang.",
        "Aucun vomissement.",
    ),
    "transit": ("Transit", "", ""),
    "mecanisme": ("Mécanisme du traumatisme", "", ""),
    "appui": (
        "Appui",
        "Appui et mobilisation du membre possibles.",
        "Appui ou mobilisation du membre impossibles.",
    ),
    "deformation": (
        "Aspect du membre",
        "Déformation, plaie ou perte de sensibilité.",
        "Ni déformation, ni plaie, ni perte de sensibilité.",
    ),
    "temperature": ("Température mesurée", "", ""),
    # L'intitulé de réponse libre est notre vocabulaire, pas celui du patient, et
    # la synthèse est relue par la règle de triage : un intitulé « Frissons »
    # suivi de « je ne sais pas » ferait apparaître « frissons » dans les raisons
    # affichées au soignant, alors que personne ne les a décrits. Les intitulés
    # restent donc hors du lexique de gravité ; les phrases canoniques, qui
    # rapportent une réponse réellement donnée, gardent le mot juste.
    "frissons": (
        "Signes cutanés et thermiques",
        "Frissons intenses ou éruption cutanée.",
        "Ni frissons intenses, ni éruption cutanée.",
    ),
    "nuque": (
        "Nuque",
        "Nuque souple.",
        "Nuque douloureuse à la flexion.",
    ),
    "intention": (
        "Intention exprimée",
        "Idées suicidaires exprimées, avec scénario.",
        "Aucune idée suicidaire exprimée.",
    ),
    "moyens": (
        "Moyens de passage à l'acte",
        "Le patient dispose de moyens pour passer à l'acte.",
        "Le patient ne dispose pas de moyens pour passer à l'acte.",
    ),
    "entourage": ("Entourage", "Patient accompagné.", "Patient seul."),
    "anciennete": ("Ancienneté des symptômes", "", ""),
    "evolution": ("Évolution", "", ""),
    "antecedents": ("Antécédents et traitements", "", ""),
}


# Mots-clés rattachant un motif à un thème. Le premier thème trouvé l'emporte,
# les familles étant ordonnées de la plus grave à la plus banale.
THEMES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("douleur_thoracique", ("thorax", "thoracique", "poitrine", "coeur", "cardiaque", "chest")),
    ("respiratoire", ("respir", "souffle", "essouffl", "asthme", "toux", "breath")),
    (
        "neurologique",
        (
            "tete",
            "cephalee",
            "mal de tete",
            "parole",
            "paralysie",
            "vertige",
            "malaise",
            "convuls",
            "head",
            "speech",
        ),
    ),
    ("psychiatrique", ("suicid", "angoisse", "deprim", "anxiete", "psy", "mal-etre")),
    (
        "traumatologie",
        (
            "chute",
            "choc",
            "entorse",
            "fracture",
            "plaie",
            "coupure",
            "brulure",
            "accident",
            "trauma",
        ),
    ),
    ("digestif", ("ventre", "abdomin", "digest", "vomiss", "diarrhee", "nausee", "stomach")),
    ("fievre", ("fievre", "temperature", "frisson", "fever")),
)


@dataclass(frozen=True)
class NextQuestion:
    """Prochaine étape du questionnaire."""

    identifiant: str | None
    texte: str | None
    termine: bool
    theme: str


def detect_theme(chief_complaint: str) -> str:
    """Rattache un motif de consultation à un thème clinique."""
    motif = normalize(chief_complaint)
    for theme, mots_cles in THEMES:
        if any(mot in motif for mot in mots_cles):
            return theme
    return "general"


def plan(chief_complaint: str) -> list[tuple[str, str]]:
    """Liste ordonnée des questions à poser pour ce motif."""
    theme = detect_theme(chief_complaint)
    specifiques = QUESTIONS_PAR_THEME.get(theme, QUESTIONS_GENERALES)
    return [*QUESTIONS_DE_GRAVITE, *specifiques]


def compile_symptoms(chief_complaint: str, answers: dict[str, str]) -> str:
    """Compile le motif et les réponses en une description exploitable.

    Chaque réponse devient une **phrase**, jamais une question suivie de son
    réponse. C'est ce qui permet au modèle de distinguer « pas de saignement »
    de « pas de difficulté à respirer », et c'est surtout ce qui évite que le
    vocabulaire des questions ne soit relu comme des symptômes par la règle de
    triage, qui s'applique ensuite à ce texte.
    """
    morceaux = [f"Motif : {chief_complaint.strip()}"]
    for identifiant, reponse in answers.items():
        phrase = _enoncer(identifiant, reponse)
        if phrase:
            morceaux.append(phrase)
    return " ".join(morceaux)


def _enoncer(identifiant: str, reponse: str) -> str:
    """Transforme une réponse en phrase, ou renvoie une chaîne vide si elle est vide.

    Une réponse binaire prend la formulation écrite pour elle. Une réponse libre
    est reportée derrière son intitulé, qui est un groupe nominal neutre et non
    la question posée.
    """
    texte = reponse.strip()
    if not texte:
        return ""
    intitule, affirmatif, negatif = RAPPORTS.get(identifiant, (identifiant, "", ""))
    binaire = _reponse_binaire(texte)
    if binaire == "oui" and affirmatif:
        return affirmatif
    if binaire == "non" and negatif:
        return negatif
    return f"{intitule} : {texte}."


# Réponse qui, pour une question de gravité, constitue un signe d'alerte.
# « Le patient est-il conscient ? non » et « Y a-t-il une difficulté à respirer ?
# oui » sont tous deux alarmants, mais l'un par la négative et l'autre par
# l'affirmative : le sens de chaque question doit être écrit, pas deviné.
REPONSE_ALARMANTE = {
    "conscience": "non",
    "respiration": "oui",
    "saignement": "oui",
    # Un « oui » à la question du déficit décrit un accident vasculaire en cours,
    # un « oui » à la céphalée en coup de tonnerre une hémorragie méningée, un
    # « non » à la parole une détresse respiratoire, un « non » à la nuque souple
    # un syndrome méningé. Continuer à poser des questions après l'une de ces
    # réponses n'a pas de sens.
    "deficit": "oui",
    "cephalee": "oui",
    "intention": "oui",
    "parole": "non",
    "nuque": "non",
    # `vomissements` en est absent à dessein : la question est double
    # (« y a-t-il des vomissements, et contiennent-ils du sang ? »), et un « oui »
    # nu ne dit pas lequel des deux. C'est le texte libre qui tranche, et la
    # règle de triage le relit.
}

AFFIRMATIONS = frozenset({"oui", "yes", "présent", "present", "positif", "positive"})
NEGATIONS = frozenset(
    {"non", "no", "aucun", "aucune", "absent", "absente", "négatif", "negatif", "rien"}
)

_MOTS = re.compile(r"[^\W\d_]+")


def _reponse_binaire(reponse: str) -> str | None:
    """Ramène une réponse à « oui », « non », ou rien si elle n'est pas binaire.

    Deux précautions, et elles sont toutes les deux nées d'un contre-exemple.

    La comparaison porte sur un **mot entier**, jamais sur un préfixe : « no »
    est une négation en anglais, et le début de « Notre », « Nous » et
    « Normalement » en français. Lire « Notre fille saigne du nez » comme un
    « non » inverse une réponse clinique sans que rien ne le signale.

    Et seule une réponse **réduite à ce seul mot** est traitée comme binaire.
    Dès qu'un soignant écrit autre chose, son texte est repris tel quel plus
    loin : c'est toujours le choix le plus sûr, puisque le texte part alors mot
    pour mot au modèle et que la règle de triage le relit, là où une réponse
    ramenée à « oui » ou « non » perd tout ce qu'elle portait.
    """
    mots = _MOTS.findall(reponse.strip().lower())
    if len(mots) != 1:
        return None
    if mots[0] in AFFIRMATIONS:
        return "oui"
    if mots[0] in NEGATIONS:
        return "non"
    return None


def has_red_flag(chief_complaint: str, answers: dict[str, str]) -> bool:
    """Dit si un signe de détresse vitale ressort déjà de la collecte.

    Attention au piège : appliquer la règle de triage à la synthèse complète
    reviendrait à la lui appliquer sur le **texte des questions**, qui contient
    par construction le vocabulaire des signes graves (« Y a-t-il une difficulté
    à respirer ? »). Le questionnaire s'arrêterait alors systématiquement dès la
    première question. On examine donc le motif et les réponses, jamais les
    questions.
    """
    if classify(chief_complaint) == VITAL:
        return True
    for identifiant, reponse in answers.items():
        # Les deux valeurs sont comparées seulement si la question attend
        # réellement une réponse binaire. Sans cette garde, une question qui n'en
        # attend pas — le mécanisme d'un traumatisme, la localisation d'une
        # douleur — confronterait deux `None` : toute réponse en texte libre y
        # passerait pour un signe de détresse vitale et arrêterait la collecte.
        attendue = REPONSE_ALARMANTE.get(identifiant)
        if attendue is not None and _reponse_binaire(reponse) == attendue:
            return True
        # La règle s'applique à **toute** réponse, y compris à celle qui commence
        # par « oui » : « oui, hémiplégie droite depuis trente minutes » porte le
        # signe dans ce qui suit le « oui », et le réserver aux réponses non
        # binaires reviendrait à ne jamais le lire.
        if classify(reponse) == VITAL:
            return True
    return False


def next_question(chief_complaint: str, answers: dict[str, str]) -> NextQuestion:
    """Détermine la prochaine question, ou signale la fin de la collecte."""
    theme = detect_theme(chief_complaint)
    if has_red_flag(chief_complaint, answers):
        # Un signe de détresse vitale est apparu : on arrête d'interroger.
        return NextQuestion(None, None, True, theme)
    for identifiant, texte in plan(chief_complaint):
        if identifiant not in answers:
            return NextQuestion(identifiant, texte, False, theme)
    return NextQuestion(None, None, True, theme)
