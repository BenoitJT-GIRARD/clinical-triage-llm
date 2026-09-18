"""Adaptive symptom questionnaire.

The agent collects symptoms through a questionnaire that adapts, and adapting means two things
here:

- **the questions depend on the complaint.** A patient arriving with chest pain and a patient
  arriving with a wound must not be asked the same things. The complaint is mapped to a
  clinical theme, and each theme has its own list of targeted questions;
- **collection stops as soon as it is no longer useful.** If a sign of life-threatening
  distress appears in an answer, questioning stops and triage fires immediately: making a heart
  attack wait through five questions would be absurd.

The logic is deterministic and readable: every question asked can be justified to a clinical
team, which is not incidental in this context.

One point that looks minor and is not: the summary handed to the model **never** copies the
question in front of its answer. Each answer becomes a declarative sentence written in advance
(the ``STATEMENTS`` table), because the text of the questions carries the whole vocabulary of
severity signs, and that summary is then read back by ``classify``. Copied verbatim, "Y a-t-il
une difficulté à respirer ? non" reads as the sign itself: a common cold with everything denied
would come out as life-threatening. Concatenating the answers alone does not work either —
"oui. non. depuis hier" is a run of words with no referent. The declarative sentence keeps the
meaning and does not carry the question.

The questions and the canonical sentences are French: they are spoken to French-speaking staff
and read back by the French triage rule. They are data, not prose about the project.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from clinical_triage.data.triage_rules import VITAL, classify, normalize

# Red-flag screening questions, asked whatever the complaint.
RED_FLAG_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("conscience", "Le patient est-il pleinement conscient et orienté ?"),
    ("respiration", "Y a-t-il une difficulté à respirer ou un essoufflement au repos ?"),
    ("saignement", "Existe-t-il un saignement actif ou important ?"),
)

# Questions specific to each clinical theme.
QUESTIONS_BY_THEME: dict[str, tuple[tuple[str, str], ...]] = {
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

# Theme used when the complaint matches no identified family.
GENERAL_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("anciennete", "Depuis combien de temps les symptômes sont-ils présents ?"),
    ("evolution", "Les symptômes s'aggravent-ils, stagnent-ils ou s'améliorent-ils ?"),
    ("antecedents", "Le patient a-t-il des antécédents ou des traitements en cours ?"),
)

# How the answer to each question is reported in the description handed to the model: a short
# label for free-text answers, and — for yes/no questions — the sentence to write in each case.
#
# Copying the question would be simpler, and would be wrong. By construction it contains the
# vocabulary of severity signs, and the compiled description is then read back by the triage
# rule: "Y a-t-il une difficulté à respirer ? non" classifies as life-threatening, with
# "difficulté à respirer" as the justification, for a patient who has just denied it. The rule,
# on the other hand, does handle a leading negation — "Pas de difficulté à respirer" — hence
# these wordings, written one by one.
STATEMENTS: dict[str, tuple[str, str, str]] = {
    # id: (label for a free-text answer, sentence if yes, sentence if no)
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
    # A free-text label is our vocabulary, not the patient's, and the summary is read back by
    # the triage rule: a label "Frissons" followed by "je ne sais pas" would make "frissons"
    # appear among the reasons shown to the nurse, when nobody described any. Labels therefore
    # stay outside the severity lexicon; the canonical sentences, which report an answer
    # actually given, keep the precise word.
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


# Keywords attaching a complaint to a theme. The first theme found wins, the families being
# ordered from the most serious to the most ordinary.
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
    """The next step of the questionnaire."""

    identifier: str | None
    text: str | None
    finished: bool
    theme: str


def detect_theme(chief_complaint: str) -> str:
    """Attach a presenting complaint to a clinical theme."""
    complaint = normalize(chief_complaint)
    for theme, keywords in THEMES:
        if any(word in complaint for word in keywords):
            return theme
    return "general"


def plan(chief_complaint: str) -> list[tuple[str, str]]:
    """The ordered list of questions to ask for this complaint."""
    theme = detect_theme(chief_complaint)
    specific = QUESTIONS_BY_THEME.get(theme, GENERAL_QUESTIONS)
    return [*RED_FLAG_QUESTIONS, *specific]


def compile_symptoms(chief_complaint: str, answers: dict[str, str]) -> str:
    """Compile the complaint and the answers into a usable description.

    Each answer becomes a **sentence**, never a question followed by its answer. That is what
    lets the model tell "no bleeding" from "no breathing difficulty", and above all what keeps
    the vocabulary of the questions from being read as symptoms by the triage rule, which is
    applied to this text next.
    """
    parts = [f"Motif : {chief_complaint.strip()}"]
    for identifier, answer in answers.items():
        sentence = _as_statement(identifier, answer)
        if sentence:
            parts.append(sentence)
    return " ".join(parts)


def _as_statement(identifier: str, answer: str) -> str:
    """Turn an answer into a sentence, or return an empty string if it is empty.

    A binary answer takes the wording written for it. A free-text answer is reported behind its
    label, which is a neutral noun phrase and not the question that was asked.
    """
    text = answer.strip()
    if not text:
        return ""
    label, affirmative, negative = STATEMENTS.get(identifier, (identifier, "", ""))
    binary = _binary_answer(text)
    if binary == "oui" and affirmative:
        return affirmative
    if binary == "non" and negative:
        return negative
    return f"{label} : {text}."


# The answer that, for a red-flag question, constitutes an alert. "Is the patient conscious?
# no" and "Is there any breathing difficulty? yes" are both alarming, but one in the negative
# and the other in the affirmative: the polarity of each question must be written, not guessed.
ALARMING_ANSWER = {
    "conscience": "non",
    "respiration": "oui",
    "saignement": "oui",
    # A "yes" to the deficit question describes a stroke in progress, a "yes" to the
    # thunderclap headache a subarachnoid haemorrhage, a "no" to speech a respiratory distress,
    # a "no" to a supple neck a meningeal syndrome. Carrying on asking questions after one of
    # those answers makes no sense.
    "deficit": "oui",
    "cephalee": "oui",
    "intention": "oui",
    "parole": "non",
    "nuque": "non",
    # `vomissements` is deliberately absent: the question is double ("is there vomiting, and
    # does it contain blood?"), and a bare "yes" does not say which. The free text decides, and
    # the triage rule reads it back.
}

AFFIRMATIONS = frozenset({"oui", "yes", "présent", "present", "positif", "positive"})
NEGATIONS = frozenset(
    {"non", "no", "aucun", "aucune", "absent", "absente", "négatif", "negatif", "rien"}
)

_WORDS = re.compile(r"[^\W\d_]+")


def _binary_answer(answer: str) -> str | None:
    """Reduce an answer to "oui", "non", or nothing when it is not binary.

    Two precautions, both born of a counter-example.

    The comparison is on a **whole word**, never a prefix: "no" is a negation in English, and
    the start of "Notre", "Nous" and "Normalement" in French. Reading "Notre fille saigne du
    nez" as a "no" inverts a clinical answer with nothing to report it.

    And only an answer **reduced to that single word** is treated as binary. As soon as a nurse
    writes anything else, their text is carried through verbatim further down: that is always
    the safer choice, since the text then goes word for word to the model and the triage rule
    reads it back, where an answer collapsed to "oui" or "non" loses everything it carried.
    """
    words = _WORDS.findall(answer.strip().lower())
    if len(words) != 1:
        return None
    if words[0] in AFFIRMATIONS:
        return "oui"
    if words[0] in NEGATIONS:
        return "non"
    return None


def has_red_flag(chief_complaint: str, answers: dict[str, str]) -> bool:
    """Say whether a sign of life-threatening distress already shows in the collection.

    Watch the trap: applying the triage rule to the full summary would amount to applying it to
    the **text of the questions**, which by construction contains the vocabulary of serious
    signs ("Y a-t-il une difficulté à respirer ?"). The questionnaire would then stop
    systematically at the first question. So the complaint and the answers are examined, never
    the questions.
    """
    if classify(chief_complaint) == VITAL:
        return True
    for identifier, answer in answers.items():
        # The two values are compared only when the question really expects a binary answer.
        # Without this guard, a question that does not — the mechanism of a trauma, the
        # location of a pain — would confront two `None`: any free-text answer would pass for a
        # sign of life-threatening distress and stop the collection.
        expected = ALARMING_ANSWER.get(identifier)
        if expected is not None and _binary_answer(answer) == expected:
            return True
        # The rule applies to **every** answer, including one that starts with "oui": "oui,
        # hémiplégie droite depuis trente minutes" carries the sign in what follows the "oui",
        # and reserving this to non-binary answers would amount to never reading it.
        if classify(answer) == VITAL:
            return True
    return False


def next_question(chief_complaint: str, answers: dict[str, str]) -> NextQuestion:
    """Determine the next question, or signal the end of collection."""
    theme = detect_theme(chief_complaint)
    if has_red_flag(chief_complaint, answers):
        # A sign of life-threatening distress has appeared: stop questioning.
        return NextQuestion(None, None, True, theme)
    for identifier, text in plan(chief_complaint):
        if identifier not in answers:
            return NextQuestion(identifier, text, False, theme)
    return NextQuestion(None, None, True, theme)
