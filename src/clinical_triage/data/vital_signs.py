"""Constantes vitales : seuils d'alerte, génération et mise en forme.

Le triage aux urgences ne repose pas uniquement sur le récit du patient : les
constantes mesurées à l'accueil (fréquence cardiaque, tension, fréquence
respiratoire, saturation, température, douleur, conscience) pèsent autant que
les symptômes. Ce module les représente explicitement, les met en texte pour le
modèle, et dit ce qu'elles déclenchent.

Les seuils adultes reprennent la logique du score d'alerte précoce NEWS2 et des
critères de tri 1-2 de l'échelle FRENCH. Les seuils pédiatriques varient avec
l'âge : on utilise une table de bandes d'âge, volontairement courte et lisible.

Ce module ne connaît rien du reste du projet : il décrit des mesures, pas des
décisions de triage. La décision est prise dans `triage_rules`.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass

# --- Bandes d'âge pédiatriques ---
# (âge max en années, FC normale, FR normale, systolique normale). Au-delà de
# 12 ans, les valeurs adultes s'appliquent.
#
# La systolique figure dans la bande au même titre que les fréquences : une
# tension d'adulte, « TA 120/75 », n'existe pas chez un nourrisson, et la règle
# la lirait pourtant comme normale.
PEDIATRIC_BANDS: tuple[tuple[int, tuple[int, int], tuple[int, int], tuple[int, int]], ...] = (
    (1, (100, 160), (30, 55), (72, 100)),
    (3, (90, 150), (22, 40), (80, 110)),
    (6, (80, 140), (20, 30), (85, 115)),
    (12, (70, 120), (18, 25), (95, 120)),
)
ADULT_HEART_RATE = (60, 100)
ADULT_RESP_RATE = (12, 20)
ADULT_SYSTOLIC = (100, 135)


@dataclass(frozen=True)
class VitalSigns:
    """Relevé de constantes tel qu'il est saisi à l'accueil des urgences.

    Chaque champ vaut `None` lorsque la mesure n'a pas été faite : à l'accueil,
    un relevé partiel est la règle plutôt que l'exception, et il faut pouvoir
    raisonner sur ce qui est disponible sans inventer le reste.
    """

    heart_rate: int | None = None  # battements par minute
    systolic_bp: int | None = None  # pression artérielle systolique, mmHg
    diastolic_bp: int | None = None  # pression artérielle diastolique, mmHg
    resp_rate: int | None = None  # cycles par minute
    spo2: int | None = None  # saturation en oxygène, %
    temperature: float | None = None  # °C
    pain_score: int | None = None  # échelle visuelle analogique, 0 à 10
    conscious: bool | None = None  # vigilance normale (éveillé et orienté)

    def is_empty(self) -> bool:
        """Dit si aucune mesure n'est disponible."""
        return all(getattr(self, champ) is None for champ in self.__dataclass_fields__)

    def render(self, lang: str) -> str:
        """Met les constantes disponibles en une ligne de texte."""
        parties: list[str] = []
        if lang == "fr":
            if self.heart_rate is not None:
                parties.append(f"FC {self.heart_rate}/min")
            if self.systolic_bp is not None and self.diastolic_bp is not None:
                parties.append(f"TA {self.systolic_bp}/{self.diastolic_bp} mmHg")
            if self.resp_rate is not None:
                parties.append(f"FR {self.resp_rate}/min")
            if self.spo2 is not None:
                parties.append(f"SpO2 {self.spo2} %")
            if self.temperature is not None:
                parties.append(f"T {self.temperature:.1f} °C")
            if self.pain_score is not None:
                parties.append(f"douleur {self.pain_score}/10")
            if self.conscious is not None:
                parties.append("vigilance normale" if self.conscious else "vigilance altérée")
            return ", ".join(parties)
        if self.heart_rate is not None:
            parties.append(f"HR {self.heart_rate}/min")
        if self.systolic_bp is not None and self.diastolic_bp is not None:
            parties.append(f"BP {self.systolic_bp}/{self.diastolic_bp} mmHg")
        if self.resp_rate is not None:
            parties.append(f"RR {self.resp_rate}/min")
        if self.spo2 is not None:
            parties.append(f"SpO2 {self.spo2}%")
        if self.temperature is not None:
            parties.append(f"T {self.temperature:.1f} °C")
        if self.pain_score is not None:
            parties.append(f"pain {self.pain_score}/10")
        if self.conscious is not None:
            parties.append("alert" if self.conscious else "altered consciousness")
        return ", ".join(parties)


def normal_ranges(age: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Plages normales pour un âge : fréquence cardiaque, respiratoire, systolique."""
    for max_age, heart_rate, resp_rate, systolic in PEDIATRIC_BANDS:
        if age <= max_age:
            return heart_rate, resp_rate, systolic
    return ADULT_HEART_RATE, ADULT_RESP_RATE, ADULT_SYSTOLIC


def critical_findings(vitals: VitalSigns, age: int) -> list[str]:
    """Anomalies imposant une prise en charge immédiate (détresse vitale).

    Les bornes basses se dérivent de la bande d'âge, comme les bornes hautes.
    Des valeurs fixes d'adulte, 8 et 40, masqueraient la bradycardie du
    nourrisson, dont la fréquence normale commence à 100 : un enfant de six mois
    à 52 battements par minute est en situation pré-arrêt.
    """
    heart_rate, resp_rate, systolic = normal_ranges(age)
    findings: list[str] = []
    if vitals.spo2 is not None and vitals.spo2 < 92:
        findings.append(f"saturation effondrée ({vitals.spo2} %)")
    if vitals.resp_rate is not None and (
        vitals.resp_rate < resp_rate[0] - 4 or vitals.resp_rate > resp_rate[1] + 8
    ):
        findings.append(f"fréquence respiratoire critique ({vitals.resp_rate}/min)")
    if vitals.heart_rate is not None and (
        vitals.heart_rate < heart_rate[0] - 20 or vitals.heart_rate > heart_rate[1] + 30
    ):
        findings.append(f"fréquence cardiaque critique ({vitals.heart_rate}/min)")
    if vitals.systolic_bp is not None and vitals.systolic_bp < systolic[0] - 10:
        findings.append(f"hypotension ({vitals.systolic_bp} mmHg de systolique)")
    # La tension est lue par le haut comme par le bas : une poussée à 230/130 est
    # une urgence au même titre qu'un effondrement tensionnel.
    if (vitals.systolic_bp is not None and vitals.systolic_bp >= 180) or (
        vitals.diastolic_bp is not None and vitals.diastolic_bp >= 110
    ):
        findings.append(
            f"poussée hypertensive sévère ({vitals.systolic_bp}/{vitals.diastolic_bp} mmHg)"
        )
    if vitals.temperature is not None and vitals.temperature < 35.0:
        findings.append(f"hypothermie ({vitals.temperature:.1f} °C)")
    if vitals.temperature is not None and vitals.temperature >= 41.0:
        findings.append(f"hyperthermie majeure ({vitals.temperature:.1f} °C)")
    if vitals.conscious is False:
        findings.append("trouble de la vigilance")
    return findings


def warning_findings(vitals: VitalSigns, age: int) -> list[str]:
    """Anomalies justifiant une évaluation médicale rapprochée, sans détresse vitale."""
    heart_rate, resp_rate, systolic = normal_ranges(age)
    findings: list[str] = []
    if vitals.spo2 is not None and 92 <= vitals.spo2 <= 94:
        findings.append(f"saturation limite ({vitals.spo2} %)")
    if vitals.resp_rate is not None and resp_rate[1] < vitals.resp_rate <= resp_rate[1] + 8:
        findings.append(f"polypnée ({vitals.resp_rate}/min)")
    if vitals.resp_rate is not None and resp_rate[0] - 4 <= vitals.resp_rate < resp_rate[0]:
        findings.append(f"bradypnée ({vitals.resp_rate}/min)")
    if vitals.heart_rate is not None and heart_rate[1] < vitals.heart_rate <= heart_rate[1] + 30:
        findings.append(f"tachycardie ({vitals.heart_rate}/min)")
    # Le ralentissement a son degré intermédiaire, comme l'accélération : sous la
    # borne basse de l'âge mais au-dessus du seuil critique, la bradycardie
    # justifie une évaluation sans relever de la détresse vitale.
    if vitals.heart_rate is not None and heart_rate[0] - 20 <= vitals.heart_rate < heart_rate[0]:
        findings.append(f"bradycardie ({vitals.heart_rate}/min)")
    if vitals.systolic_bp is not None and systolic[0] - 10 <= vitals.systolic_bp < systolic[0]:
        findings.append(f"tension basse ({vitals.systolic_bp} mmHg de systolique)")
    if (vitals.systolic_bp is not None and 160 <= vitals.systolic_bp < 180) or (
        vitals.diastolic_bp is not None and 100 <= vitals.diastolic_bp < 110
    ):
        findings.append(f"hypertension ({vitals.systolic_bp}/{vitals.diastolic_bp} mmHg)")
    if vitals.temperature is not None and vitals.temperature >= 38.5:
        findings.append(f"fièvre élevée ({vitals.temperature:.1f} °C)")
    if vitals.pain_score is not None and vitals.pain_score >= 7:
        findings.append(f"douleur intense ({vitals.pain_score}/10)")
    return findings


# --- Lecture des constantes dans un texte libre ---

# Chaque mesure est cherchée derrière les abréviations employées à l'accueil, en
# français comme en anglais. La règle de triage s'en sert pour lire les relevés
# écrits dans le récit ; sans cela, on la comparerait au modèle en la privant
# d'une information que le modèle, lui, voit.
MESURES = {
    "heart_rate": re.compile(
        r"\b(?:FC|HR|pouls|pulse|heart rate|fréquence cardiaque)\b\D{0,12}?(\d{2,3})", re.IGNORECASE
    ),
    "resp_rate": re.compile(
        r"\b(?:FR|RR|fréquence respiratoire|respiratory rate)\b\D{0,12}?(\d{1,3})", re.IGNORECASE
    ),
    "spo2": re.compile(r"\b(?:SpO2|SaO2|saturation|sats?)\b\D{0,12}?(\d{2,3})", re.IGNORECASE),
    "pain_score": re.compile(r"\b(?:douleur|pain|EVA)\b\D{0,12}?(\d{1,2})\s*/\s*10", re.IGNORECASE),
}
TENSION = re.compile(
    r"\b(?:TA|BP|tension|blood pressure)\b\D{0,12}?(\d{2,3})\s*/\s*(\d{2,3})", re.IGNORECASE
)
# Deux formes, et la distinction entre les deux est délibérée :
#
#   - avec un mot d'annonce (« T 39 °C », « fever 39C »), la partie décimale est
#     facultative : le mot lève l'ambiguïté ;
#   - sans mot d'annonce, la décimale est exigée. « 38,2 » dans une phrase est
#     presque sûrement une température ; « 38 » tout seul peut être un âge, une
#     fréquence respiratoire ou un numéro de département.
#
# La fin du nombre est marquée par « pas un autre chiffre » et non par une
# frontière de mot : « 39.1C », sans espace avant l'unité, est une écriture
# courante qu'un `\b` rejetterait.
_ANNONCE_TEMPERATURE = r"(?:T|temp(?:érature|erature)?|fièvre|fever|fébrile|febrile)"
TEMPERATURE = re.compile(
    rf"\b{_ANNONCE_TEMPERATURE}\b\D{{0,14}}?(3[3-9]|4[0-2])(?:[.,](\d))?(?!\d)"
    rf"|\b(3[3-9]|4[0-2])[.,](\d)(?!\d)",
    re.IGNORECASE,
)
# Motifs écrits sans accents : la lecture se fait sur un texte normalisé, comme
# dans la règle de triage. Une note d'accueil écrit « vigilance alteree » aussi
# souvent qu'avec les accents, et un motif accentué ne la verrait pas.
VIGILANCE_ALTEREE = re.compile(
    r"vigilance alteree|trouble de la (?:vigilance|conscience)|inconscient|non conscient"
    r"|pas conscient|desoriente|obnubile|somnolent|somnole|confus"
    r"|ne repond pas aux stimulations"
    r"|altered consciousness|unresponsive|drowsy|not rousable|not conscious",
    re.IGNORECASE,
)
VIGILANCE_NORMALE = re.compile(
    r"vigilance normale|eveille et oriente|conscient et oriente|fully alert|\balert\b",
    re.IGNORECASE,
)

# Marqueurs de négation cherchés juste avant un état de vigilance.
_NEGATION_VIGILANCE = re.compile(
    r"\b(?:sans|pas de|aucun|aucune|absence de|no|not|without|denies)\b[^.;:,]{0,20}$",
    re.IGNORECASE,
)


def _sans_accents(texte: str) -> str:
    """Minuscules sans accents, comme le fait la règle de triage."""
    decompose = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in decompose if not unicodedata.combining(c))


def _affirme(motif: re.Pattern[str], texte: str) -> bool:
    """Dit si `motif` apparaît dans `texte` sans être nié juste avant.

    La négation est prise en compte, comme l'annonce la règle de triage : sans
    cette garde, « patient sans trouble de la conscience » rendrait
    `conscious=False` et classerait un rhume banal en urgence vitale.
    """
    for trouve in motif.finditer(texte):
        amont = texte[max(0, trouve.start() - 28) : trouve.start()]
        if not _NEGATION_VIGILANCE.search(amont):
            return True
    return False


def parse(text: str) -> VitalSigns:
    """Lit dans un texte libre les constantes qui y sont écrites.

    Les mesures absentes restent à `None`. Un relevé vide est un résultat
    valide : beaucoup de descriptions n'en contiennent aucune.
    """
    valeurs: dict[str, int | float | bool | None] = {}
    for champ, motif in MESURES.items():
        trouve = motif.search(text)
        if trouve:
            valeurs[champ] = int(trouve.group(1))
    tension = TENSION.search(text)
    if tension:
        valeurs["systolic_bp"] = int(tension.group(1))
        valeurs["diastolic_bp"] = int(tension.group(2))
    temperature = TEMPERATURE.search(text)
    if temperature:
        # Le motif a deux alternatives : celle avec mot d'annonce (groupes 1 et
        # 2, la décimale pouvant manquer) et celle sans (groupes 3 et 4).
        entier = temperature.group(1) or temperature.group(3)
        decimale = temperature.group(2) or temperature.group(4) or "0"
        valeurs["temperature"] = float(f"{entier}.{decimale}")
    plat = _sans_accents(text)
    if _affirme(VIGILANCE_ALTEREE, plat):
        valeurs["conscious"] = False
    elif _affirme(VIGILANCE_NORMALE, plat):
        valeurs["conscious"] = True
    return VitalSigns(**valeurs)


AGE_EN_ANNEES = re.compile(r"\b(\d{1,3})\s*(?:ans?\b|[- ]?(?:year|yr)[- ]?old\b)", re.IGNORECASE)
AGE_EN_MOIS = re.compile(r"\b(\d{1,2})\s*(?:mois\b|[- ]?month[- ]?old\b)", re.IGNORECASE)

# Âge retenu lorsque le texte n'en donne aucun : les seuils adultes sont alors
# appliqués, ce qui est le réglage le moins risqué pour un adulte inconnu.
AGE_PAR_DEFAUT = 40

# Ce qui, juste avant un nombre, en fait une durée et non un âge : « toux depuis
# 3 mois », « enceinte de 8 mois », « asthme depuis 40 ans ». Une note d'accueil
# mêle les deux dans la même phrase, et prendre l'ancienneté pour l'âge fait
# basculer un adulte sur les seuils du nourrisson.
DUREE_AVANT_UN_NOMBRE = re.compile(
    r"\b(?:depuis|il y a|pendant|apr[eè]s|enceinte de|for|since|over the (?:last|past))\s*$",
    re.IGNORECASE,
)


def _est_une_duree(text: str, debut: int) -> bool:
    """Dit si le nombre trouvé à `debut` décrit une durée plutôt qu'un âge."""
    return DUREE_AVANT_UN_NOMBRE.search(text[max(0, debut - 24) : debut]) is not None


def parse_age(text: str) -> int:
    """Lit l'âge du patient dans le texte, en années.

    Les mois sont ramenés à des années entières, arrondies vers le bas : un
    nourrisson de dix mois compte comme 0 an, ce qui applique bien les seuils de
    la première bande pédiatrique.

    Les années sont cherchées **avant** les mois, et les durées sont écartées.
    Sans ces deux précautions, « homme de 58 ans, toux depuis 3 mois » donnerait
    0 an : le patient passerait sur les seuils du nourrisson, où une fréquence
    cardiaque à 128 est normale, et ses anomalies deviendraient invisibles.

    **Un texte sans âge rend l'âge adulte par défaut**, pas `None` : les seuils
    doivent bien être choisis, et les bandes adultes sont les plus prudentes ici.
    Une fréquence cardiaque de 130 est normale chez un nourrisson et alarmante
    chez un adulte ; en l'absence d'information, mieux vaut alerter à tort que se
    taire.
    """
    for motif, mois_par_unite in ((AGE_EN_ANNEES, 1), (AGE_EN_MOIS, 12)):
        for trouve in motif.finditer(text):
            if not _est_une_duree(text, trouve.start()):
                return int(trouve.group(1)) // mois_par_unite
    return AGE_PAR_DEFAUT


# --- Génération de relevés pour les vignettes cliniques ---

# Profils de constantes associés aux trois degrés de gravité. Chaque valeur est
# un intervalle dont on tire au hasard, autour des plages normales de l'âge.
# Deux réglages méritent un mot.
#
# La tension s'exprime en **fraction de la borne basse normale de l'âge**, pas en
# millimètres de mercure : un nourrisson en état de choc n'a pas la tension d'un
# adulte en état de choc, et 72 mmHg est un chiffre d'adulte.
#
# La température critique est **bimodale** : sans sa branche haute, les vignettes
# de sepsis, dont le motif est « fièvre avec frissons », sortiraient apyrétiques.
# Les deux branches sont cliniquement justes — un sepsis grave peut dériver dans
# un sens comme dans l'autre — mais le récit doit pouvoir imposer la sienne.
PROFILES = {
    "critique": {
        "spo2": (84, 91),
        "part_systolique": (0.72, 0.92),
        "systolique_haute": (185, 220),
        "temperature": (34.2, 34.9),
        "temperature_haute": (39.5, 41.5),
        "pain_score": (7, 10),
        "conscious_probability": 0.35,
        "heart_rate_offset": (35, 55),
        "resp_rate_offset": (9, 16),
    },
    "intermediaire": {
        "spo2": (92, 94),
        "part_systolique": (0.92, 0.99),
        "systolique_haute": (162, 179),
        "temperature": (38.5, 39.8),
        "temperature_haute": (38.5, 39.8),
        "pain_score": (5, 8),
        "conscious_probability": 1.0,
        "heart_rate_offset": (12, 28),
        "resp_rate_offset": (2, 7),
    },
    "normal": {
        "spo2": (96, 99),
        "part_systolique": (1.0, 1.0),
        "systolique_haute": None,
        "temperature": (36.3, 37.3),
        "temperature_haute": (36.3, 37.3),
        "pain_score": (0, 3),
        "conscious_probability": 1.0,
        "heart_rate_offset": (-8, 8),
        "resp_rate_offset": (-2, 2),
    },
}


# Âge à partir duquel une douleur s'auto-évalue sur une échelle de 0 à 10.
#
# L'EVA demande au patient de situer sa douleur lui-même : elle est inutilisable
# avant quatre à six ans, et en pédiatrie on emploie EVENDOL, FLACC ou l'échelle
# des visages. En dessous de cet âge, la vignette ne porte aucun score : un
# « douleur 9/10 » sur une bronchiolite du nourrisson est une mesure qui ne peut
# pas exister, et ce score déclencherait en plus le signe « douleur intense ».
AGE_MINIMUM_AUTO_EVALUATION = 6


def _douleur_auto_evaluee(age: int, plage: tuple[int, int], rng: random.Random) -> int | None:
    """Score de douleur, seulement quand le patient peut le donner lui-même."""
    if age < AGE_MINIMUM_AUTO_EVALUATION:
        return None
    return rng.randint(*plage)


def generate(
    profile: str,
    age: int,
    rng: random.Random,
    imposees: tuple[tuple[str, str], ...] = (),
    vigilance_alteree_probable: float | None = None,
) -> VitalSigns:
    """Tire un relevé de constantes cohérent avec un profil de gravité et un âge.

    Le profil `critique` ne dégrade pas toutes les constantes à la fois : on en
    choisit une ou deux, comme dans la réalité, le reste restant dans les plages
    normales. Sinon toutes les vignettes graves seraient reconnaissables à un
    relevé uniformément catastrophique.

    `imposees` liste les constantes que le **récit nomme**, avec leur sens :
    `(("temperature", "haute"),)` pour une présentation dont le motif est
    « fièvre avec frissons ». Ces constantes-là sont toujours dégradées, et dans
    le bon sens. Sans ce mécanisme, le tirage au sort laisserait sortir des sepsis
    apyrétiques et des pré-éclampsies normotendues : la description dirait une
    chose et le relevé la contredirait, dans le même exemple d'entraînement.
    """
    settings = PROFILES[profile]
    heart_rate_range, resp_rate_range, systolic_range = normal_ranges(age)
    normal_heart_rate = rng.randint(*heart_rate_range)
    normal_resp_rate = rng.randint(*resp_rate_range)
    normal_systolic = rng.randint(*systolic_range)

    if profile == "normal":
        # On borne aux plages normales de l'âge : un cas étiqueté « consultation
        # différée » ne doit jamais sortir avec une constante en alerte.
        jittered_heart_rate = normal_heart_rate + rng.randint(*settings["heart_rate_offset"])
        jittered_resp_rate = normal_resp_rate + rng.randint(*settings["resp_rate_offset"])
        return VitalSigns(
            heart_rate=min(max(jittered_heart_rate, heart_rate_range[0]), heart_rate_range[1]),
            systolic_bp=normal_systolic,
            # La diastolique reste à distance de la systolique : une différentielle
            # de quinze millimètres ne s'observe pas chez un patient qui va bien.
            diastolic_bp=min(
                rng.randint(*_diastolique_normale(systolic_range)), normal_systolic - 25
            ),
            resp_rate=min(max(jittered_resp_rate, resp_rate_range[0]), resp_rate_range[1]),
            spo2=rng.randint(*settings["spo2"]),
            temperature=round(rng.uniform(*settings["temperature"]), 1),
            pain_score=_douleur_auto_evaluee(age, settings["pain_score"], rng),
            conscious=True,
        )

    # Constantes de départ normales, puis dégradation d'une ou deux d'entre elles.
    values = {
        "heart_rate": normal_heart_rate,
        "systolic_bp": normal_systolic,
        "diastolic_bp": rng.randint(*_diastolique_normale(systolic_range)),
        "resp_rate": normal_resp_rate,
        "spo2": rng.randint(96, 99),
        "temperature": round(rng.uniform(36.3, 37.3), 1),
        "pain_score": _douleur_auto_evaluee(age, settings["pain_score"], rng),
        # La vigilance dépend de la présentation, pas de la gravité : un
        # infarctus laisse le patient parfaitement conscient. Faute d'indication,
        # on retombe sur le profil.
        "conscious": not (
            rng.random()
            < (
                vigilance_alteree_probable
                if vigilance_alteree_probable is not None
                else 1 - settings["conscious_probability"]
            )
        ),
    }

    sens = dict(imposees)
    candidates = ["spo2", "systolic_bp", "temperature", "heart_rate", "resp_rate"]
    libres = [nom for nom in candidates if nom not in sens]
    tirees = rng.sample(libres, k=min(rng.randint(1, 2), len(libres)))

    for name in [*sens, *tirees]:
        direction = sens.get(name, "basse")
        if name == "heart_rate":
            values["heart_rate"] = normal_heart_rate + rng.randint(*settings["heart_rate_offset"])
        elif name == "resp_rate":
            values["resp_rate"] = normal_resp_rate + rng.randint(*settings["resp_rate_offset"])
        elif name == "temperature":
            plage = (
                settings["temperature_haute"] if direction == "haute" else settings["temperature"]
            )
            values["temperature"] = round(rng.uniform(*plage), 1)
        elif name == "systolic_bp":
            if direction == "haute" and settings["systolique_haute"]:
                values["systolic_bp"] = rng.randint(*settings["systolique_haute"])
                values["diastolic_bp"] = rng.randint(102, 128)
            else:
                basse, haute = settings["part_systolique"]
                values["systolic_bp"] = rng.randint(
                    round(systolic_range[0] * basse), round(systolic_range[0] * haute)
                )
        else:
            values[name] = rng.randint(*settings[name])

    values["diastolic_bp"] = min(values["diastolic_bp"], values["systolic_bp"] - 25)
    return VitalSigns(**values)


def _diastolique_normale(systolic_range: tuple[int, int]) -> tuple[int, int]:
    """Plage diastolique plausible pour une systolique donnée : environ deux tiers."""
    return (round(systolic_range[0] * 0.62), round(systolic_range[1] * 0.66))
