"""Vital signs: alert thresholds, generation and rendering.

Emergency triage does not rest on the patient's narrative alone: the readings taken at
reception (heart rate, blood pressure, respiratory rate, saturation, temperature, pain,
consciousness) weigh as much as the symptoms. This module represents them explicitly, renders
them as text for the model, and says what they trigger.

The adult thresholds follow the logic of the NEWS2 early-warning score and of the sort 1-2
criteria of the FRENCH scale. The paediatric thresholds vary with age: a table of age bands,
deliberately short and readable.

This module knows nothing of the rest of the project: it describes measurements, not triage
decisions. The decision is taken in ``triage_rules``.

The finding labels are French because they travel to the nurse's screen through the API's
``rule_reasons``: they are output, not prose about the project.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass

# --- Paediatric age bands ---
# (max age in years, normal HR, normal RR, normal systolic). Above 12 years the adult values
# apply.
#
# The systolic sits in the band like the rates do: an adult blood pressure, "TA 120/75", does
# not exist in an infant, and the rule would nonetheless read it as normal.
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
    """A set of readings as entered at an emergency reception desk.

    Every field is ``None`` when the measurement was not taken: at reception a partial reading
    is the rule rather than the exception, and one must be able to reason on what is available
    without inventing the rest.
    """

    heart_rate: int | None = None  # beats per minute
    systolic_bp: int | None = None  # systolic blood pressure, mmHg
    diastolic_bp: int | None = None  # diastolic blood pressure, mmHg
    resp_rate: int | None = None  # breaths per minute
    spo2: int | None = None  # oxygen saturation, %
    temperature: float | None = None  # °C
    pain_score: int | None = None  # visual analogue scale, 0 to 10
    conscious: bool | None = None  # normal alertness (awake and oriented)

    def is_empty(self) -> bool:
        """Say whether no measurement is available."""
        return all(getattr(self, field) is None for field in self.__dataclass_fields__)

    def render(self, lang: str) -> str:
        """Render the available readings as one line of text."""
        parts: list[str] = []
        if lang == "fr":
            if self.heart_rate is not None:
                parts.append(f"FC {self.heart_rate}/min")
            if self.systolic_bp is not None and self.diastolic_bp is not None:
                parts.append(f"TA {self.systolic_bp}/{self.diastolic_bp} mmHg")
            if self.resp_rate is not None:
                parts.append(f"FR {self.resp_rate}/min")
            if self.spo2 is not None:
                parts.append(f"SpO2 {self.spo2} %")
            if self.temperature is not None:
                parts.append(f"T {self.temperature:.1f} °C")
            if self.pain_score is not None:
                parts.append(f"douleur {self.pain_score}/10")
            if self.conscious is not None:
                parts.append("vigilance normale" if self.conscious else "vigilance altérée")
            return ", ".join(parts)
        if self.heart_rate is not None:
            parts.append(f"HR {self.heart_rate}/min")
        if self.systolic_bp is not None and self.diastolic_bp is not None:
            parts.append(f"BP {self.systolic_bp}/{self.diastolic_bp} mmHg")
        if self.resp_rate is not None:
            parts.append(f"RR {self.resp_rate}/min")
        if self.spo2 is not None:
            parts.append(f"SpO2 {self.spo2}%")
        if self.temperature is not None:
            parts.append(f"T {self.temperature:.1f} °C")
        if self.pain_score is not None:
            parts.append(f"pain {self.pain_score}/10")
        if self.conscious is not None:
            parts.append("alert" if self.conscious else "altered consciousness")
        return ", ".join(parts)


def normal_ranges(age: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Normal ranges for an age: heart rate, respiratory rate, systolic."""
    for max_age, heart_rate, resp_rate, systolic in PEDIATRIC_BANDS:
        if age <= max_age:
            return heart_rate, resp_rate, systolic
    return ADULT_HEART_RATE, ADULT_RESP_RATE, ADULT_SYSTOLIC


def critical_findings(vitals: VitalSigns, age: int) -> list[str]:
    """Abnormalities that call for immediate care.

    The lower bounds derive from the age band, as the upper ones do. Fixed adult values, 8 and
    40, would hide an infant's bradycardia, whose normal rate starts at 100: a six-month-old at
    52 beats per minute is pre-arrest.
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
    # Blood pressure is read from above as well as from below: a surge to 230/130 is an
    # emergency just as a collapse is.
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
    """Abnormalities calling for prompt medical assessment, short of vital distress."""
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
    # Slowing has its intermediate degree, as acceleration does: below the lower bound for the
    # age but above the critical threshold, bradycardia warrants assessment without amounting
    # to vital distress.
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


# --- Reading vital signs out of free text ---

# Each measurement is looked for behind the abbreviations used at reception, in French as in
# English. The triage rule uses this to read the readings written into the narrative; without
# it, the rule would be compared to the model while being deprived of information the model
# does see.
MEASURES = {
    "heart_rate": re.compile(
        r"\b(?:FC|HR|pouls|pulse|heart rate|fréquence cardiaque)\b\D{0,12}?(\d{2,3})", re.IGNORECASE
    ),
    "resp_rate": re.compile(
        r"\b(?:FR|RR|fréquence respiratoire|respiratory rate)\b\D{0,12}?(\d{1,3})", re.IGNORECASE
    ),
    "spo2": re.compile(r"\b(?:SpO2|SaO2|saturation|sats?)\b\D{0,12}?(\d{2,3})", re.IGNORECASE),
    "pain_score": re.compile(r"\b(?:douleur|pain|EVA)\b\D{0,12}?(\d{1,2})\s*/\s*10", re.IGNORECASE),
}
BLOOD_PRESSURE = re.compile(
    r"\b(?:TA|BP|tension|blood pressure)\b\D{0,12}?(\d{2,3})\s*/\s*(\d{2,3})", re.IGNORECASE
)
# Two forms, and the distinction between them is deliberate:
#
#   - with a cue word ("T 39 °C", "fever 39C"), the decimal part is optional: the word removes
#     the ambiguity;
#   - without a cue word, the decimal is required. "38,2" in a sentence is almost certainly a
#     temperature; "38" alone could be an age, a respiratory rate or a postcode.
#
# The end of the number is marked by "not another digit" and not by a word boundary: "39.1C",
# with no space before the unit, is a common spelling that a `\b` would reject.
_TEMPERATURE_CUE = r"(?:T|temp(?:érature|erature)?|fièvre|fever|fébrile|febrile)"
TEMPERATURE = re.compile(
    rf"\b{_TEMPERATURE_CUE}\b\D{{0,14}}?(3[3-9]|4[0-2])(?:[.,](\d))?(?!\d)"
    rf"|\b(3[3-9]|4[0-2])[.,](\d)(?!\d)",
    re.IGNORECASE,
)
# Patterns written without accents: reading happens on a normalised text, as in the triage
# rule. A reception note writes "vigilance alteree" as often as the accented form, and an
# accented pattern would not see it.
ALTERED_CONSCIOUSNESS = re.compile(
    r"vigilance alteree|trouble de la (?:vigilance|conscience)|inconscient|non conscient"
    r"|pas conscient|desoriente|obnubile|somnolent|somnole|confus"
    r"|ne repond pas aux stimulations"
    r"|altered consciousness|unresponsive|drowsy|not rousable|not conscious",
    re.IGNORECASE,
)
NORMAL_CONSCIOUSNESS = re.compile(
    r"vigilance normale|eveille et oriente|conscient et oriente|fully alert|\balert\b",
    re.IGNORECASE,
)

# Negation markers looked for just before a consciousness state.
_NEGATION_BEFORE = re.compile(
    r"\b(?:sans|pas de|aucun|aucune|absence de|no|not|without|denies)\b[^.;:,]{0,20}$",
    re.IGNORECASE,
)


def _without_accents(text: str) -> str:
    """Lower case without accents, as the triage rule does."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _asserted(pattern: re.Pattern[str], text: str) -> bool:
    """Say whether ``pattern`` appears in ``text`` without being negated just before.

    Negation is taken into account, as the triage rule announces: without this guard, "patient
    sans trouble de la conscience" would yield ``conscious=False`` and file a common cold as
    life-threatening.
    """
    for found in pattern.finditer(text):
        upstream = text[max(0, found.start() - 28) : found.start()]
        if not _NEGATION_BEFORE.search(upstream):
            return True
    return False


def parse(text: str) -> VitalSigns:
    """Read out of free text the vital signs written in it.

    Missing measurements stay at ``None``. An empty reading is a valid result: many descriptions
    contain none.
    """
    values: dict[str, int | float | bool | None] = {}
    for field, pattern in MEASURES.items():
        found = pattern.search(text)
        if found:
            values[field] = int(found.group(1))
    pressure = BLOOD_PRESSURE.search(text)
    if pressure:
        values["systolic_bp"] = int(pressure.group(1))
        values["diastolic_bp"] = int(pressure.group(2))
    temperature = TEMPERATURE.search(text)
    if temperature:
        # The pattern has two alternatives: the one with a cue word (groups 1 and 2, the decimal
        # possibly missing) and the one without (groups 3 and 4).
        whole = temperature.group(1) or temperature.group(3)
        decimal = temperature.group(2) or temperature.group(4) or "0"
        values["temperature"] = float(f"{whole}.{decimal}")
    flat = _without_accents(text)
    if _asserted(ALTERED_CONSCIOUSNESS, flat):
        values["conscious"] = False
    elif _asserted(NORMAL_CONSCIOUSNESS, flat):
        values["conscious"] = True
    return VitalSigns(**values)


AGE_IN_YEARS = re.compile(r"\b(\d{1,3})\s*(?:ans?\b|[- ]?(?:year|yr)[- ]?old\b)", re.IGNORECASE)
AGE_IN_MONTHS = re.compile(r"\b(\d{1,2})\s*(?:mois\b|[- ]?month[- ]?old\b)", re.IGNORECASE)

# Age used when the text gives none: the adult thresholds then apply, which is the least risky
# setting for an unknown adult.
DEFAULT_AGE = 40

# What, just before a number, makes it a duration and not an age: "toux depuis 3 mois",
# "enceinte de 8 mois", "asthme depuis 40 ans". A reception note mixes both in the same
# sentence, and taking the duration for the age moves an adult onto infant thresholds.
DURATION_BEFORE_A_NUMBER = re.compile(
    r"\b(?:depuis|il y a|pendant|apr[eè]s|enceinte de|for|since|over the (?:last|past))\s*$",
    re.IGNORECASE,
)


def _is_a_duration(text: str, start: int) -> bool:
    """Say whether the number found at ``start`` describes a duration rather than an age."""
    return DURATION_BEFORE_A_NUMBER.search(text[max(0, start - 24) : start]) is not None


def parse_age(text: str) -> int:
    """Read the patient's age from the text, in years.

    Months are brought back to whole years, rounded down: a ten-month-old counts as 0, which
    correctly applies the thresholds of the first paediatric band.

    Years are looked for **before** months, and durations are discarded. Without those two
    precautions, "homme de 58 ans, toux depuis 3 mois" would give 0: the patient would move onto
    infant thresholds, where a heart rate of 128 is normal, and their abnormalities would become
    invisible.

    **A text with no age returns the default adult age**, not ``None``: the thresholds must be
    chosen one way or another, and the adult bands are the most cautious here. A heart rate of
    130 is normal in an infant and alarming in an adult; absent information, better to alert
    wrongly than to stay silent.
    """
    for pattern, months_per_unit in ((AGE_IN_YEARS, 1), (AGE_IN_MONTHS, 12)):
        for found in pattern.finditer(text):
            if not _is_a_duration(text, found.start()):
                return int(found.group(1)) // months_per_unit
    return DEFAULT_AGE


# --- Generating readings for the clinical vignettes ---

# Vital-sign profiles attached to the three degrees of severity. Each value is an interval drawn
# from, around the normal ranges for the age. Two settings deserve a word.
#
# Blood pressure is expressed as a **fraction of the lower normal bound for the age**, not in
# millimetres of mercury: an infant in shock does not have an adult's pressure, and 72 mmHg is
# an adult figure.
#
# The critical temperature is **bimodal**: without its high branch, sepsis vignettes — whose
# complaint is "fever with chills" — would come out afebrile. Both branches are clinically
# correct — severe sepsis can drift either way — but the narrative must be able to impose its
# own.
PROFILES = {
    "critical": {
        "spo2": (84, 91),
        "systolic_share": (0.72, 0.92),
        "systolic_high": (185, 220),
        "temperature": (34.2, 34.9),
        "temperature_high": (39.5, 41.5),
        "pain_score": (7, 10),
        "conscious_probability": 0.35,
        "heart_rate_offset": (35, 55),
        "resp_rate_offset": (9, 16),
    },
    "intermediate": {
        "spo2": (92, 94),
        "systolic_share": (0.92, 0.99),
        "systolic_high": (162, 179),
        "temperature": (38.5, 39.8),
        "temperature_high": (38.5, 39.8),
        "pain_score": (5, 8),
        "conscious_probability": 1.0,
        "heart_rate_offset": (12, 28),
        "resp_rate_offset": (2, 7),
    },
    "normal": {
        "spo2": (96, 99),
        "systolic_share": (1.0, 1.0),
        "systolic_high": None,
        "temperature": (36.3, 37.3),
        "temperature_high": (36.3, 37.3),
        "pain_score": (0, 3),
        "conscious_probability": 1.0,
        "heart_rate_offset": (-8, 8),
        "resp_rate_offset": (-2, 2),
    },
}


# Age from which pain is self-rated on a 0-to-10 scale.
#
# The visual analogue scale asks the patient to place their own pain: it is unusable before four
# to six years, and paediatrics uses EVENDOL, FLACC or the faces scale instead. Below this age
# the vignette carries no score: a "pain 9/10" on an infant's bronchiolitis is a measurement
# that cannot exist, and that score would additionally trigger the "severe pain" sign.
MIN_AGE_FOR_SELF_REPORT = 6


def _self_reported_pain(age: int, span: tuple[int, int], rng: random.Random) -> int | None:
    """A pain score, only when the patient can give it themselves."""
    if age < MIN_AGE_FOR_SELF_REPORT:
        return None
    return rng.randint(*span)


def generate(
    profile: str,
    age: int,
    rng: random.Random,
    forced: tuple[tuple[str, str], ...] = (),
    altered_consciousness_likely: float | None = None,
) -> VitalSigns:
    """Draw a set of readings consistent with a severity profile and an age.

    The ``critical`` profile does not degrade every reading at once: one or two are chosen, as
    in reality, the rest staying inside the normal ranges. Otherwise every serious vignette
    would be recognisable by a uniformly catastrophic reading.

    ``forced`` lists the readings the **narrative names**, with their direction:
    ``(("temperature", "high"),)`` for a presentation whose complaint is "fever with chills".
    Those readings are always degraded, and in the right direction. Without this mechanism the
    random draw would let afebrile sepsis and normotensive pre-eclampsia out: the description
    would say one thing and the reading contradict it, inside the same training example.
    """
    settings = PROFILES[profile]
    heart_rate_range, resp_rate_range, systolic_range = normal_ranges(age)
    normal_heart_rate = rng.randint(*heart_rate_range)
    normal_resp_rate = rng.randint(*resp_rate_range)
    normal_systolic = rng.randint(*systolic_range)

    if profile == "normal":
        # Bounded to the normal ranges for the age: a case labelled "deferred consultation" must
        # never come out with a reading in the alert zone.
        jittered_heart_rate = normal_heart_rate + rng.randint(*settings["heart_rate_offset"])
        jittered_resp_rate = normal_resp_rate + rng.randint(*settings["resp_rate_offset"])
        return VitalSigns(
            heart_rate=min(max(jittered_heart_rate, heart_rate_range[0]), heart_rate_range[1]),
            systolic_bp=normal_systolic,
            # The diastolic stays at a distance from the systolic: a fifteen-millimetre pulse
            # pressure is not observed in a patient who is doing well.
            diastolic_bp=min(rng.randint(*_normal_diastolic(systolic_range)), normal_systolic - 25),
            resp_rate=min(max(jittered_resp_rate, resp_rate_range[0]), resp_rate_range[1]),
            spo2=rng.randint(*settings["spo2"]),
            temperature=round(rng.uniform(*settings["temperature"]), 1),
            pain_score=_self_reported_pain(age, settings["pain_score"], rng),
            conscious=True,
        )

    # Start from normal readings, then degrade one or two of them.
    values = {
        "heart_rate": normal_heart_rate,
        "systolic_bp": normal_systolic,
        "diastolic_bp": rng.randint(*_normal_diastolic(systolic_range)),
        "resp_rate": normal_resp_rate,
        "spo2": rng.randint(96, 99),
        "temperature": round(rng.uniform(36.3, 37.3), 1),
        "pain_score": _self_reported_pain(age, settings["pain_score"], rng),
        # Consciousness depends on the presentation, not on severity: a heart attack leaves the
        # patient perfectly conscious. Absent an indication, fall back on the profile.
        "conscious": not (
            rng.random()
            < (
                altered_consciousness_likely
                if altered_consciousness_likely is not None
                else 1 - settings["conscious_probability"]
            )
        ),
    }

    directions = dict(forced)
    candidates = ["spo2", "systolic_bp", "temperature", "heart_rate", "resp_rate"]
    free = [name for name in candidates if name not in directions]
    drawn = rng.sample(free, k=min(rng.randint(1, 2), len(free)))

    for name in [*directions, *drawn]:
        direction = directions.get(name, "low")
        if name == "heart_rate":
            values["heart_rate"] = normal_heart_rate + rng.randint(*settings["heart_rate_offset"])
        elif name == "resp_rate":
            values["resp_rate"] = normal_resp_rate + rng.randint(*settings["resp_rate_offset"])
        elif name == "temperature":
            span = settings["temperature_high"] if direction == "high" else settings["temperature"]
            values["temperature"] = round(rng.uniform(*span), 1)
        elif name == "systolic_bp":
            if direction == "high" and settings["systolic_high"]:
                values["systolic_bp"] = rng.randint(*settings["systolic_high"])
                values["diastolic_bp"] = rng.randint(102, 128)
            else:
                low, high = settings["systolic_share"]
                values["systolic_bp"] = rng.randint(
                    round(systolic_range[0] * low), round(systolic_range[0] * high)
                )
        else:
            values[name] = rng.randint(*settings[name])

    values["diastolic_bp"] = min(values["diastolic_bp"], values["systolic_bp"] - 25)
    return VitalSigns(**values)


def _normal_diastolic(systolic_range: tuple[int, int]) -> tuple[int, int]:
    """Plausible diastolic range for a given systolic: about two thirds."""
    return (round(systolic_range[0] * 0.62), round(systolic_range[1] * 0.66))
