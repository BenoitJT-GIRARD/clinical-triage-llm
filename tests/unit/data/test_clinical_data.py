"""Tests of the triage corpus construction.

What is checked here is what documentation alone cannot guarantee: the balance of the corpus,
the absence of leakage between splits, the presence of the clinical metadata, and the fact that
the evaluation set never comes out on the training side.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import replace

import pytest

from clinical_triage.config import TRIAGE
from clinical_triage.data.case_generator import build_case, generate_cases
from clinical_triage.data.clinical_catalogue import (
    FORCED_VITALS,
    PRESENTATIONS,
    forced_vitals,
    presentation_by_id,
)
from clinical_triage.data.clinical_eval_set import eval_cases, eval_user_turns
from clinical_triage.data.corpus_cases import build_corpus_case, extract_vignette
from clinical_triage.data.corpus_sources import CorpusEntry
from clinical_triage.data.dataset_io import (
    check_no_leakage,
    metadata_schema,
    sft_record,
    split_train_val_test,
)
from clinical_triage.data.dpo_builder import build_preference_pairs
from clinical_triage.data.sft_builder import assemble

# --- Clinical catalogue ---


def test_the_catalogue_covers_the_three_levels():
    per_level = Counter(p.level for p in PRESENTATIONS)
    assert set(per_level) == set(TRIAGE.levels)
    assert min(per_level.values()) >= 15


def test_the_catalogue_identifiers_are_unique():
    identifiers = [p.id for p in PRESENTATIONS]
    assert len(identifiers) == len(set(identifiers))


def test_every_presentation_is_complete():
    for presentation in PRESENTATIONS:
        assert presentation.level in TRIAGE.levels
        assert presentation.vitals_profile in ("critical", "intermediate", "normal")
        assert presentation.age_range[0] <= presentation.age_range[1]
        assert presentation.complaint_fr and presentation.complaint_en
        assert len(presentation.signs_fr) >= 2 and len(presentation.signs_en) >= 2
        assert len(presentation.history_fr) >= 2 and len(presentation.history_en) >= 2
        assert presentation.onset in ("minutes", "heures", "jours", "semaines")
        assert len(presentation.justification) > 40
        assert len(presentation.recommendation) > 40


def test_lookup_by_identifier():
    assert presentation_by_id("syndrome_coronarien_aigu").level == "URGENCE_VITALE"
    with pytest.raises(KeyError):
        presentation_by_id("presentation_inexistante")


# --- Vignette generation ---


def test_the_generated_vignettes_carry_the_clinical_metadata():
    cases = generate_cases(12, "URGENCE_VITALE", "fr", random.Random(1))
    assert len(cases) == 12
    for vignette in cases:
        assert vignette.level == "URGENCE_VITALE"
        assert vignette.lang == "fr"
        assert vignette.symptoms
        assert vignette.medical_history
        assert vignette.confidence == "high"
        assert vignette.source == "clinical_vignette"


def test_the_generated_vignettes_are_unique():
    cases = generate_cases(120, "URGENCE_MODEREE", "fr", random.Random(2))
    assert len({c.user_turn for c in cases}) == len(cases)


def test_generation_respects_the_exclusions():
    """The turns reserved for the evaluation must never be regenerated."""
    excluded = eval_user_turns()
    cases = generate_cases(60, "CONSULTATION_DIFFEREE", "fr", random.Random(3), exclude=excluded)
    assert not {c.user_turn for c in cases} & excluded


def test_a_share_of_the_vignettes_is_produced_without_vital_signs():
    cases = generate_cases(200, "URGENCE_MODEREE", "en", random.Random(4))
    without_vitals = sum(1 for c in cases if c.vitals is None)
    assert 0 < without_vitals < len(cases)


# --- Extraction from the corpora ---


def test_an_exam_question_is_reduced_to_its_presentation():
    question = (
        "A 60 yr old chronic smoker presents with painless gross hematuria of 1 day duration. "
        "Investigation of choice to know the cause of hematuria"
    )
    vignette = extract_vignette(question)
    assert vignette is not None
    assert "Investigation of choice" not in vignette
    assert "chronic smoker" in vignette


def test_a_question_with_no_patient_is_discarded():
    assert extract_vignette("Levamisole is used as all except -") is None


def test_a_corpus_case_with_no_detected_sign_is_discarded():
    """A "non-urgent" label is not manufactured out of the rule's silence."""
    entry = CorpusEntry(text="", answer="", lang="en", source="medmcqa", topic="")
    description = "A 40-year-old man presents for a routine administrative certificate request."
    assert build_corpus_case(description, entry, random.Random(5)) is None


def test_a_corpus_case_with_a_sign_is_kept_with_medium_confidence():
    entry = CorpusEntry(text="", answer="", lang="en", source="medmcqa", topic="")
    description = "A 55-year-old man presents with chest pain radiating to the left arm."
    case = build_corpus_case(description, entry, random.Random(6))
    assert case is not None
    assert case.level == "URGENCE_VITALE"
    assert case.confidence == "medium"
    assert case.symptoms


# --- Assembly, splitting and leakage ---


def _corpus_examples(count: int) -> list:
    generator = random.Random(7)
    entry = CorpusEntry(text="", answer="", lang="en", source="medmcqa", topic="")
    cases = []
    for index in range(count):
        description = (
            f"A {30 + index}-year-old patient presents with chest pain since this morning."
        )
        built = build_corpus_case(description, entry, generator)
        if built is not None:
            cases.append(built)
    return cases


def test_assembly_deduplicates_and_excludes_the_evaluation_set():
    generator = random.Random(8)
    vignettes = generate_cases(30, "URGENCE_VITALE", "fr", generator)
    examples = assemble(vignettes + vignettes, _corpus_examples(10), eval_user_turns(), generator)
    turns = [e.user_turn for e in examples]
    assert len(turns) == len(set(turns))
    assert not set(turns) & eval_user_turns()


def test_the_split_leaves_no_leak():
    generator = random.Random(9)
    vignettes = []
    for level in TRIAGE.levels:
        vignettes += generate_cases(60, level, "fr", generator)
    examples = assemble(vignettes, [], set(), generator)
    splits = split_train_val_test(examples, 0.1, 0.1, seed=42)
    assert all(value == 0 for value in check_no_leakage(splits).values())
    total = sum(len(v) for v in splits.values())
    assert total == len(examples)


def test_the_leak_check_detects_a_duplicate():
    """The check must fail when there really is a leak, otherwise it serves no purpose."""
    generator = random.Random(10)
    examples = assemble(
        generate_cases(10, "URGENCE_VITALE", "fr", generator), [], set(), generator
    )
    splits = {"train": examples[:6], "test": examples[5:]}
    assert check_no_leakage(splits)["test∩train"] == 1


def test_the_leak_check_detects_a_near_duplicate():
    """Two spellings of the same case must count as a leak.

    Public corpora deliver the same case under neighbouring spellings — "A 6 year old child" and
    "A 6-year-old child". A check that only compares exact strings would let that duplicate
    spread either side of the split, and would make the test set flattering with nothing to
    report it.
    """
    generator = random.Random(11)
    first, second = assemble(
        generate_cases(2, "URGENCE_VITALE", "fr", generator), [], set(), generator
    )[:2]
    twin = replace(second, user_turn="A 6 year old child, elbow injury.")
    near = replace(first, user_turn="A 6-year-old child, elbow injury !")
    assert check_no_leakage({"train": [twin], "test": [near]})["test∩train"] == 1


def test_deduplication_discards_a_near_duplicate():
    """The same case written twice must enter the corpus once."""
    generator = random.Random(12)
    vignettes = generate_cases(1, "URGENCE_VITALE", "fr", generator)
    examples = assemble(vignettes, [], set(), generator)
    assert len(examples) == 1
    reserved = {examples[0].user_turn.upper() + " !!"}
    assert assemble(vignettes, [], reserved, generator) == []


# --- Records and metadata ---


def test_the_supervised_record_exposes_prompt_and_completion():
    """The ``messages`` column is deliberately absent: it would make the set be re-serialised
    with the model's native template, hence teach another format."""
    generator = random.Random(11)
    example = assemble(
        generate_cases(1, "URGENCE_VITALE", "fr", generator), [], set(), generator
    )[0]
    record = sft_record(example)
    assert set(record) >= {"prompt", "completion", "level", "lang", "source", "confidence"}
    assert "messages" not in record
    assert record["prompt"].endswith("<|im_start|>assistant\n")
    assert record["completion"].startswith("Niveau de priorité :")


def test_the_metadata_schema_covers_the_required_fields():
    schema = metadata_schema()
    fields = schema["sft_fields"]
    for expected in ("symptoms", "medical_history", "vitals", "source", "confidence"):
        assert expected in fields
    assert set(schema["triage_taxonomy"]) == set(TRIAGE.levels)
    for source in schema["sources"].values():
        assert source["licence"]
        assert source["origin"]


# --- Clinical evaluation set ---


def test_the_evaluation_set_is_balanced_and_bilingual():
    cases = eval_cases()
    per_level = Counter(c.level for c in cases)
    assert set(per_level) == set(TRIAGE.levels)
    assert len(set(per_level.values())) == 1  # the same n for the three levels
    per_language = Counter(c.lang for c in cases)
    assert min(per_language.values()) >= len(cases) * 0.4


def test_the_evaluation_set_contains_atypical_cases():
    cases = eval_cases()
    traps = [c for c in cases if c.case_type]
    assert len(traps) >= len(cases) * 0.3
    assert {"falsely_reassuring", "falsely_alarming", "negation", "discordant_vitals"} <= {
        c.case_type for c in traps
    }


def test_every_evaluation_case_is_documented():
    identifiers = set()
    for case in eval_cases():
        assert case.id not in identifiers
        identifiers.add(case.id)
        assert len(case.description) > 80
        assert len(case.note) > 30
        assert case.user_turn.count(case.description) == 1


# --- The vital signs must say what the narrative says ---

# What a complaint or a sign can name, and the vital sign that commits.
NAMES_A_VITAL_SIGN = {
    "temperature": r"fi[eè]vre|f[eé]brile|frisson|hyperthermie|sepsis|\bfever\b|chills",
    "spo2": r"cyanos|d[eé]satur|l[eè]vres bleues|blue lips|asphyx",
}


def _narrative(presentation) -> str:
    return " ".join(
        (
            presentation.complaint_fr,
            presentation.complaint_en,
            *presentation.signs_fr,
            *presentation.signs_en,
        )
    ).lower()


@pytest.mark.parametrize("presentation", [p for p in PRESENTATIONS if p.vitals_profile != "normal"])
def test_a_presentation_that_names_a_vital_sign_forces_it(presentation):
    """The generator degrades one or two vital signs at random.

    A "fever with chills" vignette therefore came out afebrile most of the time: the description
    and the reading contradicted each other inside the same example, and the model learnt along
    the way that vital signs mean nothing.
    """
    forced = dict(forced_vitals(presentation.id))
    narrative = _narrative(presentation)
    for vital, pattern in NAMES_A_VITAL_SIGN.items():
        if re.search(pattern, narrative):
            assert vital in forced, (
                f"{presentation.id} names « {vital} » in its narrative without forcing it"
            )


def test_every_forced_vital_sign_designates_a_known_presentation():
    identifiers = {p.id for p in PRESENTATIONS}
    assert set(FORCED_VITALS) <= identifiers


@pytest.mark.parametrize(
    "identifier", ["sepsis_grave", "syndrome_meninge", "pneumopathie_communautaire"]
)
def test_a_febrile_vignette_always_comes_out_febrile(identifier):
    presentation = presentation_by_id(identifier)
    rng = random.Random(3)
    for _ in range(12):
        case = build_case(presentation, "fr", rng)
        assert case.vitals.temperature >= 38.5, case.vitals.render("fr")


def test_a_severe_pre_eclampsia_always_comes_out_hypertensive():
    presentation = presentation_by_id("pre_eclampsie_severe")
    rng = random.Random(3)
    for _ in range(12):
        case = build_case(presentation, "fr", rng)
        assert case.vitals.systolic_bp >= 160, case.vitals.render("fr")


@pytest.mark.parametrize("identifier", ["bronchiolite_grave_nourrisson"])
def test_an_infant_does_not_have_an_adults_blood_pressure(identifier):
    """Blood pressure was drawn from adult ranges at every age.

    Infant vignettes came out at "TA 120/75" — a figure that does not exist at that age, and
    which the rule read as normal.
    """
    presentation = presentation_by_id(identifier)
    rng = random.Random(3)
    for _ in range(12):
        case = build_case(presentation, "fr", rng)
        assert case.vitals.systolic_bp <= 105, case.vitals.render("fr")


def test_the_published_schema_describes_exactly_the_delivered_columns():
    """The schema is the documentation of the dataset published on the Hub.

    It announced the supervised columns for all three files: the preference set has neither
    ``confidence`` nor ``vitals``, and the evaluation set exposes a ``description`` column that
    nothing documented. A consumer filtering on an absent column gets nothing, without
    understanding why.
    """
    from clinical_triage.data.clinical_eval_set import eval_cases
    from clinical_triage.data.dataset_io import dpo_record, eval_record

    rng = random.Random(7)
    example = assemble(generate_cases(1, TRIAGE.levels[0], "fr", rng), [], set(), rng)[0]
    pair = build_preference_pairs([example], 1, rng)[0]
    schema = metadata_schema()

    assert list(sft_record(example)) == list(schema["sft_fields"])
    assert list(dpo_record(pair)) == list(schema["dpo_fields"])
    assert list(eval_record(eval_cases()[0])) == list(schema["evaluation_fields"])


def test_the_yield_table_of_the_card_follows_the_delivered_counts():
    """The dataset card is published as is on the Hub.

    Its yield rows used to be copied by hand, and already diverged from ``metadata.json``,
    delivered next to them. The table is now written by the preparation script; this test
    observes that it has not been edited by hand since.
    """
    from clinical_triage.config import PATHS

    metadata = json.loads((PATHS.data_processed / "metadata.json").read_text(encoding="utf-8"))
    corpus_yield = metadata["statistics"]["corpus_yield"]
    card = (PATHS.data / "README.md").read_text(encoding="utf-8")
    table = card[card.index("<!-- yield:start") : card.index("<!-- yield:end -->")]

    for measure in corpus_yield.values():
        assert f"| {measure['entries_read']:,} |" in table
        assert f"{measure['cases_kept']:,}" in table
        # The loss columns add up with the cases extracted to give back the entries read: that
        # is what the surrounding text announces.
        lost = measure["funnel"]
        assert (
            lost["no_patient_presentation"]
            + lost["outside_length_bounds"]
            + lost["no_identified_sign"]
            + lost["duplicates"]
            + lost["kept"]
            == measure["entries_read"]
        )
        assert f"{lost['kept']:,}" in table
