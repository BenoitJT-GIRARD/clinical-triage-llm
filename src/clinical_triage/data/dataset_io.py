"""Serialisation, splitting and documentation of the dataset.

The format is the one the training libraries expect, and it is the same from SFT to serving:

- supervised set: ``prompt`` (the ChatML exchange up to the opening of the assistant turn) and
  ``completion`` (the expected answer). That pair lets training compute the loss on the answer
  only, with no special collator;
- preference set: ``prompt``, ``chosen``, ``rejected``, with exactly the same prompt as above.

Both sets therefore share, word for word, the format used at inference. The clinical metadata
travels in extra columns, which training ignores but which make the dataset auditable.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from clinical_triage.data.dpo_builder import PreferencePairRecord
from clinical_triage.data.sft_builder import TriageExample, case_fingerprint
from clinical_triage.prompts import SYSTEM_PROMPT, build_messages, format_chatml
from clinical_triage.utils import get_logger, path_for_log

logger = get_logger(__name__)


def _prompt_for(user_turn: str) -> str:
    """The complete ChatML prompt, opening the assistant turn."""
    return format_chatml(build_messages(user_turn), add_generation_prompt=True)


def sft_record(example: TriageExample) -> dict:
    """Serialise a triage example for supervised training."""
    return {
        "prompt": _prompt_for(example.user_turn),
        "completion": example.assistant_turn,
        "user_turn": example.user_turn,
        "level": example.level,
        "lang": example.lang,
        "source": example.source,
        "confidence": example.confidence,
        "symptoms": list(example.symptoms),
        "medical_history": list(example.medical_history),
        "vitals": example.vitals,
        "presentation_id": example.presentation_id,
    }


def dpo_record(pair: PreferencePairRecord) -> dict:
    """Serialise a preference pair for the DPO alignment."""
    return {
        "prompt": _prompt_for(pair.user_turn),
        "chosen": pair.chosen,
        "rejected": pair.rejected,
        "user_turn": pair.user_turn,
        "level": pair.level,
        "lang": pair.lang,
        "strategy": pair.strategy,
        "source": pair.source,
    }


def eval_record(case) -> dict:
    """Serialise one case of the clinical evaluation set."""
    return {
        "prompt": _prompt_for(case.user_turn),
        "completion": "",
        "user_turn": case.user_turn,
        "id": case.id,
        "level": case.level,
        "lang": case.lang,
        "case_type": case.case_type,
        "description": case.description,
        "clinical_note": case.note,
    }


# The six files the preparation produces. The list lives here because two places depend on it
# and must stay in agreement: the script that writes them, and the publication that refuses to
# start if one is missing.
DATASET_FILES = (
    "sft_train.jsonl",
    "sft_validation.jsonl",
    "sft_test.jsonl",
    "dpo_train.jsonl",
    "clinical_eval.jsonl",
    "metadata.json",
)


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write a list of records as UTF-8 JSONL, one object per line.

    Lines end in LF on every platform: the files go to the Hub as they are on disk, and a set
    built on Windows would otherwise publish other bytes than the same set built on Linux.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d lines → %s", len(rows), path_for_log(path))


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file back."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_train_val_test(
    examples: list[TriageExample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[TriageExample]]:
    """Split into train / validation / test.

    The examples have already been deduplicated on the full user turn: the same statement
    therefore cannot land on both sides of the split. :func:`check_no_leakage` verifies it
    explicitly afterwards.
    """
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    return {
        "test": shuffled[:n_test],
        "validation": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }


def check_no_leakage(splits: dict[str, list[TriageExample]]) -> dict[str, int]:
    """Count the user turns shared between two splits.

    Every value must be zero. This check is called by the preparation script, which fails if it
    is not: training data and evaluation data must not mix, and a claim of that kind has to be
    checked by code, not by documentation.

    The comparison is on the normalised form: a check that only catches exact duplicates gives
    an assurance it has not earned.
    """
    sets = {name: {case_fingerprint(e.user_turn) for e in items} for name, items in splits.items()}
    names = sorted(sets)
    overlaps: dict[str, int] = {}
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            overlaps[f"{first}∩{second}"] = len(sets[first] & sets[second])
    return overlaps


# --- The dataset card ---

# Licence and provenance of each corpus, checked on the repository page.
DOCUMENTED_SOURCES = {
    "clinical_vignette": {
        "origin": "Catalogue of clinical presentations written for this project (src/clinical_triage/data/clinical_catalogue.py)",
        "language": "fr + en",
        "licence": "MIT",
        "role": "Triage ground truth: the level comes from the presentation, not from the text.",
    },
    "mediqal": {
        "origin": "https://huggingface.co/datasets/ANR-MALADES/MediQAl",
        "language": "fr",
        "licence": "CC BY 4.0",
        "role": "The only source corpus that describes patients. Base of the authentic French vignettes.",
    },
    "medquad": {
        "origin": "https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset",
        "language": "en",
        # The Hub mirror declares no licence. This one is read on the authors' original
        # repository, github.com/abachaa/MedQuAD, whose `LICENSE.txt` is the CC BY 4.0 text and
        # whose `readme.txt` repeats it in full. Copying a licence without checking it would be
        # inventing it, hence the field that says where it was read.
        "licence": "CC BY 4.0",
        "licence_checked_on": "original repository abachaa/MedQuAD",
        "role": "Symptom descriptions, labelled by the rule (medium confidence).",
    },
    "medmcqa": {
        "origin": "https://huggingface.co/datasets/openlifescienceai/medmcqa",
        "language": "en",
        "licence": "Apache-2.0",
        "role": "Added because the other source corpora hold no English-language clinical vignettes.",
    },
    "frenchmedmcqa": {
        "origin": "https://huggingface.co/datasets/nthngdy/frenchmedmcqa",
        "language": "fr",
        # Same remark: the Parquet mirror declares nothing, the authors' repository does. The
        # mirror is used because the original repository only exposes its data through a loading
        # script, which `datasets` no longer executes.
        "licence": "Apache-2.0",
        "licence_checked_on": "original repository qanastek/frenchmedmcqa",
        "role": "Pharmacy exam questions, none of which can be labelled for triage.",
    },
    "ultramedical_preference": {
        "origin": "https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference",
        "language": "en",
        "licence": "MIT",
        "role": "External preference set, kept out of training: an independent measure of the alignment.",
    },
}


def metadata_schema() -> dict:
    """The dataset metadata schema, field by field."""
    return {
        "sft_fields": {
            "prompt": "Complete ChatML prompt (system prompt + patient turn), opening the assistant turn.",
            "completion": "Expected triage answer: level, justification, recommendation.",
            "user_turn": "The patient turn alone, useful for deduplication and audit.",
            "level": "Triage level: URGENCE_VITALE | URGENCE_MODEREE | CONSULTATION_DIFFEREE.",
            "lang": "Language of the patient description (fr | en).",
            "source": "Origin of the example (clinical_vignette, mediqal, medquad, frenchmedmcqa, medmcqa).",
            "confidence": "Confidence of the label: high (clinical catalogue) | medium (rule applied to a corpus).",
            "symptoms": "Clinical signs present in the description.",
            "medical_history": "Patient history mentioned in the description.",
            "vitals": "Vital-sign reading, empty when triage happens without measurements.",
            "presentation_id": "Identifier of the catalogue presentation, empty for corpus cases.",
        },
        # The three lists give the **exact** columns of each file, and not only what one adds to
        # another: the preference set has neither `confidence` nor `vitals`, and the evaluation
        # set exposes a `description` column that nothing documented. A consumer of the
        # published dataset who filters on an absent column gets nothing, without understanding
        # why.
        "dpo_fields": {
            "prompt": "ChatML prompt identical to the supervised set's.",
            "chosen": "Preferred answer: right level, format respected, safe course of action.",
            "rejected": "Rejected answer, same format and comparable length.",
            "user_turn": "The patient turn alone, useful for deduplication and audit.",
            "level": "Reference triage level of the case.",
            "lang": "Language of the patient description.",
            "strategy": "The defect introduced: undertriage | unsafe_recommendation | asserted_diagnosis | answered_in_english.",
            "source": "safety_preference.",
        },
        "evaluation_fields": {
            "prompt": "ChatML prompt identical to the supervised set's.",
            "completion": "Always empty: the expected answer is not given, only the level is.",
            "user_turn": "The patient turn alone, as it is submitted to the model.",
            "id": "Identifier of the evaluation case.",
            "level": "Reference triage level, written by hand.",
            "lang": "Language of the patient description.",
            "case_type": "Nature of the difficulty: empty (direct presentation), falsely_reassuring, falsely_alarming, negation, discordant_vitals.",
            "description": "The patient description alone, without the framing prompt.",
            "clinical_note": "Clinical reason for the label, for auditability and error analysis.",
        },
        "triage_taxonomy": {
            "URGENCE_VITALE": "Immediate care (levels 1 and 2 of the FRENCH scale).",
            "URGENCE_MODEREE": "Care within a few hours (levels 3 and 4).",
            "CONSULTATION_DIFFEREE": "No immediate severity criterion (level 5).",
        },
        "system_prompt": SYSTEM_PROMPT,
        "sources": DOCUMENTED_SOURCES,
    }


def write_metadata(meta: dict, path: Path) -> None:
    """Write the dataset card as indented JSON, with LF line endings like the data files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    logger.info("Metadata written → %s", path_for_log(path))
