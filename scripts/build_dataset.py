"""Build the bilingual triage dataset.

In order:

    1. clinical evaluation set: written by hand, produced first and set aside, so that no later
       stage can reuse it;
    2. public corpora: loading, extraction of the genuinely usable cases, GDPR anonymisation and
       an independent quality check;
    3. clinical vignettes: generating the complement needed to reach the target volume with a
       50/50 balance between French and English;
    4. assembly, deduplication and train / validation / test split, with a check that the splits
       are disjoint;
    5. DPO preference pairs, built from the training split alone;
    6. writing the JSONL files and the dataset card.

Usage::

    uv run python scripts/build_dataset.py
    uv run python scripts/build_dataset.py --sft-size 1200   # quick iteration
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from datetime import UTC, datetime

from clinical_triage.config import DATA, PATHS, SEED, TRIAGE
from clinical_triage.data.anonymize import (
    ENTITIES_LEFT_OUT,
    MASKED_ENTITIES,
    analyze_and_anonymize,
    audit_corpus,
)
from clinical_triage.data.case_generator import generate_cases
from clinical_triage.data.clinical_catalogue import PRESENTATIONS
from clinical_triage.data.clinical_eval_set import eval_cases, eval_user_turns
from clinical_triage.data.corpus_cases import MAX_LENGTH, extract_cases, funnel
from clinical_triage.data.corpus_sources import (
    CORPUS_NAMES,
    load_frenchmedmcqa,
    load_mediqal,
    load_medmcqa,
    load_medquad,
)
from clinical_triage.data.dataset_io import (
    DATASET_FILES,
    check_no_leakage,
    dpo_record,
    eval_record,
    metadata_schema,
    sft_record,
    split_train_val_test,
    write_jsonl,
    write_metadata,
)
from clinical_triage.data.diagnostics import (
    distinct_completions,
    metadata_leakage,
    separability,
)
from clinical_triage.data.dpo_builder import build_preference_pairs, length_balance
from clinical_triage.data.sft_builder import assemble, case_fingerprint
from clinical_triage.data.triage_rules import classify
from clinical_triage.utils import get_logger, git_revision, set_seed

logger = get_logger("build_dataset")


YIELD_MARKERS = (
    "<!-- yield:start",
    "<!-- yield:end -->",
)


def _command_run(args: argparse.Namespace) -> str:
    """The command as it was actually passed, non-default options included."""
    defaults = {"sft_size": DATA.target_sft_pairs, "dpo_size": DATA.target_dpo_pairs}
    options = []
    for name, value in vars(args).items():
        if value == defaults.get(name) or value in (0, False, None):
            continue
        flag = "--" + name.replace("_", "-")
        options.append(flag if value is True else f"{flag} {value}")
    return " ".join(["uv run python scripts/build_dataset.py", *options])


def _write_yield_into_the_card(corpus_yield: dict) -> None:
    """Rewrite the yield table of ``data/README.md`` from the counts.

    That card is published as is on the Hub, next to ``metadata.json``. The rows of the table are
    therefore written from the counts of the current build rather than typed by hand: the card
    and the metadata file published together describe the same set.
    """
    card = PATHS.data / "README.md"
    text = card.read_text(encoding="utf-8")
    opening, closing = YIELD_MARKERS
    start = text.index(opening)
    end = text.index(closing) + len(closing)

    header = (
        "| Corpus | Entries read | No patient described | Outside length bounds "
        "| No sign identified | Duplicates | Cases extracted | Cases delivered | Yield |"
    )
    rows = [
        text[start : text.index("-->", start) + 3],
        header,
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, name in CORPUS_NAMES:
        measure = corpus_yield[key]
        read, kept = measure["entries_read"], measure["cases_kept"]
        lost = measure["funnel"]
        share = f"{100 * kept / read:.1f}" if read else "0.0"
        # The best yield and the null yield are the two figures the surrounding text comments
        # on: they are set in bold.
        bold = "**" if kept == 0 or key == "mediqal" else ""
        rows.append(
            f"| {name} | {read:,} "
            f"| {lost['no_patient_presentation']:,} "
            f"| {lost['outside_length_bounds']:,} "
            f"| {lost['no_identified_sign']:,} "
            f"| {lost['duplicates']:,} "
            f"| {lost['kept']:,} "
            f"| {bold}{kept:,}{bold} | {bold}{share}%{bold} |"
        )
    rows.append(closing)

    card.write_text(text[:start] + "\n".join(rows) + text[end:], encoding="utf-8")
    logger.info("Yield table updated → %s", card)


def _distribution(items, key) -> dict[str, int]:
    """Count the occurrences of an attribute, sorted by decreasing frequency."""
    count: dict[str, int] = {}
    for item in items:
        value = str(key(item))
        count[value] = count.get(value, 0) + 1
    return dict(sorted(count.items(), key=lambda kv: -kv[1]))


def _grid(sft_size: int) -> list[tuple[str, str, int]]:
    """The six cells of the set — level × language — and the volume targeted in each.

    The remainder of the division is spread over the first cells, so that the targets sum
    exactly to the requested volume.
    """
    cells = [(level, lang) for level in TRIAGE.levels for lang in ("fr", "en")]
    base, remainder = divmod(sft_size, len(cells))
    return [
        (level, lang, base + (1 if index < remainder else 0))
        for index, (level, lang) in enumerate(cells)
    ]


def _cap_corpus_cases(cases: list, sft_size: int, rng: random.Random) -> list:
    """Bound the corpus contribution, cell by cell and then globally.

    Two bounds, and both are needed.

    **Per cell.** The extracted cases concentrate on the urgent levels and on English — the
    safety rule discards everything it cannot label, and MedMCQA is English-language. Read in
    full, it supplies more than the "urgent / English" cell has places for. Without a per-cell
    bound, the delivered set would exceed the requested volume and lose its balance between
    levels and between languages.

    **Globally.** Corpus cases carry a ``medium`` confidence label, put there by the keyword
    rule. Letting them become the majority would make the set, in essence, a transcription of
    that rule. The cap keeps them in the minority against the catalogue vignettes, whose label
    does not come from reading the text.
    """
    kept: list = []
    for level, lang, target in _grid(sft_size):
        in_cell = [c for c in cases if c.level == level and c.lang == lang]
        rng.shuffle(in_cell)
        if len(in_cell) > target:
            logger.info(
                "  %-22s %s: %d cases available, brought down to %d (cell capacity).",
                level,
                lang,
                len(in_cell),
                target,
            )
        kept += in_cell[:target]

    cap = int(sft_size * DATA.max_corpus_share)
    if len(kept) > cap:
        rng.shuffle(kept)
        kept = kept[:cap]
        logger.info(
            "  brought down to %d cases (%.0f%% of the set) to stay a minority against the "
            "catalogue.",
            len(kept),
            100 * DATA.max_corpus_share,
        )
    return kept


def _measure_answers(examples: list) -> dict:
    """Length of the expected answers, in tokens.

    It rules the generation budget out as a cause of the stopping defect: if the median answer
    sits well under the cap, the cap is not what cuts the answer short. The measurement is
    redone at every build, so that it describes the set delivered with it.
    """
    from transformers import AutoTokenizer

    from clinical_triage.config import MODEL, SERVING

    tokenizer = AutoTokenizer.from_pretrained(MODEL.base_model)
    lengths = sorted(
        len(tokenizer(e.assistant_turn, add_special_tokens=False)["input_ids"]) for e in examples
    )
    return {
        "median_tokens": lengths[len(lengths) // 2],
        "p95_tokens": lengths[int(0.95 * len(lengths))],
        "max_tokens": lengths[-1],
        "generation_cap": SERVING.max_new_tokens,
    }


def _check_description_budget(cases: list) -> dict:
    """Refuse a case longer than what the service will agree to read.

    The extraction bound is expressed in characters; the model window counts in tokens. Between
    the two lies a density, which depends on the language and the vocabulary and which a change
    of corpus can move. So the check runs on the cases actually kept, at every build: if one of
    them exceeds the budget, the model would train on a narrative the service would truncate,
    and ``MAX_LENGTH`` must come down rather than the problem be discovered in production.

    Returns the measurement, which the dataset card publishes: the published protocol reads the
    budget and the length actually observed there instead of copying them.
    """
    from transformers import AutoTokenizer

    from clinical_triage.config import MODEL, SERVING
    from clinical_triage.prompts import description_budget

    tokenizer = AutoTokenizer.from_pretrained(MODEL.base_model)
    budget = description_budget(tokenizer, MODEL.max_seq_length, SERVING.max_new_tokens)
    lengths = [len(tokenizer(c.description, add_special_tokens=False)["input_ids"]) for c in cases]
    if not lengths:
        return {"token_budget": budget}
    over = sum(1 for length in lengths if length > budget)
    logger.info(
        "  description lengths: median %d tokens, maximum %d, service budget %d.",
        sorted(lengths)[len(lengths) // 2],
        max(lengths),
        budget,
    )
    if over:
        raise SystemExit(
            f"{over} cases exceed the service's description budget ({budget} tokens, maximum "
            f"observed {max(lengths)}). Lower MAX_LENGTH in "
            "src/clinical_triage/data/corpus_cases.py."
        )
    return {
        "token_budget": budget,
        "median_tokens": sorted(lengths)[len(lengths) // 2],
        "max_tokens": max(lengths),
        "char_cap": MAX_LENGTH,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-size", type=int, default=DATA.target_sft_pairs)
    parser.add_argument("--dpo-size", type=int, default=DATA.target_dpo_pairs)
    parser.add_argument(
        "--corpus-scan",
        type=int,
        default=0,
        help=(
            "maximum number of entries read per corpus; 0 reads everything (default). It exists "
            "for quick trials only: a cap falsifies the yield table, which would then describe "
            "the cap and not the corpus."
        ),
    )
    parser.add_argument(
        "--no-anonymize", action="store_true", help="skip Presidio (development only)"
    )
    args = parser.parse_args()

    # Below this threshold, `split_train_val_test` rounds the validation and test shares to
    # zero: both files come out empty, the leak check declares them disjoint — which they
    # trivially are — and nothing reports that the delivered set can no longer measure its own
    # convergence.
    minimum = int(1 / min(DATA.val_ratio, DATA.test_ratio))
    if args.sft_size < minimum:
        raise SystemExit(
            f"--sft-size {args.sft_size} is too small: below {minimum} examples the validation "
            "and test splits would come out empty."
        )

    set_seed(SEED, include_torch=False)
    rng = random.Random(SEED)
    PATHS.ensure()
    output = PATHS.data_processed

    # --- 1. Clinical evaluation set, set aside before anything else ---
    logger.info("Step 1/6 — hand-written clinical evaluation set.")
    evaluation_cases = eval_cases()
    reserved_turns = eval_user_turns()
    write_jsonl([eval_record(c) for c in evaluation_cases], output / "clinical_eval.jsonl")
    logger.info(
        "  %d cases (%s), of which %d carry a trap.",
        len(evaluation_cases),
        _distribution(evaluation_cases, lambda c: c.level),
        sum(1 for c in evaluation_cases if c.case_type),
    )

    # --- 2. Public corpora ---
    logger.info("Step 2/6 — public corpora: loading and extracting the usable cases.")
    # MediQAl first: it is the only corpus that describes patients, and therefore the one whose
    # cases best survive the extraction.
    per_corpus = {
        "mediqal": load_mediqal(limit=args.corpus_scan),
        "medquad": load_medquad(limit=args.corpus_scan),
        "medmcqa": load_medmcqa(limit=args.corpus_scan),
        "frenchmedmcqa": load_frenchmedmcqa(limit=args.corpus_scan),
    }
    # The number of entries actually read per corpus feeds the yield table of the dataset card:
    # it comes from the read that has just happened, like the counts written next to it in
    # `metadata.json`.
    entries_read = {name: len(batch) for name, batch in per_corpus.items()}
    entries = [entry for batch in per_corpus.values() for entry in batch]
    # A corpus with no entry at all stops the build. Letting it through would deliver a dataset
    # missing a source, and the yield table would show "0" without distinguishing "unusable
    # corpus" from "corpus not downloaded": better to fail than to deliver a set whose
    # composition cannot be accounted for.
    empty = [name for name, batch in per_corpus.items() if not batch]
    if empty:
        raise SystemExit(
            f"Corpora with no entry at all: {', '.join(empty)}. Check access to the Hugging Face "
            "Hub before rebuilding the dataset."
        )
    corpus_cases = extract_cases(entries, rng)
    logger.info("  %d usable cases extracted from %d entries.", len(corpus_cases), len(entries))
    funnels = {name: funnel(batch) for name, batch in per_corpus.items()}
    for name, count in funnels.items():
        logger.info(
            "  %-14s %6d read → %5d no patient, %4d out of bounds, %4d no sign, "
            "%4d duplicates → %4d kept",
            name,
            count["entries"],
            count["no_patient_presentation"],
            count["outside_length_bounds"],
            count["no_identified_sign"],
            count["duplicates"],
            count["kept"],
        )
    corpus_cases = _cap_corpus_cases(corpus_cases, args.sft_size, rng)

    # --- 3. GDPR anonymisation of the texts coming from the corpora ---
    if args.no_anonymize:
        logger.warning("Anonymisation disabled (--no-anonymize).")
        masked_entities = 0
        residual_pii: dict[str, int] = {}
    else:
        logger.info("Step 3/6 — GDPR anonymisation of the texts coming from the corpora.")
        masked_entities = 0
        anonymised = []
        for case in corpus_cases:
            description, number = analyze_and_anonymize(case.description, case.lang)
            masked_entities += number
            turn = case.user_turn.replace(case.description, description)
            anonymised.append(replace(case, description=description, user_turn=turn))
        corpus_cases = anonymised
        residual_pii = audit_corpus([c.description for c in corpus_cases])
        logger.info(
            "  %d entities masked; independent check after masking: %s",
            masked_entities,
            residual_pii or "no residual personal data",
        )

    # Labels re-examined after masking. A corpus case's label comes from the rule applied to the
    # text **before** anonymisation; if masking removes the sign that justified it, the example
    # teaches a decision its own text no longer supports. The rule therefore runs again on the
    # text as it will be delivered, and what no longer holds is discarded.
    before_review = len(corpus_cases)
    corpus_cases = [c for c in corpus_cases if classify(c.description) == c.level]
    if before_review != len(corpus_cases):
        logger.info(
            "  %d cases discarded: masking removed the sign that carried their label.",
            before_review - len(corpus_cases),
        )

    # Deduplication of the corpus cases, here and not later. Anonymisation has just made
    # identical statements that were not, and two extractions of the same case sometimes differ
    # only by a hyphen — "A 6 year old child" against "A 6-year-old child". Discarding them now
    # is what lets the next step count correctly: deduplicating after balancing would remove
    # examples already counted, and the delivered dataset would miss its target by a few units.
    before_dedup = len(corpus_cases)
    seen: set[str] = set()
    unique = []
    for case in corpus_cases:
        # The fingerprint is on the **description**, not on the patient turn. The turn is the
        # description wrapped in one of three templates drawn at random: the same clinical case
        # under two different templates would give two different fingerprints, and the duplicate
        # would pass.
        fingerprint = case_fingerprint(case.description)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(case)
    corpus_cases = unique
    if before_dedup != len(corpus_cases):
        logger.info(
            "  %d near-duplicates discarded from the corpus cases.",
            before_dedup - len(corpus_cases),
        )

    # --- 4. Clinical vignettes: each cell of the grid is topped up ---
    # Corpus cases are kept only when a sign is spotted: they are therefore almost all urgent
    # and almost all English. Exactly what is missing in each of the six cells (level × language)
    # is generated, so that the delivered dataset is balanced on both dimensions.
    # The length check applies here, to the cases that will be delivered: after capping,
    # anonymisation, label review and deduplication. Placed higher up, it would describe the
    # extraction pool, two thirds of which is never delivered.
    description_lengths = _check_description_budget(corpus_cases)

    logger.info("Step 4/6 — generating the clinical vignettes.")
    vignettes = []
    for level, lang, target in _grid(args.sft_size):
        already = sum(1 for c in corpus_cases if c.level == level and c.lang == lang)
        missing = max(0, target - already)
        logger.info("  %-22s %s: corpus %3d, to generate %3d", level, lang, already, missing)
        vignettes += generate_cases(missing, level, lang, rng, exclude=reserved_turns)

    # --- 5. Assembly, split and preference pairs ---
    logger.info("Step 5/6 — assembly, split and preference pairs.")
    examples = assemble(vignettes, corpus_cases, reserved_turns, rng)
    logger.info("  %d unique examples.", len(examples))

    splits = split_train_val_test(examples, DATA.val_ratio, DATA.test_ratio, SEED)
    overlaps = check_no_leakage(splits)
    if any(overlaps.values()):
        raise SystemExit(f"Leak between splits: {overlaps}")
    training_turns = {e.user_turn for e in splits["train"]}
    if training_turns & reserved_turns:
        raise SystemExit("Clinical evaluation cases have landed in the training split.")
    logger.info("  splits disjoint: %s", overlaps)

    pairs = build_preference_pairs(splits["train"], args.dpo_size, rng)
    length_gap = length_balance(pairs)
    logger.info("  length balance of the pairs: %s", length_gap)

    # --- 6. Writing ---
    logger.info("Step 6/6 — writing the files.")
    for name, items in splits.items():
        write_jsonl([sft_record(e) for e in items], output / f"sft_{name}.jsonl")
    write_jsonl([dpo_record(p) for p in pairs], output / "dpo_train.jsonl")

    meta = metadata_schema()
    meta["build"] = {
        "seed": SEED,
        "git_revision": git_revision(),
        "date_utc": datetime.now(UTC).strftime("%Y-%m-%d"),
        # The command actually passed, options included. Hard-coded, it would announce a default
        # build when `--corpus-scan` or `--no-anonymize` may have produced a very different set:
        # the card would describe a run that never happened.
        "command": _command_run(args),
    }
    by_source = _distribution(examples, lambda e: e.source)
    meta["statistics"] = {
        "sft_total": len(examples),
        "sft_train": len(splits["train"]),
        "sft_validation": len(splits["validation"]),
        "sft_test": len(splits["test"]),
        "dpo_total": len(pairs),
        "clinical_evaluation": len(evaluation_cases),
        "by_level": _distribution(examples, lambda e: e.level),
        "by_language": _distribution(examples, lambda e: e.lang),
        "by_source": by_source,
        "by_confidence": _distribution(examples, lambda e: e.confidence),
        "share_with_vitals": round(sum(1 for e in examples if e.vitals) / max(1, len(examples)), 3),
        "dpo_by_strategy": _distribution(pairs, lambda p: p.strategy),
        "dpo_length_balance": length_gap,
        "evaluation_by_level": _distribution(evaluation_cases, lambda c: c.level),
        "evaluation_by_language": _distribution(evaluation_cases, lambda c: c.lang),
        "evaluation_case_types": _distribution(
            [c for c in evaluation_cases if c.case_type], lambda c: c.case_type
        ),
        "description_lengths": description_lengths,
        "answer_lengths": _measure_answers(examples),
        # Yield of each corpus: what was read, and what survives in the delivered set after
        # extraction, capping, anonymisation, label review and deduplication. It is that second
        # figure that counts, and it is the one the dataset card publishes.
        "corpus_yield": {
            name: {
                "entries_read": read,
                "cases_kept": by_source.get(name, 0),
                # Where the entries are lost, stage by stage. The overall yield does not say
                # whether what discards them belongs to the corpus or to the filter.
                "funnel": funnels[name],
            }
            for name, read in entries_read.items()
        },
    }
    # What the set contains beyond its volumes: diversity of the expected answers, share of
    # restatement in a random split, and share of the level readable from the generator's
    # metadata alone. These three measurements condition the reading of the results and are
    # taken up as they are by the published protocol.
    logger.info("  diagnostics of the produced set.")
    meta["diagnostics"] = {
        "completions": distinct_completions(examples),
        "separability": separability(examples),
        "metadata_leakage": metadata_leakage(examples, PRESENTATIONS),
    }
    meta["gdpr"] = {
        "principle": "Minimisation: no real patient data. The vignettes are synthetic, and the "
        "public corpora are research sets with no identifying data.",
        "masking": "Presidio (French and English spaCy), limited to genuinely identifying "
        f"entities ({', '.join(MASKED_ENTITIES)}).",
        "entities_left_out": ENTITIES_LEFT_OUT,
        "entities_masked": masked_entities,
        # Without anonymisation there was no check: say so, rather than let the absence of a
        # finding read as an absence of personal data.
        "independent_check": (
            "not run: anonymisation disabled (--no-anonymize)"
            if args.no_anonymize
            else residual_pii or "no residual personal data detected"
        ),
        "anonymisation_applied": not args.no_anonymize,
        "train_evaluation_overlap": overlaps,
    }
    write_metadata(meta, output / "metadata.json")
    _write_yield_into_the_card(meta["statistics"]["corpus_yield"])

    # Publication refuses to start if one of these six files is missing. Noticing it here, at
    # build time, avoids discovering it when publishing — and keeps the list and the writing in
    # agreement.
    missing_files = [name for name in DATASET_FILES if not (output / name).exists()]
    if missing_files:
        raise SystemExit("Expected files not written: " + ", ".join(missing_files))

    print(json.dumps(meta["statistics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
