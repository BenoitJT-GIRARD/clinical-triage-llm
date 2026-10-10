"""Loading the public medical corpora.

Five corpora are read: MediQAl, MedMCQA, FrenchMedMCQA and MedQuAD as sources of cases, and
UltraMedical-Preference as a source of preferences. This module only downloads them and brings
them to a common shape; deciding what is genuinely usable for triage happens in
``corpus_cases``.

Four findings, checked on the case corpora themselves, explain what follows:

- **MediQAl is the only one made of clinical vignettes.** Its ``clinical_case`` column
  describes a patient, their complaint, their history and often their vital signs on arrival.
  3,075 distinct vignettes, in French, 569 of which carry at least one readable vital sign. It
  is the only authentic French-language source.
- **MedMCQA is the English source of vignettes.** Few of its exam questions describe a
  patient, but its size makes it the only source of English clinical vignettes.
- **FrenchMedMCQA is a set of pharmacy exam questions.** The mirror used keeps only its 1,080
  single-answer questions, out of the corpus's 3,105, and only six of them describe a patient.
- **MedQuAD asks general questions about conditions**, not about patients. It serves to vary the
  English clinical vocabulary, not to supply cases.

**The corpora are read in full.** A partial read would give a yield that describes not the
source but the ceiling we set ourselves, and the published protocol would draw false conclusions
about what the corpora allow. The ``limit`` parameter exists for quick trials; at zero it bounds
nothing.

Two failures, two behaviours. **A source unreachable at opening** is reported and skipped; the
preparation script then checks that each expected corpus did supply entries, and refuses to
build a truncated dataset silently. **A read interrupted mid-way** stops everything: the
published yield is computed on the number of entries read, and a partial read absorbed in
silence would produce a table describing half a corpus while presenting it as the whole.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from clinical_triage.config import DATA
from clinical_triage.utils import get_logger

logger = get_logger(__name__)


# Readable corpus names, in the order the published documents present them. Shared by the build
# script and by the figure script: two lists would have drifted apart.
CORPUS_NAMES = (
    ("mediqal", "MediQAl"),
    ("medquad", "MedQuAD"),
    ("medmcqa", "MedMCQA"),
    ("frenchmedmcqa", "FrenchMedMCQA"),
)


@dataclass(frozen=True)
class CorpusEntry:
    """One entry of a public corpus, brought to a common shape."""

    text: str  # raw statement (question, vignette or symptom description)
    answer: str  # associated answer or explanation, when there is one
    lang: str  # "fr" or "en"
    source: str  # short name of the source corpus
    topic: str  # discipline or condition, when the corpus supplies one


def _clean(value: object) -> str:
    """Minimal cleaning: repeated whitespace and line breaks."""
    if not value:
        return ""
    return " ".join(str(value).split()).strip()


def _stream(repo: str, split: str = "train"):
    """Open a Hugging Face dataset in streaming mode, or return None when unavailable."""
    from datasets import load_dataset

    try:
        return load_dataset(repo, split=split, streaming=True)
    except Exception as exc:  # noqa: BLE001 - a missing source must not stop everything
        logger.warning("Corpus %s unavailable (%s) — source skipped.", repo, exc)
        return None


def _stream_all_splits(repo: str):
    """Chain the rows of every split of a repository.

    An exam corpus divides its questions between ``train``, ``validation`` and ``test`` for its
    own evaluation needs. That split means nothing here: we are looking for patient
    descriptions, and stopping at the training split would ignore half of them for no reason.
    """
    from datasets import load_dataset

    try:
        datasets = load_dataset(repo, streaming=True)
    except Exception as exc:  # noqa: BLE001 - a missing source must not stop everything
        logger.warning("Corpus %s unavailable (%s) — source skipped.", repo, exc)
        return
    for name in datasets:
        yield from datasets[name]


def _cap_reached(entries: list, limit: int) -> bool:
    """Say whether reading must stop. A limit of zero or less bounds nothing."""
    return limit > 0 and len(entries) >= limit


class InterruptedRead(RuntimeError):
    """A streaming read stopped mid-way, leaving the corpus incomplete."""


def _rows(dataset, repo: str):
    """Walk a stream, and refuse to return a silently truncated corpus.

    The ``try`` on opening covers the opening only: in streaming mode all network traffic happens
    during iteration. A cut at that point would propagate up to the preparation script — already
    better than a truncated corpus, but the error would not say where it happened.

    It is not absorbed either. The yield published in the dataset card is computed on the number
    of entries read: absorbing the cut would produce a table describing a partial read while
    presenting it as complete, and a set rebuilt after an incident would be indistinguishable
    from a whole one. Better a preparation to re-run than a false figure.
    """
    read = 0
    try:
        for row in dataset:
            read += 1
            yield row
    except Exception as exc:
        raise InterruptedRead(
            f"Reading {repo} was interrupted after {read} entries ({exc}). The corpus would be "
            "incomplete and the published yield false: re-run the preparation."
        ) from exc


def load_mediqal(limit: int = 0) -> list[CorpusEntry]:
    """MediQAl: annotated French clinical cases, from the ANR MALADES project.

    It is the only one of the four starting corpora that describes **patients**: the ``clinical_case`` column
    carries a complaint, a history and, one time in five, the vital signs recorded on arrival.

    Three configurations share that column — ``oeq`` (open questions), ``mcqu`` and ``mcqm``
    (multiple choice). One vignette often serves several questions: entries are deduplicated on
    the case text, without which the same patient would come back up to ten times and skew the
    balance of the set.
    """
    from datasets import load_dataset

    seen: set[str] = set()
    entries: list[CorpusEntry] = []
    for configuration in ("oeq", "mcqu", "mcqm"):
        try:
            datasets = load_dataset(DATA.corpora["mediqal"], configuration)
        except Exception as exc:  # noqa: BLE001 - a missing source must not stop everything
            logger.warning(
                "MediQAl/%s unavailable (%s) — configuration skipped.", configuration, exc
            )
            continue
        for split in datasets.values():
            for row in split:
                vignette = _clean(row.get("clinical_case"))
                if not vignette or vignette in seen:
                    continue
                seen.add(vignette)
                entries.append(
                    CorpusEntry(
                        text=vignette,
                        answer=_clean(row.get("answer")) or _clean(row.get("question")),
                        lang="fr",
                        source="mediqal",
                        topic=_clean(row.get("medical_subject")),
                    )
                )
                if _cap_reached(entries, limit):
                    logger.info("MediQAl: %d distinct vignettes loaded.", len(entries))
                    return entries
    logger.info("MediQAl: %d distinct vignettes loaded.", len(entries))
    return entries


def load_medquad(limit: int = 0) -> list[CorpusEntry]:
    """MedQuAD: English medical question-answer pairs, from the NIH websites.

    The repository used exposes two columns, ``Question`` and ``Answer``, capitalised;
    other redistributions of the same corpus name them in lower case. Both are accepted, so that
    changing mirror does not break the preparation.
    """
    dataset = _stream(DATA.corpora["medquad"])
    if dataset is None:
        return []
    entries: list[CorpusEntry] = []
    for row in _rows(dataset, DATA.corpora["medquad"]):
        question = _clean(row.get("Question") or row.get("question"))
        answer = _clean(row.get("Answer") or row.get("answer"))
        if question and answer:
            entries.append(
                CorpusEntry(
                    text=question,
                    answer=answer,
                    lang="en",
                    source="medquad",
                    topic=_clean(row.get("qtype") or row.get("question_type")),
                )
            )
        if _cap_reached(entries, limit):
            break
    logger.info("MedQuAD: %d entries loaded.", len(entries))
    return entries


def load_frenchmedmcqa(limit: int = 0) -> list[CorpusEntry]:
    """FrenchMedMCQA: French multiple-choice pharmacy questions.

    The mirror used keeps only the single-answer questions: 1,080 of the corpus's 3,105. All
    three splits are read — 595 questions in training, 164 in validation, 321 in test — since
    stopping at the first would leave out 485 questions of the only other French-language corpus.
    """
    # A generator: an unavailable source produces no row there, and the loop below terminates on
    # its own, returning an empty list.
    dataset = _rows(
        _stream_all_splits(DATA.corpora["frenchmedmcqa"]),
        DATA.corpora["frenchmedmcqa"],
    )
    # `correct_answers` is the index of the right option, counted from zero: 0 for a, 4 for e.
    # Checked against the authors' archive, which writes the answers as letters.
    options = ("a", "b", "c", "d", "e")
    entries: list[CorpusEntry] = []
    for row in dataset:
        question = _clean(row.get("question"))
        if not question:
            continue
        try:
            index = int(row.get("correct_answers"))
        except (TypeError, ValueError):
            index = -1
        correct = _clean(row.get(f"answer_{options[index]}")) if 0 <= index < len(options) else ""
        entries.append(
            CorpusEntry(
                text=question,
                answer=correct,
                lang="fr",
                source="frenchmedmcqa",
                topic="",
            )
        )
        if _cap_reached(entries, limit):
            break
    logger.info("FrenchMedMCQA: %d entries loaded.", len(entries))
    return entries


def load_medmcqa(limit: int = 0) -> list[CorpusEntry]:
    """MedMCQA: English multiple-choice questions from medical entrance exams."""
    dataset = _stream(DATA.corpora["medmcqa"])
    if dataset is None:
        return []
    options = ("opa", "opb", "opc", "opd")
    entries: list[CorpusEntry] = []
    for row in _rows(dataset, DATA.corpora["medmcqa"]):
        question = _clean(row.get("question"))
        if not question:
            continue
        try:
            index = int(row.get("cop"))
        except (TypeError, ValueError):
            index = -1
        correct = _clean(row.get(options[index])) if 0 <= index < len(options) else ""
        entries.append(
            CorpusEntry(
                text=question,
                answer=correct,
                lang="en",
                source="medmcqa",
                topic=_clean(row.get("subject_name")),
            )
        )
        if _cap_reached(entries, limit):
            break
    logger.info("MedMCQA: %d entries loaded.", len(entries))
    return entries


@dataclass(frozen=True)
class PreferencePair:
    """An annotated preference pair, as UltraMedical-Preference supplies it."""

    prompt: str
    chosen: str
    rejected: str
    label_type: str  # "easy", "hard", "length" or "human"


def _last_message(value: object) -> str:
    """Extract the content of the last message of a conversation-shaped answer."""
    import ast

    if isinstance(value, str) and value.strip().startswith("[{"):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value.strip()
    if isinstance(value, list) and value:
        last = value[-1]
        if isinstance(last, dict):
            return _clean(last.get("content"))
    return _clean(value)


def load_ultramedical_preferences(limit: int = 0) -> list[PreferencePair]:
    """UltraMedical-Preference: pairs of medical answers, test split.

    For each question the authors had several models answer, then GPT-4 scored and ranked the
    answers: the best ranked becomes the preferred one. On the test split — 777 pairs, the one the
    authors set aside for evaluation — biomedical experts reviewed and corrected those choices.
    ``label_type`` puts each pair in a group: ``easy``, ``hard``, ``length``, whose two answers
    have comparable lengths, and ``human``, where the experts overruled GPT-4.

    The file is read directly: its columns differ from the training split's, and ``datasets``
    refuses to load the whole repository.
    """
    from huggingface_hub import hf_hub_download

    repo = DATA.corpora["ultramedical_pref"]
    try:
        path = hf_hub_download(repo, "data/test.json", repo_type="dataset")
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a missing source must not stop everything
        logger.warning("Corpus %s unavailable (%s) — source skipped.", repo, exc)
        return []
    pairs: list[PreferencePair] = []
    for row in rows:
        prompt = _clean(row.get("prompt"))
        chosen = _last_message(row.get("chosen"))
        rejected = _last_message(row.get("rejected"))
        if prompt and chosen and rejected and chosen != rejected:
            pairs.append(PreferencePair(prompt, chosen, rejected, _clean(row.get("label_type"))))
        if _cap_reached(pairs, limit):
            break
    logger.info("UltraMedical-Preference: %d test pairs loaded.", len(pairs))
    return pairs
