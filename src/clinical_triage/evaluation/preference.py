"""What the preference alignment changed, measured by the quantity it optimises.

The clinical set says whether the model triages correctly. It says little about what preference
alignment changed: on three classes, two close models often produce the same decisions, and the
gap drowns in the uncertainty.

So this module reads the alignment at the point it changes: **the likelihood given to two
answers of one question**. Each answer is read twice by the same model — with the DPO adapter, and without
it, that is by the supervised model the alignment started from. The difference between the two
log-likelihoods is what the alignment changed for that answer: the implicit reward DPO
optimises. The pair is well ordered if the preferred answer
gained more from the alignment than the rejected one.

Both readings cover the same text, so what belongs to the text itself — rare vocabulary, an
author's style — cancels out in the difference. A change the alignment applies to every token,
on the other hand, accumulates with length: the summary therefore also says what share of the
pairs it settles in favour of the longer answer, and what share of the answers it lowered.

Two sets are read this way. The test split of **UltraMedical-Preference** (777 pairs; GPT-4 chose the better
answer of each, and biomedical experts reviewed the choice) measures whether anything
transferred to medicine at large; it is kept out of training, its long English essays would
teach the model to break the triage output contract. The **task-format set** of
:mod:`clinical_triage.evaluation.task_preferences` measures whether the preference learnt on
generated narratives holds on narratives written by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

from clinical_triage.config import MODEL
from clinical_triage.prompts import bound_description
from clinical_triage.utils import get_logger

logger = get_logger("preference")


@dataclass(frozen=True)
class AlignmentEffect:
    """What the alignment changed to the log-likelihood of the two answers of a pair."""

    preferred_gain: float
    rejected_gain: float
    group: str
    preferred_is_longer: bool

    @property
    def correctly_ordered(self) -> bool:
        return self.preferred_gain > self.rejected_gain


def _bounded_prompt(prompt: str, tokenizer) -> str:
    # The prompt is bounded to half the model window, so that room always remains for the answer,
    # which is the object of the measurement. The bound applies to the **text**, not to the
    # token sequence, so that the prompt tokens stay an exact prefix of the whole — which is what
    # allows the start of the answer to be located by a simple count.
    return bound_description(prompt, tokenizer, MODEL.max_seq_length // 2)[0]


def _log_likelihood(model, tokenizer, prompt: str, answer: str, device: str) -> float:
    """Log-likelihood of ``answer`` given ``prompt``, summed over its tokens.

    The sum, not the mean: it is the quantity DPO optimises, and the two readings of one answer
    cover the same tokens.
    """
    import torch

    # Bounded by the real model window. Beyond it, the forward pass would truncate the logits to
    # the window while the targets kept their original length, and reading the likelihoods would
    # fail on a dimension mismatch — UltraMedical's essays regularly exceed that window.
    bounded = _bounded_prompt(prompt, tokenizer)
    prompt_tokens = tokenizer(bounded, return_tensors="pt")
    full_tokens = tokenizer(
        bounded + answer,
        return_tensors="pt",
        truncation=True,
        max_length=MODEL.max_seq_length,
    )
    inputs = full_tokens["input_ids"].to(device)
    answer_start = prompt_tokens["input_ids"].shape[1]

    with torch.no_grad():
        logits = model(inputs).logits
    # The logit at position i predicts token i+1: shift by one.
    log_probabilities = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = inputs[0, 1:]
    kept = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    return float(kept[answer_start - 1 :].sum())


def clipped_answers(tokenizer, pairs, separator: str = "\n\n") -> int:
    """How many answers the model window cuts short.

    The comparison stays fair — both readings of one answer cover the same tokens — but it then
    covers the **beginning** of the answer. The count is published beside the result.
    """
    clipped = 0
    for pair in pairs:
        bounded = _bounded_prompt(f"{pair.prompt}{separator}", tokenizer)
        clipped += sum(
            len(tokenizer(bounded + answer)["input_ids"]) > MODEL.max_seq_length
            for answer in (pair.chosen, pair.rejected)
        )
    return clipped


def measure_alignment(
    model, tokenizer, pairs, device: str = "cuda", separator: str = "\n\n"
) -> list[AlignmentEffect]:
    """Read every answer with and without the DPO adapter, and keep the difference.

    ``model`` must carry the DPO adapter on top of the merged supervised model, as in training:
    disabling the adapter then gives back exactly the reference model, with no second load.

    ``separator`` closes the prompt. Between a free-text question and its answer, two newlines
    form a token of their own, which keeps the prompt tokens an exact prefix of the whole; a
    triage prompt already ends on the opening of the assistant turn and takes none.
    """
    effects = []
    for index, pair in enumerate(pairs):
        prompt = f"{pair.prompt}{separator}"
        after = [
            _log_likelihood(model, tokenizer, prompt, answer, device)
            for answer in (pair.chosen, pair.rejected)
        ]
        with model.disable_adapter():
            before = [
                _log_likelihood(model, tokenizer, prompt, answer, device)
                for answer in (pair.chosen, pair.rejected)
            ]
        effects.append(
            AlignmentEffect(
                preferred_gain=after[0] - before[0],
                rejected_gain=after[1] - before[1],
                group=pair.label_type,
                preferred_is_longer=len(pair.chosen) > len(pair.rejected),
            )
        )
        if (index + 1) % 100 == 0:
            logger.info("  %d/%d pairs measured", index + 1, len(pairs))
    return effects


def _share(values: list[bool]) -> float:
    return round(sum(values) / len(values), 4)


def summarize(effects: list[AlignmentEffect]) -> dict:
    """The share of correctly ordered pairs, overall and per group, with what qualifies it.

    Two shares travel with the result, because it cannot be read without them: the answers whose
    likelihood the alignment **lowered**, and the pairs it settled in favour of the **longer**
    answer.
    """
    if not effects:
        return {"n": 0, "correctly_ordered_share": 0.0, "per_group": {}}
    per_group = {}
    for group in sorted({effect.group for effect in effects}):
        subset = [effect for effect in effects if effect.group == group]
        per_group[group] = {
            "n": len(subset),
            "correctly_ordered_share": _share([effect.correctly_ordered for effect in subset]),
        }
    gains = [gain for effect in effects for gain in (effect.preferred_gain, effect.rejected_gain)]
    return {
        "n": len(effects),
        "correctly_ordered_share": _share([effect.correctly_ordered for effect in effects]),
        "per_group": per_group,
        "lowered_answers_share": _share([gain < 0 for gain in gains]),
        "longer_favoured_share": _share(
            [effect.correctly_ordered == effect.preferred_is_longer for effect in effects]
        ),
    }


def preferred_longer_share(pairs, set_aside: str = "length") -> dict:
    """How often the preferred answer is simply the longer one, outside the length-matched group.

    This is why the corpus is never trained on: an alignment learnt on pairs where the preferred
    answer is mostly the longer one learns length. The ``length`` group is set aside because its
    authors built it from answers of comparable length — it is the one group where length cannot
    decide.
    """
    considered = [pair for pair in pairs if pair.label_type != set_aside]
    longer = sum(len(pair.chosen) > len(pair.rejected) for pair in considered)
    return {
        "n": len(considered),
        "preferred_is_longer": longer,
        "share": round(longer / len(considered), 4) if considered else 0.0,
    }
