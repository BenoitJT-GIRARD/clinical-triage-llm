"""An independent measurement of the alignment, on an external preference set.

The clinical set says whether the model triages correctly. It says little about what preference
alignment changed: on three classes, two close models often produce the same decisions, and the
gap drowns in the uncertainty.

So the alignment is measured where it expresses itself: **on preferences**. For each pair of
answers annotated by humans, the likelihood the model assigns to the preferred answer is
compared with the one it assigns to the rejected answer. A better-aligned model ranks the good
answer first more often.

The set used is **UltraMedical-Preference**. It is deliberately kept **out of training**: its
answers are long English essays, which would teach the model to violate the triage output
contract. Reserved for measurement, it provides an evaluation of the alignment that depends
neither on our data nor on our labels.

Pairs labelled by answer length alone are discarded: they would measure verbosity, not clinical
preference.
"""

from __future__ import annotations

from dataclasses import dataclass

from clinical_triage.config import MODEL
from clinical_triage.prompts import bound_description
from clinical_triage.utils import get_logger

logger = get_logger("preference")


@dataclass(frozen=True)
class PreferenceScore:
    """The likelihoods assigned to the two answers of a pair.

    ``truncated_sides`` counts how many of the two answers the model window cut short — 0, 1 or
    2. It travels with the score, where a log line would bury it, because the share of
    truncated answers is published next to the result and qualifies how it must be read.
    """

    chosen: float
    rejected: float
    label_type: str
    truncated_sides: int = 0

    @property
    def correctly_ordered(self) -> bool:
        return self.chosen > self.rejected

    @property
    def margin(self) -> float:
        return self.chosen - self.rejected


def _log_likelihood(model, tokenizer, prompt: str, answer: str, device: str) -> float:
    """Mean per-token log-likelihood of ``answer``, given ``prompt``.

    Normalised by length: without that, the comparison would mechanically favour the shorter
    answer, and length would be measured a second time, substance never.
    """
    import torch

    # The bounds come from the model's real window. Beyond it, the forward pass would truncate
    # the logits to the window while the targets kept their original length, and reading the
    # likelihoods would fail on a dimension mismatch — UltraMedical's essays regularly exceed
    # that window.
    #
    # The prompt is bounded to half the window so that room always remains for the answer, which
    # is the object of the measurement. The bound applies to the **text**, not to the token
    # sequence, so that the prompt tokens stay an exact prefix of the whole — which is what
    # allows the start of the answer to be located by a simple count.
    bounded_prompt, _ = bound_description(prompt, tokenizer, MODEL.max_seq_length // 2)
    prompt_tokens = tokenizer(bounded_prompt, return_tensors="pt")
    full_tokens = tokenizer(
        bounded_prompt + answer,
        return_tensors="pt",
        truncation=True,
        max_length=MODEL.max_seq_length,
    )
    inputs = full_tokens["input_ids"].to(device)
    answer_start = prompt_tokens["input_ids"].shape[1]
    if inputs.shape[1] <= answer_start:
        return float("-inf")

    with torch.no_grad():
        logits = model(inputs).logits
    # The logit at position i predicts token i+1: shift by one.
    log_probabilities = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = inputs[0, 1:]
    kept = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    answer_only = kept[answer_start - 1 :]
    return float(answer_only.mean())


def truncated_sides(tokenizer, pair, prompt: str) -> int:
    """How many of the pair's two answers the model window cuts short.

    UltraMedical's essays regularly exceed the window. The comparison stays fair — the same
    bound applies to the preferred and to the rejected answer, and the score is normalised by
    length — but it then covers the **beginning** of each answer. The published result must be
    able to say so with a number, and not pass over it in silence.
    """
    bounded, _ = bound_description(prompt, tokenizer, MODEL.max_seq_length // 2)
    return sum(
        len(tokenizer(bounded + answer)["input_ids"]) > MODEL.max_seq_length
        for answer in (pair.chosen, pair.rejected)
    )


def score_pairs(model, tokenizer, pairs, device: str = "cuda") -> list[PreferenceScore]:
    """Assign each pair the likelihoods of its two answers."""
    scores = []
    for index, pair in enumerate(pairs):
        # The two newlines are not cosmetic. `_log_likelihood` locates the start of the answer
        # by counting the tokens of the prompt alone, which assumes those tokens form an
        # **exact prefix** of the whole. Sub-word tokenisation can merge the last character of
        # the prompt with the first of the answer and shift the boundary by one token; a
        # separator that forms its own token prevents that. Checked on the corpus: no broken
        # prefix.
        prompt = f"{pair.prompt}\n\n"
        scores.append(
            PreferenceScore(
                chosen=_log_likelihood(model, tokenizer, prompt, pair.chosen, device),
                rejected=_log_likelihood(model, tokenizer, prompt, pair.rejected, device),
                label_type=pair.label_type,
                truncated_sides=truncated_sides(tokenizer, pair, prompt),
            )
        )
        if (index + 1) % 25 == 0:
            logger.info("  %d/%d pairs scored", index + 1, len(pairs))
    truncated = sum(score.truncated_sides for score in scores)
    if truncated:
        logger.info(
            "  %d answers out of %d exceed the window: the comparison covers their beginning.",
            truncated,
            2 * len(pairs),
        )
    return scores


def summarize(scores: list[PreferenceScore]) -> dict:
    """Aggregate the scores: share of correctly ordered pairs and mean margin."""
    if not scores:
        return {"n": 0, "correctly_ordered_share": 0.0, "mean_margin": 0.0}
    correctly_ordered = sum(score.correctly_ordered for score in scores)
    per_difficulty = {}
    for level in sorted({score.label_type for score in scores}):
        subset = [s for s in scores if s.label_type == level]
        per_difficulty[level] = {
            "n": len(subset),
            "correctly_ordered_share": round(
                sum(s.correctly_ordered for s in subset) / len(subset), 4
            ),
        }
    truncated = sum(score.truncated_sides for score in scores)
    return {
        "n": len(scores),
        "correctly_ordered_share": round(correctly_ordered / len(scores), 4),
        "mean_margin": round(sum(score.margin for score in scores) / len(scores), 4),
        "per_difficulty": per_difficulty,
        # How much of what was compared the model window cut short. Published rather than
        # logged: it is the first thing that qualifies a result sitting below chance.
        "truncated_answers": truncated,
        "truncated_share": round(truncated / (2 * len(scores)), 4),
        # The pair-by-pair detail, in set order. Two models are evaluated on the **same** pairs:
        # comparing their two proportions would treat them as independent measurements, when
        # all the information is in the pairs where they diverge.
        "correctly_ordered": [score.correctly_ordered for score in scores],
    }
