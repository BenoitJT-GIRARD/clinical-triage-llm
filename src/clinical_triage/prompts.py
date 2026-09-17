"""The agent's dialogue format — single source of truth.

The same template serves supervised fine-tuning, preference alignment, evaluation and the
service. That is deliberate: the slightest divergence between the format learnt and the format
served degrades the answers without any training metric reporting it.

Two precautions matter:

- our own ChatML template is installed on the tokenizer (:func:`install_chat_template`) and
  saved with the model. Qwen3's native template inserts a ``<think></think>`` block before the
  assistant's answer; the training libraries apply that template as soon as a dataset exposes a
  ``messages`` column, which would teach the model a format inference never reproduces;
- ``<|im_end|>`` is declared as the end-of-sequence token
  (:func:`install_end_of_turn_token`). The Qwen3-1.7B-Base tokenizer ships with
  ``<|endoftext|>`` as end of sequence: without this correction, generation never stops at the
  end of the triage answer.

The prompt and the answer format are French because the model answers in French: they are the
data the model was trained on, not prose about the project.
"""

from __future__ import annotations

import re
import unicodedata

from clinical_triage.config import TRIAGE

# --- System prompt: role, scope, output format and guard rails ---

SYSTEM_PROMPT = (
    "Tu es l'assistant de triage médical du service des urgences. "
    "Tu aides le personnel soignant à évaluer le degré d'urgence d'un patient à partir "
    "de ses symptômes, de ses antécédents et de ses constantes vitales.\n\n"
    "Tu réponds TOUJOURS en français et selon le format suivant :\n"
    "Niveau de priorité : <URGENCE_VITALE | URGENCE_MODEREE | CONSULTATION_DIFFEREE>\n"
    "Justification : <explication clinique courte et claire>\n"
    "Recommandation : <conduite à tenir pour le patient et le personnel>\n\n"
    "Règles de sécurité :\n"
    "- En cas de doute ou de signe de gravité, surclasse le niveau d'urgence.\n"
    "- Ne pose jamais de diagnostic définitif : tu fournis une aide à la décision.\n"
    "- Rappelle de contacter le 15 (SAMU) devant tout signe vital engagé."
)

# --- ChatML markers (Qwen family) ---

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Jinja template equivalent to `format_chatml`, installed on the tokenizer so that the training
# libraries produce exactly the same text.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def build_messages(user_content: str, system: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    """Build the role/content message list of one triage exchange."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def format_chatml(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    """Serialise messages in ChatML.

    With ``add_generation_prompt``, the assistant turn is opened to trigger generation
    (inference). Otherwise the whole exchange is serialised, answer included (training).
    """
    parts = [f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages]
    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def completion_payload(symptoms: str, model: str) -> dict:
    """Request body for vLLM's OpenAI-compatible completion API.

    The gateway and the performance bench both call it, and that is the point: assembled on
    each side, the body would stay identical field for field until the day one of them gains a
    generation parameter — and the published latencies would then describe a request the
    service no longer sends, with nothing to report it.
    """
    from clinical_triage.config import SERVING

    return {
        "model": model,
        "prompt": format_chatml(build_messages(symptoms), add_generation_prompt=True),
        "max_tokens": SERVING.max_new_tokens,
        "temperature": SERVING.temperature,
        "stop": [IM_END],
    }


def description_budget(tokenizer, max_seq_length: int, max_new_tokens: int) -> int:
    """How many tokens the window leaves to the patient description.

    The window must hold three things: the framing (system prompt and dialogue markers), the
    patient description, and the answer to produce. The framing is measured rather than
    estimated — an empty prompt is serialised and counted.
    """
    framing = len(
        tokenizer(format_chatml(build_messages(""), add_generation_prompt=True))["input_ids"]
    )
    return max(0, max_seq_length - max_new_tokens - framing)


def bound_description(text: str, tokenizer, budget: int) -> tuple[str, bool]:
    """Bring a description back inside the token budget, and say whether it was cut.

    Without this bound, a description longer than the model's window does not produce a
    degraded answer: it **interrupts generation** with a tensor-dimension error, at the exact
    moment a nurse is waiting for one. The API contract accepts up to four thousand characters,
    which is more than enough to exceed the window.

    The cut falls at the end of the text, never on the system prompt nor on the question that
    follows it: a triage whose prompt has disappeared no longer answers in the expected format.
    The second element of the pair says a cut happened — it must not stay in the code, it must
    reach the nurse, who alone knows whether what is missing mattered.
    """
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(tokens) <= budget:
        return text, False
    return tokenizer.decode(tokens[:budget], skip_special_tokens=True), True


def install_chat_template(tokenizer) -> None:
    """Install our ChatML template on the tokenizer (replacing the model's)."""
    tokenizer.chat_template = CHAT_TEMPLATE


def install_end_of_turn_token(tokenizer) -> int:
    """Declare ``<|im_end|>`` as end of sequence and return its id.

    The tokenizer is also the reference of the merged model exported to vLLM: fixing the end of
    sequence here is enough to make generation stop in every inference engine.
    """
    end_id = tokenizer.convert_tokens_to_ids(IM_END)
    tokenizer.eos_token = IM_END
    if tokenizer.pad_token is None:
        tokenizer.pad_token = IM_END
    return end_id


def prepare_tokenizer(tokenizer) -> int:
    """Apply both format corrections and return the end-of-turn id."""
    install_chat_template(tokenizer)
    return install_end_of_turn_token(tokenizer)


# --- Formatting and reading a triage answer ---


def build_target_response(level: str, justification: str, recommendation: str) -> str:
    """Format the model's target answer in the expected layout."""
    if level not in TRIAGE.levels:
        raise ValueError(f"Unknown triage level: {level!r}")
    return (
        f"Niveau de priorité : {level}\n"
        f"Justification : {justification}\n"
        f"Recommandation : {recommendation}"
    )


def strip_accents(text: str) -> str:
    """Drop accents so that reading an answer tolerates spelling.

    Used for **detection**, never for display: text that goes through here must not come back
    out to a user. The safety-check module uses it to compare its patterns against a text
    normalised the same way.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


_LEVEL_LINE = re.compile(
    r"niveau\s+de\s+priorite\s*:\s*(URGENCE_VITALE|URGENCE_MODEREE|CONSULTATION_DIFFEREE)",
    re.IGNORECASE,
)
_JUSTIFICATION_LINE = re.compile(r"justification\s*:\s*(.+)", re.IGNORECASE)
_RECOMMENDATION_LINE = re.compile(r"recommandation\s*:\s*(.+)", re.IGNORECASE)


def extract_level(generated_text: str) -> str | None:
    """Extract the priority level the model announced.

    Returns ``None`` if no valid level is found, or if the answer announces several different
    ones: in both cases the answer is unusable by the hospital information system and must
    count as off-format.
    """
    found = {m.group(1).upper() for m in _LEVEL_LINE.finditer(strip_accents(generated_text))}
    if len(found) != 1:
        return None
    return found.pop()


def parse_response(generated_text: str) -> dict[str, str | None]:
    """Split a generated answer into level, justification and recommendation.

    Both texts are returned **as the model wrote them**, accents included: they are displayed
    on the nurse's screen, and "oedeme aigu du poumon" would be a visible mistake there. Only
    the level line is read on a normalised text, because its label carries an accent
    ("priorité") that the model may omit; "Justification" and "Recommandation" carry none, and
    are therefore matched directly on the original text.
    """
    justification = _JUSTIFICATION_LINE.search(generated_text)
    recommendation = _RECOMMENDATION_LINE.search(generated_text)
    return {
        "level": extract_level(generated_text),
        "justification": justification.group(1).strip() if justification else None,
        "recommendation": recommendation.group(1).strip() if recommendation else None,
    }


# What, at the start of a line, signals that generation has left the answer: a new dialogue
# turn, a template marker, or the first words of the system prompt. A hijacked prompt produces
# exactly that.
_DIALOGUE_RESUMES = re.compile(
    r"^\s*(?:<\|im_(?:start|end)\|>|human\s*:|assistant\s*:|system\s*:|user\s*:"
    r"|tu es l'assistant|you are the)",
    re.IGNORECASE,
)

# Lines kept when the answer carries no recommendation. The contract has three; one more
# tolerates a justification that spills over.
LINES_WITHOUT_RECOMMENDATION = 4


def truncate_to_answer(generated_text: str) -> str:
    """Cut everything the model produces after the recommendation line.

    The service's safety net: if generation overruns (missed end token, repetition), the nurse
    and the audit log receive only the expected answer block, never the system prompt nor a
    free-running text.

    The net **fails closed**. Stopping at the recommendation line alone would return a
    generation that produces none in full, system prompt included — precisely the case of a
    hijacked prompt. Two guards are added: the cut happens before any resumed dialogue, and
    failing a recommendation, only the first lines are kept.
    """
    text = generated_text.split(IM_END)[0].strip()
    kept: list[str] = []
    for line in text.splitlines():
        if _DIALOGUE_RESUMES.match(line):
            break
        kept.append(line)
        if _RECOMMENDATION_LINE.match(strip_accents(line.strip())):
            return "\n".join(kept).strip()
    return "\n".join(kept[:LINES_WITHOUT_RECOMMENDATION]).strip()
