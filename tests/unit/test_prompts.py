"""Tests of the dialogue format and of reading an answer back.

The prompt and the answers are French: they are the model's contract, and the assertions below
match them character for character.
"""

from __future__ import annotations

import pytest

from clinical_triage.prompts import (
    CHAT_TEMPLATE,
    IM_END,
    LINES_WITHOUT_RECOMMENDATION,
    bound_description,
    build_messages,
    build_target_response,
    description_budget,
    extract_level,
    format_chatml,
    parse_response,
    truncate_to_answer,
)


class FakeTokenizer:
    """A minimal tokenizer, enough to check the template installation."""

    def __init__(self) -> None:
        self.chat_template = "the model's native template"
        self.eos_token = "<|endoftext|>"
        self.pad_token = None

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|im_end|>": 151645, "<|endoftext|>": 151643}[token]


def test_messages_start_with_the_system_prompt():
    messages = build_messages("Douleur thoracique depuis une heure.")
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "thoracique" in messages[1]["content"]


def test_a_generation_prompt_opens_the_assistant_turn():
    prompt = format_chatml(build_messages("Mal de tête léger."), add_generation_prompt=True)
    assert prompt.endswith("<|im_start|>assistant\n")


def test_a_complete_exchange_ends_with_the_end_token():
    messages = build_messages("Toux sèche.") + [{"role": "assistant", "content": "Réponse."}]
    assert format_chatml(messages).rstrip("\n").endswith(IM_END)


def test_the_installed_template_reproduces_format_chatml_exactly():
    """The Jinja template and the Python function must produce the same text.

    That is the guarantee that training, which goes through the template, and serving, which
    goes through the function, see the same format.
    """
    jinja2 = pytest.importorskip("jinja2")
    messages = build_messages("Vertiges depuis ce matin.") + [
        {"role": "assistant", "content": "Niveau de priorité : URGENCE_MODEREE"}
    ]
    template = jinja2.Template(CHAT_TEMPLATE)
    assert template.render(messages=messages, add_generation_prompt=False) == format_chatml(
        messages
    )
    assert template.render(messages=messages[:2], add_generation_prompt=True) == format_chatml(
        messages[:2], add_generation_prompt=True
    )


def test_prepare_tokenizer_fixes_the_end_token():
    from clinical_triage.prompts import prepare_tokenizer

    tokenizer = FakeTokenizer()
    identifier = prepare_tokenizer(tokenizer)
    assert identifier == 151645
    assert tokenizer.eos_token == IM_END
    assert tokenizer.pad_token == IM_END
    assert tokenizer.chat_template == CHAT_TEMPLATE


def test_a_target_answer_refuses_an_unknown_level():
    with pytest.raises(ValueError):
        build_target_response("URGENCE_INCONNUE", "x", "y")


def test_level_extraction_tolerates_missing_accents():
    assert (
        extract_level("Niveau de priorité : URGENCE_VITALE\nJustification : ...")
        == "URGENCE_VITALE"
    )
    assert extract_level("Niveau de priorite : URGENCE_MODEREE") == "URGENCE_MODEREE"


def test_level_extraction_refuses_an_ambiguous_answer():
    """Two different levels in one answer make it unusable."""
    answer = "Niveau de priorité : URGENCE_VITALE\n...\nNiveau de priorité : CONSULTATION_DIFFEREE"
    assert extract_level(answer) is None


def test_level_extraction_accepts_an_identical_repetition():
    answer = "Niveau de priorité : URGENCE_VITALE\n...\nNiveau de priorité : URGENCE_VITALE"
    assert extract_level(answer) == "URGENCE_VITALE"


def test_an_answer_with_no_level_extracts_none():
    assert extract_level("Réponse hors format, sans niveau.") is None


def test_the_three_parts_are_read_back():
    """Both texts are returned as the model wrote them, accents included.

    They appear on the nurse's screen: "Fievre elevee" would be a visible mistake there. Only
    the level line is read on a normalised text, because its label carries an accent the model
    may omit.
    """
    answer = build_target_response("URGENCE_MODEREE", "Fièvre élevée.", "Évaluation sous 4 heures.")
    parts = parse_response(answer)
    assert parts["level"] == "URGENCE_MODEREE"
    assert parts["justification"] == "Fièvre élevée."
    assert parts["recommendation"] == "Évaluation sous 4 heures."


def test_the_level_is_read_even_when_the_model_drops_the_accent():
    without_accent = (
        "Niveau de priorite : URGENCE_VITALE\n"
        "Justification : détresse respiratoire.\n"
        "Recommandation : déchocage."
    )
    parts = parse_response(without_accent)
    assert parts["level"] == "URGENCE_VITALE"
    assert parts["justification"] == "détresse respiratoire."


def test_truncation_cuts_after_the_recommendation():
    """The service's safety net: nothing gets through after the recommendation."""
    generated = (
        "Niveau de priorité : URGENCE_VITALE\n"
        "Justification : Signes de détresse.\n"
        "Recommandation : Appeler le 15.\n"
        "Human: Tu es l'assistant de triage médical du service des urgences..."
    )
    truncated = truncate_to_answer(generated)
    assert truncated.endswith("Appeler le 15.")
    assert "assistant de triage" not in truncated


def test_truncation_cuts_at_the_end_token():
    assert truncate_to_answer(f"Réponse.{IM_END}suite parasite") == "Réponse."


def test_truncation_leaves_a_compliant_answer_intact():
    answer = build_target_response("CONSULTATION_DIFFEREE", "Rien de grave.", "Médecin traitant.")
    assert truncate_to_answer(answer) == answer


# --- The truncation net must fail closed ---


def test_a_generation_without_a_recommendation_is_not_returned_whole():
    """The net used to stop at the recommendation line alone.

    A generation producing none — precisely the case of a hijacked prompt — was therefore
    returned whole, system prompt included, shown to the nurse and written to the audit log as
    the triage answer.
    """
    leak = (
        "Tu es l'assistant de triage médical du service des urgences.\n"
        "Tu aides le personnel soignant à évaluer le degré d'urgence.\n"
        "Bonjour."
    )
    assert truncate_to_answer(leak) == ""


def test_truncation_cuts_when_a_dialogue_turn_resumes():
    generated = (
        "Niveau de priorité : URGENCE_VITALE\n"
        "<|im_start|>system\n"
        "Tu es l'assistant de triage médical du service."
    )
    assert truncate_to_answer(generated) == "Niveau de priorité : URGENCE_VITALE"


def test_an_answer_without_a_recommendation_is_bounded_to_a_few_lines():
    generated = "Niveau de priorité : URGENCE_MODEREE\nJustification : fièvre.\n" + "Blabla.\n" * 20
    assert len(truncate_to_answer(generated).splitlines()) <= LINES_WITHOUT_RECOMMENDATION


def test_a_justification_spanning_two_lines_survives_truncation():
    """The net must not close on a compliant answer that spills over by one line."""
    generated = (
        "Niveau de priorité : URGENCE_VITALE\n"
        "Justification : douleur thoracique constrictive\n"
        "avec sueurs profuses.\n"
        "Recommandation : déchocage immédiat."
    )
    assert truncate_to_answer(generated).endswith("Recommandation : déchocage immédiat.")


# --- The model window is a constraint, not a suggestion ---


class BoundedTokenizer:
    """A minimal tokenizer: one token per word, enough to exercise the bound."""

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": text.split()}

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


def test_a_description_within_budget_is_left_alone():
    text = "Douleur thoracique depuis vingt minutes"
    bounded, cut = bound_description(text, BoundedTokenizer(), budget=10)
    assert bounded == text
    assert cut is False


def test_a_description_that_is_too_long_is_cut_and_reported():
    """Without this bound, generation stops on a dimension error.

    The "two-page paste" robustness check found it: 825 prompt tokens for a 768 window, and the
    service falls over instead of returning a degraded answer.
    """
    text = " ".join(["mot"] * 50)
    bounded, cut = bound_description(text, BoundedTokenizer(), budget=10)
    assert cut is True
    assert len(bounded.split()) == 10


def test_the_budget_is_measured_rather_than_estimated():
    """The framing — system prompt and markers — is counted, not guessed."""
    budget = description_budget(BoundedTokenizer(), max_seq_length=1000, max_new_tokens=200)
    framing = len(format_chatml(build_messages(""), add_generation_prompt=True).split())
    assert budget == 1000 - 200 - framing


def test_a_negative_budget_is_brought_back_to_zero():
    assert description_budget(BoundedTokenizer(), max_seq_length=10, max_new_tokens=200) == 0
