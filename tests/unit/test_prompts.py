"""Tests du format de dialogue et de la lecture des réponses."""

from __future__ import annotations

import pytest

from clinical_triage.prompts import (
    CHAT_TEMPLATE,
    IM_END,
    LINES_WITHOUT_RECOMMENDATION,
    bound_description,
    description_budget,
    build_messages,
    build_target_response,
    extract_level,
    format_chatml,
    parse_response,
    truncate_to_answer,
)


class TokenizerFictif:
    """Tokenizer minimal, suffisant pour vérifier l'installation du gabarit."""

    def __init__(self) -> None:
        self.chat_template = "gabarit natif du modèle"
        self.eos_token = "<|endoftext|>"
        self.pad_token = None

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|im_end|>": 151645, "<|endoftext|>": 151643}[token]


def test_messages_commencent_par_la_consigne_systeme():
    messages = build_messages("Douleur thoracique depuis une heure.")
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "thoracique" in messages[1]["content"]


def test_invite_de_generation_ouvre_le_tour_assistant():
    invite = format_chatml(build_messages("Mal de tête léger."), add_generation_prompt=True)
    assert invite.endswith("<|im_start|>assistant\n")


def test_echange_complet_se_termine_par_le_jeton_de_fin():
    messages = build_messages("Toux sèche.") + [{"role": "assistant", "content": "Réponse."}]
    assert format_chatml(messages).rstrip("\n").endswith(IM_END)


def test_le_gabarit_installe_reproduit_exactement_format_chatml():
    """Le gabarit Jinja et la fonction Python doivent produire le même texte.

    C'est la garantie que l'entraînement, qui passe par le gabarit, et le service,
    qui passe par la fonction, voient bien le même format.
    """
    jinja2 = pytest.importorskip("jinja2")
    messages = build_messages("Vertiges depuis ce matin.") + [
        {"role": "assistant", "content": "Niveau de priorité : URGENCE_MODEREE"}
    ]
    gabarit = jinja2.Template(CHAT_TEMPLATE)
    assert gabarit.render(messages=messages, add_generation_prompt=False) == format_chatml(messages)
    assert gabarit.render(messages=messages[:2], add_generation_prompt=True) == format_chatml(
        messages[:2], add_generation_prompt=True
    )


def test_prepare_tokenizer_corrige_le_jeton_de_fin():
    from clinical_triage.prompts import prepare_tokenizer

    tokenizer = TokenizerFictif()
    identifiant = prepare_tokenizer(tokenizer)
    assert identifiant == 151645
    assert tokenizer.eos_token == IM_END
    assert tokenizer.pad_token == IM_END
    assert tokenizer.chat_template == CHAT_TEMPLATE


def test_reponse_cible_refuse_un_niveau_inconnu():
    with pytest.raises(ValueError):
        build_target_response("URGENCE_INCONNUE", "x", "y")


def test_extraction_du_niveau_tolere_les_accents():
    assert (
        extract_level("Niveau de priorité : URGENCE_VITALE\nJustification : ...")
        == "URGENCE_VITALE"
    )
    assert extract_level("Niveau de priorite : URGENCE_MODEREE") == "URGENCE_MODEREE"


def test_extraction_du_niveau_refuse_une_reponse_ambigue():
    """Deux niveaux différents dans une réponse la rendent inexploitable."""
    reponse = "Niveau de priorité : URGENCE_VITALE\n...\nNiveau de priorité : CONSULTATION_DIFFEREE"
    assert extract_level(reponse) is None


def test_extraction_du_niveau_accepte_une_repetition_identique():
    reponse = "Niveau de priorité : URGENCE_VITALE\n...\nNiveau de priorité : URGENCE_VITALE"
    assert extract_level(reponse) == "URGENCE_VITALE"


def test_extraction_du_niveau_absente():
    assert extract_level("Réponse hors format, sans niveau.") is None


def test_lecture_des_trois_parties():
    """Les deux textes rendus sont ceux du modèle, accents compris.

    Ils s'affichent à l'écran du soignant : « Fievre elevee » y serait une faute
    visible. Seule la ligne de niveau se lit sur un texte normalisé, parce que
    son étiquette porte un accent que le modèle peut omettre.
    """
    reponse = build_target_response(
        "URGENCE_MODEREE", "Fièvre élevée.", "Évaluation sous 4 heures."
    )
    parties = parse_response(reponse)
    assert parties["level"] == "URGENCE_MODEREE"
    assert parties["justification"] == "Fièvre élevée."
    assert parties["recommendation"] == "Évaluation sous 4 heures."


def test_le_niveau_se_lit_meme_si_le_modele_oublie_l_accent_de_priorite():
    sans_accent = (
        "Niveau de priorite : URGENCE_VITALE\n"
        "Justification : détresse respiratoire.\n"
        "Recommandation : déchocage."
    )
    parties = parse_response(sans_accent)
    assert parties["level"] == "URGENCE_VITALE"
    assert parties["justification"] == "détresse respiratoire."


def test_troncature_coupe_apres_la_recommandation():
    """Le filet de sécurité du service : rien ne passe après la recommendation."""
    genere = (
        "Niveau de priorité : URGENCE_VITALE\n"
        "Justification : Signes de détresse.\n"
        "Recommandation : Appeler le 15.\n"
        "Human: Tu es l'assistant de triage médical du Centre Hospitalier..."
    )
    tronque = truncate_to_answer(genere)
    assert tronque.endswith("Appeler le 15.")
    assert "assistant de triage" not in tronque


def test_troncature_coupe_au_jeton_de_fin():
    assert truncate_to_answer(f"Réponse.{IM_END}suite parasite") == "Réponse."


def test_troncature_laisse_intacte_une_reponse_conforme():
    reponse = build_target_response("CONSULTATION_DIFFEREE", "Rien de grave.", "Médecin traitant.")
    assert truncate_to_answer(reponse) == reponse


# --- Le filet de troncature doit échouer fermé ---


def test_une_generation_sans_recommandation_ne_rend_pas_tout():
    """Le filet ne s'arrêtait qu'à la ligne de recommendation.

    Une génération qui n'en produit aucune — le cas même d'une consigne
    détournée — était donc rendue entière, consigne système comprise, affichée
    au soignant et écrite au journal d'audit comme étant la réponse de triage.
    """
    fuite = (
        "Tu es l'assistant de triage médical du service des urgences.\n"
        "Tu aides le personnel soignant à évaluer le degré d'urgence.\n"
        "Bonjour."
    )
    assert truncate_to_answer(fuite) == ""


def test_la_troncature_coupe_a_la_reprise_d_un_tour_de_dialogue():
    genere = (
        "Niveau de priorité : URGENCE_VITALE\n"
        "<|im_start|>system\n"
        "Tu es l'assistant de triage médical du service."
    )
    assert truncate_to_answer(genere) == "Niveau de priorité : URGENCE_VITALE"


def test_une_reponse_sans_recommandation_est_bornee_a_quelques_lignes():
    genere = "Niveau de priorité : URGENCE_MODEREE\nJustification : fièvre.\n" + "Blabla.\n" * 20
    assert len(truncate_to_answer(genere).splitlines()) <= LINES_WITHOUT_RECOMMENDATION


def test_une_justification_sur_deux_lignes_survit_a_la_troncature():
    """Le filet ne doit pas se refermer sur une réponse conforme qui déborde d'une ligne."""
    genere = (
        "Niveau de priorité : URGENCE_VITALE\n"
        "Justification : douleur thoracique constrictive\n"
        "avec sueurs profuses.\n"
        "Recommandation : déchocage immédiat."
    )
    assert truncate_to_answer(genere).endswith("Recommandation : déchocage immédiat.")


# --- La fenêtre du modèle est une contrainte, pas une suggestion ---


class TokenizerBorne:
    """Tokenizer minimal : un jeton par mot, suffisant pour éprouver la borne."""

    def __call__(self, texte, add_special_tokens=True):
        return {"input_ids": texte.split()}

    def decode(self, jetons, skip_special_tokens=True):
        return " ".join(jetons)


def test_une_description_dans_le_budget_n_est_pas_touchee():
    texte = "Douleur thoracique depuis vingt minutes"
    borne, coupee = bound_description(texte, TokenizerBorne(), budget=10)
    assert borne == texte
    assert coupee is False


def test_une_description_trop_longue_est_coupee_et_signalee():
    """Sans cette borne, la génération s'interrompt sur une erreur de dimension.

    C'est le contrôle de robustesse « copier-coller de deux pages » qui l'a
    trouvé : 825 jetons d'invite pour une fenêtre de 768, et le service tombe au
    lieu de rendre une réponse dégradée.
    """
    texte = " ".join(["mot"] * 50)
    borne, coupee = bound_description(texte, TokenizerBorne(), budget=10)
    assert coupee is True
    assert len(borne.split()) == 10


def test_le_budget_se_mesure_plutot_qu_il_ne_s_estime():
    """L'encadrement — consigne système et marqueurs — se compte, il ne se devine pas."""
    budget = description_budget(TokenizerBorne(), max_seq_length=1000, max_new_tokens=200)
    encadrement = len(format_chatml(build_messages(""), add_generation_prompt=True).split())
    assert budget == 1000 - 200 - encadrement


def test_un_budget_negatif_est_ramene_a_zero():
    assert description_budget(TokenizerBorne(), max_seq_length=10, max_new_tokens=200) == 0
