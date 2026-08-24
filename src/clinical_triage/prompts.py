"""Format de dialogue de l'agent de triage — source unique de vérité.

Le même gabarit sert au fine-tuning supervisé, à l'alignement par préférences,
à l'évaluation et au service. C'est volontaire : la moindre divergence entre le
format appris et le format servi dégrade les réponses sans qu'aucune métrique
d'entraînement ne le signale.

Deux précautions importantes :

- On installe notre propre gabarit ChatML sur le tokenizer (`install_chat_template`)
  et on le sauvegarde avec le modèle. Le gabarit natif de Qwen3 insère un bloc
  `<think></think>` devant la réponse de l'assistant ; les bibliothèques
  d'entraînement appliquent ce gabarit dès qu'un jeu de données expose une
  colonne `messages`, ce qui ferait apprendre au modèle un format que l'inférence
  ne reproduit jamais.
- On déclare `<|im_end|>` comme jeton de fin de séquence
  (`install_end_of_turn_token`). Le tokenizer de Qwen3-1.7B-Base sort d'usine
  avec `<|endoftext|>` comme fin de séquence : sans cette correction, la
  génération ne s'arrête jamais à la fin de la réponse de triage.
"""

from __future__ import annotations

import re
import unicodedata

from clinical_triage.config import TRIAGE

# --- Consigne système : rôle, périmètre, format de sortie et garde-fous ---

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

# --- Marqueurs ChatML (famille Qwen) ---

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Gabarit Jinja équivalent à `format_chatml`, installé sur le tokenizer pour que
# les bibliothèques d'entraînement produisent exactement le même texte.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def build_messages(user_content: str, system: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    """Construit la liste de messages (rôle/contenu) d'un échange de triage."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def format_chatml(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    """Sérialise des messages au format ChatML.

    Avec `add_generation_prompt`, on ouvre le tour assistant pour déclencher la
    génération (inférence). Sinon l'échange complet est sérialisé, réponse
    comprise (cas de l'entraînement).
    """
    parts = [f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages]
    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def completion_payload(symptomes: str, modele: str) -> dict:
    """Corps de requête pour l'API de complétion compatible OpenAI de vLLM.

    La passerelle et le banc de performance l'appellent tous les deux, et c'est
    tout l'intérêt : assemblé de chaque côté, le corps de requête resterait
    identique champ pour champ jusqu'au jour où l'un gagne un paramètre de
    génération, et les latences publiées décriraient alors une requête que le
    service n'émet plus, sans que rien ne le signale.
    """
    from clinical_triage.config import SERVING

    return {
        "model": modele,
        "prompt": format_chatml(build_messages(symptomes), add_generation_prompt=True),
        "max_tokens": SERVING.max_new_tokens,
        "temperature": SERVING.temperature,
        "stop": [IM_END],
    }


def budget_de_description(tokenizer, max_seq_length: int, max_new_tokens: int) -> int:
    """Nombre de jetons laissés à la description du patient dans la fenêtre.

    La fenêtre doit loger trois choses : l'encadrement (consigne système et
    marqueurs de dialogue), la description du patient, et la réponse à produire.
    L'encadrement se mesure plutôt qu'il ne s'estime — on sérialise une invite
    vide et on la compte.
    """
    encadrement = len(
        tokenizer(format_chatml(build_messages(""), add_generation_prompt=True))["input_ids"]
    )
    return max(0, max_seq_length - max_new_tokens - encadrement)


def borner_la_description(texte: str, tokenizer, budget: int) -> tuple[str, bool]:
    """Ramène une description dans le budget de jetons, et dit si elle a été coupée.

    Sans cette borne, une description plus longue que la fenêtre du modèle ne
    produit pas une réponse dégradée : elle **interrompt la génération** par une
    erreur de dimension de tenseur, au moment précis où un soignant attend une
    réponse. Le contrat d'API accepte jusqu'à quatre mille caractères, soit
    largement de quoi dépasser la fenêtre.

    La coupe porte sur la fin du texte, et jamais sur la consigne système ni sur
    la question qui la suit : un triage dont la consigne a disparu ne répond plus
    au format attendu. Le second élément du couple dit qu'une coupe a eu lieu —
    il ne doit pas rester dans le code, il doit remonter jusqu'au soignant, qui
    est seul à savoir si ce qui manque comptait.
    """
    jetons = tokenizer(texte, add_special_tokens=False)["input_ids"]
    if len(jetons) <= budget:
        return texte, False
    return tokenizer.decode(jetons[:budget], skip_special_tokens=True), True


def install_chat_template(tokenizer) -> None:
    """Installe notre gabarit ChatML sur le tokenizer (remplace celui du modèle)."""
    tokenizer.chat_template = CHAT_TEMPLATE


def install_end_of_turn_token(tokenizer) -> int:
    """Déclare `<|im_end|>` comme fin de séquence et renvoie son identifiant.

    Le tokenizer sert aussi de référence au modèle fusionné exporté vers vLLM :
    corriger la fin de séquence ici suffit à faire s'arrêter la génération dans
    tous les moteurs d'inférence.
    """
    end_id = tokenizer.convert_tokens_to_ids(IM_END)
    tokenizer.eos_token = IM_END
    if tokenizer.pad_token is None:
        tokenizer.pad_token = IM_END
    return end_id


def prepare_tokenizer(tokenizer) -> int:
    """Applique les deux corrections de format et renvoie l'identifiant de fin de tour."""
    install_chat_template(tokenizer)
    return install_end_of_turn_token(tokenizer)


# --- Mise en forme et lecture de la réponse de triage ---


def build_target_response(level: str, justification: str, recommendation: str) -> str:
    """Met en forme la réponse cible du modèle selon le format imposé."""
    if level not in TRIAGE.levels:
        raise ValueError(f"Niveau de triage inconnu : {level!r}")
    return (
        f"Niveau de priorité : {level}\n"
        f"Justification : {justification}\n"
        f"Recommandation : {recommendation}"
    )


def strip_accents(text: str) -> str:
    """Retire les accents pour rendre la lecture d'une réponse tolérante à l'orthographe.

    Sert au **repérage**, jamais à la restitution : un texte passé par ici ne
    doit pas ressortir vers un utilisateur. Le module de contrôle de sécurité
    s'en sert pour comparer ses motifs à un texte normalisé comme eux.
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
    """Extrait le niveau de priorité annoncé par le modèle.

    Renvoie `None` si aucun niveau valide n'est trouvé, ou si la réponse en
    annonce plusieurs différents : dans les deux cas la réponse est inexploitable
    par le système d'information hospitalier et doit compter comme hors format.
    """
    found = {m.group(1).upper() for m in _LEVEL_LINE.finditer(strip_accents(generated_text))}
    if len(found) != 1:
        return None
    return found.pop()


def parse_response(generated_text: str) -> dict[str, str | None]:
    """Décompose une réponse générée en niveau, justification et recommandation.

    Les deux textes sont rendus **tels que le modèle les a écrits**, accents
    compris : ils s'affichent à l'écran du soignant, et « oedeme aigu du poumon »
    y serait une faute visible. Seule la ligne de niveau se lit sur un texte
    normalisé, parce que son étiquette porte un accent (« priorité ») que le
    modèle peut omettre ; « Justification » et « Recommandation » n'en ont pas,
    et se repèrent donc directement sur le texte d'origine.
    """
    justification = _JUSTIFICATION_LINE.search(generated_text)
    recommendation = _RECOMMENDATION_LINE.search(generated_text)
    return {
        "level": extract_level(generated_text),
        "justification": justification.group(1).strip() if justification else None,
        "recommendation": recommendation.group(1).strip() if recommendation else None,
    }


# Ce qui, en début de ligne, signale que la génération a quitté la réponse : un
# nouveau tour de dialogue, un marqueur de gabarit, ou le premier mot de la
# consigne système. Une consigne détournée produit exactement cela.
_REPRISE_DE_DIALOGUE = re.compile(
    r"^\s*(?:<\|im_(?:start|end)\|>|human\s*:|assistant\s*:|system\s*:|user\s*:"
    r"|tu es l'assistant|you are the)",
    re.IGNORECASE,
)

# Lignes conservées quand la réponse ne comporte pas de recommandation. Le
# contrat en compte trois ; une de plus tolère une justification qui déborde.
LIGNES_SANS_RECOMMANDATION = 4


def truncate_to_answer(generated_text: str) -> str:
    """Coupe tout ce que le modèle produit après la ligne de recommandation.

    Filet de sécurité du service : si la génération déborde (jeton de fin manqué,
    répétition), le personnel soignant et le journal d'audit ne reçoivent que le
    bloc de réponse attendu, jamais la consigne système ni un texte en roue libre.

    Le filet **échoue fermé**. S'arrêter à la seule ligne de recommandation
    rendrait entière, consigne système comprise, une génération qui n'en produit
    aucune — le cas même d'une consigne détournée. Deux garde-fous s'y ajoutent :
    on coupe avant toute reprise de dialogue, et à défaut de recommandation on ne
    garde que les premières lignes.
    """
    text = generated_text.split(IM_END)[0].strip()
    kept: list[str] = []
    for line in text.splitlines():
        if _REPRISE_DE_DIALOGUE.match(line):
            break
        kept.append(line)
        if _RECOMMENDATION_LINE.match(strip_accents(line.strip())):
            return "\n".join(kept).strip()
    return "\n".join(kept[:LIGNES_SANS_RECOMMANDATION]).strip()
