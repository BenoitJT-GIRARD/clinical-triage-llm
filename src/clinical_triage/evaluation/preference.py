"""Mesure indépendante de l'alignement, sur un jeu de préférences externe.

Le jeu clinique dit si le modèle trie juste. Il ne dit pas grand-chose de ce que
l'alignement par préférences a changé : sur trois classes, deux modèles proches
produisent souvent les mêmes décisions, et l'écart se noie dans l'incertitude.

On mesure donc l'alignement là où il s'exprime : **sur des préférences**. Pour
chaque paire de réponses annotée par des humains, on compare la vraisemblance
que le modèle attribue à la réponse préférée et à la réponse rejetée. Un modèle
mieux aligné classe plus souvent la bonne réponse devant l'autre.

Le jeu utilisé est **UltraMedical-Preference**, le corpus de préférences
médicales du cahier des charges. Il est délibérément tenu **hors de
l'entraînement** : ses réponses sont de longues dissertations en anglais, qui
apprendraient au modèle à violer le contrat de sortie du triage. Réservé à la
mesure, il fournit en revanche une évaluation de l'alignement qui ne dépend ni
de nos données, ni de nos étiquettes.

Les paires étiquetées par la seule longueur de la réponse sont écartées : elles
mesureraient la verbosité, pas la préférence clinique.
"""

from __future__ import annotations

from dataclasses import dataclass

from clinical_triage.config import MODEL
from clinical_triage.prompts import borner_la_description
from clinical_triage.utils import get_logger

logger = get_logger("preference")


@dataclass(frozen=True)
class PreferenceScore:
    """Vraisemblances attribuées aux deux réponses d'une paire."""

    chosen: float
    rejected: float
    label_type: str

    @property
    def bien_ordonnee(self) -> bool:
        return self.chosen > self.rejected

    @property
    def marge(self) -> float:
        return self.chosen - self.rejected


def _log_vraisemblance(model, tokenizer, prompt: str, reponse: str, device: str) -> float:
    """Log-vraisemblance moyenne par jeton de `reponse`, sachant `prompt`.

    On normalise par la longueur : sans cela, la comparaison favoriserait
    mécaniquement la réponse la plus courte, et l'on mesurerait encore la
    longueur au lieu du fond.
    """
    import torch

    # Les bornes sont tirées de la fenêtre réelle du modèle. Au-delà, le passage
    # avant tronquerait les logits à la fenêtre pendant que les cibles
    # garderaient leur longueur d'origine, et la lecture des vraisemblances
    # échouerait sur un écart de dimension — les dissertations d'UltraMedical
    # dépassent régulièrement cette fenêtre.
    #
    # L'invite est bornée à la moitié de la fenêtre pour qu'il reste toujours de
    # la place à la réponse, qui est l'objet de la mesure. La borne s'applique au
    # **texte**, et non à la suite de jetons, pour que les jetons de l'invite
    # restent un préfixe exact de ceux de l'ensemble — c'est ce qui permet de
    # localiser le début de la réponse par un simple comptage.
    invite_bornee, _ = borner_la_description(prompt, tokenizer, MODEL.max_seq_length // 2)
    jetons_invite = tokenizer(invite_bornee, return_tensors="pt")
    jetons_complets = tokenizer(
        invite_bornee + reponse,
        return_tensors="pt",
        truncation=True,
        max_length=MODEL.max_seq_length,
    )
    entrees = jetons_complets["input_ids"].to(device)
    debut_reponse = jetons_invite["input_ids"].shape[1]
    if entrees.shape[1] <= debut_reponse:
        return float("-inf")

    with torch.no_grad():
        logits = model(entrees).logits
    # Le logit à la position i prédit le jeton i+1 : on décale d'un cran.
    log_probabilites = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    cibles = entrees[0, 1:]
    retenues = log_probabilites.gather(1, cibles.unsqueeze(1)).squeeze(1)
    reponse_seule = retenues[debut_reponse - 1 :]
    return float(reponse_seule.mean())


def compter_les_ecretees(tokenizer, pairs) -> int:
    """Nombre de réponses dont la fenêtre du modèle coupe la fin.

    Les dissertations d'UltraMedical dépassent régulièrement la fenêtre. La
    comparaison reste juste — la même borne s'applique à la réponse préférée et à
    la rejetée, et le score est normalisé par la longueur — mais elle porte alors
    sur le **début** de chaque réponse. Le rapport doit pouvoir le dire avec un
    chiffre plutôt que de le passer sous silence.
    """
    ecretees = 0
    for paire in pairs:
        invite, _ = borner_la_description(
            f"{paire.prompt}\n\n", tokenizer, MODEL.max_seq_length // 2
        )
        for reponse in (paire.chosen, paire.rejected):
            if len(tokenizer(invite + reponse)["input_ids"]) > MODEL.max_seq_length:
                ecretees += 1
    return ecretees


def score_pairs(model, tokenizer, pairs, device: str = "cuda") -> list[PreferenceScore]:
    """Attribue à chaque paire les vraisemblances des deux réponses."""
    ecretees = compter_les_ecretees(tokenizer, pairs)
    if ecretees:
        logger.info(
            "  %d réponses sur %d dépassent la fenêtre : la comparaison porte sur leur début.",
            ecretees,
            2 * len(pairs),
        )
    scores = []
    for index, paire in enumerate(pairs):
        # Les deux sauts de ligne ne sont pas cosmétiques. `_log_vraisemblance`
        # localise le début de la réponse en comptant les jetons de l'invite
        # seule, ce qui suppose que ces jetons forment un **préfixe exact** de
        # ceux de l'ensemble. Un découpage en sous-mots peut fusionner le dernier
        # caractère de l'invite avec le premier de la réponse et décaler la
        # frontière d'un jeton ; un séparateur qui forme son propre jeton l'en
        # empêche. Vérifié sur le corpus : aucune rupture de préfixe.
        invite = f"{paire.prompt}\n\n"
        scores.append(
            PreferenceScore(
                chosen=_log_vraisemblance(model, tokenizer, invite, paire.chosen, device),
                rejected=_log_vraisemblance(model, tokenizer, invite, paire.rejected, device),
                label_type=paire.label_type,
            )
        )
        if (index + 1) % 25 == 0:
            logger.info("  %d/%d paires évaluées", index + 1, len(pairs))
    return scores


def summarize(scores: list[PreferenceScore]) -> dict:
    """Agrège les scores : part de paires bien ordonnées et marge moyenne."""
    if not scores:
        return {"n": 0, "part_bien_ordonnees": 0.0, "marge_moyenne": 0.0}
    bien_ordonnees = sum(score.bien_ordonnee for score in scores)
    par_difficulte = {}
    for niveau in sorted({score.label_type for score in scores}):
        sous_ensemble = [s for s in scores if s.label_type == niveau]
        par_difficulte[niveau] = {
            "n": len(sous_ensemble),
            "part_bien_ordonnees": round(
                sum(s.bien_ordonnee for s in sous_ensemble) / len(sous_ensemble), 4
            ),
        }
    return {
        "n": len(scores),
        "part_bien_ordonnees": round(bien_ordonnees / len(scores), 4),
        "marge_moyenne": round(sum(score.marge for score in scores) / len(scores), 4),
        "par_difficulte": par_difficulte,
        # Le détail paire par paire, dans l'ordre du jeu. Deux modèles sont
        # évalués sur les **mêmes** paires : comparer leurs deux proportions
        # reviendrait à les traiter comme des mesures indépendantes, alors que
        # l'information est entièrement dans les paires où ils divergent.
        "bien_ordonnees": [score.bien_ordonnee for score in scores],
    }
