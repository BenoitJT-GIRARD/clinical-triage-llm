"""Références auxquelles comparer le modèle.

Un chiffre d'exactitude ne veut rien dire seul. Quatre références encadrent le
résultat du modèle et disent s'il apporte quelque chose :

- **la classe majoritaire** : que donnerait un système qui répond toujours la
  même chose ? Sur un jeu déséquilibré, cette référence est étonnamment haute,
  et beaucoup de résultats publiés ne la dépassent que de peu ;
- **la prudence maximale** : que donnerait un système qui classe tout en urgence
  vitale ? Il ne sous-trie jamais — son taux de sous-triage est nul — mais il
  envoie tout le monde au déchocage. Cette référence rappelle qu'on ne juge pas
  la sécurité sans regarder le coût de la prudence ;
- **la règle explicite** : que donnerait le système qu'un service peut déployer
  en un après-midi, sans modèle de langage ?
- **un classifieur classique** : que donnerait un sac de n-grammes entraîné sur
  les mêmes paires que le modèle ? C'est la référence décisive. Battre une règle
  à mots-clés ne dit rien de l'intérêt du fine-tuning, puisqu'une régression
  logistique entraînée en une seconde et demie la bat aussi. Sans cette
  comparaison, l'évaluation ne peut pas répondre à la question qu'elle pose.

Le classifieur classique ne produit qu'un niveau. Le contrat de sortie du service
en demande trois champs — niveau, justification, recommandation — dont il ne
fabrique aucun des deux derniers. La comparaison porte donc sur la seule
dimension où une référence existe, et le rapport doit le dire.
"""

from __future__ import annotations

from collections import Counter

from clinical_triage.data.triage_rules import classify
from clinical_triage.utils import get_logger

logger = get_logger(__name__)


def majority_class(train_levels: list[str], nombre_de_cas: int) -> list[str]:
    """Prédit toujours le niveau le plus fréquent du jeu d'entraînement."""
    plus_frequent = Counter(train_levels).most_common(1)[0][0]
    return [plus_frequent] * nombre_de_cas


def always_critical(nombre_de_cas: int) -> list[str]:
    """Prédit toujours l'urgence vitale : prudence maximale, engorgement maximal."""
    return ["URGENCE_VITALE"] * nombre_de_cas


def explicit_rule(descriptions: list[str]) -> list[str]:
    """Applique la règle de triage explicite, constantes vitales comprises."""
    return [classify(description) for description in descriptions]


# Configurations mises en concurrence. Le choix se fait sur le jeu de validation,
# jamais sur le jeu d'évaluation : une référence réglée sur le jeu qui la juge
# n'est plus une référence, c'est un ajustement a posteriori.
CONFIGURATIONS_CLASSIQUES = (
    ("mots_unigrammes", "word", (1, 1)),
    ("mots_bigrammes", "word", (1, 2)),
    ("caracteres_3_5", "char_wb", (3, 5)),
)


def _fabriquer(analyseur: str, ngram: tuple[int, int]):
    """Vectorisation en n-grammes pondérés puis séparateur linéaire."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    return make_pipeline(
        TfidfVectorizer(analyzer=analyseur, ngram_range=ngram, min_df=2),
        LinearSVC(),
    )


def classical_classifier(
    train_textes: list[str],
    train_niveaux: list[str],
    validation_textes: list[str],
    validation_niveaux: list[str],
    descriptions: list[str],
) -> tuple[list[str], dict]:
    """Entraîne un classifieur classique sur les mêmes données que le modèle.

    Protocole, arrêté avant de regarder le moindre résultat :

    1. les configurations candidates sont entraînées sur le **jeu
       d'entraînement**, celui-là même que voit le modèle ;
    2. celle qui obtient la meilleure exactitude **sur le jeu de validation** est
       retenue — le jeu d'évaluation clinique n'est jamais consulté ;
    3. elle prédit une fois, sur les descriptions demandées.

    La référence n'est pas réentraînée sur entraînement + validation, bien que ce
    soit l'usage : elle verrait 500 exemples de plus que le modèle, et la
    comparaison porterait sur deux volumes de données différents.

    Renvoie les prédictions et la trace du choix, pour que le rapport publie la
    configuration retenue et son score de sélection plutôt qu'un chiffre nu.
    """
    resultats = []
    for nom, analyseur, ngram in CONFIGURATIONS_CLASSIQUES:
        modele = _fabriquer(analyseur, ngram)
        modele.fit(train_textes, train_niveaux)
        predits = modele.predict(validation_textes)
        exactitude = sum(p == v for p, v in zip(predits, validation_niveaux, strict=True)) / len(
            validation_niveaux
        )
        resultats.append((exactitude, nom, analyseur, ngram))
    resultats.sort(reverse=True, key=lambda r: r[0])
    exactitude_de_selection, nom, analyseur, ngram = resultats[0]
    logger.info(
        "  référence classique : %s retenue (validation %.4f) parmi %s",
        nom,
        exactitude_de_selection,
        ", ".join(f"{n}={e:.4f}" for e, n, _, _ in resultats),
    )

    retenu = _fabriquer(analyseur, ngram)
    retenu.fit(train_textes, train_niveaux)
    trace = {
        "configuration": nom,
        "exactitude_de_selection": round(exactitude_de_selection, 4),
        "candidates": {n: round(e, 4) for e, n, _, _ in resultats},
        "exemples_d_entrainement": len(train_textes),
    }
    return list(retenu.predict(descriptions)), trace
