"""Ce que le jeu produit contient réellement, mesuré plutôt qu'affirmé.

Un dataset se décrit d'ordinaire par ses volumes. Trois autres mesures disent ce
que ces volumes valent, et elles conditionnent la lecture de tous les résultats
publiés ensuite :

- **la diversité des cibles.** Les vignettes sont dérivées d'un catalogue de
  présentations, et toutes celles qui viennent d'une même présentation partagent
  la même réponse attendue. Compter les réponses distinctes dit combien de
  décisions différentes le modèle a réellement vues ;
- **la séparabilité.** Un découpage aléatoire place des reformulations d'un même
  cas de part et d'autre de la frontière : ce qu'on mesure alors sur le jeu de
  test est la reconnaissance de cas déjà vus. Le même calcul, découpé par
  présentation d'origine, mesure le transfert à des cas nouveaux. L'écart entre
  les deux chiffres est la part de restitution contenue dans le premier ;
- **la fuite par les métadonnées.** Le générateur tire le délai d'installation et
  le profil de constantes de la présentation, donc du niveau. Un vote majoritaire
  sur ces deux seuls champs mesure ce que le niveau doit à la forme du gabarit
  plutôt qu'au contenu clinique.

Ces trois nombres sont écrits dans la carte du dataset et repris par le rapport.
Ils ne flattent pas le projet ; c'est précisément pourquoi ils sont calculés à
chaque construction plutôt que laissés à l'appréciation du lecteur.
"""

from __future__ import annotations

from collections import Counter, defaultdict


def _reponse_attendue(exemple) -> str:
    """Le texte que le modèle doit produire, quel que soit le format d'entrée.

    Les exemples circulent sous deux formes dans le projet : l'objet
    `TriageExample`, qui nomme ce champ `assistant_turn`, et la ligne JSONL
    relue depuis le disque, qui le nomme `completion`.
    """
    if isinstance(exemple, dict):
        return exemple.get("completion") or exemple.get("assistant_turn", "")
    return getattr(exemple, "assistant_turn", None) or getattr(exemple, "completion", "")


def _champ(exemple, nom: str, defaut=""):
    """Lit un champ, que l'exemple soit un objet ou une ligne JSONL."""
    if isinstance(exemple, dict):
        return exemple.get(nom, defaut)
    return getattr(exemple, nom, defaut)


def completions_distinctes(exemples: list) -> dict:
    """Compte les réponses attendues réellement distinctes, et leur répétition."""
    par_source: dict[str, list[str]] = defaultdict(list)
    for exemple in exemples:
        origine = "vignettes" if _champ(exemple, "source") == "vignette_clinique" else "corpus"
        par_source[origine].append(_reponse_attendue(exemple))

    detail = {}
    for origine, reponses in par_source.items():
        distinctes = len(set(reponses))
        detail[origine] = {
            "exemples": len(reponses),
            "reponses_distinctes": distinctes,
            "repetition_moyenne": round(len(reponses) / distinctes, 1) if distinctes else 0.0,
        }
    return {
        "total": len({_reponse_attendue(e) for e in exemples}),
        "par_origine": detail,
    }


def separabilite(exemples: list, decoupages: int = 5) -> dict:
    """Exactitude d'un classifieur simple, en découpage aléatoire puis groupé.

    Le classifieur n'a pas d'intérêt en soi : il sert de sonde. S'il atteint la
    perfection en découpage aléatoire et redescend nettement en découpage groupé
    par présentation, c'est que le jeu de test aléatoire contient surtout des
    reformulations de cas vus à l'entraînement.

    Renvoie un dictionnaire vide si scikit-learn n'est pas disponible : la mesure
    est un diagnostic, pas une étape dont dépend la construction du jeu.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
    except ImportError:
        return {}

    vignettes = [
        e
        for e in exemples
        if _champ(e, "source") == "vignette_clinique" and _champ(e, "presentation_id")
    ]
    if len(vignettes) < decoupages * 2:
        return {}
    textes = [_champ(e, "user_turn") for e in vignettes]
    niveaux = [_champ(e, "level") for e in vignettes]
    groupes = [_champ(e, "presentation_id") for e in vignettes]
    if len(set(groupes)) < decoupages:
        return {}

    sonde = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=2000)
    )
    try:
        aleatoire = cross_val_score(sonde, textes, niveaux, cv=decoupages).mean()
        groupee = cross_val_score(
            sonde, textes, niveaux, cv=GroupKFold(n_splits=decoupages), groups=groupes
        ).mean()
    except ValueError:
        # Un découpage groupé peut isoler un pli qui ne contient qu'un seul
        # niveau ; la sonde ne sait alors rien apprendre. C'est un diagnostic :
        # il renonce et le dit, il ne fait pas échouer la construction du jeu.
        return {}
    return {
        "vignettes": len(vignettes),
        "presentations": len(set(groupes)),
        "decoupage_aleatoire": round(float(aleatoire), 4),
        "decoupage_par_presentation": round(float(groupee), 4),
        "ecart": round(float(aleatoire - groupee), 4),
    }


def fuite_par_metadonnees(exemples: list, presentations) -> dict:
    """Part du niveau prédictible par le seul délai et le seul profil de constantes.

    Ces deux champs sont choisis par le générateur d'après la présentation, donc
    d'après le niveau. Un patient qui consulte depuis trois semaines avec des
    constantes normales relève effectivement d'une consultation différée : la
    corrélation est en partie clinique. Elle est néanmoins mesurable, et un
    modèle peut s'y appuyer au lieu de lire le motif.
    """
    fiches = {p.id: p for p in presentations}
    cellules: dict[tuple[str, str], Counter] = defaultdict(Counter)
    total = 0
    for exemple in exemples:
        fiche = fiches.get(_champ(exemple, "presentation_id"))
        if fiche is None:
            continue
        cellules[(fiche.delai, fiche.vitals_profile)][_champ(exemple, "level")] += 1
        total += 1
    if not total:
        return {}

    justes = sum(compte.most_common(1)[0][1] for compte in cellules.values())
    homogenes = sum(sum(c.values()) for c in cellules.values() if len(c) == 1)
    return {
        "vignettes": total,
        "cellules": len(cellules),
        "exactitude_du_vote_majoritaire": round(justes / total, 4),
        "part_en_cellule_homogene": round(homogenes / total, 4),
    }
