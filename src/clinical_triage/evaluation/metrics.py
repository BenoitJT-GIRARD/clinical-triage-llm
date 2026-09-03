"""Métriques d'évaluation du triage.

Quatre familles, dans l'ordre d'importance pour un service d'urgences :

1. **sécurité clinique** — le taux de sous-triage : proportion de cas réellement
   urgents que le modèle classe à un niveau de gravité inférieur. C'est la faute
   qui tue, et elle ne se lit pas dans l'exactitude globale ;
2. **performance** — exactitude et F1 par niveau, avec l'intervalle de confiance.
   Sur soixante cas, un écart de trois points n'est pas un écart : donner
   l'exactitude sans son intervalle laisse croire le contraire ;
3. **format** — proportion de réponses exploitables par le système d'information,
   et proportion de générations qui se terminent d'elles-mêmes ;
4. **latence** — moyenne, médiane et 95ᵉ centile.

La matrice de confusion comporte une colonne supplémentaire, `HORS_FORMAT`. Une
réponse illisible n'est pas une prédiction : la ranger dans une classe de triage
fausserait la lecture, dans un sens ou dans l'autre.
"""

from __future__ import annotations

from math import ceil, comb, sqrt

from clinical_triage.config import TRIAGE

HORS_FORMAT = "HORS_FORMAT"


def wilson_interval(succes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance à 95 % d'une proportion, méthode de Wilson.

    On préfère Wilson à l'approximation normale — ±SEM — parce qu'il reste
    correct sur de petits effectifs et près de 0 ou de 1, exactement le régime
    d'un jeu d'évaluation clinique de quelques dizaines de cas. L'approximation
    normale y produit des bornes hors de [0, 1] : sur 2 sous-triages pour 40 cas
    urgents, elle donne [-1,8 % ; 11,8 %], une borne négative sur un taux.

    Sans observation, l'intervalle est **[0 ; 1]** et non [0 ; 0] : zéro mesure
    n'établit pas une proportion nulle, elle n'établit rien. Un intervalle de
    largeur nulle affirmerait une certitude absolue tirée du vide — la faute même
    que cette fonction existe pour éviter.
    """
    if total == 0:
        return (0.0, 1.0)
    proportion = succes / total
    denominateur = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominateur
    demi_largeur = (
        z * sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominateur
    )
    return (round(max(0.0, centre - demi_largeur), 4), round(min(1.0, centre + demi_largeur), 4))


# Effectif en dessous duquel aucun taux n'est publiable.
#
# Sur trois cas, l'intervalle exact d'une réussite parfaite est [0,29 ; 1,00] :
# il couvre 71 % de l'échelle. Une barre surmontée d'une moustache pareille se
# lit « mesuré avec incertitude » alors que la lecture juste est « non
# mesurable ». En dessous de ce seuil, on publie la fraction brute.
EFFECTIF_MINIMUM_POUR_UN_TAUX = 6


def _binomiale_cumulee(k: int, n: int, p: float) -> float:
    """P(X <= k) pour X ~ Binomiale(n, p), calculée exactement.

    Somme directe des termes : `n` ne dépasse pas quelques centaines dans ce
    projet, et une somme exacte évite d'avoir à justifier une approximation.
    """
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(succes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalle **exact** d'une proportion, méthode de Clopper-Pearson.

    Wilson est le bon choix par défaut : plus court, et de couverture correcte en
    moyenne. Deux situations demandent mieux, et c'est ce que cette fonction
    apporte — une couverture garantie ≥ 95 % pour **toute** valeur de p, au prix
    d'un intervalle plus large :

    - **la métrique de sécurité.** Sur le sous-triage, une couverture qui descend
      sous 95 % pour certaines valeurs de p n'est pas acceptable : c'est la seule
      mesure du rapport dont une sous-estimation se paie en patients ;
    - **les tout petits effectifs.** Wilson y est nettement anticonservateur à la
      frontière : sur 3 réussites en 3 essais il annonce une borne basse de 0,44
      là où l'exact donne 0,29. Quinze points de fausse précision sur le
      sous-groupe le plus dangereux du jeu d'évaluation.

    Les bornes se cherchent par dichotomie sur la loi binomiale cumulée, ce qui
    évite d'introduire une dépendance à une bibliothèque de statistiques pour
    deux appels — et se relit sans connaître la fonction bêta incomplète.
    """
    if total == 0:
        return (0.0, 1.0)

    def _dichotomie(cible, croissante: bool) -> float:
        basse, haute = 0.0, 1.0
        for _ in range(60):
            milieu = (basse + haute) / 2
            if (cible(milieu) > 0) == croissante:
                haute = milieu
            else:
                basse = milieu
        return (basse + haute) / 2

    # Borne basse : le p pour lequel observer au moins `succes` succès n'a plus
    # que alpha/2 de chance. Borne haute : le p pour lequel en observer au plus
    # `succes` n'a plus que alpha/2 de chance.
    basse = (
        0.0
        if succes == 0
        else _dichotomie(lambda p: (1 - _binomiale_cumulee(succes - 1, total, p)) - alpha / 2, True)
    )
    haute = (
        1.0
        if succes == total
        else _dichotomie(lambda p: _binomiale_cumulee(succes, total, p) - alpha / 2, False)
    )
    return (round(basse, 4), round(haute, 4))


def difference_newcombe(
    succes_a: int, total_a: int, succes_b: int, total_b: int, z: float = 1.96
) -> tuple[float, float, float]:
    """Écart entre deux proportions **indépendantes**, avec son intervalle.

    Renvoie (écart, borne basse, borne haute). L'écart est `a - b`.

    Cette fonction existe parce que lire deux intervalles marginaux et regarder
    s'ils se recouvrent est un **sophisme**, et qu'il donne ici la conclusion
    inverse de la bonne. Le non-recouvrement prouve une différence ; le
    recouvrement ne prouve rien. Sur les chiffres du projet — 465/500 contre
    51/60 — les deux intervalles de Wilson se chevauchent largement, et pourtant
    l'intervalle de l'écart exclut zéro.

    Quand l'écart est la question posée, c'est l'écart qu'il faut estimer. La
    méthode est celle de Newcombe, qui compose les intervalles de Wilson des deux
    proportions : elle hérite de leur bon comportement près de 0 et de 1, là où
    l'approximation normale sur la différence échoue.

    Les deux échantillons doivent être **indépendants**. Deux systèmes évalués
    sur les mêmes cas ne le sont pas : c'est `mcnemar_exact` qu'il leur faut.
    """
    proportion_a = succes_a / total_a if total_a else 0.0
    proportion_b = succes_b / total_b if total_b else 0.0
    basse_a, haute_a = wilson_interval(succes_a, total_a, z)
    basse_b, haute_b = wilson_interval(succes_b, total_b, z)

    ecart = proportion_a - proportion_b
    vers_le_bas = sqrt((proportion_a - basse_a) ** 2 + (haute_b - proportion_b) ** 2)
    vers_le_haut = sqrt((haute_a - proportion_a) ** 2 + (proportion_b - basse_b) ** 2)
    return (round(ecart, 4), round(ecart - vers_le_bas, 4), round(ecart + vers_le_haut, 4))


def intervalle_de_proportion(succes: int, total: int) -> tuple[float, float] | None:
    """Intervalle à publier pour une proportion, ou `None` si l'effectif l'interdit.

    Une seule porte d'entrée, pour que le choix de l'estimateur soit pris au même
    endroit partout et puisse être relu :

    - moins de six observations : **rien**. L'intervalle exact y couvre les deux
      tiers de l'échelle ; le publier donnerait à une non-mesure l'apparence
      d'une mesure. L'appelant écrit la fraction brute ;
    - jusqu'à trente : **Clopper-Pearson**, exact, parce que Wilson est encore
      anticonservateur près des bornes sur ces effectifs ;
    - au-delà : **Wilson**, plus court et de couverture correcte.
    """
    if total < EFFECTIF_MINIMUM_POUR_UN_TAUX:
        return None
    if total <= 30:
        return clopper_pearson(succes, total)
    return wilson_interval(succes, total)


def mcnemar_exact(seulement_a: int, seulement_b: int) -> float:
    """Test de McNemar exact sur deux mesures **appariées**, bilatéral.

    Deux systèmes évalués sur les mêmes cas ne se comparent pas comme deux
    proportions indépendantes : ce qui porte l'information, ce sont les seuls cas
    où ils divergent. `seulement_a` compte ceux que le premier réussit et que le
    second manque, `seulement_b` l'inverse ; les accords n'entrent pas dans le
    calcul.

    Sous l'hypothèse qu'aucun des deux n'est meilleur, chaque désaccord tombe
    d'un côté ou de l'autre à pile ou face. On lit donc directement la loi
    binomiale, sans approximation : sur une poignée de désaccords, la correction
    de continuité du χ² n'a pas de sens.

    Renvoie 1.0 quand les deux systèmes ne divergent nulle part : il n'y a alors
    rien à départager.
    """
    discordants = seulement_a + seulement_b
    if discordants == 0:
        return 1.0
    plus_rare = min(seulement_a, seulement_b)
    queue = sum(comb(discordants, k) for k in range(plus_rare + 1)) / 2**discordants
    return min(1.0, 2 * queue)


def _as_label(prediction: str | None) -> str:
    """Ramène une prédiction à une classe, `HORS_FORMAT` compris."""
    return prediction if prediction in TRIAGE.levels else HORS_FORMAT


def format_compliance(preds: list[str | None]) -> float:
    """Proportion de réponses dont le niveau de priorité est extractible."""
    if not preds:
        return 0.0
    return sum(p in TRIAGE.levels for p in preds) / len(preds)


def accuracy(gold: list[str], preds: list[str | None]) -> float:
    """Exactitude du niveau de triage. Une réponse hors format compte comme fausse."""
    if not gold:
        return 0.0
    return sum(g == p for g, p in zip(gold, preds, strict=True)) / len(gold)


def undertriage_rate(gold: list[str], preds: list[str | None]) -> tuple[float, int, int]:
    """Taux de sous-triage, avec le détail (taux, nombre de fautes, cas concernés).

    Parmi les cas réellement urgents — vitaux ou modérés — proportion de ceux que
    le modèle classe à un niveau strictement moins grave. Une réponse hors format
    y compte comme un sous-triage : elle ne déclenche aucune prise en charge.

    Sur un cas **non urgent**, en revanche, une réponse illisible n'entre dans
    aucun des deux taux de faute : le sous-triage ne regarde pas ces cas-là, et
    le surclassement les écarte aussi. Elle n'apparaît alors que dans le respect
    du format et dans l'exactitude — c'est pourquoi le tableau du rapport publie
    ces colonnes côte à côte, un surclassement à 0 % pouvant sinon se lire comme
    une qualité alors qu'il décrit un système muet.
    """
    urgents = [(g, p) for g, p in zip(gold, preds, strict=True) if TRIAGE.severity[g] >= 1]
    if not urgents:
        return (0.0, 0, 0)
    fautes = 0
    for attendu, predit in urgents:
        rang_predit = TRIAGE.severity.get(predit, -1) if predit is not None else -1
        if rang_predit < TRIAGE.severity[attendu]:
            fautes += 1
    return (fautes / len(urgents), fautes, len(urgents))


def overtriage_rate(gold: list[str], preds: list[str | None]) -> float:
    """Taux de surclassement : cas classés plus graves qu'ils ne le sont.

    Le surclassement n'est pas dangereux pour le patient, mais il engorge le
    service : c'est le coût de la prudence, et il se mesure.

    Le dénominateur est **l'ensemble des cas**, pas seulement ceux qui peuvent
    être surclassés. C'est la convention des échelles de triage, et c'est aussi
    la seule qui rende le chiffre directement lisible par un chef de service :
    « sur cent patients présentés, tant sont envoyés plus haut qu'il ne fallait ».
    Une réponse hors format ne compte pas ici — elle est déjà comptée comme un
    sous-triage.
    """
    if not gold:
        return 0.0
    fautes = 0
    for attendu, predit in zip(gold, preds, strict=True):
        if predit in TRIAGE.levels and TRIAGE.severity[predit] > TRIAGE.severity[attendu]:
            fautes += 1
    return fautes / len(gold)


def per_class(gold: list[str], preds: list[str | None]) -> dict[str, dict[str, float]]:
    """Précision, rappel, F1 et effectif pour chaque niveau de triage.

    Une réserve à connaître avant de citer la **précision** : c'est une valeur
    prédictive positive, et elle dépend entièrement de la prévalence du jeu sur
    lequel on la mesure. Le jeu d'évaluation clinique est équilibré à dessein —
    vingt cas par niveau — alors qu'un service d'urgences voit une toute autre
    répartition. La précision publiée ici décrit donc ce plan d'évaluation, pas
    ce que l'infirmière d'accueil observerait : la transposer telle quelle à un
    service réel serait une faute de lecture. Le **rappel**, lui, ne dépend pas
    de la prévalence et se transpose ; c'est la raison pour laquelle le rapport
    publie la matrice de confusion et le sous-triage plutôt que ces trois
    nombres, qui restent dans le fichier de résultats pour l'analyse.
    """
    etiquettes = [_as_label(p) for p in preds]
    resultat: dict[str, dict[str, float]] = {}
    for niveau in TRIAGE.levels:
        vrais_positifs = sum(
            g == niveau and p == niveau for g, p in zip(gold, etiquettes, strict=True)
        )
        predits = sum(p == niveau for p in etiquettes)
        reels = sum(g == niveau for g in gold)
        precision = vrais_positifs / predits if predits else 0.0
        rappel = vrais_positifs / reels if reels else 0.0
        f1 = 2 * precision * rappel / (precision + rappel) if precision + rappel else 0.0
        resultat[niveau] = {
            "precision": round(precision, 4),
            "rappel": round(rappel, 4),
            "f1": round(f1, 4),
            "effectif": reels,
        }
    return resultat


def confusion(gold: list[str], preds: list[str | None]) -> dict:
    """Matrice de confusion : lignes = vérité, colonnes = prédiction et hors format.

    Renvoie trois clés : `lignes` et `colonnes` donnent l'ordre d'affichage, et
    `matrice[reel][predit]` donne l'effectif. Les en-têtes voyagent avec les
    chiffres parce que le résultat finit en JSON puis en figure : les relire
    depuis la configuration ferait dépendre un rapport archivé d'un ordre de
    niveaux susceptible de changer.
    """
    colonnes = [*TRIAGE.levels, HORS_FORMAT]
    matrice = {niveau: dict.fromkeys(colonnes, 0) for niveau in TRIAGE.levels}
    for attendu, predit in zip(gold, preds, strict=True):
        matrice[attendu][_as_label(predit)] += 1
    return {"lignes": list(TRIAGE.levels), "colonnes": colonnes, "matrice": matrice}


def _centile(triees: list[float], part: float) -> float:
    """Centile par rang le plus proche : la plus petite mesure sous laquelle
    tombe au moins `part` des observations.

    Cette convention n'interpole jamais entre deux mesures : le chiffre publié
    est toujours une latence réellement observée. C'est ce qu'on veut d'un
    rapport de performance, où une valeur inventée entre deux points de mesure
    serait invérifiable.
    """
    rang = ceil(part * len(triees))
    return triees[min(len(triees) - 1, max(0, rang - 1))]


def latency_summary(latences_ms: list[float]) -> dict[str, float]:
    """Moyenne, médiane et 95ᵉ centile des latences observées.

    La médiane est le centile 50 au sens ci-dessus : sur un effectif pair, elle
    vaut la mesure du bas du couple central plutôt que leur moyenne, pour la
    même raison — on ne publie que des valeurs mesurées.
    """
    if not latences_ms:
        return {"moyenne_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    triees = sorted(latences_ms)
    return {
        "moyenne_ms": round(sum(triees) / len(triees), 1),
        "p50_ms": round(_centile(triees, 0.50), 1),
        "p95_ms": round(_centile(triees, 0.95), 1),
    }


def summarize(
    gold: list[str],
    preds: list[str | None],
    latences_ms: list[float] | None = None,
    arrets_propres: list[bool] | None = None,
) -> dict:
    """Agrège toutes les métriques en un dictionnaire, intervalles compris."""
    n = len(gold)
    justes = sum(g == p for g, p in zip(gold, preds, strict=True))
    taux_sous_triage, fautes, urgents = undertriage_rate(gold, preds)
    resume = {
        "n": n,
        "exactitude": round(accuracy(gold, preds), 4),
        "exactitude_ic95": wilson_interval(justes, n),
        "sous_triage": round(taux_sous_triage, 4),
        # Clopper-Pearson et non Wilson : c'est la seule mesure du rapport dont
        # une sous-estimation se paie en patients, et l'exact garantit sa
        # couverture pour toute valeur du taux, là où Wilson ne la tient qu'en
        # moyenne. L'intervalle est un peu plus large ; c'est le prix juste.
        "sous_triage_ic95": clopper_pearson(fautes, urgents),
        "sous_triage_detail": {"fautes": fautes, "cas_urgents": urgents},
        "surclassement": round(overtriage_rate(gold, preds), 4),
        "respect_format": round(format_compliance(preds), 4),
        "par_niveau": per_class(gold, preds),
        "confusion": confusion(gold, preds),
    }
    if arrets_propres is not None:
        resume["arrets_propres"] = round(sum(arrets_propres) / max(1, len(arrets_propres)), 4)
    if latences_ms is not None:
        resume["latence"] = latency_summary(latences_ms)
    return resume


def subgroup_accuracy(
    gold: list[str], preds: list[str | None], groupes: list[str]
) -> dict[str, dict]:
    """Exactitude et sous-triage détaillés par sous-groupe (langue, type de piège...)."""
    resultat: dict[str, dict] = {}
    for nom in sorted(set(groupes)):
        indices = [i for i, g in enumerate(groupes) if g == nom]
        sous_gold = [gold[i] for i in indices]
        sous_preds = [preds[i] for i in indices]
        justes = sum(g == p for g, p in zip(sous_gold, sous_preds, strict=True))
        taux, fautes, urgents = undertriage_rate(sous_gold, sous_preds)
        resultat[nom or "presentation_directe"] = {
            "n": len(indices),
            "exactitude": round(accuracy(sous_gold, sous_preds), 4),
            "exactitude_ic95": intervalle_de_proportion(justes, len(indices)),
            "sous_triage": round(taux, 4),
            # Clopper-Pearson directement, sans passer par
            # `intervalle_de_proportion` : le sous-triage est la mesure de
            # sécurité, et c'est par sous-groupe que ses effectifs sont les plus
            # faibles. L'intervalle exact y est publié même sous l'effectif
            # minimum.
            "sous_triage_ic95": clopper_pearson(fautes, urgents),
            "sous_triage_detail": {"fautes": fautes, "cas_urgents": urgents},
        }
    return resultat
