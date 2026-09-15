"""Figures du projet.

Chaque figure est produite à partir des fichiers de résultats, jamais de valeurs
recopiées : si une évaluation est relancée, les figures changent avec elle.

Quelques règles tenues partout, pour que les neuf figures forment un ensemble :

- **aucune estimation n'est tracée nue.** Une proportion mesurée sur quelques
  dizaines de cas sort avec son intervalle de confiance ; en dessous de six
  observations elle sort comme une fraction brute, parce qu'un intervalle y
  couvrirait les deux tiers de l'échelle et donnerait à une non-mesure
  l'apparence d'une mesure. Le choix de l'estimateur est pris une seule fois,
  dans `metrics.intervalle_de_proportion`, et se relit là ;
- **un comptage exhaustif ne porte pas de barre d'erreur.** Le dataset livré et
  les cellules des matrices de confusion sont dénombrés, pas estimés : leur
  ajouter une moustache inventerait de l'aléa. La figure le dit en toutes
  lettres, pour qu'on ne lise pas l'absence d'incertitude comme un oubli ;
- **chaque axe porte son libellé et son unité**, et l'effectif figure au pied de
  la figure. Un titre de panneau ne remplace ni l'un ni l'autre : une exactitude
  sur trois cas et une exactitude sur trente-deux ne se lisent pas de la même
  façon, et rien d'autre sur le dessin ne les distingue ;
- **les nombres sont écrits à la française**, virgule décimale comprise, sur les
  étiquettes comme sur les graduations ;
- **trois couleurs au maximum**, toujours dans le même ordre et toujours pour la
  même chose, et jamais la couleur seule pour porter une information : une
  hachure ou une étiquette la double ;
- **jamais deux échelles sur un même graphique.** Deux grandeurs de nature
  différente donnent deux panneaux côte à côte, pas deux axes verticaux ;
- **la grille s'efface.** Elle aide à lire, elle ne doit pas se voir.

Sur le choix de l'estimateur, une précision qui a son importance : l'usage
courant « moyenne ± erreur type » ne s'applique **pas** ici. Il décrit la
précision d'une moyenne de grandeurs continues ; or presque tout ce qui est
tracé dans ce rapport est une proportion binomiale sur de petits effectifs, où
l'approximation normale sort de l'intervalle [0, 1] — deux sous-triages sur
quarante cas urgents donnent une borne basse négative — et annonce une certitude
absolue dès que la proportion vaut 0 ou 1. Wilson est donc l'estimateur par
défaut, l'intervalle exact de Clopper-Pearson prend le relais sur les petits
effectifs et sur la métrique de sécurité, et les latences restent décrites par
leurs centiles, qui ne sont pas des moyennes.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from clinical_triage.evaluation.metrics import (
    EFFECTIF_MINIMUM_POUR_UN_TAUX,
    clopper_pearson,
    difference_newcombe,
    intervalle_de_proportion,
)

# Palette catégorielle, dans l'ordre. Les trois teintes ont été **mesurées**, pas
# choisies à l'œil, et voici ce que la mesure donne sur fond clair :
#
#   - écart minimal entre deux teintes voisines en vision normale : ΔE 27,6 ;
#   - le même écart en deutéranopie : ΔE 9,2, au-dessus du seuil de 8 ;
#   - contraste du vert sur le fond : 2,74 pour 1, **sous le seuil de 3 pour 1**.
#
# Ce dernier point n'est pas ignoré : il impose que la couleur ne porte jamais
# seule une information. C'est la règle tenue partout ici — chaque marque est
# doublée d'une valeur écrite, d'une légende et, pour les barres non mesurables,
# d'une hachure qui survit à l'impression en niveaux de gris.
BLEU, ORANGE, VERT = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = (BLEU, ORANGE, VERT)

# Encres : le texte ne porte jamais la couleur d'une série.
ENCRE = "#1a1a19"
ENCRE_SECONDAIRE = "#52514e"
GRILLE = "#dcdcda"

# Rampe séquentielle bleue, du plus clair au plus foncé, pour les matrices.
RAMPE_BLEUE = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]

# Largeur commune à toutes les figures, en pouces.
#
# Le rapport les réduit toutes à la même largeur de colonne : une figure large de
# 12,6 pouces y est donc rétrécie deux fois plus qu'une figure de 8, et la même
# déclaration de 9 points devient 4,4 points sur l'une et 6,9 sur l'autre — la
# première est illisible à l'impression. Une largeur unique rend le facteur de
# réduction identique, et les tailles de police ci-dessous sont choisies pour ce
# facteur-là : elles paraissent grandes à l'écran, et tombent juste sur le papier.
LARGEUR_FIGURE = 9.0

# Hachure des barres dont l'effectif interdit d'estimer un taux. La couleur ne
# suffirait pas : elle disparaît à l'impression en niveaux de gris.
HACHURE_NON_MESURABLE = "///"

NIVEAUX_COURTS = {
    "URGENCE_VITALE": "Vitale",
    "URGENCE_MODEREE": "Modérée",
    "CONSULTATION_DIFFEREE": "Différée",
    "HORS_FORMAT": "Hors format",
}

# Les trois niveaux que le dataset contient, énumérés plutôt que découpés dans
# le dictionnaire d'affichage ci-dessus : celui-ci porte en plus `HORS_FORMAT`,
# qui est une colonne légitime des matrices de confusion et n'a rien à faire
# dans un comptage du corpus. Une tranche positionnelle sur ce dictionnaire
# ferait dépendre l'exhaustivité du comptage de l'ordre d'écriture des clés.
NIVEAUX_DU_DATASET = ("URGENCE_VITALE", "URGENCE_MODEREE", "CONSULTATION_DIFFEREE")

# Abréviations des noms de systèmes sur les graduations. Les noms complets
# restent en légende et dans le texte : ce sont les graduations qui manquent de
# place, six systèmes ne tenant pas côte à côte sous un panneau de quatre pouces.
SYSTEMES_COURTS = {
    "Classe majoritaire": "Majoritaire",
    "Prudence maximale": "Prudence max.",
    "Règle explicite": "Règle",
    "Classifieur classique": "Classifieur\nclassique",
    "Qwen3-1.7B-Base": "Base",
    "SFT + LoRA": "SFT",
    "SFT + LoRA + DPO (adaptateur)": "SFT+DPO\n(adapt.)",
    "SFT + LoRA + DPO (fusionné)": "SFT+DPO\n(fusionné)",
    "SFT + LoRA + DPO": "SFT+DPO",
}


def _graduations_de_systemes(axe, noms: list[str], positions: list[float]) -> None:
    """Pose les noms de systèmes abrégés, inclinés et alignés à droite."""
    axe.set_xticks(positions, [SYSTEMES_COURTS.get(nom, nom) for nom in noms])
    axe.tick_params(axis="x", rotation=30)
    for etiquette in axe.get_xticklabels():
        etiquette.set_horizontalalignment("right")


# Noms d'affichage des corpus. Les identifiants techniques se ressemblent trop —
# `medmcqa`, `mediqal` et `medquad` diffèrent de deux lettres, en minuscules et
# inclinés sous une barre — pour qu'un lecteur rattache chaque barre à sa source.
SOURCES_COURTES = {
    # Ces vignettes sont engendrées : le libellé le dit, sous peine de faire
    # passer un corpus construit pour un corpus collecté.
    "vignette_clinique": "Catalogue (généré)",
    "mediqal": "MediQAl",
    "medquad": "MedQuAD",
    "medmcqa": "MedMCQA",
    "frenchmedmcqa": "FrenchMedMCQA",
}


def appliquer_style() -> None:
    """Règle matplotlib une fois pour toutes les figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "font.size": 12,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlecolor": ENCRE,
            "axes.labelcolor": ENCRE_SECONDAIRE,
            "axes.edgecolor": GRILLE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRILLE,
            "grid.linewidth": 0.6,
            "xtick.color": ENCRE_SECONDAIRE,
            "ytick.color": ENCRE_SECONDAIRE,
            "legend.frameon": False,
            "legend.fontsize": 11,
        }
    )


# --- Écriture des nombres, et mentions de méthode ---


def _fr(valeur: float, decimales: int = 2) -> str:
    """Écrit un nombre à la française : virgule décimale, milliers espacés."""
    texte = f"{valeur:,.{decimales}f}".replace(",", " ").replace(".", ",")
    return texte


VIRGULE_DECIMALE = FuncFormatter(lambda valeur, _: f"{valeur:g}".replace(".", ","))


def _graduations_francaises(*axes, abscisse: bool = False) -> None:
    """Pose la virgule décimale sur les graduations.

    L'abscisse est laissée de côté par défaut : elle porte le plus souvent des
    noms de systèmes ou de catégories. Les courbes d'entraînement, elles, la
    graduent en époques, où le point décimal anglais détonnerait.
    """
    for axe in axes:
        axe.yaxis.set_major_formatter(VIRGULE_DECIMALE)
        if abscisse:
            axe.xaxis.set_major_formatter(VIRGULE_DECIMALE)


def _pied_de_figure(figure, mentions: list[str]) -> None:
    """Écrit sous la figure l'effectif, l'estimateur et les réserves de lecture.

    Une figure quitte le rapport : elle part dans les
    carnets, dans une diapositive. Elle doit donc porter elle-même ce qu'il
    faut pour la lire — combien de cas, quelle incertitude, et ce que le
    protocole ne permet pas d'affirmer.

    Le pied est posé **sous** le cadre, six points plus bas, et c'est la boîte de
    rendu ajustée au contenu qui l'inclut à l'enregistrement. Écrit dans le cadre,
    il faudrait lui réserver une bande de hauteur devinée : trop courte, une
    mention qui gagne une ligne passe sous les intitulés d'axe ; trop longue, elle
    est prise sur le tracé.
    """
    # Le repli se calcule sur la largeur réelle de la figure : `wrap=True` de
    # matplotlib coupe sur la largeur de l'axe, pas sur celle du dessin, et
    # ferait déborder la dernière mention hors de l'image.
    largeur, hauteur = figure.get_size_inches()
    figure.text(
        0.006,
        -6 / (hauteur * 72),
        textwrap.fill("   ·   ".join(mentions), width=int(largeur * 17)),
        ha="left",
        va="top",
        fontsize=11,
        color=ENCRE_SECONDAIRE,
    )


def _compte(mesure: dict) -> tuple[int, int]:
    """Nombre de réussites et effectif d'une mesure d'exactitude.

    Les fichiers de résultats publient une proportion arrondie et l'effectif ;
    l'entier de réussites s'en déduit sans ambiguïté tant que l'effectif reste
    très inférieur à la précision de l'arrondi, ce qui est le cas partout ici.
    """
    effectif = int(mesure.get("n", 0))
    return round(mesure.get("exactitude", 0.0) * effectif), effectif


def _barre_de_proportion(
    axe,
    positions: list[float],
    mesures: list[dict],
    couleur: str,
    largeur: float,
    etiquette_serie: str | None = None,
) -> None:
    """Trace des barres de proportion avec l'incertitude que l'effectif autorise.

    Trois cas, et c'est tout l'objet de cette fonction :

    - effectif suffisant : barre pleine, moustache asymétrique de l'intervalle,
      valeur écrite au-dessus de la borne haute ;
    - effectif trop faible : barre hachurée, **aucune moustache**, et la fraction
      brute pour étiquette. Un intervalle y couvrirait les deux tiers de
      l'échelle, et se lirait « mesuré avec incertitude » là où la lecture juste
      est « non mesurable » ;
    - mesure absente : rien n'est tracé, et la marque « n.d. » prend la place.
      Une case vide dessinée comme une barre à zéro se lit comme un échec total,
      ce qui est une affirmation, alors que la donnée manque.
    """
    for position, mesure in zip(positions, mesures, strict=True):
        if not mesure:
            axe.text(
                position,
                0.02,
                "n.d.",
                ha="center",
                va="bottom",
                fontsize=10,
                style="italic",
                color=ENCRE_SECONDAIRE,
            )
            continue

        succes, effectif = _compte(mesure)
        valeur = mesure["exactitude"]
        intervalle = intervalle_de_proportion(succes, effectif)
        mesurable = intervalle is not None

        axe.bar(
            position,
            valeur,
            width=largeur,
            color=couleur,
            hatch=None if mesurable else HACHURE_NON_MESURABLE,
            edgecolor="white" if mesurable else couleur,
            linewidth=0 if mesurable else 0.8,
            alpha=1.0 if mesurable else 0.45,
            label=etiquette_serie,
        )
        etiquette_serie = None  # une seule entrée de légende par série

        if mesurable:
            basse, haute = intervalle
            axe.errorbar(
                position,
                valeur,
                yerr=[[valeur - basse], [haute - valeur]],
                fmt="none",
                ecolor=ENCRE_SECONDAIRE,
                capsize=3.5,
                linewidth=1.1,
            )
            axe.text(
                position,
                haute + 0.025,
                _fr(valeur, 2),
                ha="center",
                va="bottom",
                fontsize=10.5,
                color=ENCRE,
            )
        else:
            axe.text(
                position,
                valeur + 0.025,
                f"{succes}/{effectif}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                style="italic",
                color=ENCRE_SECONDAIRE,
            )


def _legende_des_series(axe, noms: list[str]) -> None:
    """Pose une légende dont les vignettes ne dépendent pas de la première barre.

    Matplotlib reprend le style de la première marque d'une série : quand
    celle-ci est une barre hachurée — un sous-groupe trop petit pour porter un
    taux — toute la série apparaît hachurée en légende, et la hachure cesse de
    signifier « non mesurable ».

    Elle est posée sur la figure, au-dessus du tracé : dans le tracé, elle
    recouvrirait les étiquettes de valeur des barres les plus hautes ; sur
    l'axe, elle se superposerait au titre du panneau.
    """
    from matplotlib.patches import Patch

    axe.get_figure().legend(
        handles=[
            Patch(facecolor=couleur, label=nom) for nom, couleur in zip(noms, SERIES, strict=False)
        ],
        loc="upper center",
        ncol=len(noms),
        frameon=False,
    )


# Les clés du jeu d'évaluation servent d'identifiant : sans accent, en minuscules
# et soudées par des soulignés. Une figure ne se lit pas dans cet alphabet-là.
NOMS_DE_NATURES = {
    "constantes_discordantes": "constantes discordantes",
    "faux_alarmant": "faux alarmant",
    "faux_rassurant": "faux rassurant",
    "negation": "négation",
    "presentation_directe": "présentation directe",
}


def _nom_de_nature(cle: str) -> str:
    """Intitulé lisible d'une nature de cas, à défaut la clé rendue présentable."""
    return NOMS_DE_NATURES.get(cle, cle.replace("_", " "))


def _mention_estimateur(effectifs: list[int]) -> str:
    """Nomme l'estimateur réellement employé, selon les effectifs rencontrés."""
    utiles = [n for n in effectifs if n]
    if not utiles:
        return "effectifs inconnus"
    morceaux = []
    if any(n > 30 for n in utiles):
        morceaux.append("Wilson")
    if any(EFFECTIF_MINIMUM_POUR_UN_TAUX <= n <= 30 for n in utiles):
        morceaux.append("Clopper-Pearson exact")
    mention = "intervalles de confiance à 95 % (" + ", ".join(morceaux) + ")" if morceaux else ""
    if any(n < EFFECTIF_MINIMUM_POUR_UN_TAUX for n in utiles):
        reserve = (
            f"sous {EFFECTIF_MINIMUM_POUR_UN_TAUX} cas, fraction brute et barre hachurée : "
            "effectif insuffisant pour estimer un taux"
        )
        return f"{mention} ; {reserve}" if mention else reserve
    return mention


def _etiqueter(axe, barres, format_valeur="{:.0f}", decalage=0.01) -> None:
    """Écrit la valeur au-dessus de chaque barre, à la française."""
    hauteur_max = max([barre.get_height() for barre in barres] or [0])
    for barre in barres:
        axe.text(
            barre.get_x() + barre.get_width() / 2,
            barre.get_height() + hauteur_max * decalage,
            format_valeur.format(barre.get_height()).replace(".", ",").replace(",", " ", 0),
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=ENCRE,
        )


# Hauteur laissée libre au-dessus de la donnée la plus haute d'un panneau : de
# quoi écrire l'étiquette de valeur sans qu'elle monte sur le titre. L'unité est
# celle de l'axe — une proportion sur un panneau, un écart de proportions sur
# l'autre — et la réserve vaut, sur les deux cadres, deux à trois fois la hauteur
# de l'étiquette qu'elle abrite.
MARGE_D_ETIQUETTE = 0.15


def _reserver_la_marge_haute(axe, sommets: list[float]) -> None:
    """Remonte la borne haute d'un panneau au-dessus de ses étiquettes de valeur.

    La marge est **additive**, et non proportionnelle au sommet : un panneau dont
    les valeurs peuvent être négatives — l'écart de la figure 8 le peut, un modèle
    qui transfère mieux qu'il ne restitue — verrait une marge proportionnelle
    poser la borne haute sous ses propres données.
    """
    if sommets:
        axe.set_ylim(top=max(sommets) + MARGE_D_ETIQUETTE)


def _enregistrer(figure, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # La boîte de rendu est ajustée au contenu réellement dessiné. Sans cela, un
    # intitulé d'axe vertical plus haut que son axe dépasse du dessin et se
    # retrouve coupé au bord de l'image : la mise en page, elle, ne sait pas
    # rétrécir un axe en dessous de sa hauteur minimale. C'est aussi ce qui donne
    # sa place au pied de figure, écrit sous le cadre.
    figure.savefig(destination, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return destination


# --- 1. Composition du dataset ---


def composition_dataset(statistiques: dict, destination: Path) -> Path:
    """Trois partitions du même corpus : par niveau, par langue, par source.

    Ce sont des **comptages exhaustifs** du jeu livré, pas des estimations :
    aucune barre d'erreur, et la figure le dit. Les trois panneaux partagent leur
    axe vertical, ce qui rend les trois partitions comparables d'un coup d'œil.
    """
    niveaux = statistiques["repartition_niveaux"]
    langues = statistiques["repartition_langues"]
    sources = statistiques["repartition_sources"]
    confiance = statistiques.get("repartition_confiance", {})

    # Indexation directe, sans valeur de repli : une métadonnée incomplète doit
    # faire échouer la construction de la figure, jamais produire une barre à
    # zéro. « Le corpus ne contient aucune urgence vitale » est une affirmation
    # clinique, et ce n'est pas ce qu'une clé manquante veut dire.
    valeurs_niveaux = [niveaux[niveau] for niveau in NIVEAUX_DU_DATASET]
    total = sum(valeurs_niveaux)

    figure, (gauche, milieu, droite) = plt.subplots(
        1, 3, figsize=(LARGEUR_FIGURE, 3.0), sharey=True
    )

    barres = gauche.bar(
        [NIVEAUX_COURTS[n] for n in NIVEAUX_DU_DATASET], valeurs_niveaux, color=BLEU, width=0.55
    )
    _etiqueter(gauche, barres)
    # Trois libellés de huit lettres ne tiennent pas côte à côte sous un panneau
    # de deux pouces et demi : ils s'inclinent.
    gauche.tick_params(axis="x", rotation=20)
    for etiquette in gauche.get_xticklabels():
        etiquette.set_horizontalalignment("right")
    gauche.set_title("Niveaux de triage")
    gauche.set_xlabel("Niveau (gravité décroissante)")
    gauche.set_ylabel("Nombre d'exemples")

    barres = milieu.bar(
        ["Français", "Anglais"], [langues["fr"], langues["en"]], color=VERT, width=0.55
    )
    _etiqueter(milieu, barres)
    milieu.set_title("Langues")
    milieu.set_xlabel("Langue de la description")

    noms = list(sources)
    barres = droite.bar(
        [SOURCES_COURTES.get(nom, nom) for nom in noms],
        [sources[nom] for nom in noms],
        color=ORANGE,
        width=0.55,
    )
    _etiqueter(droite, barres)
    droite.set_title("Sources")
    droite.set_xlabel("Corpus d'origine")
    # Inclinaison marquée et alignement à droite : à 12°, quatre noms de corpus
    # se chevauchent sur un panneau de trois pouces.
    droite.tick_params(axis="x", rotation=30)
    for etiquette in droite.get_xticklabels():
        etiquette.set_horizontalalignment("right")

    for axe in (gauche, milieu, droite):
        axe.grid(axis="x", visible=False)

    mentions = [
        f"n = {_fr(total, 0)} exemples",
        "comptage exhaustif : pas d'incertitude d'échantillonnage à représenter",
    ]
    if confiance:
        mentions.append(
            f"{_fr(confiance.get('haute', 0), 0)} étiquettes de confiance haute (catalogue "
            f"clinique) et {_fr(confiance.get('moyenne', 0), 0)} de confiance moyenne "
            "(règle appliquée aux corpus)"
        )
    _pied_de_figure(figure, mentions)
    figure.tight_layout()
    return _enregistrer(figure, destination)


# --- 2. Réglage des hyperparamètres ---


def reglage_hyperparametres(comparaison: dict, destination: Path) -> Path:
    """Exactitude de triage et perte de validation de chaque variante testée.

    Les quatre variantes sont évaluées sur **les mêmes** cas de validation, et la
    variante retenue est choisie sur ces mêmes cas : son exactitude est donc
    optimiste, et la figure le dit plutôt que de laisser lire un classement.
    """
    variantes = comparaison["variantes"]
    cas = comparaison.get("protocole", {}).get("cas_evalues", 0)
    noms = [v["variante"] for v in variantes]
    positions = list(range(len(noms)))

    figure, (gauche, droite) = plt.subplots(1, 2, figsize=(LARGEUR_FIGURE, 2.8))

    mesures = [{"exactitude": v["exactitude_triage"], "n": cas} for v in variantes]
    _barre_de_proportion(gauche, positions, mesures, BLEU, 0.55)
    gauche.set_xticks(positions, noms)
    gauche.set_title("Exactitude de triage")
    gauche.set_xlabel("Variante (rang LoRA · taux d'apprentissage)")
    gauche.set_ylabel("Exactitude (proportion)")
    gauche.set_ylim(0, 1.18)

    pertes = [v["eval_loss"] or 0 for v in variantes]
    barres = droite.bar(noms, pertes, color=ORANGE, width=0.55)
    _etiqueter(droite, barres, "{:.3f}")
    droite.set_title("Perte de validation")
    droite.set_xlabel("Variante (rang LoRA · taux d'apprentissage)")
    droite.set_ylabel("Entropie croisée")

    for axe in (gauche, droite):
        axe.tick_params(axis="x", rotation=12)
        axe.grid(axis="x", visible=False)
    _graduations_francaises(gauche, droite)

    _pied_de_figure(
        figure,
        [
            f"n = {cas} cas de validation, identiques pour les quatre variantes",
            _mention_estimateur([cas]),
            "comparaison appariée : des intervalles qui se recouvrent n'excluent pas un écart réel",
            "la variante retenue est choisie sur ces mêmes cas — exactitude optimiste",
        ],
    )
    figure.tight_layout()
    return _enregistrer(figure, destination)


# --- 3. Apprentissage supervisé ---


def apprentissage_sft(historique: list[dict], destination: Path) -> Path:
    """Perte d'entraînement et perte de validation au fil des époques."""
    entrainement = [(e["epoch"], e["loss"]) for e in historique if "loss" in e]
    validation = [(e["epoch"], e["eval_loss"]) for e in historique if "eval_loss" in e]

    figure, axe = plt.subplots(figsize=(LARGEUR_FIGURE, 2.8))
    axe.plot(*zip(*entrainement, strict=True), color=BLEU, linewidth=2, label="entraînement")
    if validation:
        # Points non reliés : deux évaluations ne décrivent pas une trajectoire
        # continue, et une ligne pleine invite à interpoler entre elles.
        axe.plot(
            *zip(*validation, strict=True),
            color=ORANGE,
            linewidth=0,
            marker="o",
            markersize=7,
            label="validation",
        )
        for epoque, perte in validation:
            axe.annotate(
                _fr(perte, 4),
                (epoque, perte),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=10.5,
                color=ENCRE,
            )
    axe.set_xlabel("Époque")
    axe.set_ylabel("Perte")
    axe.set_title("Fine-tuning supervisé : convergence")
    axe.legend(loc="upper right")
    _graduations_francaises(axe, abscisse=True)

    _pied_de_figure(
        figure,
        [
            "exécution unique (graine 42) : la dispersion entre exécutions n'est pas mesurée",
            "les points de validation portent sur le même jeu — leur comparaison est appariée",
            "entropie croisée sans unité, moyennée par jeton et non par exemple",
        ],
    )
    figure.tight_layout()
    return _enregistrer(figure, destination)


# --- 4. Alignement par préférences ---


def alignement_dpo(historique: list[dict], destination: Path, paires: int = 0) -> Path:
    """Perte d'alignement et préférences correctement ordonnées.

    Deux grandeurs sans rapport d'échelle : deux panneaux, jamais deux axes.
    """
    perte = [(e["epoch"], e["loss"]) for e in historique if "loss" in e]
    precision = [
        (e["epoch"], e["eval_rewards/accuracies"])
        for e in historique
        if "eval_rewards/accuracies" in e
    ]
    figure, (gauche, droite) = plt.subplots(1, 2, figsize=(LARGEUR_FIGURE, 3.2))
    if perte:
        gauche.plot(*zip(*perte, strict=True), color=BLEU, linewidth=2)
    gauche.set_title("Perte d'alignement")
    gauche.set_xlabel("Époque")
    gauche.set_ylabel("Perte DPO")

    if precision:
        droite.plot(
            *zip(*precision, strict=True),
            color=VERT,
            linewidth=0,
            marker="o",
            markersize=6,
        )
        # L'intervalle ne porte que sur le point publié. Une bande continue le
        # long de la courbe se lirait comme une garantie simultanée sur toute la
        # trajectoire, ce qu'une succession d'intervalles ponctuels n'est pas.
        epoque_finale, valeur_finale = precision[-1]
        intervalle = intervalle_de_proportion(round(valeur_finale * paires), paires)
        if intervalle:
            basse, haute = intervalle
            droite.errorbar(
                epoque_finale,
                valeur_finale,
                yerr=[[valeur_finale - basse], [haute - valeur_finale]],
                fmt="none",
                ecolor=ENCRE_SECONDAIRE,
                capsize=4,
                linewidth=1.1,
            )
        # Étiquette posée du côté où le point laisse de la place : au-dessus d'une
        # part proche de 1,00 — la valeur attendue ici — elle chevaucherait le
        # titre, et sous un point proche de zéro elle sortirait du cadre pour
        # tomber dans les graduations.
        droite.annotate(
            _fr(valeur_finale, 2),
            (epoque_finale, valeur_finale),
            textcoords="offset points",
            xytext=(-6, -18) if valeur_finale > 0.5 else (-6, 12),
            ha="right",
            fontsize=10.5,
            color=ENCRE,
        )
    droite.set_ylim(0, 1.15)
    droite.set_title("Validation : paires bien ordonnées")
    droite.set_xlabel("Époque")
    droite.set_ylabel("Part des paires")
    _graduations_francaises(gauche, droite, abscisse=True)

    mentions = ["exécution unique (graine 42) : dispersion entre exécutions non mesurée"]
    if paires:
        mentions.insert(0, f"n = {paires} paires de validation")
        mentions.append(_mention_estimateur([paires]) + ", sur le point final seulement")
    _pied_de_figure(figure, mentions)
    figure.tight_layout()
    return _enregistrer(figure, destination)


# --- 5. Comparaison des systèmes ---


def comparaison_systemes(mesures: dict[str, dict], destination: Path) -> Path:
    """Exactitude et sous-triage, chacun avec l'intervalle que son effectif autorise.

    Le sous-triage porte un intervalle **exact** de Clopper-Pearson, et non de
    Wilson : c'est la seule mesure du rapport dont une sous-estimation se paie en
    patients, et l'exact garantit sa couverture pour toute valeur du taux.
    """
    noms = list(mesures)
    positions = list(range(len(noms)))

    figure, (gauche, droite) = plt.subplots(1, 2, figsize=(LARGEUR_FIGURE, 3.6))

    _barre_de_proportion(gauche, positions, [mesures[n] for n in noms], BLEU, 0.55)
    _graduations_de_systemes(gauche, noms, positions)
    gauche.set_title("Exactitude sur le jeu clinique")
    gauche.set_xlabel("Système")
    gauche.set_ylabel("Exactitude (proportion)")
    gauche.set_ylim(0, 1.2)

    # Le sous-triage s'écrit en fraction : « 2/40 » transporte l'effectif avec la
    # valeur, là où « 0,05 » laisse croire à une résolution que la mesure n'a
    # pas — un seul cas vaut deux points et demi sur quarante.
    sommets: list[float] = []
    for position, nom in zip(positions, noms, strict=True):
        detail = mesures[nom].get("sous_triage_detail", {})
        fautes = detail.get("fautes", 0)
        urgents = detail.get("cas_urgents", 0)
        valeur = mesures[nom].get("sous_triage", 0.0)
        droite.bar(position, valeur, width=0.55, color=ORANGE)
        sommets.append(valeur)
        if urgents:
            basse, haute = clopper_pearson(fautes, urgents)
            sommets.append(haute)
            droite.errorbar(
                position,
                valeur,
                yerr=[[valeur - basse], [haute - valeur]],
                fmt="none",
                ecolor=ENCRE_SECONDAIRE,
                capsize=3.5,
                linewidth=1.1,
            )
            droite.text(
                position,
                haute + 0.012,
                f"{fautes}/{urgents}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                color=ENCRE,
            )
    # La fraction est écrite au-dessus de la borne haute de l'intervalle : sans
    # marge réservée, l'axe se cadre sur les seules données et l'étiquette du
    # système le plus haut monte sur le titre.
    _reserver_la_marge_haute(droite, sommets)
    _graduations_de_systemes(droite, noms, positions)
    droite.set_title("Sous-triage des cas urgents")
    droite.set_xlabel("Système")
    droite.set_ylabel("Sous-triage (proportion)")

    for axe in (gauche, droite):
        axe.grid(axis="x", visible=False)
    _graduations_francaises(gauche, droite)

    effectifs = [m.get("n", 0) for m in mesures.values()]
    urgents = next(
        (
            m["sous_triage_detail"]["cas_urgents"]
            for m in mesures.values()
            if m.get("sous_triage_detail")
        ),
        0,
    )
    _pied_de_figure(
        figure,
        [
            f"n = {max(effectifs, default=0)} cas, dont {urgents} urgents pour le sous-triage",
            f"exactitude : {_mention_estimateur(effectifs)}",
            "sous-triage : intervalle exact de Clopper-Pearson, à couverture garantie",
            "systèmes évalués sur les mêmes cas : leur comparaison est appariée",
        ],
    )
    figure.tight_layout()
    return _enregistrer(figure, destination)


# --- 6. Matrices de confusion ---


def matrices_confusion(confusions: dict[str, dict], destination: Path) -> Path:
    """Une matrice par système, avec une colonne pour les réponses hors format.

    Les panneaux partagent une échelle de couleur et une barre d'échelle : sans
    elles, deux cellules de même teinte dans deux panneaux voisins comptent des
    nombres différents, et la comparaison visuelle — la seule raison de les
    mettre côte à côte — induit en erreur.
    """
    from matplotlib.colors import LinearSegmentedColormap

    rampe = LinearSegmentedColormap.from_list("bleu", RAMPE_BLEUE)
    noms = list(confusions)
    # `constrained` plutôt que `tight_layout` : ce dernier ne sait pas tenir
    # compte d'une barre d'échelle attachée à plusieurs panneaux et la laisse
    # mordre sur le dernier. Les panneaux partagent leur axe vertical : répété,
    # il empiéterait sur les cellules du panneau voisin.
    figure, axes = plt.subplots(
        1, len(noms), figsize=(LARGEUR_FIGURE, 3.4), layout="constrained", sharey=True
    )
    if len(noms) == 1:
        axes = [axes]

    grilles = {}
    for nom in noms:
        donnees = confusions[nom]
        lignes, colonnes = donnees["lignes"], donnees["colonnes"]
        grilles[nom] = [
            [donnees["matrice"][ligne][colonne] for colonne in colonnes] for ligne in lignes
        ]
    maximum = max((max(rangee) for grille in grilles.values() for rangee in grille), default=0) or 1

    for axe, nom in zip(axes, noms, strict=True):
        donnees = confusions[nom]
        lignes, colonnes = donnees["lignes"], donnees["colonnes"]
        grille = grilles[nom]
        image = axe.imshow(grille, cmap=rampe, vmin=0, vmax=maximum)
        axe.set_xticks(
            range(len(colonnes)), [NIVEAUX_COURTS[c] for c in colonnes], rotation=30, ha="right"
        )
        axe.set_yticks(range(len(lignes)), [NIVEAUX_COURTS[ligne] for ligne in lignes])
        # Titre replié : un nom de système plus large que son panneau déborderait
        # sur l'échelle de couleur posée à droite de la figure.
        axe.set_title(textwrap.fill(nom, 18))
        axe.set_xlabel("Niveau prédit")
        axe.grid(visible=False)
        for i, rangee in enumerate(grille):
            for j, valeur in enumerate(rangee):
                axe.text(
                    j,
                    i,
                    str(valeur),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if valeur > maximum * 0.6 else ENCRE,
                )
    axes[0].set_ylabel("Niveau réel")

    barre = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.05)
    barre.set_label("Nombre de cas", color=ENCRE_SECONDAIRE, fontsize=8.5)
    barre.ax.tick_params(labelsize=8, color=ENCRE_SECONDAIRE)
    barre.outline.set_visible(False)

    total = sum(sum(rangee) for rangee in next(iter(grilles.values()), [[0]]))
    _pied_de_figure(
        figure,
        [
            f"n = {total} cas par matrice",
            "comptage exhaustif : pas d'incertitude d'échantillonnage",
            "effectifs de lignes fixés par le plan d'évaluation",
            "échelle de couleur commune aux panneaux",
        ],
    )
    return _enregistrer(figure, destination)


# --- 7. Performance selon le type de cas ---


def performance_par_piege(par_systeme: dict[str, dict], destination: Path) -> Path:
    """Exactitude par nature de cas, avec l'effectif de chaque sous-groupe.

    Ces sous-groupes vont de trois à trente-deux cas. Sans leur effectif écrit
    sous la barre et sans l'incertitude au-dessus, quatre barres de même hauteur
    se lisent comme quatre mesures de même valeur — alors que l'une repose sur
    trente-deux observations et l'autre sur trois.
    """
    categories = sorted({categorie for mesures in par_systeme.values() for categorie in mesures})
    noms = list(par_systeme)
    largeur = 0.8 / max(1, len(noms))

    figure, axe = plt.subplots(figsize=(LARGEUR_FIGURE, 3.6))
    effectifs: list[int] = []
    for index, (nom, couleur) in enumerate(zip(noms, SERIES, strict=False)):
        positions = [i + index * largeur - 0.4 + largeur / 2 for i in range(len(categories))]
        # `.get(categorie)` sans valeur de repli : une catégorie non mesurée pour
        # ce système doit rester vide, pas devenir une barre à zéro.
        mesures = [par_systeme[nom].get(categorie) for categorie in categories]
        effectifs += [m.get("n", 0) for m in mesures if m]
        _barre_de_proportion(axe, positions, mesures, couleur, largeur * 0.9, nom)

    tailles = {
        categorie: max(
            (par_systeme[nom][categorie]["n"] for nom in noms if categorie in par_systeme[nom]),
            default=0,
        )
        for categorie in categories
    }
    # Les intitulés sont repliés : côte à côte sur une seule ligne, « constantes
    # discordantes » et « faux alarmant » se toucheraient.
    axe.set_xticks(
        range(len(categories)),
        [f"{textwrap.fill(_nom_de_nature(c), 14)}\n(n = {tailles[c]})" for c in categories],
    )
    axe.set_ylim(0, 1.25)
    axe.set_xlabel("Nature du cas", labelpad=10)
    axe.set_ylabel("Exactitude (proportion)")
    axe.set_title("Exactitude selon la nature du cas")
    axe.grid(axis="x", visible=False)
    _legende_des_series(axe, noms)
    _graduations_francaises(axe)

    _pied_de_figure(
        figure,
        [
            f"effectifs par sous-groupe : {', '.join(f'{_nom_de_nature(c)} {tailles[c]}' for c in categories)}",
            _mention_estimateur(effectifs),
            "sous-groupes de composition différente : leurs exactitudes ne sont pas comparables entre elles",
        ],
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    return _enregistrer(figure, destination)


# --- 8. Généralisation ---


def generalisation(interne: dict[str, dict], clinique: dict[str, dict], destination: Path) -> Path:
    """Restitution sur le jeu interne, transfert sur le jeu clinique, et leur écart.

    Le troisième panneau est celui qui répond à la question posée. Lire les deux
    premiers en regardant si leurs intervalles se recouvrent est un sophisme, et
    il donne ici la conclusion inverse de la bonne : deux intervalles marginaux
    qui se chevauchent sont parfaitement compatibles avec un écart réel. L'écart
    s'estime directement, par la méthode de Newcombe.
    """
    noms = [nom for nom in clinique if nom in interne]
    positions = list(range(len(noms)))
    largeur = 0.36

    figure, (gauche, droite) = plt.subplots(
        1, 2, figsize=(LARGEUR_FIGURE, 3.9), gridspec_kw={"width_ratios": [1.45, 1]}
    )

    _barre_de_proportion(
        gauche,
        [p - largeur / 2 for p in positions],
        [interne[n] for n in noms],
        BLEU,
        largeur,
        "jeu de test interne (restitution)",
    )
    _barre_de_proportion(
        gauche,
        [p + largeur / 2 for p in positions],
        [clinique[n] for n in noms],
        ORANGE,
        largeur,
        "jeu clinique indépendant",
    )
    gauche.set_xticks(positions, noms)
    gauche.set_ylim(0, 1.2)
    gauche.set_xlabel("Système")
    gauche.set_ylabel("Exactitude (proportion)")
    gauche.set_title("Restitution et transfert")
    gauche.grid(axis="x", visible=False)
    # Vignettes construites à part — sinon elles reprennent le style de la
    # première barre — et posées sur la figure, au-dessus des deux panneaux :
    # accrochées à l'axe de gauche, elles couvriraient les deux titres.
    from matplotlib.patches import Patch

    figure.legend(
        handles=[
            Patch(facecolor=BLEU, label="jeu de test interne (restitution)"),
            Patch(facecolor=ORANGE, label="jeu clinique indépendant"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
    )

    ecarts, basses, hautes = [], [], []
    for nom in noms:
        succes_i, total_i = _compte(interne[nom])
        succes_c, total_c = _compte(clinique[nom])
        ecart, basse, haute = difference_newcombe(succes_i, total_i, succes_c, total_c)
        ecarts.append(ecart)
        basses.append(ecart - basse)
        hautes.append(haute - ecart)

    droite.axhline(0, color=ENCRE_SECONDAIRE, linewidth=1)
    droite.errorbar(
        positions,
        ecarts,
        yerr=[basses, hautes],
        fmt="o",
        markersize=7,
        color=VERT,
        ecolor=ENCRE_SECONDAIRE,
        capsize=4,
        linewidth=1.2,
    )
    for position, ecart, haut in zip(positions, ecarts, hautes, strict=True):
        droite.text(
            position,
            ecart + haut + 0.012,
            _fr(ecart, 2),
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=ENCRE,
        )
    # La borne haute est posée explicitement, comme sur le panneau de gauche :
    # cadré sur les seules données, l'axe laisserait les étiquettes de valeur
    # monter sur le titre. Le zéro entre dans le calcul parce que la ligne de
    # référence y est tracée, et que le panneau perdrait ce à quoi il compare si
    # elle sortait du cadre.
    sommets = [ecart + haut for ecart, haut in zip(ecarts, hautes, strict=True)]
    _reserver_la_marge_haute(droite, [*sommets, 0.0])
    droite.set_xticks(positions, noms)
    droite.set_xlabel("Système")
    droite.set_ylabel("Écart")
    droite.set_title("Écart et son intervalle")
    droite.grid(axis="x", visible=False)

    for axe in (gauche, droite):
        axe.tick_params(axis="x", rotation=12)
    _graduations_francaises(gauche, droite)

    _pied_de_figure(
        figure,
        [
            (
                f"n = {_compte(interne[noms[0]])[1] if noms else 0} cas internes, "
                f"{_compte(clinique[noms[0]])[1] if noms else 0} cas cliniques"
            ),
            (
                "écart = exactitude interne − exactitude clinique, en points de "
                "proportion ; intervalle de Newcombe, un intervalle qui exclut zéro "
                "établit l'écart"
            ),
            (
                "le jeu interne reprend les présentations vues à l'entraînement : il mesure "
                "la restitution, non la généralisation"
            ),
        ],
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    return _enregistrer(figure, destination)


# --- 9. Performance de l'endpoint ---


def latence_endpoint(mesures: dict[str, dict], destination: Path) -> Path:
    """Latence par niveau de charge, et débit correspondant.

    Les latences sont décrites par leurs centiles et non par une moyenne : une
    distribution de temps de réponse est asymétrique à queue lourde, et sa
    moyenne est tirée par la queue. Aucun intervalle n'est tracé sur le 95ᵉ
    centile — sur quelques dizaines de requêtes, il n'en existe pas qui soit
    honnête, et le pied de figure le dit plutôt que d'en inventer un.
    """
    charges = [nom.replace("concurrence_", "") for nom in mesures]
    p50 = [mesures[nom]["p50_ms"] for nom in mesures]
    p95 = [mesures[nom]["p95_ms"] for nom in mesures]
    debits = [mesures[nom]["debit_req_par_s"] for nom in mesures]
    requetes = [mesures[nom].get("requetes", 0) for nom in mesures]

    figure, (gauche, droite) = plt.subplots(1, 2, figsize=(LARGEUR_FIGURE, 2.9))
    positions = list(range(len(charges)))
    largeur = 0.36
    barres_p50 = gauche.bar(
        [p - largeur / 2 for p in positions], p50, width=largeur, color=BLEU, label="médiane"
    )
    barres_p95 = gauche.bar(
        [p + largeur / 2 for p in positions], p95, width=largeur, color=ORANGE, label="95ᵉ centile"
    )
    for barres in (barres_p50, barres_p95):
        _etiqueter(gauche, barres, "{:.0f}")
    gauche.set_xticks(positions, charges)
    gauche.set_xlabel("Requêtes simultanées")
    gauche.set_ylabel("Latence (ms)")
    gauche.set_title("Latence perçue")
    gauche.grid(axis="x", visible=False)
    # Dans le tracé, la légende recouvrirait les étiquettes du premier couple de
    # barres : elle passe au-dessus des deux panneaux.
    figure.legend(loc="upper center", ncol=2, frameon=False)

    barres = droite.bar(charges, debits, color=VERT, width=0.5)
    _etiqueter(droite, barres, "{:.1f}")
    droite.set_xlabel("Requêtes simultanées")
    droite.set_ylabel("Débit (req/s)")
    droite.set_title("Débit")
    droite.grid(axis="x", visible=False)
    _graduations_francaises(gauche, droite)

    _pied_de_figure(
        figure,
        [
            f"n = {max(requetes, default=0)} requêtes par niveau de charge, mesurées sur POST /triage",
            (
                "centiles par rang le plus proche ; aucun intervalle sur le 95ᵉ centile : "
                "l'effectif ne permet pas d'en établir un"
            ),
            "débit : fenêtre unique par niveau de charge, non répétée — aucune dispersion mesurable",
        ],
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    return _enregistrer(figure, destination)
