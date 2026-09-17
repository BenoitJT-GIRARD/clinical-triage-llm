"""Tests des métriques d'évaluation et des références."""

from __future__ import annotations

import pytest

from clinical_triage.evaluation.baselines import (
    always_critical,
    classical_classifier,
    explicit_rule,
    majority_class,
)
from clinical_triage.evaluation.metrics import (
    HORS_FORMAT,
    accuracy,
    clopper_pearson,
    confusion,
    difference_newcombe,
    format_compliance,
    interval_for_proportion,
    latency_summary,
    mcnemar_exact,
    overtriage_rate,
    per_class,
    subgroup_accuracy,
    summarize,
    undertriage_rate,
    wilson_interval,
)

VITALE = "URGENCE_VITALE"
MODEREE = "URGENCE_MODEREE"
DIFFEREE = "CONSULTATION_DIFFEREE"


def test_exactitude_parfaite():
    attendus = [VITALE, DIFFEREE]
    assert accuracy(attendus, attendus) == 1.0


def test_une_reponse_hors_format_compte_comme_fausse():
    assert accuracy([VITALE, MODEREE], [VITALE, None]) == 0.5
    assert format_compliance([VITALE, None, MODEREE]) == 2 / 3


def test_le_sous_triage_ne_compte_que_les_cas_urgents():
    taux, fautes, urgents = undertriage_rate([VITALE, VITALE, DIFFEREE], [DIFFEREE, VITALE, VITALE])
    assert (taux, fautes, urgents) == (0.5, 1, 2)


def test_une_reponse_hors_format_compte_comme_un_sous_triage():
    """Une réponse illisible ne déclenche aucune prise en charge."""
    taux, fautes, _ = undertriage_rate([VITALE], [None])
    assert (taux, fautes) == (1.0, 1)


def test_le_surclassement_n_est_pas_un_sous_triage():
    assert undertriage_rate([DIFFEREE], [VITALE])[0] == 0.0
    assert overtriage_rate([DIFFEREE], [VITALE]) == 1.0


def test_intervalle_de_wilson():
    borne_basse, borne_haute = wilson_interval(57, 60)
    assert borne_basse < 57 / 60 < borne_haute
    assert 0.85 < borne_basse < 0.95


def test_l_intervalle_se_resserre_quand_l_effectif_augmente():
    petit = wilson_interval(9, 10)
    grand = wilson_interval(900, 1000)
    assert (grand[1] - grand[0]) < (petit[1] - petit[0])


def test_la_matrice_de_confusion_isole_les_reponses_hors_format():
    """Ranger une réponse illisible dans une classe de triage fausserait la lecture."""
    matrice = confusion([VITALE, VITALE, DIFFEREE], [VITALE, None, DIFFEREE])
    assert HORS_FORMAT in matrice["colonnes"]
    assert matrice["matrice"][VITALE][HORS_FORMAT] == 1
    assert matrice["matrice"][VITALE][VITALE] == 1
    assert matrice["matrice"][DIFFEREE][DIFFEREE] == 1


def test_precision_rappel_par_niveau():
    resultats = per_class([VITALE, VITALE, MODEREE], [VITALE, MODEREE, MODEREE])
    assert resultats[VITALE]["rappel"] == 0.5
    assert resultats[VITALE]["precision"] == 1.0
    assert resultats[MODEREE]["effectif"] == 1


def test_resume_complet():
    attendus = [VITALE, MODEREE, DIFFEREE]
    resume = summarize(
        attendus, attendus, latences_ms=[100.0, 120.0, 110.0], arrets_propres=[True, True, False]
    )
    for cle in (
        "exactitude",
        "exactitude_ic95",
        "sous_triage",
        "respect_format",
        "par_niveau",
        "confusion",
        "latence",
    ):
        assert cle in resume
    assert resume["arrets_propres"] == round(2 / 3, 4)
    assert resume["latence"]["p50_ms"] == 110.0


def test_latence_sur_une_serie_vide():
    assert latency_summary([]) == {"moyenne_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}


def test_les_centiles_de_latence_sont_des_mesures_reelles():
    """Le 95ᵉ centile conditionne un critère de mise en service : il ne s'invente pas.

    Sur soixante mesures de 1 à 60 ms, le rang le plus proche désigne la 57ᵉ valeur
    — la plus petite sous laquelle tombent au moins 95 % des observations.
    """
    resume = latency_summary([float(i) for i in range(1, 61)])
    assert resume["p95_ms"] == 57.0
    assert resume["p50_ms"] == 30.0


def test_la_mediane_ne_moyenne_pas_le_couple_central():
    """Sur un effectif pair, on publie une latence observée, pas leur moyenne."""
    assert latency_summary([10.0, 20.0, 30.0, 40.0])["p50_ms"] == 20.0


def test_detail_par_sous_groupe():
    attendus = [VITALE, VITALE, DIFFEREE, DIFFEREE]
    predits = [VITALE, DIFFEREE, DIFFEREE, DIFFEREE]
    detail = subgroup_accuracy(attendus, predits, ["fr", "fr", "en", "en"])
    assert detail["fr"]["exactitude"] == 0.5
    assert detail["en"]["exactitude"] == 1.0
    assert detail["fr"]["sous_triage_detail"]["fautes"] == 1


def test_le_groupe_sans_nom_est_nomme():
    detail = subgroup_accuracy([VITALE], [VITALE], [""])
    assert "presentation_directe" in detail


# --- Références ---


def test_reference_classe_majoritaire():
    predictions = majority_class([DIFFEREE, DIFFEREE, VITALE], 3)
    assert predictions == [DIFFEREE] * 3


def test_reference_prudence_maximale_ne_sous_trie_jamais():
    attendus = [VITALE, MODEREE, DIFFEREE]
    predictions = always_critical(len(attendus))
    assert undertriage_rate(attendus, predictions)[0] == 0.0
    assert overtriage_rate(attendus, predictions) > 0.5


def test_reference_regle_explicite():
    descriptions = ["Douleur thoracique et sueurs.", "Rhume banal depuis deux jours."]
    assert explicit_rule(descriptions) == [VITALE, DIFFEREE]


def _jeu_separable(nombre: int) -> tuple[list[str], list[str]]:
    """Deux vocabulaires disjoints : un sac de mots doit y reussir sans peine."""
    textes = [f"douleur thoracique sueurs cas {i}" for i in range(nombre)]
    niveaux = [VITALE] * nombre
    textes += [f"rhume banal repos cas {i}" for i in range(nombre)]
    niveaux += [DIFFEREE] * nombre
    return textes, niveaux


def test_la_reference_classique_apprend_et_predit():
    pytest.importorskip("sklearn")
    textes, niveaux = _jeu_separable(20)
    predictions, trace = classical_classifier(
        textes, niveaux, textes, niveaux, ["douleur thoracique sueurs cas 99"]
    )
    assert predictions == [VITALE]
    # La référence n'est pas réentraînée sur entraînement + validation :
    # elle verrait plus d'exemples que le modèle auquel on la compare.
    assert trace["exemples_d_entrainement"] == len(textes)


def test_la_reference_classique_choisit_sa_configuration_sur_la_validation():
    """Le jeu d'évaluation ne doit jamais entrer dans le choix de la référence.

    Une référence réglée sur le jeu qui la juge n'en est plus une. La trace
    publie donc la configuration retenue et le score qui l'a fait retenir, tous
    deux issus du seul jeu de validation.
    """
    pytest.importorskip("sklearn")
    textes, niveaux = _jeu_separable(20)
    _, trace = classical_classifier(textes, niveaux, textes, niveaux, textes[:1])
    assert trace["configuration"] in trace["candidates"]
    assert trace["candidates"][trace["configuration"]] == max(trace["candidates"].values())
    assert 0.0 <= trace["exactitude_de_selection"] <= 1.0


# --- Comparaison de deux systèmes sur les mêmes cas ---


def test_sans_desaccord_rien_ne_departage():
    """Deux systèmes qui ne divergent nulle part ne se comparent pas."""
    assert mcnemar_exact(0, 0) == 1.0


@pytest.mark.parametrize(
    ("gagnees", "perdues", "attendu"),
    [(2, 0, 0.5), (5, 0, 0.0625), (10, 0, 0.001953125), (3, 3, 1.0)],
)
def test_le_test_de_mcnemar_lit_la_loi_binomiale(gagnees, perdues, attendu):
    """Sur une poignée de désaccords, l'approximation du khi-deux n'a pas de sens.

    Chaque désaccord tombe d'un côté ou de l'autre à pile ou face sous
    l'hypothèse nulle : la loi binomiale se lit directement.
    """
    assert mcnemar_exact(gagnees, perdues) == pytest.approx(attendu)


def test_le_test_est_bilateral_et_symetrique():
    assert mcnemar_exact(9, 1) == mcnemar_exact(1, 9)


def test_neuf_gains_contre_un_sont_significatifs_douze_contre_dix_non():
    """C'est toute la différence entre un écart et un écart démontré.

    Les deux paires de chiffres donnent pourtant un écart de proportions du même
    ordre : sans test apparié, on conclurait dans les deux cas.
    """
    assert mcnemar_exact(9, 1) <= 0.05
    assert mcnemar_exact(12, 10) > 0.05


def test_un_systeme_muet_obtient_le_meilleur_score_de_surclassement():
    """Les trois colonnes du tableau de résultats ne se lisent qu'ensemble.

    Un système qui ne produit rien d'exploitable n'entre dans aucun des deux taux
    de faute sur les cas non urgents : il affiche donc 0 % de surclassement — le
    meilleur score possible de cette colonne — alors qu'il est inutilisable. Le
    respect du format, publié juste à côté, est ce qui l'empêche de se lire comme
    une qualité, et le rapport le dit.
    """
    gold = [VITALE, VITALE, MODEREE, MODEREE, DIFFEREE, DIFFEREE]
    muet: list[str | None] = [None] * 6

    assert overtriage_rate(gold, muet) == 0.0
    assert undertriage_rate(gold, muet)[0] == 1.0
    resume = summarize(gold, muet)
    assert resume["exactitude"] == 0.0
    assert resume["respect_format"] == 0.0


# --- Estimateurs d'incertitude : le bon outil pour chaque grandeur ---


def test_sans_observation_l_intervalle_est_l_ignorance_totale():
    """Zéro mesure n'établit pas une proportion nulle, elle n'établit rien.

    La fonction rendait [0 ; 0] : un intervalle de largeur nulle tiré du vide,
    c'est-à-dire une certitude absolue — exactement la faute qu'un intervalle
    de confiance existe pour empêcher.
    """
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert clopper_pearson(0, 0) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("succes", "total", "attendu"),
    [
        (3, 3, (0.2924, 1.0)),
        (4, 4, (0.3976, 1.0)),
        (2, 40, (0.0061, 0.1692)),
        (0, 40, (0.0, 0.0881)),
        (51, 60, (0.7343, 0.929)),
    ],
)
def test_l_intervalle_exact_vaut_ses_valeurs_de_reference(succes, total, attendu):
    assert clopper_pearson(succes, total) == attendu


def test_l_intervalle_exact_concorde_avec_la_loi_beta():
    """Contrôle indépendant : la dichotomie doit retrouver le quantile de la loi bêta.

    L'implémentation cherche ses bornes sur la loi binomiale cumulée plutôt que
    d'ajouter une dépendance pour deux appels. Ce test vérifie que ce choix ne
    coûte pas de précision, en la comparant à la formule fermée.
    """
    stats = pytest.importorskip("scipy.stats")
    for succes, total in [(3, 3), (1, 7), (17, 40), (19, 20), (32, 32), (10, 10)]:
        basse = 0.0 if succes == 0 else stats.beta.ppf(0.025, succes, total - succes + 1)
        haute = 1.0 if succes == total else stats.beta.ppf(0.975, succes + 1, total - succes)
        obtenu = clopper_pearson(succes, total)
        assert abs(obtenu[0] - basse) < 1e-4, (succes, total)
        assert abs(obtenu[1] - haute) < 1e-4, (succes, total)


def test_l_exact_est_plus_prudent_que_wilson_sur_les_petits_effectifs():
    """C'est la raison d'être de Clopper-Pearson sur les sous-groupes minuscules."""
    exact_basse = clopper_pearson(3, 3)[0]
    wilson_basse = wilson_interval(3, 3)[0]
    assert exact_basse < wilson_basse
    assert round(wilson_basse - exact_basse, 2) >= 0.14


def test_aucun_taux_n_est_publie_sous_six_observations():
    """Sur trois cas, l'intervalle exact couvre 71 % de l'échelle.

    Une moustache pareille se lit « mesuré avec incertitude » quand la lecture
    juste est « non mesurable ». L'appelant doit écrire la fraction brute.
    """
    assert interval_for_proportion(3, 3) is None
    assert interval_for_proportion(4, 4) is None
    assert interval_for_proportion(5, 5) is None
    assert interval_for_proportion(6, 6) is not None


def test_la_politique_choisit_l_exact_puis_wilson():
    assert interval_for_proportion(19, 20) == clopper_pearson(19, 20)
    assert interval_for_proportion(51, 60) == wilson_interval(51, 60)


def test_l_ecart_se_lit_sur_son_intervalle_et_non_sur_le_recouvrement():
    """Le sophisme du recouvrement donne ici la conclusion inverse de la bonne.

    Sur les chiffres du projet — 465/500 contre 51/60 — les deux intervalles de
    Wilson se chevauchent largement, et pourtant l'intervalle de l'écart exclut
    zéro : l'écart de huit points est réel.
    """
    interne = wilson_interval(465, 500)
    clinique = wilson_interval(51, 60)
    assert interne[0] <= clinique[1], "les intervalles marginaux se recouvrent bien"

    ecart, basse, haute = difference_newcombe(465, 500, 51, 60)
    assert ecart == 0.08
    assert (basse, haute) == (0.0063, 0.1927)
    assert basse > 0, "l'intervalle de l'écart exclut zéro"


def test_un_ecart_nul_a_un_intervalle_qui_contient_zero():
    ecart, basse, haute = difference_newcombe(30, 60, 30, 60)
    assert ecart == 0.0
    assert basse < 0 < haute
