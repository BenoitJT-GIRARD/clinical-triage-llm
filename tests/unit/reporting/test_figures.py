"""Tests de la production des figures.

Les figures sont des binaires : personne ne les relit pixel à pixel. Ces tests vérifient
qu'elles se construisent à partir de données de la forme attendue.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from clinical_triage.reporting import figures

STATISTIQUES = {
    "repartition_niveaux": {
        "URGENCE_VITALE": 1666,
        "URGENCE_MODEREE": 1666,
        "CONSULTATION_DIFFEREE": 1666,
    },
    "repartition_langues": {"fr": 2499, "en": 2499},
    "repartition_confiance": {"haute": 4188, "moyenne": 810},
    "repartition_sources": {"vignette_clinique": 4188, "medmcqa": 643, "medquad": 167},
}

MESURES = {
    "Règle explicite": {
        "n": 60,
        "exactitude": 0.58,
        "exactitude_ic95": [0.45, 0.70],
        "sous_triage": 0.42,
        "sous_triage_detail": {"fautes": 17, "cas_urgents": 40},
        "respect_format": 1.0,
        "surclassement": 0.05,
    },
    "SFT + LoRA + DPO": {
        "n": 60,
        "exactitude": 0.80,
        "exactitude_ic95": [0.68, 0.88],
        "sous_triage": 0.15,
        "sous_triage_detail": {"fautes": 6, "cas_urgents": 40},
        "respect_format": 1.0,
        "surclassement": 0.12,
    },
}


@pytest.fixture(autouse=True)
def style():
    figures.appliquer_style()


def test_figure_composition_dataset(tmp_path):
    chemin = figures.composition_dataset(STATISTIQUES, tmp_path / "composition.png")
    assert chemin.exists()
    assert chemin.read_bytes().startswith(b"\x89PNG")


def test_figure_comparaison_systemes(tmp_path):
    chemin = figures.comparaison_systemes(MESURES, tmp_path / "comparaison.png")
    assert chemin.stat().st_size > 5000


def test_figure_matrices_confusion(tmp_path):
    confusion = {
        "lignes": ["URGENCE_VITALE", "URGENCE_MODEREE", "CONSULTATION_DIFFEREE"],
        "colonnes": [
            "URGENCE_VITALE",
            "URGENCE_MODEREE",
            "CONSULTATION_DIFFEREE",
            "HORS_FORMAT",
        ],
        "matrice": {
            "URGENCE_VITALE": {
                "URGENCE_VITALE": 15,
                "URGENCE_MODEREE": 3,
                "CONSULTATION_DIFFEREE": 1,
                "HORS_FORMAT": 1,
            },
            "URGENCE_MODEREE": {
                "URGENCE_VITALE": 2,
                "URGENCE_MODEREE": 16,
                "CONSULTATION_DIFFEREE": 2,
                "HORS_FORMAT": 0,
            },
            "CONSULTATION_DIFFEREE": {
                "URGENCE_VITALE": 1,
                "URGENCE_MODEREE": 2,
                "CONSULTATION_DIFFEREE": 17,
                "HORS_FORMAT": 0,
            },
        },
    }
    chemin = figures.matrices_confusion({"SFT + LoRA + DPO": confusion}, tmp_path / "confusion.png")
    assert chemin.exists()


def test_figures_de_courbes(tmp_path):
    historique_sft = [
        {"epoch": 0.5, "loss": 1.2},
        {"epoch": 1.0, "loss": 0.8, "eval_loss": 0.9},
        {"epoch": 2.0, "loss": 0.5, "eval_loss": 0.6},
    ]
    assert figures.apprentissage_sft(historique_sft, tmp_path / "sft.png").exists()

    historique_dpo = [
        {"epoch": 0.5, "loss": 0.6},
        {"epoch": 1.0, "loss": 0.3, "eval_rewards/accuracies": 0.92, "eval_rewards/margins": 2.1},
    ]
    assert figures.alignement_dpo(historique_dpo, tmp_path / "dpo.png").exists()


def test_figures_de_synthese(tmp_path):
    comparaison = {
        "variantes": [
            {
                "variante": "r16_lr2e-4",
                "exactitude_triage": 0.82,
                "eval_loss": 0.41,
                "parametres_entrainables": 17432576,
                "duree_s": 500.0,
                "memoire_gpu_max_go": 9.3,
            },
            {
                "variante": "r8_lr2e-4",
                "exactitude_triage": 0.78,
                "eval_loss": 0.45,
                "parametres_entrainables": 8716288,
                "duree_s": 480.0,
                "memoire_gpu_max_go": 8.9,
            },
        ]
    }
    assert figures.reglage_hyperparametres(comparaison, tmp_path / "reglage.png").exists()

    par_piege = {
        "SFT + LoRA + DPO": {
            "presentation_directe": {"exactitude": 0.88, "n": 32},
            "faux_rassurant": {"exactitude": 0.60, "n": 10},
        }
    }
    assert figures.performance_par_piege(par_piege, tmp_path / "pieges.png").exists()
    assert figures.generalisation(
        {"SFT + LoRA + DPO": {"exactitude": 0.95, "n": 500}},
        {"SFT + LoRA + DPO": {"exactitude": 0.80, "n": 60}},
        tmp_path / "gener.png",
    ).exists()

    banc = {
        "concurrence_1": {
            "p50_ms": 420.0,
            "p95_ms": 610.0,
            "debit_req_par_s": 2.3,
            "tokens_par_s": 180.0,
        },
        "concurrence_4": {
            "p50_ms": 700.0,
            "p95_ms": 980.0,
            "debit_req_par_s": 6.1,
            "tokens_par_s": 470.0,
        },
    }
    assert figures.latence_endpoint(banc, tmp_path / "latence.png").exists()


# --- Ce que les figures ont le droit d'affirmer ---


def test_un_sous_groupe_trop_petit_ne_porte_pas_de_moustache(tmp_path):
    """Trois cas ne permettent pas d'estimer un taux.

    L'intervalle exact d'une réussite parfaite sur trois cas est [0,29 ; 1,00] :
    il couvre 71 % de l'échelle. Une barre surmontée d'une moustache pareille se
    lit « mesuré avec incertitude » quand la lecture juste est « non mesurable ».
    La figure trace alors une barre hachurée et la fraction brute.
    """
    from matplotlib.container import ErrorbarContainer

    par_piege = {
        "modèle": {
            "constantes_discordantes": {"exactitude": 1.0, "n": 3},
            "presentation_directe": {"exactitude": 0.94, "n": 32},
        }
    }
    figures.appliquer_style()
    figure, axe = plt.subplots()
    figures._barre_de_proportion(
        axe,
        [0, 1],
        [
            par_piege["modèle"]["constantes_discordantes"],
            par_piege["modèle"]["presentation_directe"],
        ],
        figures.BLEU,
        0.5,
    )
    moustaches = [c for c in axe.containers if isinstance(c, ErrorbarContainer)]
    assert len(moustaches) == 1, "une seule des deux barres est mesurable"

    hachurees = [barre for barre in axe.patches if barre.get_hatch()]
    assert len(hachurees) == 1
    textes = [t.get_text() for t in axe.texts]
    assert "3/3" in textes, "la fraction brute remplace le taux"
    plt.close(figure)


def test_une_mesure_absente_n_est_pas_dessinee_comme_un_zero(tmp_path):
    """Une catégorie non mesurée était tracée comme une barre à zéro étiquetée « 0.00 ».

    Indiscernable d'un système qui se serait trompé sur tous les cas du
    sous-groupe : une absence présentée comme un échec mesuré.
    """
    figures.appliquer_style()
    figure, axe = plt.subplots()
    figures._barre_de_proportion(
        axe, [0, 1], [None, {"exactitude": 0.9, "n": 20}], figures.BLEU, 0.5
    )
    assert len(axe.patches) == 1, "rien n'est tracé pour la mesure absente"
    assert "n.d." in [t.get_text() for t in axe.texts]
    plt.close(figure)


def test_les_figures_portent_leur_effectif_et_leur_estimateur(tmp_path):
    """Une figure quitte le rapport : elle doit se lire seule."""
    chemin = figures.comparaison_systemes(MESURES, tmp_path / "comparaison.png")
    assert chemin.exists()

    figure, _ = plt.subplots()
    figures._pied_de_figure(figure, ["n = 60 cas", "Wilson"])
    assert any("n = 60" in t.get_text() for t in figure.texts)
    plt.close(figure)


def test_les_nombres_sont_ecrits_a_la_francaise():
    assert figures._fr(0.917, 2) == "0,92"
    assert figures._fr(5000, 0) == "5 000"
    assert figures._fr(1.5, 1) == "1,5"
