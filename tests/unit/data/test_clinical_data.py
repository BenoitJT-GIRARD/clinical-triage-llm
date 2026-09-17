"""Tests de la construction du corpus de triage.

On vérifie ici ce que le cahier des charges exige et que la documentation ne peut
pas garantir seule : l'équilibre du corpus, l'absence de fuite entre les jeux,
la présence des métadonnées cliniques, et le fait que le jeu d'évaluation ne
sorte jamais du côté de l'entraînement.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import replace

import pytest

from clinical_triage.config import TRIAGE
from clinical_triage.data.case_generator import build_case, generate_cases
from clinical_triage.data.clinical_catalogue import (
    CONSTANTES_IMPOSEES,
    PRESENTATIONS,
    forced_vitals,
    presentation_by_id,
)
from clinical_triage.data.clinical_eval_set import eval_cases, eval_user_turns
from clinical_triage.data.corpus_cases import build_corpus_case, extract_vignette
from clinical_triage.data.corpus_sources import CorpusEntry
from clinical_triage.data.dataset_io import (
    check_no_leakage,
    metadata_schema,
    sft_record,
    split_train_val_test,
)
from clinical_triage.data.dpo_builder import build_preference_pairs
from clinical_triage.data.sft_builder import assemble

# --- Catalogue clinique ---


def test_le_catalogue_couvre_les_trois_niveaux():
    par_niveau = Counter(p.level for p in PRESENTATIONS)
    assert set(par_niveau) == set(TRIAGE.levels)
    assert min(par_niveau.values()) >= 15


def test_les_identifiants_du_catalogue_sont_uniques():
    identifiants = [p.id for p in PRESENTATIONS]
    assert len(identifiants) == len(set(identifiants))


def test_chaque_presentation_est_complete():
    for presentation in PRESENTATIONS:
        assert presentation.level in TRIAGE.levels
        assert presentation.vitals_profile in ("critique", "intermediaire", "normal")
        assert presentation.age_range[0] <= presentation.age_range[1]
        assert presentation.complaint_fr and presentation.complaint_en
        assert len(presentation.signs_fr) >= 2 and len(presentation.signs_en) >= 2
        assert len(presentation.history_fr) >= 2 and len(presentation.history_en) >= 2
        assert presentation.onset in ("minutes", "heures", "jours", "semaines")
        assert len(presentation.justification) > 40
        assert len(presentation.recommendation) > 40


def test_recherche_par_identifiant():
    assert presentation_by_id("syndrome_coronarien_aigu").level == "URGENCE_VITALE"
    with pytest.raises(KeyError):
        presentation_by_id("presentation_inexistante")


# --- Génération de vignettes ---


def test_les_vignettes_generees_portent_les_metadonnees_cliniques():
    cas = generate_cases(12, "URGENCE_VITALE", "fr", random.Random(1))
    assert len(cas) == 12
    for vignette in cas:
        assert vignette.level == "URGENCE_VITALE"
        assert vignette.lang == "fr"
        assert vignette.symptomes
        assert vignette.antecedents
        assert vignette.confiance == "haute"
        assert vignette.source == "vignette_clinique"


def test_les_vignettes_generees_sont_uniques():
    cas = generate_cases(120, "URGENCE_MODEREE", "fr", random.Random(2))
    assert len({c.user_turn for c in cas}) == len(cas)


def test_la_generation_respecte_les_exclusions():
    """Les tours réservés à l'évaluation ne doivent jamais être régénérés."""
    exclus = eval_user_turns()
    cas = generate_cases(60, "CONSULTATION_DIFFEREE", "fr", random.Random(3), exclude=exclus)
    assert not {c.user_turn for c in cas} & exclus


def test_une_part_des_vignettes_est_produite_sans_constantes():
    cas = generate_cases(200, "URGENCE_MODEREE", "en", random.Random(4))
    sans_constantes = sum(1 for c in cas if c.constantes is None)
    assert 0 < sans_constantes < len(cas)


# --- Extraction depuis les corpus ---


def test_une_question_d_examen_est_reduite_a_la_presentation():
    question = (
        "A 60 yr old chronic smoker presents with painless gross hematuria of 1 day duration. "
        "Investigation of choice to know the cause of hematuria"
    )
    vignette = extract_vignette(question)
    assert vignette is not None
    assert "Investigation of choice" not in vignette
    assert "chronic smoker" in vignette


def test_une_question_sans_patient_est_ecartee():
    assert extract_vignette("Levamisole is used as all except -") is None


def test_un_cas_de_corpus_sans_signe_detecte_est_ecarte():
    """On ne fabrique pas une étiquette « non urgent » à partir du silence de la règle."""
    entree = CorpusEntry(text="", answer="", lang="en", source="medmcqa", topic="")
    description = "A 40-year-old man presents for a routine administrative certificate request."
    assert build_corpus_case(description, entree, random.Random(5)) is None


def test_un_cas_de_corpus_avec_signe_est_conserve_avec_une_confiance_moyenne():
    entree = CorpusEntry(text="", answer="", lang="en", source="medmcqa", topic="")
    description = "A 55-year-old man presents with chest pain radiating to the left arm."
    cas = build_corpus_case(description, entree, random.Random(6))
    assert cas is not None
    assert cas.level == "URGENCE_VITALE"
    assert cas.confiance == "moyenne"
    assert cas.symptomes


# --- Assemblage, découpage et fuite ---


def _corpus_exemples(nombre: int) -> list:
    generateur = random.Random(7)
    entree = CorpusEntry(text="", answer="", lang="en", source="medmcqa", topic="")
    cas = []
    for index in range(nombre):
        description = (
            f"A {30 + index}-year-old patient presents with chest pain since this morning."
        )
        construit = build_corpus_case(description, entree, generateur)
        if construit is not None:
            cas.append(construit)
    return cas


def test_l_assemblage_deduplique_et_exclut_l_evaluation():
    generateur = random.Random(8)
    vignettes = generate_cases(30, "URGENCE_VITALE", "fr", generateur)
    exemples = assemble(vignettes + vignettes, _corpus_exemples(10), eval_user_turns(), generateur)
    tours = [e.user_turn for e in exemples]
    assert len(tours) == len(set(tours))
    assert not set(tours) & eval_user_turns()


def test_le_decoupage_ne_laisse_aucune_fuite():
    generateur = random.Random(9)
    vignettes = []
    for niveau in TRIAGE.levels:
        vignettes += generate_cases(60, niveau, "fr", generateur)
    exemples = assemble(vignettes, [], set(), generateur)
    decoupages = split_train_val_test(exemples, 0.1, 0.1, seed=42)
    assert all(valeur == 0 for valeur in check_no_leakage(decoupages).values())
    total = sum(len(v) for v in decoupages.values())
    assert total == len(exemples)


def test_le_controle_de_fuite_detecte_un_doublon():
    """Le contrôle doit échouer quand il y a vraiment une fuite, sinon il ne sert à rien."""
    generateur = random.Random(10)
    exemples = assemble(
        generate_cases(10, "URGENCE_VITALE", "fr", generateur), [], set(), generateur
    )
    decoupages = {"train": exemples[:6], "test": exemples[5:]}
    assert check_no_leakage(decoupages)["test∩train"] == 1


def test_le_controle_de_fuite_detecte_un_quasi_doublon():
    """Deux écritures du même cas doivent compter comme une fuite.

    Les corpus publics livrent le même cas sous des orthographes voisines — « A 6
    year old child » et « A 6-year-old child ». Un contrôle qui ne compare qu'à
    l'identique laisserait ce doublon se répartir de part et d'autre du découpage,
    et rendrait le jeu de test complaisant sans que rien ne le signale.
    """
    generateur = random.Random(11)
    premier, second = assemble(
        generate_cases(2, "URGENCE_VITALE", "fr", generateur), [], set(), generateur
    )[:2]
    jumeau = replace(second, user_turn="A 6 year old child, elbow injury.")
    presque = replace(premier, user_turn="A 6-year-old child, elbow injury !")
    assert check_no_leakage({"train": [jumeau], "test": [presque]})["test∩train"] == 1


def test_la_deduplication_ecarte_un_quasi_doublon():
    """Le même cas écrit deux fois ne doit entrer qu'une fois dans le corpus."""
    generateur = random.Random(12)
    vignettes = generate_cases(1, "URGENCE_VITALE", "fr", generateur)
    exemples = assemble(vignettes, [], set(), generateur)
    assert len(exemples) == 1
    reserve = {exemples[0].user_turn.upper() + " !!"}
    assert assemble(vignettes, [], reserve, generateur) == []


# --- Enregistrements et métadonnées ---


def test_l_enregistrement_supervise_expose_prompt_et_completion():
    """La colonne `messages` est volontairement absente : elle ferait re-sérialiser
    le jeu avec le gabarit natif du modèle, donc apprendre un autre format."""
    generateur = random.Random(11)
    exemple = assemble(
        generate_cases(1, "URGENCE_VITALE", "fr", generateur), [], set(), generateur
    )[0]
    enregistrement = sft_record(exemple)
    assert set(enregistrement) >= {"prompt", "completion", "level", "lang", "source", "confiance"}
    assert "messages" not in enregistrement
    assert enregistrement["prompt"].endswith("<|im_start|>assistant\n")
    assert enregistrement["completion"].startswith("Niveau de priorité :")


def test_le_schema_de_metadonnees_couvre_les_champs_exiges():
    schema = metadata_schema()
    champs = schema["champs_sft"]
    for attendu in ("symptomes", "antecedents", "constantes", "source", "confiance"):
        assert attendu in champs
    assert set(schema["taxonomie_triage"]) == set(TRIAGE.levels)
    for source in schema["sources"].values():
        assert source["licence"]
        assert source["origine"]


# --- Jeu d'évaluation clinique ---


def test_le_jeu_d_evaluation_est_equilibre_et_bilingue():
    cas = eval_cases()
    par_niveau = Counter(c.level for c in cas)
    assert set(par_niveau) == set(TRIAGE.levels)
    assert len(set(par_niveau.values())) == 1  # même effectif pour les trois niveaux
    par_langue = Counter(c.lang for c in cas)
    assert min(par_langue.values()) >= len(cas) * 0.4


def test_le_jeu_d_evaluation_contient_des_cas_atypiques():
    cas = eval_cases()
    pieges = [c for c in cas if c.piege]
    assert len(pieges) >= len(cas) * 0.3
    assert {"faux_rassurant", "faux_alarmant", "negation", "constantes_discordantes"} <= {
        c.piege for c in pieges
    }


def test_chaque_cas_d_evaluation_est_documente():
    identifiants = set()
    for cas in eval_cases():
        assert cas.id not in identifiants
        identifiants.add(cas.id)
        assert len(cas.description) > 80
        assert len(cas.note) > 30
        assert cas.user_turn.count(cas.description) == 1


# --- Les constantes doivent dire la même chose que le récit ---

# Ce qu'un motif ou un signe peut nommer, et la constante que cela engage.
NOMME_UNE_CONSTANTE = {
    "temperature": r"fi[eè]vre|f[eé]brile|frisson|hyperthermie|sepsis|\bfever\b|chills",
    "spo2": r"cyanos|d[eé]satur|l[eè]vres bleues|blue lips|asphyx",
}


def _recit(presentation) -> str:
    return " ".join(
        (
            presentation.complaint_fr,
            presentation.complaint_en,
            *presentation.signs_fr,
            *presentation.signs_en,
        )
    ).lower()


@pytest.mark.parametrize("presentation", [p for p in PRESENTATIONS if p.vitals_profile != "normal"])
def test_une_presentation_qui_nomme_une_constante_l_impose(presentation):
    """Le générateur dégrade une ou deux constantes au hasard.

    Une vignette « fièvre avec frissons » sortait donc apyrétique la plupart du
    temps : la description et le relevé se contredisaient dans le même exemple,
    et le modèle apprenait au passage que les constantes ne veulent rien dire.
    """
    imposees = dict(forced_vitals(presentation.id))
    recit = _recit(presentation)
    for constante, motif in NOMME_UNE_CONSTANTE.items():
        if re.search(motif, recit):
            assert constante in imposees, (
                f"{presentation.id} nomme « {constante} » dans son récit sans l'imposer"
            )


def test_chaque_constante_imposee_designe_une_presentation_connue():
    identifiants = {p.id for p in PRESENTATIONS}
    assert set(CONSTANTES_IMPOSEES) <= identifiants


@pytest.mark.parametrize(
    "identifiant", ["sepsis_grave", "syndrome_meninge", "pneumopathie_communautaire"]
)
def test_une_vignette_febrile_sort_toujours_febrile(identifiant):
    presentation = presentation_by_id(identifiant)
    rng = random.Random(3)
    for _ in range(12):
        cas = build_case(presentation, "fr", rng)
        assert cas.constantes.temperature >= 38.5, cas.constantes.render("fr")


def test_une_pre_eclampsie_severe_sort_toujours_hypertendue():
    presentation = presentation_by_id("pre_eclampsie_severe")
    rng = random.Random(3)
    for _ in range(12):
        cas = build_case(presentation, "fr", rng)
        assert cas.constantes.systolic_bp >= 160, cas.constantes.render("fr")


@pytest.mark.parametrize("identifiant", ["bronchiolite_grave_nourrisson"])
def test_un_nourrisson_n_a_pas_la_tension_d_un_adulte(identifiant):
    """La tension était tirée de constantes d'adulte à tout âge.

    Les vignettes de nourrisson sortaient à « TA 120/75 » — un chiffre qui
    n'existe pas à cet âge, et que la règle lisait comme normal.
    """
    presentation = presentation_by_id(identifiant)
    rng = random.Random(3)
    for _ in range(12):
        cas = build_case(presentation, "fr", rng)
        assert cas.constantes.systolic_bp <= 105, cas.constantes.render("fr")


def test_le_schema_publie_decrit_exactement_les_colonnes_livrees():
    """Le schéma est la documentation du dataset publié sur le Hub.

    Il annonçait les colonnes du jeu supervisé pour les trois fichiers : le jeu
    de préférences n'a ni `confiance` ni `constantes`, et le jeu d'évaluation
    expose une colonne `description` que rien ne documentait. Un consommateur
    qui filtre sur une colonne absente n'obtient rien, sans comprendre pourquoi.
    """
    from clinical_triage.data.clinical_eval_set import eval_cases
    from clinical_triage.data.dataset_io import dpo_record, eval_record

    rng = random.Random(7)
    exemple = assemble(generate_cases(1, TRIAGE.levels[0], "fr", rng), [], set(), rng)[0]
    paire = build_preference_pairs([exemple], 1, rng)[0]
    schema = metadata_schema()

    assert list(sft_record(exemple)) == list(schema["champs_sft"])
    assert list(dpo_record(paire)) == list(schema["champs_dpo"])
    assert list(eval_record(eval_cases()[0])) == list(schema["champs_evaluation"])


def test_le_tableau_de_rendement_de_la_carte_suit_les_comptages_livres():
    """La carte du dataset est publiée telle quelle sur le Hub.

    Ses trois lignes de rendement y étaient recopiées à la main, et divergeaient
    déjà de `metadata.json`, livré à côté d'elles. Le tableau est maintenant
    écrit par le script de préparation ; ce test constate qu'il n'a pas été
    remodifié à la main depuis.
    """
    from clinical_triage.config import PATHS

    metadonnees = json.loads((PATHS.data_processed / "metadata.json").read_text(encoding="utf-8"))
    rendement = metadonnees["statistiques"]["rendement_corpus"]
    carte = (PATHS.data / "README.md").read_text(encoding="utf-8")
    tableau = carte[carte.index("<!-- rendement:debut") : carte.index("<!-- rendement:fin -->")]

    def lisible(valeur: int) -> str:
        return f"{valeur:,}".replace(",", " ")

    for mesure in rendement.values():
        assert f"| {lisible(mesure['entrees_lues'])} |" in tableau
        assert lisible(mesure["cas_retenus"]) in tableau
        # Les colonnes de perte s'additionnent avec les cas extraits pour
        # retrouver les entrées lues : c'est ce que le texte autour annonce.
        perdus = mesure["entonnoir"]
        assert (
            perdus["sans_presentation_de_patient"]
            + perdus["hors_bornes_de_longueur"]
            + perdus["sans_signe_identifie"]
            + perdus["doublons"]
            + perdus["retenus"]
            == mesure["entrees_lues"]
        )
        assert lisible(perdus["retenus"]) in tableau
