"""Tests du questionnaire adaptatif."""

from __future__ import annotations

import pytest

from clinical_triage.config import TRIAGE
from clinical_triage.data.triage_rules import DEFERRED, classify, explain
from clinical_triage.serving.questionnaire import (
    QUESTIONS_DE_GRAVITE,
    QUESTIONS_GENERALES,
    QUESTIONS_PAR_THEME,
    RAPPORTS,
    REPONSE_ALARMANTE,
    _reponse_binaire,
    compile_symptoms,
    detect_theme,
    has_red_flag,
    next_question,
    plan,
)


@pytest.mark.parametrize(
    ("motif", "theme"),
    [
        ("douleur dans la poitrine", "douleur_thoracique"),
        ("essoufflement depuis hier", "respiratoire"),
        ("mal de tête violent", "neurologique"),
        ("idées suicidaires", "psychiatrique"),
        ("chute de vélo", "traumatologie"),
        ("mal au ventre", "digestif"),
        ("fièvre depuis deux jours", "fievre"),
        ("démarche administrative", "general"),
    ],
)
def test_le_motif_determine_le_theme(motif, theme):
    assert detect_theme(motif) == theme


def test_les_questions_dependent_du_motif():
    """C'est ce qui rend le questionnaire adaptatif plutôt que fixe."""
    thoracique = [identifiant for identifiant, _ in plan("douleur dans la poitrine")]
    traumatisme = [identifiant for identifiant, _ in plan("entorse de la cheville")]
    assert thoracique != traumatisme
    assert "irradiation" in thoracique
    assert "appui" in traumatisme


def test_les_questions_de_gravite_sont_posees_en_premier_quel_que_soit_le_motif():
    for motif in ("mal de gorge", "douleur dans la poitrine", "chute de vélo"):
        premieres = [identifiant for identifiant, _ in plan(motif)][: len(QUESTIONS_DE_GRAVITE)]
        assert premieres == [identifiant for identifiant, _ in QUESTIONS_DE_GRAVITE]


def test_un_signe_vital_arrete_immediatement_la_collecte():
    etape = next_question("douleur thoracique intense", {})
    assert etape.termine is True
    assert etape.identifiant is None


def test_un_motif_benin_declenche_des_questions():
    etape = next_question("mal de gorge léger", {})
    assert etape.termine is False
    assert etape.identifiant and etape.texte


def test_la_collecte_se_termine_quand_tout_a_ete_repondu():
    reponses = {identifiant: "non" for identifiant, _ in plan("fatigue")}
    assert next_question("fatigue", reponses).termine is True


def test_un_signe_vital_apparu_en_cours_de_collecte_arrete_le_questionnaire():
    etape = next_question("fatigue", {"conscience": "non, le patient est inconscient"})
    assert etape.termine is True


def test_la_synthese_conserve_le_sens_de_chaque_reponse():
    """Ne garder que les réponses produirait « non. non. oui », sans référent."""
    synthese = compile_symptoms("toux", {"respiration": "non", "temperature": "38,5 depuis hier"})
    assert "toux" in synthese
    assert "Pas de difficulté à respirer" in synthese
    assert "38,5 depuis hier" in synthese


def test_une_reponse_negative_est_ecrite_comme_une_negation():
    """Une négation post-posée serait relue comme le symptôme lui-même.

    C'est le piège que ce module déjoue déjà pour l'arrêt anticipé, et qui
    reparaissait ici : la description compilée est relue par la règle de triage
    dans le service, et « Difficulté à respirer : non » y est lu comme une
    difficulté à respirer.
    """
    synthese = compile_symptoms("toux", {"respiration": "non"})
    assert "Difficulté à respirer :" not in synthese
    assert synthese.endswith("Pas de difficulté à respirer ni d'essoufflement au repos.")


@pytest.mark.parametrize(
    "motif",
    [
        "gene dans la poitrine",
        "petite toux",
        "mal de tete leger",
        "ventre un peu gonfle",
        "petite coupure au doigt",
        "fievre a 38",
        "un peu d angoisse",
        "fatigue generale",
    ],
)
def test_des_reponses_rassurantes_n_aggravent_jamais_le_verdict_de_la_regle(motif):
    """Un rhume dont tout est nié ne doit pas ressortir en urgence vitale.

    Un cas parcourt les huit thèmes : c'est le texte des questions qui, recopié
    dans la description, faisait basculer la règle.
    """
    # « non » n'est pas rassurant partout : à « le patient est-il conscient ? »,
    # c'est la réponse alarmante. On répond donc l'inverse de ce que
    # `REPONSE_ALARMANTE` déclare pour chaque question.
    rassurantes = {
        identifiant: "oui" if REPONSE_ALARMANTE.get(identifiant) == "non" else "non"
        for identifiant, _ in plan(motif)
    }
    description = compile_symptoms(motif, rassurantes)
    assert TRIAGE.severity[classify(description)] <= TRIAGE.severity[classify(motif)]


def test_chaque_question_sait_comment_reporter_sa_reponse():
    """Une question sans formulation retomberait sur son identifiant brut."""
    posees = {identifiant for identifiant, _ in QUESTIONS_DE_GRAVITE}
    for questions in QUESTIONS_PAR_THEME.values():
        posees.update(identifiant for identifiant, _ in questions)
    posees.update(identifiant for identifiant, _ in QUESTIONS_GENERALES)
    assert posees <= set(RAPPORTS)


def test_les_reponses_vides_sont_ignorees():
    synthese = compile_symptoms("toux", {"respiration": "   ", "saignement": "non"})
    assert "respirer" not in synthese
    assert "saignement" in synthese.lower()


# --- Lecture d'une réponse oui / non ---


@pytest.mark.parametrize(
    "reponse",
    [
        "Notre fille saigne du nez en abondance depuis 20 minutes",
        "Nous ne savons pas",
        "Normalement non",
        "Nouvelle crise depuis ce matin",
    ],
)
def test_un_mot_qui_commence_par_no_n_est_pas_une_negation(reponse):
    """« no » est une négation en anglais, et le début de « Notre » en français.

    La comparaison portait sur un préfixe : « Notre fille saigne du nez » était
    lu comme un « non », et la synthèse transmise au modèle écrivait « Aucun
    saignement actif » — l'exact contraire de ce que l'accompagnant venait de
    déclarer.
    """
    assert _reponse_binaire(reponse) is None


@pytest.mark.parametrize(
    ("attendu", "reponse"),
    [
        ("oui", "oui"),
        ("oui", "Oui"),
        ("non", "non"),
        ("non", "  NON  "),
        ("non", "Aucun"),
        ("non", "no"),
        ("non", "rien"),
        (None, "oui, beaucoup de sang"),
        (None, "peut-etre"),
    ],
)
def test_seule_une_reponse_reduite_a_un_mot_est_binaire(attendu, reponse):
    """Dès qu'il y a autre chose, le texte est repris tel quel : c'est plus sûr."""
    assert _reponse_binaire(reponse) == attendu


def test_une_reponse_libre_arrive_intacte_dans_la_synthese():
    synthese = compile_symptoms(
        "Chute de velo", {"saignement": "Notre fille saigne du nez en abondance depuis 20 minutes"}
    )
    assert "saigne du nez en abondance" in synthese
    assert "Aucun saignement" not in synthese


# --- Dépistage du signe vital ---


@pytest.mark.parametrize(
    ("motif", "reponses"),
    [
        ("mal de tete", {"deficit": "oui"}),
        ("mal de tete", {"cephalee": "oui"}),
        ("angoisse", {"intention": "oui"}),
        ("toux", {"parole": "non"}),
        ("fievre", {"nuque": "non"}),
        ("mal de tete", {"deficit": "oui, hemiplegie droite depuis 30 minutes"}),
        ("chute", {"saignement": "Notre fille saigne beaucoup du nez"}),
    ],
)
def test_une_reponse_alarmante_arrete_la_collecte(motif, reponses):
    """Cinq questions sur huit ne déclenchaient rien, faute d'être répertoriées.

    Et la règle de triage n'était appliquée qu'aux réponses non binaires : tout
    ce qui suivait un « oui » n'était jamais lu.
    """
    assert has_red_flag(motif, reponses)
    assert next_question(motif, reponses).termine


@pytest.mark.parametrize(
    ("motif", "reponses"),
    [
        ("rhume", {"conscience": "oui", "respiration": "non", "saignement": "non"}),
        ("entorse", {"appui": "non", "deformation": "non"}),
        ("fievre", {"nuque": "oui", "frissons": "non"}),
        ("angoisse", {"intention": "non", "moyens": "non"}),
        ("ventre", {"vomissements": "oui"}),
    ],
)
def test_une_collecte_rassurante_se_poursuit(motif, reponses):
    assert not has_red_flag(motif, reponses)


def test_aucun_intitule_de_reponse_libre_ne_declenche_la_regle_a_lui_seul():
    """L'intitulé est notre vocabulaire, pas celui du patient.

    La synthèse compilée est relue par la règle de triage, et son avis s'affiche
    à côté de celui du modèle. « Idées suicidaires : je ne sais pas » y faisait
    apparaître « idees suicidaires » dans les raisons montrées au soignant, pour
    un patient qui n'avait rien exprimé du tout : une justification fabriquée par
    la formulation du questionnaire lui-même.
    """
    for identifiant, (intitule, _, _) in RAPPORTS.items():
        phrase = f"{intitule} : je ne sais pas."
        assert classify(phrase) == DEFERRED, (identifiant, intitule, explain(phrase))


def test_une_reponse_reellement_alarmante_reste_detectee():
    """Les phrases canoniques, elles, rapportent ce que le patient a dit."""
    for identifiant in ("intention", "frissons", "respiration"):
        _, affirmatif, _ = RAPPORTS[identifiant]
        assert classify(affirmatif) != DEFERRED, identifiant


def test_une_reponse_libre_a_une_question_non_binaire_n_alarme_pas():
    """Toutes les questions n'attendent pas un oui ou un non.

    Le mécanisme d'un traumatisme se raconte en toutes lettres. Comparé sans
    garde, il valait `None` des deux côtés — réponse non binaire et question sans
    réponse alarmante — et le questionnaire s'arrêtait sur une détresse vitale
    qui n'avait pas été décrite.
    """
    motif = "entorse de la cheville apres une chute"
    reponses = {
        "conscience": "oui",
        "respiration": "non",
        "saignement": "non",
        "mecanisme": "chute de sa hauteur dans l escalier",
    }

    assert has_red_flag(motif, reponses) is False
    assert next_question(motif, reponses).termine is False


def test_une_reponse_libre_qui_decrit_un_signe_grave_alarme_toujours():
    """La garde ne doit pas rendre le questionnaire sourd au contenu."""
    motif = "chute a domicile"
    reponses = {"mecanisme": "chute de quatre metres, le patient est inconscient"}

    assert has_red_flag(motif, reponses) is True
