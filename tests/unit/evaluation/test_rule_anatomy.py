"""Why the explicit rule misses an urgent case, on narratives built for each cause."""

from __future__ import annotations

from clinical_triage.evaluation.rule_anatomy import CAUSES, cause_of_miss, rule_misses


def test_a_sign_discarded_by_a_negation_is_named():
    assert cause_of_miss("Homme de 55 ans, il n'a pas de douleur thoracique.") == CAUSES[0]


def test_a_lesser_sign_read_is_named():
    assert cause_of_miss("Femme de 40 ans, FC 112, se sent mal depuis ce matin.") == CAUSES[1]


def test_a_narrative_the_rule_cannot_read_is_named():
    assert cause_of_miss("Il est tout gris et ne dit plus rien depuis tout à l'heure.") == CAUSES[2]


def test_the_counts_cover_urgent_cases_alone():
    cases = [
        {
            "id": "a",
            "level": "URGENCE_VITALE",
            "case_type": "falsely_reassuring",
            "description": "Il est tout gris et ne dit plus rien depuis tout à l'heure.",
        },
        {
            "id": "b",
            "level": "URGENCE_VITALE",
            "case_type": "",
            "description": "Homme de 70 ans, arrêt cardiaque devant les secours.",
        },
        {
            "id": "c",
            "level": "CONSULTATION_DIFFEREE",
            "case_type": "",
            "description": "Rhume depuis deux jours.",
        },
    ]
    anatomy = rule_misses(cases)
    assert (anatomy["urgent_cases"], anatomy["missed"]) == (2, 1)
    assert anatomy["per_cause"]["nothing read"] == 1
    assert anatomy["per_case_type"] == {"falsely_reassuring": 1}
