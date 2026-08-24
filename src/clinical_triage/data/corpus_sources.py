"""Chargement des corpus médicaux publics exigés par le cahier des charges.

Quatre corpus sont demandés : MediQAl, FrenchMedMCQA, MedQuAD et
UltraMedical-Preference. Ce module se contente de les télécharger et de les
ramener à un format commun ; le tri de ce qui est réellement exploitable pour du
triage se fait dans `corpus_cases`.

Trois constats de terrain, vérifiés sur les corpus eux-mêmes, expliquent la suite :

- **MediQAl est le seul à contenir de vraies vignettes cliniques.** Sa colonne
  `clinical_case` décrit un patient, son motif, ses antécédents et souvent ses
  constantes d'entrée. 3 075 vignettes distinctes, en français, dont 569 dont on
  sait lire au moins une constante. C'est la seule source authentique francophone.
- **FrenchMedMCQA compte 1 080 questions** sur ses trois découpages, et six
  d'entre elles seulement décrivent un patient : c'est un jeu de questions
  d'examen de pharmacie.
- **MedQuAD pose des questions générales sur des pathologies**, pas sur des
  patients. Il sert à varier le vocabulaire clinique anglophone, pas à fournir
  des cas.

MedMCQA complète le volume anglophone de vignettes, puisque les trois corpus
exploitables du cahier des charges sont soit francophones, soit sans patient.

**Les corpus sont lus intégralement.** Une lecture partielle donnerait un
rendement qui ne décrit pas la source mais le plafond qu'on s'est fixé, et le
rapport en tirerait des conclusions fausses sur ce que les corpus permettent.
Le paramètre `limit` existe pour les essais rapides ; à zéro, il ne borne rien.

Deux pannes, deux conduites. **Une source injoignable à l'ouverture** est
signalée et ignorée ; le script de préparation vérifie ensuite que chaque corpus
attendu a bien fourni des entrées, et refuse de construire un dataset amputé sans
le dire. **Une lecture interrompue en route**, elle, arrête tout : le rendement
publié se calcule sur le nombre d'entrées lues, et une lecture partielle absorbée
en silence produirait un tableau qui décrit la moitié d'un corpus en la présentant
comme le corpus entier.
"""

from __future__ import annotations

from dataclasses import dataclass

from clinical_triage.config import DATA
from clinical_triage.utils import get_logger

logger = get_logger(__name__)


# Noms lisibles des corpus, dans l'ordre ou les livrables les presentent.
# Partages par le script de construction et par celui du rapport : deux listes
# auraient fini par diverger.
NOMS_DE_CORPUS = (
    ("mediqal", "MediQAl"),
    ("medquad", "MedQuAD"),
    ("medmcqa", "MedMCQA"),
    ("frenchmedmcqa", "FrenchMedMCQA"),
)


@dataclass(frozen=True)
class CorpusEntry:
    """Entrée d'un corpus public, ramenée à un format commun."""

    text: str  # énoncé brut (question, vignette ou description de symptômes)
    answer: str  # réponse ou explication associée, quand elle existe
    lang: str  # "fr" ou "en"
    source: str  # nom court du corpus d'origine
    topic: str  # discipline ou pathologie, quand le corpus la fournit


def _clean(value: object) -> str:
    """Nettoyage minimal : espaces multiples et retours à la ligne."""
    if not value:
        return ""
    return " ".join(str(value).split()).strip()


def _stream(repo: str, split: str = "train"):
    """Ouvre un jeu Hugging Face en streaming, ou renvoie None s'il est indisponible."""
    from datasets import load_dataset

    try:
        return load_dataset(repo, split=split, streaming=True)
    except Exception as exc:  # noqa: BLE001 - une source absente ne doit pas tout arrêter
        logger.warning("Corpus %s indisponible (%s) — source ignorée.", repo, exc)
        return None


def _stream_tous_les_decoupages(repo: str):
    """Enchaîne les lignes de tous les découpages d'un dépôt.

    Un corpus d'examen répartit ses questions entre `train`, `validation` et
    `test` pour ses propres besoins d'évaluation. Ce découpage n'a aucun sens
    ici : on cherche des descriptions de patients, et s'arrêter au découpage
    d'entraînement reviendrait à en ignorer la moitié sans raison.
    """
    from datasets import load_dataset

    try:
        jeux = load_dataset(repo, streaming=True)
    except Exception as exc:  # noqa: BLE001 - une source absente ne doit pas tout arrêter
        logger.warning("Corpus %s indisponible (%s) — source ignorée.", repo, exc)
        return
    for nom in jeux:
        yield from jeux[nom]


def _plafond_atteint(entrees: list, limite: int) -> bool:
    """Dit si la lecture doit s'arrêter. Une limite nulle ou négative ne borne rien."""
    return limite > 0 and len(entrees) >= limite


class LectureInterrompue(RuntimeError):
    """Une lecture en flux s'est arrêtée en route, corpus incomplet."""


def _lignes(dataset, repo: str):
    """Parcourt un flux, et refuse de rendre un corpus tronqué en silence.

    Le `try` des ouvertures ne couvre que l'ouverture : en streaming, tout le
    trafic réseau a lieu pendant l'itération. Une coupure à ce moment-là
    remonterait jusqu'au script de préparation — c'est déjà mieux qu'un corpus
    amputé, mais l'erreur ne dirait pas où elle s'est produite.

    On ne l'absorbe pas pour autant. Le rendement publié dans la carte du
    dataset est calculé sur le nombre d'entrées lues : absorber la coupure
    produirait un tableau qui décrit une lecture partielle en la présentant
    comme complète, et un jeu reconstruit après incident serait indiscernable
    d'un jeu entier. Mieux vaut une préparation à relancer qu'un chiffre faux.
    """
    lues = 0
    try:
        for ligne in dataset:
            lues += 1
            yield ligne
    except Exception as exc:
        raise LectureInterrompue(
            f"Lecture de {repo} interrompue après {lues} entrées ({exc}). "
            "Le corpus serait incomplet et le rendement publié, faux : relancez la préparation."
        ) from exc


def load_mediqal(limit: int = 0) -> list[CorpusEntry]:
    """MediQAl : cas cliniques français annotés, issus du projet ANR MALADES.

    C'est le seul corpus du cahier des charges qui décrive des **patients** :
    la colonne `clinical_case` porte un motif, des antécédents et, une fois sur
    cinq, les constantes relevées à l'entrée.

    Trois configurations partagent cette colonne — `oeq` (questions ouvertes),
    `mcqu` et `mcqm` (choix multiples). Une même vignette y sert souvent plusieurs
    questions : on dédoublonne sur le texte du cas, sans quoi le même patient
    reviendrait jusqu'à dix fois dans le corpus et fausserait l'équilibre du jeu.
    """
    from datasets import load_dataset

    vus: set[str] = set()
    entries: list[CorpusEntry] = []
    for configuration in ("oeq", "mcqu", "mcqm"):
        try:
            jeux = load_dataset(DATA.corpora["mediqal"], configuration)
        except Exception as exc:  # noqa: BLE001 - une source absente ne doit pas tout arrêter
            logger.warning(
                "MediQAl/%s indisponible (%s) — configuration ignorée.", configuration, exc
            )
            continue
        for decoupage in jeux.values():
            for row in decoupage:
                vignette = _clean(row.get("clinical_case"))
                if not vignette or vignette in vus:
                    continue
                vus.add(vignette)
                entries.append(
                    CorpusEntry(
                        text=vignette,
                        answer=_clean(row.get("answer")) or _clean(row.get("question")),
                        lang="fr",
                        source="mediqal",
                        topic=_clean(row.get("medical_subject")),
                    )
                )
                if _plafond_atteint(entries, limit):
                    logger.info("MediQAl : %d vignettes distinctes chargées.", len(entries))
                    return entries
    logger.info("MediQAl : %d vignettes distinctes chargées.", len(entries))
    return entries


def load_medquad(limit: int = 0) -> list[CorpusEntry]:
    """MedQuAD : questions-réponses médicales en anglais, issues des sites du NIH.

    Le dépôt désigné par le cahier des charges expose deux colonnes, `Question`
    et `Answer`, avec une majuscule ; d'autres redistributions du même corpus les
    nomment en minuscules. On accepte les deux, pour que changer de miroir ne
    casse pas la préparation.
    """
    dataset = _stream(DATA.corpora["medquad"])
    if dataset is None:
        return []
    entries: list[CorpusEntry] = []
    for row in _lignes(dataset, DATA.corpora["medquad"]):
        question = _clean(row.get("Question") or row.get("question"))
        answer = _clean(row.get("Answer") or row.get("answer"))
        if question and answer:
            entries.append(
                CorpusEntry(
                    text=question,
                    answer=answer,
                    lang="en",
                    source="medquad",
                    topic=_clean(row.get("qtype") or row.get("question_type")),
                )
            )
        if _plafond_atteint(entries, limit):
            break
    logger.info("MedQuAD : %d entrées chargées.", len(entries))
    return entries


def load_frenchmedmcqa(limit: int = 0) -> list[CorpusEntry]:
    """FrenchMedMCQA : questions à choix multiples de pharmacie, en français.

    Les trois découpages sont lus : 595 questions en entraînement, 164 en
    validation, 321 en test. Se limiter au premier laisserait de côté 485
    questions du seul autre corpus francophone du cahier des charges.
    """
    # Générateur : une source indisponible n'y produit aucune ligne, et la boucle
    # ci-dessous se termine d'elle-même en renvoyant une liste vide.
    dataset = _lignes(
        _stream_tous_les_decoupages(DATA.corpora["frenchmedmcqa"]),
        DATA.corpora["frenchmedmcqa"],
    )
    # `correct_answers` encode les bonnes options, soit en lettres, soit en chiffres.
    chiffre_vers_lettre = {"1": "a", "2": "b", "3": "c", "4": "d", "5": "e"}
    entries: list[CorpusEntry] = []
    for row in dataset:
        question = _clean(row.get("question"))
        if not question:
            continue
        brut = str(row.get("correct_answers") or "").lower()
        lettres = [chiffre_vers_lettre.get(c, c) for c in brut if c.isalnum()]
        bonnes = [_clean(row.get(f"answer_{lettre}")) for lettre in lettres]
        entries.append(
            CorpusEntry(
                text=question,
                answer="; ".join(a for a in bonnes if a),
                lang="fr",
                source="frenchmedmcqa",
                topic="",
            )
        )
        if _plafond_atteint(entries, limit):
            break
    logger.info("FrenchMedMCQA : %d entrées chargées.", len(entries))
    return entries


def load_medmcqa(limit: int = 0) -> list[CorpusEntry]:
    """MedMCQA : questions à choix multiples d'internat médical, en anglais."""
    dataset = _stream(DATA.corpora["medmcqa"])
    if dataset is None:
        return []
    options = ("opa", "opb", "opc", "opd")
    entries: list[CorpusEntry] = []
    for row in _lignes(dataset, DATA.corpora["medmcqa"]):
        question = _clean(row.get("question"))
        if not question:
            continue
        try:
            index = int(row.get("cop"))
        except (TypeError, ValueError):
            index = -1
        bonne = _clean(row.get(options[index])) if 0 <= index < len(options) else ""
        entries.append(
            CorpusEntry(
                text=question,
                answer=bonne,
                lang="en",
                source="medmcqa",
                topic=_clean(row.get("subject_name")),
            )
        )
        if _plafond_atteint(entries, limit):
            break
    logger.info("MedMCQA : %d entrées chargées.", len(entries))
    return entries


@dataclass(frozen=True)
class PreferencePair:
    """Paire de préférence annotée, telle que la fournit UltraMedical-Preference."""

    prompt: str
    chosen: str
    rejected: str
    label_type: str  # "hard", "easy" ou "length"


def _last_message(value: object) -> str:
    """Extrait le contenu du dernier message d'une réponse au format conversation."""
    import ast

    if isinstance(value, str) and value.strip().startswith("[{"):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value.strip()
    if isinstance(value, list) and value:
        dernier = value[-1]
        if isinstance(dernier, dict):
            return _clean(dernier.get("content"))
    return _clean(value)


def load_ultramedical_preferences(
    limit: int, drop_length_labels: bool = True
) -> list[PreferencePair]:
    """UltraMedical-Preference : paires de réponses médicales annotées par préférence.

    Le corpus indique dans `label_type` sur quel critère la préférence a été
    établie. Un tiers des paires est étiqueté `length` : la réponse préférée
    l'est parce qu'elle est plus longue. Entraîner un alignement sur ce signal
    apprend au modèle que « plus long vaut mieux », ce qui, sur un agent dont la
    réponse tient en trois lignes, se traduit par une génération qui ne s'arrête
    plus. On écarte donc ces paires par défaut.
    """
    dataset = _stream(DATA.corpora["ultramedical_pref"])
    if dataset is None:
        return []
    pairs: list[PreferencePair] = []
    ecartees = 0
    for row in _lignes(dataset, DATA.corpora["ultramedical_pref"]):
        label_type = _clean(row.get("label_type"))
        if drop_length_labels and label_type == "length":
            ecartees += 1
            continue
        prompt = _clean(row.get("prompt"))
        chosen = _last_message(row.get("chosen"))
        rejected = _last_message(row.get("rejected"))
        if prompt and chosen and rejected and chosen != rejected:
            pairs.append(PreferencePair(prompt, chosen, rejected, label_type))
        if len(pairs) >= limit:
            break
    logger.info(
        "UltraMedical-Preference : %d paires retenues, %d écartées (préférence de longueur).",
        len(pairs),
        ecartees,
    )
    return pairs
