"""Construction du dataset bilingue de triage médical.

Déroulé, dans l'ordre :

    1. jeu d'évaluation clinique : écrit à la main, sorti en premier et mis de
       côté, pour qu'aucune étape suivante ne puisse le réutiliser ;
    2. corpus publics : chargement, extraction des cas réellement exploitables,
       anonymisation RGPD et contrôle qualité indépendant ;
    3. vignettes cliniques : génération du complément nécessaire pour atteindre
       le volume cible avec un équilibre 50/50 entre français et anglais ;
    4. assemblage, déduplication et découpage train / validation / test, avec
       vérification que les découpages sont disjoints ;
    5. paires de préférences DPO, construites à partir du seul jeu
       d'entraînement ;
    6. écriture des fichiers JSONL et de la carte du dataset.

Usage :
    uv run python scripts/01_build_dataset.py
    uv run python scripts/01_build_dataset.py --sft-size 1200   # itération rapide
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from datetime import UTC, datetime

from clinical_triage.config import DATA, PATHS, SEED, TRIAGE
from clinical_triage.data.anonymize import (
    ENTITES_ECARTEES,
    ENTITES_MASQUEES,
    analyze_and_anonymize,
    audit_corpus,
)
from clinical_triage.data.case_generator import generate_cases
from clinical_triage.data.clinical_catalogue import PRESENTATIONS
from clinical_triage.data.clinical_eval_set import eval_cases, eval_user_turns
from clinical_triage.data.corpus_cases import MAX_LENGTH, entonnoir, extract_cases
from clinical_triage.data.corpus_sources import (
    NOMS_DE_CORPUS,
    load_frenchmedmcqa,
    load_mediqal,
    load_medmcqa,
    load_medquad,
)
from clinical_triage.data.dataset_io import (
    FICHIERS_DU_DATASET,
    check_no_leakage,
    dpo_record,
    eval_record,
    metadata_schema,
    sft_record,
    split_train_val_test,
    write_jsonl,
    write_metadata,
)
from clinical_triage.data.diagnostics import (
    completions_distinctes,
    fuite_par_metadonnees,
    separabilite,
)
from clinical_triage.data.dpo_builder import build_preference_pairs, length_balance
from clinical_triage.data.sft_builder import assemble, empreinte_de_cas
from clinical_triage.data.triage_rules import classify
from clinical_triage.utils import get_logger, revision_git, set_seed

logger = get_logger("build_dataset")


BORNES_DU_RENDEMENT = (
    "<!-- rendement:debut",
    "<!-- rendement:fin -->",
)


def _commande_executee(args: argparse.Namespace) -> str:
    """La commande telle qu'elle a été passée, options non vides comprises."""
    defauts = {"sft_size": DATA.target_sft_pairs, "dpo_size": DATA.target_dpo_pairs}
    options = []
    for nom, valeur in vars(args).items():
        if valeur == defauts.get(nom) or valeur in (0, False, None):
            continue
        tiret = "--" + nom.replace("_", "-")
        options.append(tiret if valeur is True else f"{tiret} {valeur}")
    return " ".join(["uv run python scripts/01_build_dataset.py", *options])


def _lisible(valeur: int) -> str:
    """Milliers séparés par une espace, comme le veut l'usage français."""
    return f"{valeur:,}".replace(",", " ")


def _ecrire_le_rendement_dans_la_carte(rendement: dict) -> None:
    """Réécrit le tableau de rendement de `data/README.md` depuis les comptages.

    Cette carte est publiée telle quelle sur le Hub, à côté de `metadata.json`.
    Les trois lignes du tableau sont donc écrites depuis les comptages de la
    construction en cours, et non saisies à la main : la carte et le fichier de
    métadonnées publiés ensemble décrivent le même jeu.
    """
    carte = PATHS.data / "README.md"
    texte = carte.read_text(encoding="utf-8")
    ouverture, fermeture = BORNES_DU_RENDEMENT
    debut = texte.index(ouverture)
    fin = texte.index(fermeture) + len(fermeture)

    entete = (
        "| Corpus | Entrées lues | Sans patient décrit | Hors bornes de longueur "
        "| Sans signe identifié | Doublons | Cas extraits | Cas livrés | Rendement |"
    )
    lignes = [
        texte[debut : texte.index("-->", debut) + 3],
        entete,
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for identifiant, nom in NOMS_DE_CORPUS:
        mesure = rendement[identifiant]
        lues, retenus = mesure["entrees_lues"], mesure["cas_retenus"]
        perdus = mesure["entonnoir"]
        # Décimale à la virgule pour le rendement, milliers séparés par une
        # espace pour les comptages : `_lisible` ne touche qu'à ces derniers,
        # sinon elle défait la virgule du rendement juste à côté.
        part = f"{100 * retenus / lues:.1f}".replace(".", ",") if lues else "0,0"
        # Le meilleur rendement et le rendement nul sont les deux chiffres que
        # le texte qui suit commente : ils sont mis en valeur.
        gras = "**" if retenus == 0 or identifiant == "mediqal" else ""
        lignes.append(
            f"| {nom} | {_lisible(lues)} "
            f"| {_lisible(perdus['sans_presentation_de_patient'])} "
            f"| {_lisible(perdus['hors_bornes_de_longueur'])} "
            f"| {_lisible(perdus['sans_signe_identifie'])} "
            f"| {_lisible(perdus['doublons'])} "
            f"| {_lisible(perdus['retenus'])} "
            f"| {gras}{_lisible(retenus)}{gras} | {gras}{part} %{gras} |"
        )
    lignes.append(fermeture)

    carte.write_text(texte[:debut] + "\n".join(lignes) + texte[fin:], encoding="utf-8")
    logger.info("Tableau de rendement mis à jour → %s", carte)


def _repartition(items, cle) -> dict[str, int]:
    """Compte les occurrences d'un attribut, triées par ordre décroissant."""
    compte: dict[str, int] = {}
    for item in items:
        valeur = str(cle(item))
        compte[valeur] = compte.get(valeur, 0) + 1
    return dict(sorted(compte.items(), key=lambda kv: -kv[1]))


def _grille_des_cases(sft_size: int) -> list[tuple[str, str, int]]:
    """Les six cases du jeu — niveau × langue — et le volume visé dans chacune.

    Le reste de la division est réparti sur les premières cases, pour que la
    somme des cibles tombe exactement sur le volume demandé.
    """
    cases = [(niveau, langue) for niveau in TRIAGE.levels for langue in ("fr", "en")]
    base, reste = divmod(sft_size, len(cases))
    return [
        (niveau, langue, base + (1 if index < reste else 0))
        for index, (niveau, langue) in enumerate(cases)
    ]


def _limiter_les_cas_de_corpus(cas: list, sft_size: int, rng: random.Random) -> list:
    """Borne l'apport des corpus, case par case puis globalement.

    Deux bornes, et il faut les deux.

    **Par case.** Les cas extraits se concentrent sur les niveaux urgents et sur
    l'anglais — la règle de sécurité écarte tout ce qu'elle ne sait pas étiqueter,
    et MedMCQA est anglophone. Lu intégralement, il en fournit plus que la case
    « urgence modérée / anglais » n'a de places. Sans borne par case, le jeu livré
    dépasserait le volume demandé et perdrait son équilibre entre niveaux et
    entre langues.

    **Globalement.** Les cas de corpus portent une étiquette de confiance
    `moyenne`, posée par la règle à mots-clés. Les laisser devenir majoritaires
    ferait du jeu, pour l'essentiel, une transcription de cette règle. Le plafond
    les maintient minoritaires face aux vignettes du catalogue, dont l'étiquette
    ne vient pas d'une lecture du texte.
    """
    retenus: list = []
    for niveau, langue, cible in _grille_des_cases(sft_size):
        de_la_case = [c for c in cas if c.level == niveau and c.lang == langue]
        rng.shuffle(de_la_case)
        if len(de_la_case) > cible:
            logger.info(
                "  %-22s %s : %d cas disponibles, ramenés à %d (capacité de la case).",
                niveau,
                langue,
                len(de_la_case),
                cible,
            )
        retenus += de_la_case[:cible]

    plafond = int(sft_size * DATA.max_corpus_share)
    if len(retenus) > plafond:
        rng.shuffle(retenus)
        retenus = retenus[:plafond]
        logger.info(
            "  ramenés à %d cas (%.0f %% du jeu) pour rester minoritaires face au catalogue.",
            len(retenus),
            100 * DATA.max_corpus_share,
        )
    return retenus


def _mesurer_les_reponses(exemples: list) -> dict:
    """Longueur des réponses attendues, en jetons.

    Le rapport s'en sert pour écarter le budget de génération comme cause du
    défaut d'arrêt : si la réponse médiane tient largement sous le plafond, ce
    n'est pas lui qui coupe la réponse. La mesure est refaite à chaque
    construction, pour qu'elle décrive le jeu livré avec elle.
    """
    from transformers import AutoTokenizer

    from clinical_triage.config import MODEL, SERVING

    tokenizer = AutoTokenizer.from_pretrained(MODEL.base_model)
    longueurs = sorted(
        len(tokenizer(e.assistant_turn, add_special_tokens=False)["input_ids"]) for e in exemples
    )
    return {
        "mediane_jetons": longueurs[len(longueurs) // 2],
        "p95_jetons": longueurs[int(0.95 * len(longueurs))],
        "maximum_jetons": longueurs[-1],
        "plafond_de_generation": SERVING.max_new_tokens,
    }


def _verifier_le_budget_de_description(cas: list) -> dict:
    """Refuse un cas plus long que ce que le service acceptera de lire.

    La borne d'extraction est exprimée en caractères ; la fenêtre du modèle se
    compte en jetons. Entre les deux il y a une densité, qui dépend de la langue
    et du vocabulaire, et qu'un changement de corpus peut déplacer. On vérifie
    donc sur les cas réellement retenus, à chaque construction : si l'un d'eux
    dépasse le budget, le modèle s'entraînerait sur un récit que le service
    tronquerait, et il faut abaisser `MAX_LENGTH` plutôt que le découvrir en
    production.

    Renvoie la mesure, que la carte du dataset publie : le rapport y lit le
    budget et la longueur réellement observée au lieu de les recopier.
    """
    from transformers import AutoTokenizer

    from clinical_triage.config import MODEL, SERVING
    from clinical_triage.prompts import budget_de_description

    tokenizer = AutoTokenizer.from_pretrained(MODEL.base_model)
    budget = budget_de_description(tokenizer, MODEL.max_seq_length, SERVING.max_new_tokens)
    longueurs = [len(tokenizer(c.description, add_special_tokens=False)["input_ids"]) for c in cas]
    if not longueurs:
        return {"budget_jetons": budget}
    depassements = sum(1 for longueur in longueurs if longueur > budget)
    logger.info(
        "  longueur des descriptions : médiane %d jetons, maximum %d, budget du service %d.",
        sorted(longueurs)[len(longueurs) // 2],
        max(longueurs),
        budget,
    )
    if depassements:
        raise SystemExit(
            f"{depassements} cas dépassent le budget de description du service "
            f"({budget} jetons, maximum observé {max(longueurs)}). "
            "Abaissez MAX_LENGTH dans src/clinical_triage/data/corpus_cases.py."
        )
    return {
        "budget_jetons": budget,
        "mediane_jetons": sorted(longueurs)[len(longueurs) // 2],
        "maximum_jetons": max(longueurs),
        "borne_caracteres": MAX_LENGTH,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-size", type=int, default=DATA.target_sft_pairs)
    parser.add_argument("--dpo-size", type=int, default=DATA.target_dpo_pairs)
    parser.add_argument(
        "--corpus-scan",
        type=int,
        default=0,
        help=(
            "nombre maximum d'entrées lues par corpus ; 0 lit tout (défaut). "
            "N'existe que pour les essais rapides : un plafond fausse le tableau "
            "de rendement, qui décrirait alors le plafond et non le corpus."
        ),
    )
    parser.add_argument(
        "--no-anonymize", action="store_true", help="ignorer Presidio (mise au point)"
    )
    args = parser.parse_args()

    # Sous ce seuil, `split_train_val_test` arrondit les parts de validation et
    # de test à zéro : les deux fichiers sortent vides, le contrôle de fuite les
    # déclare disjoints — ce qu'ils sont, trivialement — et rien ne signale que
    # le jeu livré n'a plus de quoi mesurer sa propre convergence.
    minimum = int(1 / min(DATA.val_ratio, DATA.test_ratio))
    if args.sft_size < minimum:
        raise SystemExit(
            f"--sft-size {args.sft_size} est trop petit : sous {minimum} exemples, les jeux de "
            "validation et de test sortiraient vides."
        )

    set_seed(SEED, include_torch=False)
    rng = random.Random(SEED)
    PATHS.ensure()
    sortie = PATHS.data_processed

    # --- 1. Jeu d'évaluation clinique, mis de côté avant toute autre étape ---
    logger.info("Étape 1/6 — jeu d'évaluation clinique rédigé à la main.")
    cas_evaluation = eval_cases()
    tours_reserves = eval_user_turns()
    write_jsonl([eval_record(c) for c in cas_evaluation], sortie / "clinical_eval.jsonl")
    logger.info(
        "  %d cas (%s), dont %d avec piège.",
        len(cas_evaluation),
        _repartition(cas_evaluation, lambda c: c.level),
        sum(1 for c in cas_evaluation if c.piege),
    )

    # --- 2. Corpus publics ---
    logger.info("Étape 2/6 — corpus publics : chargement et extraction des cas exploitables.")
    # MediQAl d'abord : c'est le seul corpus qui décrive des patients, et donc
    # celui dont les cas survivent le mieux à l'extraction.
    par_corpus = {
        "mediqal": load_mediqal(limit=args.corpus_scan),
        "medquad": load_medquad(limit=args.corpus_scan),
        "medmcqa": load_medmcqa(limit=args.corpus_scan),
        "frenchmedmcqa": load_frenchmedmcqa(limit=args.corpus_scan),
    }
    # Le nombre d'entrées réellement lues par corpus alimente le tableau de
    # rendement de la carte du dataset : il vient de la lecture qui vient d'avoir
    # lieu, comme les comptages écrits à côté de lui dans `metadata.json`.
    entrees_lues = {nom: len(lot) for nom, lot in par_corpus.items()}
    entrees = [entree for lot in par_corpus.values() for entree in lot]
    # Un corpus sans aucune entrée arrête la construction. Le laisser passer
    # livrerait un dataset amputé d'une source, et le tableau de rendement
    # afficherait « 0 » sans distinguer « corpus inexploitable » de « corpus non
    # téléchargé » : on échoue plutôt que de livrer un jeu dont on ne sait pas
    # dire d'où vient la composition.
    vides = [nom for nom, lot in par_corpus.items() if not lot]
    if vides:
        raise SystemExit(
            f"Corpus sans aucune entrée : {', '.join(vides)}. "
            "Vérifiez l'accès au Hugging Face Hub avant de reconstruire le dataset."
        )
    cas_corpus = extract_cases(entrees, rng)
    logger.info("  %d cas exploitables extraits sur %d entrées.", len(cas_corpus), len(entrees))
    entonnoirs = {nom: entonnoir(lot) for nom, lot in par_corpus.items()}
    for nom, compte in entonnoirs.items():
        logger.info(
            "  %-14s %6d lues → %5d sans patient, %4d hors bornes, %4d sans signe, "
            "%4d doublons → %4d retenus",
            nom,
            compte["entrees"],
            compte["sans_presentation_de_patient"],
            compte["hors_bornes_de_longueur"],
            compte["sans_signe_identifie"],
            compte["doublons"],
            compte["retenus"],
        )
    cas_corpus = _limiter_les_cas_de_corpus(cas_corpus, args.sft_size, rng)

    # --- 3. Anonymisation RGPD des textes issus des corpus ---
    if args.no_anonymize:
        logger.warning("Anonymisation désactivée (--no-anonymize).")
        entites_masquees = 0
        pii_residuelles: dict[str, int] = {}
    else:
        logger.info("Étape 3/6 — anonymisation RGPD des textes issus des corpus.")
        entites_masquees = 0
        anonymises = []
        for case in cas_corpus:
            description, nombre = analyze_and_anonymize(case.description, case.lang)
            entites_masquees += nombre
            tour = case.user_turn.replace(case.description, description)
            anonymises.append(replace(case, description=description, user_turn=tour))
        cas_corpus = anonymises
        pii_residuelles = audit_corpus([c.description for c in cas_corpus])
        logger.info(
            "  %d entités masquées ; contrôle indépendant après masquage : %s",
            entites_masquees,
            pii_residuelles or "aucune donnée personnelle résiduelle",
        )

    # Ré-examen des étiquettes après masquage. L'étiquette d'un cas de corpus
    # vient de la règle appliquée au texte **avant** anonymisation ; si le
    # masquage retire le signe qui la justifiait, l'exemple enseigne une décision
    # que son propre texte ne soutient plus. On repasse donc la règle sur le
    # texte tel qu'il sera livré, et on écarte ce qui ne tient plus.
    avant_reexamen = len(cas_corpus)
    cas_corpus = [c for c in cas_corpus if classify(c.description) == c.level]
    if avant_reexamen != len(cas_corpus):
        logger.info(
            "  %d cas écartés : le masquage a retiré le signe qui portait leur étiquette.",
            avant_reexamen - len(cas_corpus),
        )

    # Déduplication des cas de corpus, ici et pas plus tard. L'anonymisation
    # vient de rendre identiques des énoncés qui ne l'étaient pas, et deux
    # extractions du même cas ne diffèrent parfois que par une césure — « A 6
    # year old child » contre « A 6-year-old child ». Les écarter maintenant est
    # ce qui permet à l'étape suivante de compter juste : dédupliquer après
    # l'équilibrage retirerait des exemples déjà comptés, et le dataset livré
    # manquerait sa cible de quelques unités.
    avant_dedoublonnage = len(cas_corpus)
    vus: set[str] = set()
    uniques = []
    for case in cas_corpus:
        # L'empreinte porte sur la **description**, pas sur le tour patient.
        # Le tour est la description enveloppée dans l'un des trois gabarits
        # tirés au hasard : deux fois le même cas clinique sous deux gabarits
        # différents donnerait deux empreintes différentes, et le doublon
        # passerait.
        empreinte = empreinte_de_cas(case.description)
        if empreinte in vus:
            continue
        vus.add(empreinte)
        uniques.append(case)
    cas_corpus = uniques
    if avant_dedoublonnage != len(cas_corpus):
        logger.info(
            "  %d quasi-doublons écartés des cas de corpus.",
            avant_dedoublonnage - len(cas_corpus),
        )

    # --- 4. Vignettes cliniques : on complète chaque case de la grille ---
    # Les cas issus des corpus ne sont conservés que lorsqu'un signe est repéré :
    # ils sont donc presque tous urgents et presque tous anglophones. On génère
    # exactement ce qu'il manque dans chacune des six cases (niveau × langue)
    # pour que le dataset livré soit équilibré sur les deux dimensions.
    # Le contrôle de longueur porte ici, sur les cas qui seront livrés : après
    # plafonnement, anonymisation, ré-examen des étiquettes et déduplication.
    # Placé plus haut, il décrirait le vivier d'extraction, dont les deux tiers
    # ne sont jamais livrés.
    longueur_des_descriptions = _verifier_le_budget_de_description(cas_corpus)

    logger.info("Étape 4/6 — génération des vignettes cliniques.")
    vignettes = []
    for niveau, langue, cible in _grille_des_cases(args.sft_size):
        deja = sum(1 for c in cas_corpus if c.level == niveau and c.lang == langue)
        manquant = max(0, cible - deja)
        logger.info("  %-22s %s : corpus %3d, à générer %3d", niveau, langue, deja, manquant)
        vignettes += generate_cases(manquant, niveau, langue, rng, exclude=tours_reserves)

    # --- 5. Assemblage, découpage et paires de préférences ---
    logger.info("Étape 5/6 — assemblage, découpage et paires de préférences.")
    exemples = assemble(vignettes, cas_corpus, tours_reserves, rng)
    logger.info("  %d exemples uniques.", len(exemples))

    decoupages = split_train_val_test(exemples, DATA.val_ratio, DATA.test_ratio, SEED)
    chevauchements = check_no_leakage(decoupages)
    if any(chevauchements.values()):
        raise SystemExit(f"Fuite entre découpages : {chevauchements}")
    tours_entrainement = {e.user_turn for e in decoupages["train"]}
    if tours_entrainement & tours_reserves:
        raise SystemExit("Des cas d'évaluation clinique se retrouvent dans le jeu d'entraînement.")
    logger.info("  découpages disjoints : %s", chevauchements)

    paires = build_preference_pairs(decoupages["train"], args.dpo_size, rng)
    equilibre_longueurs = length_balance(paires)
    logger.info("  équilibre des longueurs des paires : %s", equilibre_longueurs)

    # --- 6. Écriture ---
    logger.info("Étape 6/6 — écriture des fichiers.")
    for nom, items in decoupages.items():
        write_jsonl([sft_record(e) for e in items], sortie / f"sft_{nom}.jsonl")
    write_jsonl([dpo_record(p) for p in paires], sortie / "dpo_train.jsonl")

    meta = metadata_schema()
    meta["construction"] = {
        "graine": SEED,
        "revision_git": revision_git(),
        "date_utc": datetime.now(UTC).strftime("%Y-%m-%d"),
        # La commande réellement passée, options comprises. Écrite en dur, elle
        # annoncerait une construction par défaut alors que `--corpus-scan` ou
        # `--no-anonymize` ont pu produire un tout autre jeu : la carte
        # décrirait une exécution qui n'a pas eu lieu.
        "commande": _commande_executee(args),
    }
    par_source = _repartition(exemples, lambda e: e.source)
    meta["statistiques"] = {
        "sft_total": len(exemples),
        "sft_train": len(decoupages["train"]),
        "sft_validation": len(decoupages["validation"]),
        "sft_test": len(decoupages["test"]),
        "dpo_total": len(paires),
        "evaluation_clinique": len(cas_evaluation),
        "repartition_niveaux": _repartition(exemples, lambda e: e.level),
        "repartition_langues": _repartition(exemples, lambda e: e.lang),
        "repartition_sources": par_source,
        "repartition_confiance": _repartition(exemples, lambda e: e.confiance),
        "part_avec_constantes": round(
            sum(1 for e in exemples if e.constantes) / max(1, len(exemples)), 3
        ),
        "dpo_repartition_strategies": _repartition(paires, lambda p: p.strategie),
        "dpo_equilibre_longueurs": equilibre_longueurs,
        "evaluation_repartition_niveaux": _repartition(cas_evaluation, lambda c: c.level),
        "evaluation_repartition_langues": _repartition(cas_evaluation, lambda c: c.lang),
        "evaluation_pieges": _repartition(
            [c for c in cas_evaluation if c.piege], lambda c: c.piege
        ),
        # Rendement de chaque corpus : ce qui a été lu, et ce qui subsiste dans
        # le jeu livré après extraction, plafonnement, anonymisation, ré-examen
        # des étiquettes et déduplication. C'est ce second chiffre qui compte, et
        # c'est celui que la carte du dataset publie.
        "longueur_des_descriptions": longueur_des_descriptions,
        "longueur_des_reponses": _mesurer_les_reponses(exemples),
        "rendement_corpus": {
            nom: {
                "entrees_lues": lues,
                "cas_retenus": par_source.get(nom, 0),
                # Où les entrées se perdent, étape par étape. Le rendement global
                # ne dit pas si ce qui les écarte tient au corpus ou au filtre.
                "entonnoir": entonnoirs[nom],
            }
            for nom, lues in entrees_lues.items()
        },
    }
    # Ce que le jeu contient au-delà de ses volumes : diversité des réponses
    # attendues, part de restitution dans un découpage aléatoire, et part du
    # niveau lisible sur les seules métadonnées du générateur. Ces trois mesures
    # conditionnent la lecture des résultats et sont reprises telles quelles par
    # le rapport.
    logger.info("  diagnostics du jeu produit.")
    meta["diagnostics"] = {
        "completions": completions_distinctes(exemples),
        "separabilite": separabilite(exemples),
        "fuite_par_metadonnees": fuite_par_metadonnees(exemples, PRESENTATIONS),
    }
    meta["rgpd"] = {
        "principe": "Minimisation : aucune donnée patient réelle. Les vignettes sont synthétiques, "
        "les corpus publics sont des jeux de recherche sans donnée identifiante.",
        "masquage": "Presidio (spaCy français et anglais), limité aux entités réellement identifiantes "
        f"({', '.join(ENTITES_MASQUEES)}).",
        "entites_ecartees": ENTITES_ECARTEES,
        "entites_masquees_nombre": entites_masquees,
        # Sans anonymisation, il n'y a pas eu de contrôle : le dire, plutôt que
        # de laisser lire l'absence de trouvaille comme une absence de donnée
        # personnelle.
        "controle_independant": (
            "non exécuté : anonymisation désactivée (--no-anonymize)"
            if args.no_anonymize
            else pii_residuelles or "aucune donnée personnelle résiduelle détectée"
        ),
        "anonymisation_appliquee": not args.no_anonymize,
        "separation_entrainement_evaluation": chevauchements,
    }
    write_metadata(meta, sortie / "metadata.json")
    _ecrire_le_rendement_dans_la_carte(meta["statistiques"]["rendement_corpus"])

    # La publication refuse de partir s'il manque un de ces six fichiers. Le
    # constater ici, à la construction, évite de le découvrir au moment de
    # publier — et garde la liste et l'écriture d'accord.
    manquants = [nom for nom in FICHIERS_DU_DATASET if not (sortie / nom).exists()]
    if manquants:
        raise SystemExit("Fichiers attendus non écrits : " + ", ".join(manquants))

    print(json.dumps(meta["statistiques"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
