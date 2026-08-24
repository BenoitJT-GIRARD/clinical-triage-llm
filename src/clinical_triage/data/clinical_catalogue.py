"""Catalogue des présentations cliniques servant de vérité terrain au dataset.

Aucun corpus médical public n'est annoté en niveaux de triage. Étiqueter des
textes par simple présence de mots-clés produit des exemples absurdes et, pire,
une évaluation circulaire : le modèle réapprend la règle qui a fabriqué les
étiquettes, et la règle seule obtient alors de meilleurs scores que le modèle.

On part donc dans l'autre sens. Chaque entrée décrit une **présentation type**
rencontrée à l'accueil des urgences, avec son niveau de triage de référence. Le
générateur de vignettes (`case_generator`) habille ensuite cette présentation
d'un âge, d'antécédents, de constantes et d'une formulation ; l'étiquette vient
de la présentation, jamais d'une relecture du texte produit.

Le rattachement aux niveaux suit l'échelle FRENCH utilisée dans les services
d'urgence français : tris 1-2 (prise en charge immédiate) → `URGENCE_VITALE`,
tris 3-4 (délai de quelques heures) → `URGENCE_MODEREE`, tri 5 → `CONSULTATION_DIFFEREE`.

Le catalogue est volontairement écrit à la main et relu cas par cas : le brief
demande de privilégier la qualité des annotations sur le volume. Il couvre
l'adulte, l'enfant, la femme enceinte, la traumatologie, la psychiatrie et les
motifs de médecine générale qui encombrent les urgences.

Limite assumée : ces présentations ont été rédigées par un ingénieur, pas par un
médecin urgentiste. Elles suffisent à un POC ; une validation clinique est un
prérequis à tout usage réel (voir la feuille de route du rapport technique).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Presentation:
    """Présentation type d'un patient et niveau de triage de référence."""

    id: str
    level: str
    # Profil de constantes attendu : "critique", "intermediaire" ou "normal".
    vitals_profile: str
    age_range: tuple[int, int]
    motif_fr: str
    motif_en: str
    signes_fr: tuple[str, ...]
    signes_en: tuple[str, ...]
    antecedents_fr: tuple[str, ...]
    antecedents_en: tuple[str, ...]
    # Échelle de temps d'installation : "minutes", "heures", "jours", "semaines".
    delai: str
    # Justification et recommandation sont toujours en français : c'est la langue
    # de réponse imposée à l'agent, quelle que soit la langue de la description.
    justification: str
    recommandation: str


PRESENTATIONS: tuple[Presentation, ...] = (
    # ------------------------------------------------------------------
    # URGENCE VITALE — prise en charge immédiate
    # ------------------------------------------------------------------
    Presentation(
        id="syndrome_coronarien_aigu",
        level="URGENCE_VITALE",
        vitals_profile="critique",
        age_range=(45, 88),
        motif_fr="douleur thoracique constrictive",
        motif_en="crushing chest pain",
        signes_fr=(
            "irradiation dans le bras gauche",
            "sueurs profuses",
            "nausées",
            "angoisse de mort",
        ),
        signes_en=(
            "radiating to the left arm",
            "profuse sweating",
            "nausea",
            "sense of impending doom",
        ),
        antecedents_fr=(
            "hypertension artérielle",
            "tabagisme actif",
            "diabète de type 2",
            "hypercholestérolémie",
        ),
        antecedents_en=("hypertension", "current smoker", "type 2 diabetes", "high cholesterol"),
        delai="minutes",
        justification=(
            "Douleur thoracique constrictive avec irradiation brachiale et sueurs : tableau "
            "évocateur d'un syndrome coronarien aigu, dont le pronostic dépend du délai de "
            "reperfusion."
        ),
        recommandation=(
            "Prise en charge immédiate en salle de déchocage, électrocardiogramme dans les dix "
            "minutes, appel du cardiologue et du 15 (SAMU) si la prise en charge ne peut être "
            "immédiate."
        ),
    ),
    Presentation(
        id="avc_deficit_focal",
        level="URGENCE_VITALE",
        vitals_profile="intermediaire",
        age_range=(55, 92),
        motif_fr="déficit moteur brutal d'un hémicorps",
        motif_en="sudden weakness on one side of the body",
        signes_fr=(
            "troubles de la parole",
            "déviation de la bouche",
            "perte d'équilibre",
            "vision double",
        ),
        signes_en=("slurred speech", "facial droop", "loss of balance", "double vision"),
        antecedents_fr=(
            "fibrillation auriculaire",
            "hypertension artérielle",
            "accident ischémique transitoire",
        ),
        antecedents_en=(
            "atrial fibrillation",
            "hypertension",
            "previous transient ischaemic attack",
        ),
        delai="minutes",
        justification=(
            "Déficit neurologique focal d'installation brutale : suspicion d'accident vasculaire "
            "cérébral, avec une fenêtre thérapeutique de thrombolyse très courte."
        ),
        recommandation=(
            "Alerte thrombolyse immédiate, imagerie cérébrale en urgence, contact de l'unité "
            "neurovasculaire et du 15 (SAMU). Noter précisément l'heure de début des symptômes."
        ),
    ),
    Presentation(
        id="asthme_aigu_grave",
        level="URGENCE_VITALE",
        vitals_profile="critique",
        age_range=(8, 70),
        motif_fr="crise d'asthme avec difficulté à terminer ses phrases",
        motif_en="asthma attack, unable to finish sentences",
        signes_fr=(
            "tirage intercostal",
            "sifflements audibles",
            "cyanose des lèvres",
            "position assise penchée en avant",
        ),
        signes_en=(
            "intercostal retractions",
            "audible wheezing",
            "blue lips",
            "sitting forward to breathe",
        ),
        antecedents_fr=(
            "asthme sévère",
            "hospitalisation antérieure en réanimation",
            "allergie aux acariens",
        ),
        antecedents_en=("severe asthma", "previous intensive care admission", "dust mite allergy"),
        delai="heures",
        justification=(
            "L'impossibilité de terminer une phrase et le tirage signent une crise d'asthme aiguë "
            "grave, avec risque d'épuisement respiratoire à court terme."
        ),
        recommandation=(
            "Oxygénothérapie et bronchodilatateurs nébulisés sans délai, corticothérapie "
            "systémique, surveillance continue et appel du 15 (SAMU) en cas d'aggravation."
        ),
    ),
    Presentation(
        id="choc_anaphylactique",
        level="URGENCE_VITALE",
        vitals_profile="critique",
        age_range=(3, 75),
        motif_fr="malaise avec gonflement du visage après une piqûre d'insecte",
        motif_en="collapse with facial swelling after an insect sting",
        signes_fr=(
            "urticaire généralisée",
            "gêne à la déglutition",
            "voix rauque",
            "chute de la tension",
        ),
        signes_en=(
            "generalised hives",
            "difficulty swallowing",
            "hoarse voice",
            "dropping blood pressure",
        ),
        antecedents_fr=(
            "allergie connue aux hyménoptères",
            "asthme",
            "porteur d'un stylo d'adrénaline",
        ),
        antecedents_en=(
            "known wasp venom allergy",
            "asthma",
            "carries an adrenaline auto-injector",
        ),
        delai="minutes",
        justification=(
            "Atteinte cutanée, respiratoire et hémodynamique après exposition à un allergène : "
            "choc anaphylactique, dont le traitement ne souffre aucun délai."
        ),
        recommandation=(
            "Adrénaline intramusculaire immédiate, position allongée jambes surélevées, "
            "oxygénothérapie et appel du 15 (SAMU). Surveillance prolongée du fait du risque de "
            "réaction biphasique."
        ),
    ),
    Presentation(
        id="hemorragie_digestive_haute",
        level="URGENCE_VITALE",
        vitals_profile="critique",
        age_range=(40, 90),
        motif_fr="vomissements de sang rouge",
        motif_en="vomiting bright red blood",
        signes_fr=("pâleur marquée", "selles noires", "vertiges en se levant", "soif intense"),
        signes_en=("marked pallor", "black stools", "dizziness on standing", "intense thirst"),
        antecedents_fr=(
            "cirrhose",
            "ulcère gastroduodénal",
            "traitement anti-inflammatoire au long cours",
        ),
        antecedents_en=("liver cirrhosis", "peptic ulcer", "long-term anti-inflammatory treatment"),
        delai="heures",
        justification=(
            "Hématémèse avec signes de mauvaise tolérance : hémorragie digestive haute active, "
            "avec risque de choc hémorragique."
        ),
        recommandation=(
            "Deux voies veineuses de gros calibre, bilan avec groupage et commande de culots "
            "globulaires, avis gastro-entérologique urgent pour endoscopie."
        ),
    ),
    Presentation(
        id="sepsis_grave",
        level="URGENCE_VITALE",
        vitals_profile="critique",
        age_range=(35, 92),
        motif_fr="fièvre avec frissons et confusion",
        motif_en="fever with shivering and confusion",
        signes_fr=(
            "marbrures des genoux",
            "extrémités froides",
            "somnolence",
            "diminution des urines",
        ),
        signes_en=("mottled knees", "cold extremities", "drowsiness", "reduced urine output"),
        antecedents_fr=("immunodépression", "chimiothérapie en cours", "sonde urinaire à demeure"),
        antecedents_en=("immunosuppression", "ongoing chemotherapy", "indwelling urinary catheter"),
        delai="heures",
        justification=(
            "Fièvre associée à des signes d'hypoperfusion et à une altération de la vigilance : "
            "sepsis avec défaillance d'organe, dont la mortalité croît de façon horaire."
        ),
        recommandation=(
            "Hémocultures puis antibiothérapie probabiliste dans l'heure, remplissage vasculaire, "
            "mesure du lactate et avis réanimateur immédiat."
        ),
    ),
    Presentation(
        id="syndrome_meninge",
        level="URGENCE_VITALE",
        vitals_profile="intermediaire",
        age_range=(1, 45),
        motif_fr="fièvre élevée avec raideur de la nuque",
        motif_en="high fever with neck stiffness",
        signes_fr=(
            "céphalées intenses",
            "gêne à la lumière",
            "somnolence",
            "taches violacées sur la peau",
        ),
        signes_en=(
            "severe headache",
            "discomfort in bright light",
            "drowsiness",
            "purple skin blotches",
        ),
        antecedents_fr=("vaccination incomplète", "otite récente", "vie en collectivité"),
        antecedents_en=(
            "incomplete vaccination",
            "recent ear infection",
            "lives in shared accommodation",
        ),
        delai="heures",
        justification=(
            "Association fièvre, raideur de nuque et trouble de la vigilance : syndrome méningé "
            "fébrile, avec suspicion de purpura fulminans en présence de lésions cutanées."
        ),
        recommandation=(
            "Isolement gouttelettes, antibiothérapie sans attendre la ponction lombaire en cas de "
            "purpura, avis infectiologique et réanimateur immédiat, appel du 15 (SAMU)."
        ),
    ),
    Presentation(
        id="traumatisme_cranien_grave",
        level="URGENCE_VITALE",
        vitals_profile="critique",
        age_range=(15, 85),
        motif_fr="chute d'une hauteur avec perte de connaissance",
        motif_en="fall from height with loss of consciousness",
        signes_fr=(
            "vomissements répétés",
            "confusion persistante",
            "plaie du cuir chevelu",
            "amnésie des faits",
        ),
        signes_en=(
            "repeated vomiting",
            "persistent confusion",
            "scalp wound",
            "no memory of the fall",
        ),
        antecedents_fr=("traitement anticoagulant", "consommation d'alcool", "chutes à répétition"),
        antecedents_en=("anticoagulant therapy", "alcohol use", "recurrent falls"),
        delai="heures",
        justification=(
            "Traumatisme crânien avec perte de connaissance, vomissements et confusion : risque "
            "d'hématome intracrânien, majoré par le traitement anticoagulant."
        ),
        recommandation=(
            "Immobilisation du rachis cervical, scanner cérébral en urgence, surveillance "
            "neurologique rapprochée et avis neurochirurgical."
        ),
    ),
    Presentation(
        id="crise_convulsive_prolongee",
        level="URGENCE_VITALE",
        vitals_profile="critique",
        age_range=(2, 70),
        motif_fr="convulsions qui se prolongent au-delà de cinq minutes",
        motif_en="seizures lasting more than five minutes",
        signes_fr=(
            "morsure de langue",
            "perte d'urines",
            "respiration bruyante",
            "absence de reprise de conscience",
        ),
        signes_en=(
            "tongue biting",
            "incontinence",
            "noisy breathing",
            "not regaining consciousness",
        ),
        antecedents_fr=("épilepsie connue", "arrêt récent du traitement", "sevrage alcoolique"),
        antecedents_en=("known epilepsy", "recently stopped medication", "alcohol withdrawal"),
        delai="minutes",
        justification=(
            "Crise convulsive prolongée sans reprise de conscience : état de mal épileptique, avec "
            "risque de souffrance cérébrale et d'atteinte des voies aériennes."
        ),
        recommandation=(
            "Protection des voies aériennes, oxygénothérapie, benzodiazépine sans délai, "
            "glycémie capillaire et appel du 15 (SAMU)."
        ),
    ),
    Presentation(
        id="intoxication_medicamenteuse_volontaire",
        level="URGENCE_VITALE",
        vitals_profile="critique",
        age_range=(14, 60),
        motif_fr="ingestion volontaire d'une boîte de médicaments",
        motif_en="deliberate ingestion of a box of tablets",
        signes_fr=(
            "somnolence croissante",
            "vomissements",
            "propos incohérents",
            "ralentissement respiratoire",
        ),
        signes_en=("increasing drowsiness", "vomiting", "incoherent speech", "slow breathing"),
        antecedents_fr=("dépression", "tentative de suicide antérieure", "rupture récente"),
        antecedents_en=("depression", "previous suicide attempt", "recent break-up"),
        delai="heures",
        justification=(
            "Intoxication médicamenteuse volontaire récente avec retentissement neurologique : "
            "risque toxique immédiat et risque suicidaire élevé."
        ),
        recommandation=(
            "Prise en charge somatique immédiate, contact du centre antipoison, surveillance "
            "continue et évaluation psychiatrique dès la stabilisation. Ne pas laisser le patient seul."
        ),
    ),
    Presentation(
        id="pre_eclampsie_severe",
        level="URGENCE_VITALE",
        vitals_profile="intermediaire",
        age_range=(18, 44),
        motif_fr="céphalées intenses au troisième trimestre de grossesse",
        motif_en="severe headache in the third trimester of pregnancy",
        signes_fr=(
            "mouches devant les yeux",
            "douleur en barre sous les côtes",
            "œdèmes du visage",
            "prise de poids rapide",
        ),
        signes_en=(
            "visual floaters",
            "band-like pain under the ribs",
            "facial swelling",
            "rapid weight gain",
        ),
        antecedents_fr=("première grossesse", "hypertension gravidique", "grossesse gémellaire"),
        antecedents_en=("first pregnancy", "pregnancy-induced hypertension", "twin pregnancy"),
        delai="heures",
        justification=(
            "Céphalées, troubles visuels et douleur épigastrique en fin de grossesse : "
            "pré-éclampsie sévère, avec risque d'éclampsie et de souffrance fœtale."
        ),
        recommandation=(
            "Transfert immédiat en maternité de niveau adapté, mesure de la tension et de la "
            "protéinurie, surveillance du rythme cardiaque fœtal, avis obstétrical sans délai."
        ),
    ),
    Presentation(
        id="rupture_anevrisme_aorte",
        level="URGENCE_VITALE",
        vitals_profile="critique",
        age_range=(60, 90),
        motif_fr="douleur abdominale brutale avec malaise",
        motif_en="sudden abdominal pain with collapse",
        signes_fr=(
            "masse abdominale battante",
            "douleur irradiant dans le dos",
            "pâleur",
            "extrémités froides",
        ),
        signes_en=(
            "pulsatile abdominal mass",
            "pain radiating to the back",
            "pallor",
            "cold extremities",
        ),
        antecedents_fr=(
            "anévrisme de l'aorte connu",
            "tabagisme ancien",
            "artérite des membres inférieurs",
        ),
        antecedents_en=("known aortic aneurysm", "former smoker", "peripheral artery disease"),
        delai="minutes",
        justification=(
            "Douleur abdominale brutale, masse battante et mauvaise tolérance hémodynamique : "
            "rupture d'anévrisme de l'aorte abdominale jusqu'à preuve du contraire."
        ),
        recommandation=(
            "Urgence chirurgicale vitale : alerter le bloc et le chirurgien vasculaire, transfusion "
            "anticipée, appel du 15 (SAMU) sans délai."
        ),
    ),
    Presentation(
        id="embolie_pulmonaire",
        level="URGENCE_VITALE",
        vitals_profile="critique",
        age_range=(30, 85),
        motif_fr="essoufflement brutal avec douleur au côté",
        motif_en="sudden breathlessness with side pain",
        signes_fr=(
            "douleur au mollet",
            "accélération du pouls",
            "malaise à l'effort",
            "crachats sanglants",
        ),
        signes_en=("calf pain", "racing pulse", "collapse on exertion", "coughing up blood"),
        antecedents_fr=(
            "chirurgie récente",
            "immobilisation prolongée",
            "contraception œstroprogestative",
            "cancer en cours de traitement",
        ),
        antecedents_en=(
            "recent surgery",
            "prolonged immobilisation",
            "combined oral contraception",
            "cancer under treatment",
        ),
        delai="heures",
        justification=(
            "Dyspnée brutale avec douleur pleurale et facteur de risque thromboembolique : "
            "suspicion d'embolie pulmonaire, potentiellement grave d'emblée."
        ),
        recommandation=(
            "Oxygénothérapie, score de probabilité clinique, angioscanner en urgence et "
            "anticoagulation dès la suspicion forte, après avis médical."
        ),
    ),
    Presentation(
        id="acidocetose_diabetique",
        level="URGENCE_VITALE",
        vitals_profile="critique",
        age_range=(10, 60),
        motif_fr="soif intense avec respiration rapide et profonde",
        motif_en="intense thirst with deep rapid breathing",
        signes_fr=(
            "haleine fruitée",
            "douleurs abdominales",
            "somnolence",
            "urines très abondantes",
        ),
        signes_en=(
            "fruity breath",
            "abdominal pain",
            "drowsiness",
            "passing large amounts of urine",
        ),
        antecedents_fr=(
            "diabète de type 1",
            "arrêt des injections d'insuline",
            "infection récente",
        ),
        antecedents_en=("type 1 diabetes", "stopped insulin injections", "recent infection"),
        delai="heures",
        justification=(
            "Syndrome polyuro-polydipsique avec polypnée et somnolence chez un diabétique : "
            "acidocétose diabétique, urgence métabolique."
        ),
        recommandation=(
            "Glycémie et cétonémie capillaires immédiates, réhydratation intraveineuse, insuline "
            "en continu et surveillance du potassium en unité de soins continus."
        ),
    ),
    Presentation(
        id="hypoglycemie_severe",
        level="URGENCE_VITALE",
        vitals_profile="critique",
        age_range=(20, 88),
        motif_fr="malaise avec sueurs et propos incohérents",
        motif_en="collapse with sweating and confused speech",
        signes_fr=(
            "tremblements",
            "pâleur",
            "agressivité inhabituelle",
            "difficulté à se réveiller",
        ),
        signes_en=("tremor", "pallor", "unusual aggression", "difficult to rouse"),
        antecedents_fr=("diabète traité par insuline", "repas sauté", "insuffisance rénale"),
        antecedents_en=("insulin-treated diabetes", "missed meal", "kidney failure"),
        delai="minutes",
        justification=(
            "Trouble de la conscience avec signes adrénergiques chez un patient sous insuline : "
            "hypoglycémie sévère, réversible mais rapidement délétère pour le cerveau."
        ),
        recommandation=(
            "Glycémie capillaire immédiate, resucrage par voie intraveineuse si trouble de la "
            "conscience, surveillance jusqu'à normalisation puis recherche de la cause."
        ),
    ),
    Presentation(
        id="occlusion_intestinale",
        level="URGENCE_VITALE",
        vitals_profile="intermediaire",
        age_range=(40, 88),
        motif_fr="arrêt des gaz et des selles avec ventre distendu",
        motif_en="no bowel movements or wind with a distended abdomen",
        signes_fr=(
            "vomissements fécaloïdes",
            "douleurs en crampes",
            "ventre tendu",
            "absence de bruits intestinaux",
        ),
        signes_en=("faeculent vomiting", "cramping pain", "rigid abdomen", "absent bowel sounds"),
        antecedents_fr=("chirurgie abdominale antérieure", "hernie non opérée", "cancer colique"),
        antecedents_en=("previous abdominal surgery", "untreated hernia", "colon cancer"),
        delai="heures",
        justification=(
            "Arrêt du transit avec distension et vomissements : occlusion intestinale, avec risque "
            "d'ischémie digestive et de perforation."
        ),
        recommandation=(
            "À jeun strict, sonde nasogastrique en aspiration, correction hydroélectrolytique, "
            "scanner abdominal et avis chirurgical urgent."
        ),
    ),
    Presentation(
        id="brulure_etendue",
        level="URGENCE_VITALE",
        vitals_profile="critique",
        age_range=(5, 70),
        motif_fr="brûlure étendue du tronc et des bras",
        motif_en="extensive burns to the torso and arms",
        signes_fr=(
            "peau cartonnée par endroits",
            "suies autour du nez",
            "voix modifiée",
            "douleur majeure",
        ),
        signes_en=(
            "leathery skin in places",
            "soot around the nose",
            "changed voice",
            "severe pain",
        ),
        antecedents_fr=("accident domestique", "incendie en espace clos", "épilepsie"),
        antecedents_en=("domestic accident", "fire in a closed space", "epilepsy"),
        delai="minutes",
        justification=(
            "Brûlure étendue avec signes d'inhalation de fumées : risque d'obstruction des voies "
            "aériennes et de choc hypovolémique dans les heures qui suivent."
        ),
        recommandation=(
            "Refroidissement puis couverture stérile, oxygénothérapie à haut débit, remplissage "
            "vasculaire et transfert vers un centre de traitement des brûlés via le 15 (SAMU)."
        ),
    ),
    Presentation(
        id="hemorragie_du_post_partum",
        level="URGENCE_VITALE",
        vitals_profile="critique",
        age_range=(18, 44),
        motif_fr="saignement abondant après un accouchement récent",
        motif_en="heavy bleeding after a recent delivery",
        signes_fr=("caillots volumineux", "vertiges", "pâleur", "pouls filant"),
        signes_en=("large clots", "dizziness", "pallor", "thready pulse"),
        antecedents_fr=(
            "accouchement il y a moins de 48 heures",
            "grossesse gémellaire",
            "césarienne",
        ),
        antecedents_en=("delivery within the last 48 hours", "twin pregnancy", "caesarean section"),
        delai="heures",
        justification=(
            "Saignement abondant du post-partum avec retentissement hémodynamique : hémorragie de "
            "la délivrance, première cause de mortalité maternelle évitable."
        ),
        recommandation=(
            "Appel immédiat de l'équipe obstétricale et anesthésique, deux voies veineuses, "
            "utérotoniques, commande de produits sanguins et transfert en salle de naissance."
        ),
    ),
    Presentation(
        id="bronchiolite_grave_nourrisson",
        level="URGENCE_VITALE",
        vitals_profile="critique",
        age_range=(0, 1),
        motif_fr="nourrisson qui respire vite et refuse de boire",
        motif_en="infant breathing fast and refusing to feed",
        signes_fr=(
            "battement des ailes du nez",
            "creusement entre les côtes",
            "pauses respiratoires",
            "lèvres bleutées",
        ),
        signes_en=("nasal flaring", "chest indrawing", "pauses in breathing", "bluish lips"),
        antecedents_fr=(
            "prématurité",
            "épidémie de bronchiolite en cours",
            "frère ou sœur enrhumé",
        ),
        antecedents_en=("premature birth", "ongoing bronchiolitis season", "sibling with a cold"),
        delai="jours",
        justification=(
            "Détresse respiratoire du nourrisson avec signes de lutte, pauses respiratoires et "
            "refus alimentaire : bronchiolite grave, à risque d'épuisement rapide."
        ),
        recommandation=(
            "Oxygénothérapie, désobstruction rhinopharyngée, monitorage continu et hospitalisation "
            "en unité pédiatrique ; appel du 15 (SAMU) en cas d'apnée."
        ),
    ),
    Presentation(
        id="plaie_penetrante_thoracique",
        level="URGENCE_VITALE",
        vitals_profile="critique",
        age_range=(16, 60),
        motif_fr="plaie par arme blanche au thorax",
        motif_en="stab wound to the chest",
        signes_fr=(
            "difficulté à respirer",
            "plaie qui aspire l'air",
            "agitation",
            "pâleur extrême",
        ),
        signes_en=("difficulty breathing", "sucking chest wound", "agitation", "extreme pallor"),
        antecedents_fr=("agression", "consommation d'alcool", "aucun antécédent connu"),
        antecedents_en=("assault", "alcohol use", "no known medical history"),
        delai="minutes",
        justification=(
            "Plaie pénétrante thoracique avec détresse respiratoire : suspicion de pneumothorax "
            "compressif ou d'hémothorax, pronostic vital engagé en quelques minutes."
        ),
        recommandation=(
            "Pansement occlusif trois côtés, oxygénothérapie, voies veineuses, alerte "
            "chirurgicale et déchocage immédiat. Signalement judiciaire à prévoir."
        ),
    ),
    Presentation(
        id="deshydratation_severe_nourrisson",
        level="URGENCE_VITALE",
        vitals_profile="critique",
        age_range=(0, 2),
        motif_fr="diarrhée profuse chez un nourrisson devenu somnolent",
        motif_en="profuse diarrhoea in an infant who has become drowsy",
        signes_fr=(
            "yeux creux",
            "pli cutané persistant",
            "couches sèches depuis huit heures",
            "pleurs sans larmes",
        ),
        signes_en=(
            "sunken eyes",
            "skin tenting",
            "dry nappies for eight hours",
            "crying without tears",
        ),
        antecedents_fr=(
            "gastro-entérite en crèche",
            "vomissements associés",
            "faible poids de naissance",
        ),
        antecedents_en=("gastroenteritis at nursery", "associated vomiting", "low birth weight"),
        delai="jours",
        justification=(
            "Déshydratation sévère du nourrisson avec somnolence et anurie : risque de collapsus, "
            "la tolérance de l'enfant à la perte hydrique étant très faible."
        ),
        recommandation=(
            "Voie veineuse et réhydratation intraveineuse immédiate, pesée, ionogramme et "
            "hospitalisation en pédiatrie."
        ),
    ),
    Presentation(
        id="crise_suicidaire_aigue",
        level="URGENCE_VITALE",
        vitals_profile="normal",
        age_range=(14, 70),
        motif_fr="idées suicidaires avec un scénario précis et des moyens à disposition",
        motif_en="suicidal thoughts with a specific plan and means at hand",
        signes_fr=("propos d'adieu", "isolement récent", "refus de soins", "agitation"),
        signes_en=("saying goodbye", "recent withdrawal", "refusing care", "agitation"),
        antecedents_fr=("trouble dépressif", "tentative récente", "consommation d'alcool"),
        antecedents_en=("depressive disorder", "recent attempt", "alcohol use"),
        delai="heures",
        justification=(
            "Crise suicidaire avec scénario construit et moyens accessibles : le risque de passage "
            "à l'acte est immédiat, même en l'absence de toute anomalie des constantes."
        ),
        recommandation=(
            "Ne jamais laisser le patient seul, retirer les moyens létaux, évaluation "
            "psychiatrique en urgence et hospitalisation si le risque persiste."
        ),
    ),
    # ------------------------------------------------------------------
    # URGENCE MODEREE — prise en charge sous quelques heures
    # ------------------------------------------------------------------
    Presentation(
        id="suspicion_appendicite",
        level="URGENCE_MODEREE",
        vitals_profile="intermediaire",
        age_range=(6, 45),
        motif_fr="douleur de la fosse iliaque droite",
        motif_en="pain in the lower right abdomen",
        signes_fr=("nausées", "perte d'appétit", "douleur à la décompression", "fièvre modérée"),
        signes_en=("nausea", "loss of appetite", "rebound tenderness", "moderate fever"),
        antecedents_fr=(
            "aucun antécédent chirurgical",
            "épisodes douloureux similaires",
            "constipation récente",
        ),
        antecedents_en=("no previous surgery", "similar pain episodes", "recent constipation"),
        delai="heures",
        justification=(
            "Douleur de la fosse iliaque droite avec signes d'irritation péritonéale et fièvre : "
            "suspicion d'appendicite aiguë, à confirmer avant qu'elle ne se complique."
        ),
        recommandation=(
            "Patient à jeun, antalgie, bilan biologique et imagerie abdominale, avis chirurgical "
            "dans les heures qui suivent. Réévaluer immédiatement en cas de défense généralisée."
        ),
    ),
    Presentation(
        id="colique_nephretique",
        level="URGENCE_MODEREE",
        vitals_profile="intermediaire",
        age_range=(20, 70),
        motif_fr="douleur lombaire en coup de poignard irradiant vers l'aine",
        motif_en="stabbing flank pain radiating to the groin",
        signes_fr=(
            "agitation permanente",
            "nausées",
            "envies fréquentes d'uriner",
            "sang dans les urines",
        ),
        signes_en=(
            "constant restlessness",
            "nausea",
            "frequent urge to urinate",
            "blood in the urine",
        ),
        antecedents_fr=(
            "calculs rénaux connus",
            "faible consommation d'eau",
            "épisode identique il y a deux ans",
        ),
        antecedents_en=(
            "known kidney stones",
            "low fluid intake",
            "identical episode two years ago",
        ),
        delai="heures",
        justification=(
            "Douleur lombaire paroxystique irradiant vers l'aine avec hématurie : colique "
            "néphrétique, très douloureuse mais sans détresse vitale en l'absence de fièvre."
        ),
        recommandation=(
            "Antalgie par anti-inflammatoire en l'absence de contre-indication, bandelette "
            "urinaire, imagerie et réévaluation. Retour immédiat en cas de fièvre ou d'anurie."
        ),
    ),
    Presentation(
        id="fracture_poignet_deplacee",
        level="URGENCE_MODEREE",
        vitals_profile="intermediaire",
        age_range=(8, 85),
        motif_fr="poignet déformé après une chute sur la main",
        motif_en="deformed wrist after falling onto the hand",
        signes_fr=(
            "gonflement important",
            "impossibilité de bouger les doigts sans douleur",
            "hématome",
            "doigts bien colorés",
        ),
        signes_en=(
            "marked swelling",
            "cannot move fingers without pain",
            "bruising",
            "fingers well perfused",
        ),
        antecedents_fr=("ostéoporose", "chute de sa hauteur", "pratique sportive"),
        antecedents_en=("osteoporosis", "fall from standing height", "plays sport"),
        delai="heures",
        justification=(
            "Déformation et impotence fonctionnelle après un traumatisme direct : fracture du "
            "poignet probable, sans atteinte vasculo-nerveuse immédiate."
        ),
        recommandation=(
            "Immobilisation par attelle, antalgie, radiographie puis avis orthopédique pour "
            "réduction. Surveiller la coloration et la sensibilité des doigts."
        ),
    ),
    Presentation(
        id="pyelonephrite",
        level="URGENCE_MODEREE",
        vitals_profile="intermediaire",
        age_range=(16, 75),
        motif_fr="fièvre avec douleur du dos et brûlures urinaires",
        motif_en="fever with back pain and burning on urination",
        signes_fr=("frissons", "urines troubles", "douleur à la percussion lombaire", "nausées"),
        signes_en=("shivering", "cloudy urine", "tenderness over the kidney", "nausea"),
        antecedents_fr=("infections urinaires à répétition", "grossesse en cours", "diabète"),
        antecedents_en=("recurrent urinary infections", "current pregnancy", "diabetes"),
        delai="jours",
        justification=(
            "Fièvre associée à une douleur lombaire et à des signes urinaires : pyélonéphrite "
            "aiguë, qui impose une antibiothérapie rapide sans être d'emblée une détresse vitale."
        ),
        recommandation=(
            "Bandelette et examen cytobactériologique des urines, hémocultures si frissons, "
            "antibiothérapie adaptée dans les heures qui suivent, hospitalisation si grossesse."
        ),
    ),
    Presentation(
        id="crise_asthme_moderee",
        level="URGENCE_MODEREE",
        vitals_profile="intermediaire",
        age_range=(5, 65),
        motif_fr="gêne respiratoire avec sifflements, phrases complètes possibles",
        motif_en="wheezy breathing, still able to speak in full sentences",
        signes_fr=(
            "toux sèche",
            "oppression thoracique",
            "amélioration partielle après inhalateur",
            "pas de cyanose",
        ),
        signes_en=("dry cough", "chest tightness", "partial relief after inhaler", "no cyanosis"),
        antecedents_fr=("asthme d'effort", "rhinite allergique", "infection virale en cours"),
        antecedents_en=("exercise-induced asthma", "allergic rhinitis", "current viral infection"),
        delai="heures",
        justification=(
            "Crise d'asthme avec parole conservée et réponse partielle au bronchodilatateur : "
            "exacerbation modérée, à traiter et surveiller sans urgence vitale immédiate."
        ),
        recommandation=(
            "Bronchodilatateurs inhalés répétés, corticothérapie orale courte, réévaluation à "
            "une heure. Consultation immédiate si la parole devient difficile."
        ),
    ),
    Presentation(
        id="plaie_profonde_suturable",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(4, 80),
        motif_fr="plaie profonde de l'avant-bras nécessitant des points",
        motif_en="deep forearm wound needing stitches",
        signes_fr=(
            "saignement contrôlé par compression",
            "berges nettes",
            "mobilité conservée",
            "sensibilité normale",
        ),
        signes_en=(
            "bleeding controlled by pressure",
            "clean wound edges",
            "movement preserved",
            "normal sensation",
        ),
        antecedents_fr=(
            "vaccination antitétanique à jour",
            "accident de bricolage",
            "aucun traitement",
        ),
        antecedents_en=("tetanus vaccination up to date", "DIY accident", "no medication"),
        delai="heures",
        justification=(
            "Plaie profonde mais sans saignement actif ni atteinte tendineuse ou nerveuse : "
            "suture nécessaire dans les six heures pour limiter le risque infectieux."
        ),
        recommandation=(
            "Lavage abondant, exploration de la plaie, suture et vérification du statut "
            "antitétanique. Consignes de surveillance de l'infection à la sortie."
        ),
    ),
    Presentation(
        id="entorse_cheville_grave",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(10, 60),
        motif_fr="cheville très douloureuse après une torsion, appui impossible",
        motif_en="very painful ankle after twisting it, unable to bear weight",
        signes_fr=(
            "gonflement immédiat",
            "hématome sous la malléole",
            "douleur à la palpation osseuse",
            "pied bien coloré",
        ),
        signes_en=(
            "immediate swelling",
            "bruising below the ankle bone",
            "bony tenderness",
            "foot well perfused",
        ),
        antecedents_fr=("entorses répétées", "sport de pivot", "surpoids"),
        antecedents_en=("repeated sprains", "pivoting sport", "overweight"),
        delai="heures",
        justification=(
            "Impossibilité d'appui avec douleur osseuse après torsion : les critères d'Ottawa "
            "imposent une radiographie pour éliminer une fracture."
        ),
        recommandation=(
            "Glace, immobilisation et surélévation, antalgie, radiographie de cheville puis "
            "orientation vers une consultation orthopédique."
        ),
    ),
    Presentation(
        id="gastro_enterite_deshydratation_moderee",
        level="URGENCE_MODEREE",
        vitals_profile="intermediaire",
        age_range=(1, 80),
        motif_fr="diarrhées et vomissements depuis deux jours avec fatigue",
        motif_en="diarrhoea and vomiting for two days with fatigue",
        signes_fr=(
            "bouche sèche",
            "urines rares et foncées",
            "douleurs abdominales diffuses",
            "vigilance conservée",
        ),
        signes_en=(
            "dry mouth",
            "scant dark urine",
            "diffuse abdominal pain",
            "alert and responsive",
        ),
        antecedents_fr=("repas suspect", "épidémie familiale", "traitement diurétique"),
        antecedents_en=("suspect meal", "family outbreak", "diuretic treatment"),
        delai="jours",
        justification=(
            "Pertes digestives prolongées avec signes de déshydratation modérée mais vigilance "
            "normale : réhydratation nécessaire dans les heures qui suivent."
        ),
        recommandation=(
            "Réhydratation orale fractionnée, ionogramme si terrain fragile, surveillance du "
            "poids et des urines. Retour immédiat en cas de somnolence ou de sang dans les selles."
        ),
    ),
    Presentation(
        id="migraine_severe",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(15, 60),
        motif_fr="céphalée pulsatile invalidante avec nausées",
        motif_en="disabling throbbing headache with nausea",
        signes_fr=(
            "gêne à la lumière et au bruit",
            "aura visuelle régressive",
            "examen neurologique normal",
            "épisodes identiques connus",
        ),
        signes_en=(
            "light and noise sensitivity",
            "resolving visual aura",
            "normal neurological examination",
            "known identical episodes",
        ),
        antecedents_fr=(
            "migraine avec aura",
            "antécédents familiaux",
            "traitement de crise inefficace aujourd'hui",
        ),
        antecedents_en=(
            "migraine with aura",
            "family history",
            "usual treatment ineffective today",
        ),
        delai="heures",
        justification=(
            "Céphalée typique d'une migraine déjà connue, avec examen neurologique normal : pas de "
            "critère de gravité, mais douleur invalidante justifiant une prise en charge rapide."
        ),
        recommandation=(
            "Antalgie adaptée au repos dans le calme et l'obscurité, antiémétique si besoin. "
            "Imagerie uniquement si la céphalée change de caractère ou s'installe brutalement."
        ),
    ),
    Presentation(
        id="fibrillation_auriculaire_toleree",
        level="URGENCE_MODEREE",
        vitals_profile="intermediaire",
        age_range=(50, 88),
        motif_fr="palpitations irrégulières depuis ce matin",
        motif_en="irregular palpitations since this morning",
        signes_fr=(
            "fatigue à l'effort",
            "pas de douleur thoracique",
            "tension conservée",
            "pouls irrégulier",
        ),
        signes_en=(
            "tiredness on exertion",
            "no chest pain",
            "blood pressure maintained",
            "irregular pulse",
        ),
        antecedents_fr=(
            "hypertension artérielle",
            "apnées du sommeil",
            "consommation récente d'alcool",
        ),
        antecedents_en=("hypertension", "sleep apnoea", "recent alcohol intake"),
        delai="heures",
        justification=(
            "Arythmie bien tolérée sur le plan hémodynamique, sans douleur thoracique ni signe "
            "d'insuffisance cardiaque : évaluation cardiologique nécessaire mais non immédiate."
        ),
        recommandation=(
            "Électrocardiogramme, ionogramme et bilan thyroïdien, évaluation du risque "
            "thromboembolique et avis cardiologique dans la journée."
        ),
    ),
    Presentation(
        id="zona_ophtalmique",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(45, 88),
        motif_fr="éruption douloureuse sur le front et la paupière",
        motif_en="painful rash on the forehead and eyelid",
        signes_fr=(
            "vésicules groupées d'un seul côté",
            "œil rouge",
            "larmoiement",
            "douleur à type de brûlure",
        ),
        signes_en=("clustered blisters on one side", "red eye", "watery eye", "burning pain"),
        antecedents_fr=("varicelle dans l'enfance", "immunodépression", "stress récent"),
        antecedents_en=("childhood chickenpox", "immunosuppression", "recent stress"),
        delai="jours",
        justification=(
            "Zona du territoire ophtalmique : le risque de complication cornéenne impose un "
            "traitement antiviral dans les 72 heures et un avis spécialisé rapide."
        ),
        recommandation=(
            "Antiviral par voie orale sans attendre, antalgie, avis ophtalmologique dans les 24 "
            "heures. Éviter tout contact avec des personnes non immunisées."
        ),
    ),
    Presentation(
        id="abces_dentaire",
        level="URGENCE_MODEREE",
        vitals_profile="intermediaire",
        age_range=(12, 70),
        motif_fr="joue gonflée avec douleur dentaire intense",
        motif_en="swollen cheek with severe toothache",
        signes_fr=(
            "fièvre modérée",
            "difficulté à ouvrir la bouche",
            "ganglion sous la mâchoire",
            "déglutition possible",
        ),
        signes_en=(
            "moderate fever",
            "difficulty opening the mouth",
            "swollen gland under the jaw",
            "able to swallow",
        ),
        antecedents_fr=("carie négligée", "suivi dentaire irrégulier", "diabète"),
        antecedents_en=("neglected cavity", "irregular dental care", "diabetes"),
        delai="jours",
        justification=(
            "Cellulite d'origine dentaire avec fièvre et limitation de l'ouverture buccale : "
            "infection à traiter rapidement, sans signe d'extension cervicale pour l'instant."
        ),
        recommandation=(
            "Antibiothérapie et antalgie, avis stomatologique pour drainage. Consultation "
            "immédiate si la déglutition ou la respiration deviennent difficiles."
        ),
    ),
    Presentation(
        id="otite_moyenne_aigue_enfant",
        level="URGENCE_MODEREE",
        vitals_profile="intermediaire",
        age_range=(1, 8),
        motif_fr="enfant qui pleure en se tenant l'oreille avec de la fièvre",
        motif_en="child crying and holding the ear, with fever",
        signes_fr=(
            "réveils nocturnes",
            "rhume depuis trois jours",
            "bon contact",
            "alimentation conservée",
        ),
        signes_en=("waking at night", "cold for three days", "interacts normally", "still eating"),
        antecedents_fr=("otites à répétition", "vie en collectivité", "vaccination à jour"),
        antecedents_en=("recurrent ear infections", "attends nursery", "vaccinations up to date"),
        delai="jours",
        justification=(
            "Otalgie fébrile chez un enfant en bon état général : otite moyenne aiguë probable, "
            "à examiner et traiter dans la journée sans critère de gravité."
        ),
        recommandation=(
            "Examen otoscopique, antalgie et antipyrétique, antibiothérapie selon l'âge et "
            "l'aspect du tympan. Réévaluation à 48 heures."
        ),
    ),
    Presentation(
        id="pneumopathie_communautaire",
        level="URGENCE_MODEREE",
        vitals_profile="intermediaire",
        age_range=(25, 80),
        motif_fr="toux grasse et fièvre avec point de côté",
        motif_en="productive cough and fever with chest pain on breathing",
        signes_fr=("crachats colorés", "essoufflement à l'effort", "frissons", "vigilance normale"),
        signes_en=("coloured sputum", "breathless on exertion", "shivering", "fully alert"),
        antecedents_fr=("tabagisme", "bronchopneumopathie chronique", "grippe récente"),
        antecedents_en=("smoking", "chronic lung disease", "recent influenza"),
        delai="jours",
        justification=(
            "Foyer infectieux pulmonaire probable avec fièvre et douleur pleurale, sans détresse "
            "respiratoire ni trouble de la vigilance : traitement à instaurer dans la journée."
        ),
        recommandation=(
            "Radiographie thoracique, évaluation de la gravité par score clinique, "
            "antibiothérapie probabiliste et réévaluation à 48 heures."
        ),
    ),
    Presentation(
        id="lombalgie_febrile",
        level="URGENCE_MODEREE",
        vitals_profile="intermediaire",
        age_range=(30, 80),
        motif_fr="douleur lombaire avec fièvre",
        motif_en="low back pain with fever",
        signes_fr=(
            "raideur du dos",
            "sueurs nocturnes",
            "pas de déficit moteur",
            "marche possible",
        ),
        signes_en=("stiff back", "night sweats", "no weakness", "able to walk"),
        antecedents_fr=("injection récente", "toxicomanie intraveineuse", "immunodépression"),
        antecedents_en=("recent injection", "intravenous drug use", "immunosuppression"),
        delai="jours",
        justification=(
            "Lombalgie fébrile : drapeau rouge imposant d'éliminer une spondylodiscite ou un "
            "abcès, même en l'absence de déficit neurologique."
        ),
        recommandation=(
            "Bilan inflammatoire et hémocultures, imagerie du rachis, avis spécialisé dans la "
            "journée. Consultation immédiate en cas de déficit moteur ou de troubles sphinctériens."
        ),
    ),
    Presentation(
        id="traumatisme_cranien_sous_anticoagulant",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(65, 92),
        motif_fr="chute de sa hauteur avec choc à la tête, sans perte de connaissance",
        motif_en="fall from standing height with a head knock, no loss of consciousness",
        signes_fr=(
            "bosse frontale",
            "céphalée légère",
            "pas de vomissement",
            "examen neurologique normal",
        ),
        signes_en=(
            "forehead lump",
            "mild headache",
            "no vomiting",
            "normal neurological examination",
        ),
        antecedents_fr=(
            "traitement anticoagulant oral",
            "fibrillation auriculaire",
            "chutes récentes",
        ),
        antecedents_en=("oral anticoagulant therapy", "atrial fibrillation", "recent falls"),
        delai="heures",
        justification=(
            "Traumatisme crânien sans signe de gravité immédiat, mais le traitement anticoagulant "
            "impose un scanner dans l'heure et une surveillance : l'hématome sous-dural du sujet "
            "âgé anticoagulé est justement asymptomatique à la phase initiale."
        ),
        recommandation=(
            "Scanner cérébral dans l'heure, surveillance neurologique d'au moins quatre heures, "
            "consignes écrites de surveillance à la sortie et réévaluation du rapport "
            "bénéfice-risque du traitement anticoagulant."
        ),
    ),
    Presentation(
        id="brulure_second_degre_limitee",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(3, 75),
        motif_fr="brûlure de l'avant-bras par liquide bouillant",
        motif_en="forearm scald from boiling liquid",
        signes_fr=(
            "phlyctènes",
            "douleur vive",
            "surface inférieure à une paume",
            "pas d'atteinte du visage",
        ),
        signes_en=("blisters", "sharp pain", "smaller than one palm", "face not involved"),
        antecedents_fr=(
            "accident de cuisine",
            "vaccination antitétanique à jour",
            "aucun traitement",
        ),
        antecedents_en=("kitchen accident", "tetanus vaccination up to date", "no medication"),
        delai="heures",
        justification=(
            "Brûlure du deuxième degré de surface limitée, sans atteinte des zones fonctionnelles "
            "ni signe d'inhalation : soins locaux nécessaires dans les heures qui suivent."
        ),
        recommandation=(
            "Refroidissement, antalgie, pansement gras et évaluation de la profondeur à 48 heures. "
            "Consultation spécialisée si la cicatrisation traîne."
        ),
    ),
    Presentation(
        id="erysipele_jambe",
        level="URGENCE_MODEREE",
        vitals_profile="intermediaire",
        age_range=(40, 88),
        motif_fr="jambe rouge, chaude et douloureuse avec fièvre",
        motif_en="red, hot, painful leg with fever",
        signes_fr=(
            "placard bien limité",
            "ganglion inguinal",
            "frissons",
            "pas de nécrose cutanée",
        ),
        signes_en=(
            "well-demarcated red patch",
            "swollen groin gland",
            "shivering",
            "no skin necrosis",
        ),
        antecedents_fr=("insuffisance veineuse", "mycose entre les orteils", "surpoids"),
        antecedents_en=("venous insufficiency", "athlete's foot", "overweight"),
        delai="jours",
        justification=(
            "Dermohypodermite bactérienne aiguë fébrile sans signe de gravité locale : "
            "antibiothérapie à débuter rapidement, surveillance de l'extension."
        ),
        recommandation=(
            "Antibiothérapie antistreptococcique, repos jambe surélevée, délimitation au crayon "
            "de la zone rouge et réévaluation à 48 heures. Retour immédiat si douleur disproportionnée."
        ),
    ),
    Presentation(
        id="corps_etranger_oculaire",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(16, 65),
        motif_fr="sensation de corps étranger dans l'œil après meulage",
        motif_en="foreign body sensation in the eye after grinding metal",
        signes_fr=(
            "œil rouge et larmoyant",
            "gêne à la lumière",
            "vision conservée",
            "clignements incessants",
        ),
        signes_en=("red watery eye", "light sensitivity", "vision preserved", "constant blinking"),
        antecedents_fr=(
            "travail sans lunettes de protection",
            "aucun antécédent ophtalmologique",
            "port de lentilles",
        ),
        antecedents_en=("working without safety glasses", "no eye history", "wears contact lenses"),
        delai="heures",
        justification=(
            "Corps étranger cornéen probable après projection métallique : risque d'abcès ou de "
            "rouille cornéenne, à retirer rapidement, sans menace immédiate pour la vision."
        ),
        recommandation=(
            "Mesure de l'acuité visuelle, examen à la fluorescéine, retrait par un praticien "
            "entraîné et avis ophtalmologique dans les 24 heures."
        ),
    ),
    Presentation(
        id="vertige_rotatoire_recent",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(35, 80),
        motif_fr="vertiges rotatoires avec vomissements depuis ce matin",
        motif_en="spinning dizziness with vomiting since this morning",
        signes_fr=(
            "aggravation aux mouvements de tête",
            "marche instable",
            "pas de déficit moteur",
            "audition normale",
        ),
        signes_en=(
            "worse on head movement",
            "unsteady walking",
            "no limb weakness",
            "normal hearing",
        ),
        antecedents_fr=("épisodes similaires", "migraine", "hypertension artérielle"),
        antecedents_en=("similar episodes", "migraine", "hypertension"),
        delai="heures",
        justification=(
            "Syndrome vestibulaire aigu : l'absence de déficit focal oriente vers une cause "
            "périphérique, mais une origine centrale doit être écartée par l'examen."
        ),
        recommandation=(
            "Examen neurologique et manœuvres vestibulaires, antiémétique, imagerie cérébrale si "
            "le moindre signe central apparaît. Éviter la conduite jusqu'à résolution."
        ),
    ),
    Presentation(
        id="crise_drepanocytaire",
        level="URGENCE_MODEREE",
        vitals_profile="intermediaire",
        age_range=(5, 45),
        motif_fr="douleurs osseuses diffuses chez un patient drépanocytaire",
        motif_en="widespread bone pain in a patient with sickle cell disease",
        signes_fr=(
            "douleur habituelle mais plus intense",
            "pas de fièvre élevée",
            "respiration normale",
            "hydratation insuffisante",
        ),
        signes_en=(
            "usual pain but more intense",
            "no high fever",
            "normal breathing",
            "poorly hydrated",
        ),
        antecedents_fr=(
            "drépanocytose homozygote",
            "crises vaso-occlusives répétées",
            "épisode déclenché par le froid",
        ),
        antecedents_en=(
            "sickle cell anaemia",
            "recurrent vaso-occlusive crises",
            "triggered by cold",
        ),
        delai="heures",
        justification=(
            "Crise vaso-occlusive typique sans syndrome thoracique aigu ni fièvre élevée : "
            "antalgie urgente mais pas de défaillance d'organe constatée."
        ),
        recommandation=(
            "Antalgie de palier adapté sans délai, hydratation, oxygénothérapie si la saturation "
            "baisse, recherche d'un facteur déclenchant. Surveillance respiratoire rapprochée."
        ),
    ),
    Presentation(
        id="reaction_allergique_cutanee",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(2, 70),
        motif_fr="plaques d'urticaire étendues après la prise d'un médicament",
        motif_en="widespread hives after taking a medication",
        signes_fr=(
            "démangeaisons intenses",
            "pas de gêne respiratoire",
            "déglutition normale",
            "tension conservée",
        ),
        signes_en=(
            "intense itching",
            "no breathing difficulty",
            "normal swallowing",
            "blood pressure maintained",
        ),
        antecedents_fr=(
            "antibiotique débuté la veille",
            "terrain allergique",
            "aucune allergie connue",
        ),
        antecedents_en=("antibiotic started yesterday", "allergic background", "no known allergy"),
        delai="heures",
        justification=(
            "Urticaire médicamenteuse étendue sans atteinte respiratoire ni hémodynamique : "
            "réaction allergique à traiter et à surveiller, sans critère d'anaphylaxie."
        ),
        recommandation=(
            "Arrêt du médicament suspect, antihistaminique, surveillance de deux heures. "
            "Consultation immédiate si gonflement du visage, gêne à avaler ou essoufflement."
        ),
    ),
    Presentation(
        id="douleur_thoracique_parietale",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(18, 55),
        motif_fr="douleur thoracique reproduite à la palpation après un effort de musculation",
        motif_en="chest pain reproduced by pressing, after weight training",
        signes_fr=(
            "douleur au mouvement du bras",
            "pas de sueurs",
            "pas d'essoufflement",
            "électrocardiogramme normal",
        ),
        signes_en=(
            "pain on arm movement",
            "no sweating",
            "no breathlessness",
            "normal electrocardiogram",
        ),
        antecedents_fr=(
            "aucun facteur de risque cardiovasculaire",
            "sport intensif récent",
            "non-fumeur",
        ),
        antecedents_en=("no cardiovascular risk factors", "recent intensive sport", "non-smoker"),
        delai="jours",
        justification=(
            "Douleur thoracique reproductible à la palpation, d'allure pariétale, chez un sujet "
            "jeune sans facteur de risque : l'origine coronarienne reste à écarter formellement."
        ),
        recommandation=(
            "Électrocardiogramme et dosage de troponine pour éliminer une cause coronarienne, "
            "antalgie simple puis retour à domicile avec consignes de surveillance."
        ),
    ),
    Presentation(
        id="hyperglycemie_sans_cetose",
        level="URGENCE_MODEREE",
        vitals_profile="intermediaire",
        age_range=(30, 80),
        motif_fr="glycémie très élevée au doigt avec soif et fatigue",
        motif_en="very high finger-prick glucose with thirst and fatigue",
        signes_fr=("urines fréquentes", "vision floue", "pas de cétones", "vigilance normale"),
        signes_en=("frequent urination", "blurred vision", "no ketones", "fully alert"),
        antecedents_fr=("diabète de type 2", "oubli du traitement", "corticothérapie récente"),
        antecedents_en=("type 2 diabetes", "missed medication", "recent steroid treatment"),
        delai="jours",
        justification=(
            "Déséquilibre glycémique franc sans cétose ni trouble de la vigilance : adaptation "
            "thérapeutique nécessaire rapidement, sans urgence vitale."
        ),
        recommandation=(
            "Hydratation, recherche de cétones, ionogramme, adaptation du traitement et avis "
            "diabétologique. Recherche d'un facteur déclenchant infectieux."
        ),
    ),
    Presentation(
        id="agitation_anxieuse",
        level="URGENCE_MODEREE",
        vitals_profile="intermediaire",
        age_range=(16, 65),
        motif_fr="crise d'angoisse avec sensation d'étouffement",
        motif_en="panic attack with a feeling of suffocation",
        signes_fr=(
            "fourmillements des mains",
            "respiration rapide",
            "peur de mourir",
            "examen clinique normal",
        ),
        signes_en=(
            "tingling hands",
            "rapid breathing",
            "fear of dying",
            "normal physical examination",
        ),
        antecedents_fr=(
            "trouble anxieux",
            "épisodes identiques",
            "consommation de café importante",
        ),
        antecedents_en=("anxiety disorder", "identical episodes", "high caffeine intake"),
        delai="heures",
        justification=(
            "Tableau typique d'attaque de panique déjà connue, avec examen clinique normal : la "
            "prise en charge est rapide mais les causes organiques doivent être écartées."
        ),
        recommandation=(
            "Réassurance dans un endroit calme, contrôle de la respiration, électrocardiogramme "
            "et glycémie pour éliminer une cause organique, orientation vers un suivi psychologique."
        ),
    ),
    # ------------------------------------------------------------------
    # CONSULTATION DIFFEREE — pas de critère de gravité immédiat
    # ------------------------------------------------------------------
    Presentation(
        id="rhinopharyngite",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(2, 70),
        motif_fr="nez bouché et mal de gorge depuis deux jours",
        motif_en="blocked nose and sore throat for two days",
        signes_fr=("éternuements", "fatigue légère", "pas de fièvre", "appétit conservé"),
        signes_en=("sneezing", "mild tiredness", "no fever", "normal appetite"),
        antecedents_fr=("aucun antécédent notable", "épisodes hivernaux habituels", "non-fumeur"),
        antecedents_en=("no notable history", "usual winter episodes", "non-smoker"),
        delai="jours",
        justification=(
            "Infection virale bénigne des voies aériennes supérieures, sans fièvre ni signe "
            "respiratoire de gravité : aucun critère d'urgence."
        ),
        recommandation=(
            "Traitement symptomatique et lavages de nez, consultation chez le médecin traitant "
            "si les symptômes persistent au-delà de dix jours ou si une fièvre apparaît."
        ),
    ),
    Presentation(
        id="lombalgie_mecanique",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(25, 70),
        motif_fr="douleur du bas du dos après avoir porté une charge",
        motif_en="lower back pain after lifting a heavy load",
        signes_fr=(
            "douleur soulagée au repos",
            "pas de fièvre",
            "pas de déficit des jambes",
            "marche possible",
        ),
        signes_en=("pain relieved by rest", "no fever", "no leg weakness", "able to walk"),
        antecedents_fr=("travail physique", "épisodes similaires", "sédentarité"),
        antecedents_en=("physical work", "similar episodes", "sedentary lifestyle"),
        delai="jours",
        justification=(
            "Lombalgie commune sans drapeau rouge : ni fièvre, ni déficit neurologique, ni "
            "traumatisme violent, ni altération de l'état général."
        ),
        recommandation=(
            "Antalgie simple, poursuite d'une activité adaptée plutôt que le repos strict, "
            "consultation du médecin traitant si la douleur dépasse quatre semaines."
        ),
    ),
    Presentation(
        id="renouvellement_ordonnance",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(35, 85),
        motif_fr="demande de renouvellement d'un traitement chronique",
        motif_en="request to renew a long-term prescription",
        signes_fr=(
            "aucun symptôme aigu",
            "traitement bien supporté",
            "constantes habituelles",
            "bon état général",
        ),
        signes_en=(
            "no acute symptoms",
            "medication well tolerated",
            "usual observations",
            "good general condition",
        ),
        antecedents_fr=("hypertension équilibrée", "hypothyroïdie substituée", "suivi régulier"),
        antecedents_en=(
            "well-controlled hypertension",
            "treated hypothyroidism",
            "regular follow-up",
        ),
        delai="semaines",
        justification=(
            "Demande administrative sans symptôme aigu : ce motif relève de la médecine de ville "
            "et non du service d'urgence."
        ),
        recommandation=(
            "Orienter vers le médecin traitant ou le pharmacien pour une dispensation de "
            "dépannage, expliquer le circuit adapté hors urgences."
        ),
    ),
    Presentation(
        id="eczema_poussee",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(1, 60),
        motif_fr="plaques sèches qui démangent aux plis des coudes",
        motif_en="dry itchy patches in the elbow creases",
        signes_fr=("peau épaissie", "grattage nocturne", "pas de suintement", "pas de fièvre"),
        signes_en=("thickened skin", "scratching at night", "no oozing", "no fever"),
        antecedents_fr=("dermatite atopique", "asthme dans la famille", "arrêt récent des crèmes"),
        antecedents_en=("atopic dermatitis", "asthma in the family", "recently stopped creams"),
        delai="semaines",
        justification=(
            "Poussée d'eczéma chronique sans signe de surinfection ni retentissement général : "
            "situation dermatologique courante, non urgente."
        ),
        recommandation=(
            "Reprise des émollients et d'un dermocorticoïde adapté, consultation programmée chez "
            "le dermatologue ou le médecin traitant."
        ),
    ),
    Presentation(
        id="conjonctivite_simple",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(2, 70),
        motif_fr="œil rouge collé le matin depuis deux jours",
        motif_en="red eye, sticky in the morning, for two days",
        signes_fr=(
            "sécrétions claires",
            "vision normale",
            "pas de douleur profonde",
            "pas de gêne à la lumière",
        ),
        signes_en=("clear discharge", "normal vision", "no deep pain", "no light sensitivity"),
        antecedents_fr=(
            "contage familial",
            "pas de port de lentilles",
            "aucun antécédent oculaire",
        ),
        antecedents_en=("family member affected", "no contact lenses", "no eye history"),
        delai="jours",
        justification=(
            "Conjonctivite banale : vision conservée, absence de douleur profonde et de "
            "photophobie, ce qui écarte les causes ophtalmologiques graves."
        ),
        recommandation=(
            "Lavages oculaires au sérum physiologique et mesures d'hygiène. Consultation rapide "
            "si baisse de vision, douleur intense ou photophobie."
        ),
    ),
    Presentation(
        id="plaie_superficielle_doigt",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(5, 80),
        motif_fr="petite coupure au doigt avec un couteau de cuisine",
        motif_en="small finger cut from a kitchen knife",
        signes_fr=(
            "saignement arrêté",
            "plaie superficielle",
            "mobilité et sensibilité normales",
            "berges propres",
        ),
        signes_en=(
            "bleeding stopped",
            "superficial wound",
            "normal movement and sensation",
            "clean edges",
        ),
        antecedents_fr=(
            "vaccination antitétanique à jour",
            "aucun traitement",
            "accident domestique",
        ),
        antecedents_en=("tetanus vaccination up to date", "no medication", "domestic accident"),
        delai="heures",
        justification=(
            "Plaie superficielle sans saignement actif ni atteinte tendineuse ou nerveuse : "
            "aucun geste urgent nécessaire."
        ),
        recommandation=(
            "Nettoyage, désinfection et pansement, surveillance des signes d'infection. "
            "Consultation si rougeur, chaleur ou douleur croissante."
        ),
    ),
    Presentation(
        id="allergie_saisonniere",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(6, 60),
        motif_fr="éternuements et yeux qui piquent au printemps",
        motif_en="sneezing and itchy eyes in spring",
        signes_fr=(
            "nez qui coule clair",
            "pas de fièvre",
            "respiration normale",
            "symptômes en extérieur",
        ),
        signes_en=("clear runny nose", "no fever", "normal breathing", "symptoms outdoors"),
        antecedents_fr=(
            "rhinite allergique connue",
            "antécédents familiaux d'allergie",
            "traitement habituel épuisé",
        ),
        antecedents_en=(
            "known allergic rhinitis",
            "family history of allergy",
            "ran out of usual treatment",
        ),
        delai="semaines",
        justification=(
            "Rhinoconjonctivite allergique saisonnière typique, sans signe respiratoire bas ni "
            "retentissement général."
        ),
        recommandation=(
            "Antihistaminique et lavages de nez, consultation programmée pour un bilan "
            "allergologique si les symptômes sont invalidants chaque année."
        ),
    ),
    Presentation(
        id="douleur_dentaire_ancienne",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(15, 75),
        motif_fr="douleur dentaire intermittente depuis trois semaines",
        motif_en="on-and-off toothache for three weeks",
        signes_fr=(
            "sensibilité au froid",
            "pas de gonflement",
            "pas de fièvre",
            "ouverture de bouche normale",
        ),
        signes_en=("sensitivity to cold", "no swelling", "no fever", "normal mouth opening"),
        antecedents_fr=("carie connue", "rendez-vous dentaire prévu", "suivi irrégulier"),
        antecedents_en=("known cavity", "dental appointment booked", "irregular follow-up"),
        delai="semaines",
        justification=(
            "Douleur dentaire chronique sans signe infectieux local ni général : relève du "
            "chirurgien-dentiste et non du service d'urgence."
        ),
        recommandation=(
            "Antalgie simple et consultation dentaire programmée. Retour aux urgences en cas de "
            "gonflement du visage, de fièvre ou de difficulté à avaler."
        ),
    ),
    Presentation(
        id="constipation_chronique",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(25, 85),
        motif_fr="constipation depuis plusieurs semaines avec ballonnements",
        motif_en="constipation for several weeks with bloating",
        signes_fr=("ventre souple", "gaz présents", "pas de sang", "poids stable"),
        signes_en=("soft abdomen", "passing wind", "no blood", "stable weight"),
        antecedents_fr=("alimentation pauvre en fibres", "sédentarité", "traitement par opiacés"),
        antecedents_en=("low-fibre diet", "sedentary lifestyle", "opioid treatment"),
        delai="semaines",
        justification=(
            "Constipation chronique avec transit gazeux conservé et abdomen souple : aucun signe "
            "d'occlusion ni d'alerte digestive."
        ),
        recommandation=(
            "Mesures hygiéno-diététiques et laxatif doux, consultation programmée chez le médecin "
            "traitant. Consultation urgente si arrêt des gaz, vomissements ou sang dans les selles."
        ),
    ),
    Presentation(
        id="verrue_plantaire",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(8, 55),
        motif_fr="petite excroissance douloureuse sous le pied",
        motif_en="small painful growth under the foot",
        signes_fr=(
            "gêne à la marche prolongée",
            "pas de rougeur",
            "pas de fièvre",
            "aspect stable",
        ),
        signes_en=("discomfort on long walks", "no redness", "no fever", "unchanged appearance"),
        antecedents_fr=("fréquentation de la piscine", "aucun antécédent", "diabète absent"),
        antecedents_en=("swimming pool use", "no medical history", "no diabetes"),
        delai="semaines",
        justification=(
            "Lésion cutanée bénigne d'évolution lente, sans signe infectieux ni terrain à risque : "
            "aucun caractère urgent."
        ),
        recommandation=(
            "Traitement kératolytique en pharmacie, consultation dermatologique programmée si "
            "persistance ou si la lésion se modifie."
        ),
    ),
    Presentation(
        id="fatigue_chronique",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(20, 70),
        motif_fr="fatigue depuis plusieurs semaines sans autre symptôme",
        motif_en="tiredness for several weeks with no other symptom",
        signes_fr=(
            "sommeil perturbé",
            "pas de perte de poids",
            "pas de fièvre",
            "examen clinique normal",
        ),
        signes_en=("disturbed sleep", "no weight loss", "no fever", "normal physical examination"),
        antecedents_fr=(
            "charge de travail importante",
            "aucun traitement",
            "bilan sanguin ancien normal",
        ),
        antecedents_en=("heavy workload", "no medication", "previous blood tests normal"),
        delai="semaines",
        justification=(
            "Asthénie isolée sans signe d'alerte associé : bilan à organiser en ville, aucun "
            "élément ne justifie un passage aux urgences."
        ),
        recommandation=(
            "Consultation programmée chez le médecin traitant pour un bilan étiologique, "
            "hygiène de sommeil et évaluation de la charge de travail."
        ),
    ),
    Presentation(
        id="tendinite_epaule",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(30, 70),
        motif_fr="douleur de l'épaule à l'élévation du bras depuis un mois",
        motif_en="shoulder pain when raising the arm for a month",
        signes_fr=(
            "douleur nocturne modérée",
            "pas de traumatisme",
            "pas de fièvre",
            "force conservée",
        ),
        signes_en=("moderate night pain", "no injury", "no fever", "strength preserved"),
        antecedents_fr=("travail répétitif", "pratique du tennis", "aucun traitement"),
        antecedents_en=("repetitive work", "plays tennis", "no medication"),
        delai="semaines",
        justification=(
            "Tendinopathie d'installation progressive sans traumatisme, sans fièvre et sans "
            "déficit de force : pathologie chronique relevant du suivi ambulatoire."
        ),
        recommandation=(
            "Antalgie, repos relatif et kinésithérapie, consultation programmée avec le médecin "
            "traitant ou un rhumatologue."
        ),
    ),
    Presentation(
        id="acne_inflammatoire",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(13, 30),
        motif_fr="boutons inflammatoires du visage et du dos",
        motif_en="inflamed spots on the face and back",
        signes_fr=(
            "évolution progressive",
            "pas de fièvre",
            "retentissement esthétique",
            "pas d'abcès",
        ),
        signes_en=("gradual course", "no fever", "cosmetic impact", "no abscess"),
        antecedents_fr=(
            "acné depuis l'adolescence",
            "traitements locaux inefficaces",
            "aucun antécédent médical",
        ),
        antecedents_en=(
            "acne since adolescence",
            "topical treatments ineffective",
            "no medical history",
        ),
        delai="semaines",
        justification=(
            "Dermatose chronique sans signe infectieux aigu : prise en charge dermatologique "
            "programmée, aucun critère d'urgence."
        ),
        recommandation=(
            "Consultation dermatologique programmée pour adapter le traitement de fond, "
            "poursuite des soins locaux dans l'intervalle."
        ),
    ),
    Presentation(
        id="reflux_gastro_oesophagien",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(25, 70),
        motif_fr="brûlures remontant derrière le sternum après les repas",
        motif_en="burning rising behind the breastbone after meals",
        signes_fr=(
            "aggravation en position allongée",
            "pas d'irradiation au bras",
            "pas de sueurs",
            "poids stable",
        ),
        signes_en=("worse when lying down", "no arm radiation", "no sweating", "stable weight"),
        antecedents_fr=("reflux connu", "surpoids", "repas tardifs"),
        antecedents_en=("known reflux", "overweight", "late meals"),
        delai="semaines",
        justification=(
            "Symptomatologie de reflux typique, rythmée par les repas et la position, sans signe "
            "d'alarme digestif ni élément en faveur d'une douleur coronarienne."
        ),
        recommandation=(
            "Mesures hygiéno-diététiques et traitement antiacide d'épreuve, consultation "
            "programmée. Consultation urgente en cas de difficulté à avaler ou d'amaigrissement."
        ),
    ),
    Presentation(
        id="suivi_tension_stable",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(45, 85),
        motif_fr="tension mesurée un peu élevée à la maison, sans symptôme",
        motif_en="slightly high blood pressure measured at home, no symptoms",
        signes_fr=(
            "pas de céphalée",
            "pas de trouble visuel",
            "pas de douleur thoracique",
            "tension normale à l'accueil",
        ),
        signes_en=(
            "no headache",
            "no visual disturbance",
            "no chest pain",
            "normal reading on arrival",
        ),
        antecedents_fr=("hypertension traitée", "automesure récente", "suivi régulier"),
        antecedents_en=("treated hypertension", "recent home monitoring", "regular follow-up"),
        delai="semaines",
        justification=(
            "Chiffres tensionnels isolés sans aucun signe de retentissement : il n'existe pas "
            "d'urgence hypertensive en l'absence de symptôme."
        ),
        recommandation=(
            "Poursuite du traitement et de l'automesure, consultation programmée pour adapter le "
            "traitement. Consultation urgente en cas de céphalée intense ou de trouble visuel."
        ),
    ),
    Presentation(
        id="entorse_doigt_benigne",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(10, 60),
        motif_fr="doigt douloureux après un choc au ballon",
        motif_en="painful finger after a ball impact",
        signes_fr=(
            "léger gonflement",
            "flexion possible",
            "pas de déformation",
            "pas d'hématome important",
        ),
        signes_en=("slight swelling", "able to bend it", "no deformity", "no significant bruising"),
        antecedents_fr=("sport de ballon", "aucun antécédent", "vaccination à jour"),
        antecedents_en=("ball sport", "no medical history", "vaccinations up to date"),
        delai="heures",
        justification=(
            "Traumatisme digital bénin : mobilité conservée et absence de déformation rendent la "
            "fracture très improbable."
        ),
        recommandation=(
            "Glace, syndactylie et antalgie simple. Consultation si la douleur persiste au-delà "
            "d'une semaine ou si une déformation apparaît."
        ),
    ),
    Presentation(
        id="mycose_ongle",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(35, 80),
        motif_fr="ongle de pied épaissi et jauni depuis des mois",
        motif_en="thickened yellow toenail for months",
        signes_fr=(
            "pas de douleur",
            "pas de rougeur du pourtour",
            "pas de fièvre",
            "évolution très lente",
        ),
        signes_en=("no pain", "no redness around the nail", "no fever", "very slow course"),
        antecedents_fr=(
            "fréquentation de vestiaires collectifs",
            "pas de diabète",
            "chaussures fermées",
        ),
        antecedents_en=("uses shared changing rooms", "no diabetes", "closed shoes"),
        delai="semaines",
        justification=(
            "Onychomycose d'évolution chronique, sans douleur ni signe inflammatoire, chez un "
            "patient sans terrain à risque podologique."
        ),
        recommandation=(
            "Prélèvement mycologique et traitement local en consultation programmée, mesures "
            "d'hygiène des pieds."
        ),
    ),
    Presentation(
        id="certificat_sport",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(8, 60),
        motif_fr="demande de certificat médical d'aptitude au sport",
        motif_en="request for a sports fitness certificate",
        signes_fr=(
            "aucun symptôme",
            "activité physique régulière",
            "examen normal",
            "constantes normales",
        ),
        signes_en=(
            "no symptoms",
            "regular physical activity",
            "normal examination",
            "normal observations",
        ),
        antecedents_fr=(
            "aucun antécédent",
            "pas de traitement",
            "pas d'antécédent familial cardiaque",
        ),
        antecedents_en=(
            "no medical history",
            "no medication",
            "no family history of heart disease",
        ),
        delai="semaines",
        justification=(
            "Demande administrative sans plainte médicale : ce motif ne relève pas du service "
            "d'urgence."
        ),
        recommandation=(
            "Orienter vers le médecin traitant pour une consultation dédiée, expliquer que les "
            "urgences ne délivrent pas ce type de certificat."
        ),
    ),
    Presentation(
        id="insomnie_chronique",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(25, 80),
        motif_fr="difficultés d'endormissement depuis plusieurs mois",
        motif_en="difficulty falling asleep for several months",
        signes_fr=("réveils nocturnes", "fatigue diurne", "pas d'idées noires", "examen normal"),
        signes_en=("night waking", "daytime tiredness", "no dark thoughts", "normal examination"),
        antecedents_fr=("stress professionnel", "écrans le soir", "consommation de café"),
        antecedents_en=("work stress", "screens in the evening", "caffeine intake"),
        delai="semaines",
        justification=(
            "Trouble du sommeil chronique sans souffrance psychiatrique aiguë ni idée suicidaire : "
            "prise en charge programmée en ville."
        ),
        recommandation=(
            "Conseils d'hygiène du sommeil, consultation programmée chez le médecin traitant. "
            "Consultation urgente si apparition d'idées suicidaires."
        ),
    ),
    Presentation(
        id="hemorroides_non_compliquees",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(25, 75),
        motif_fr="gêne anale avec un peu de sang sur le papier",
        motif_en="anal discomfort with a little blood on the paper",
        signes_fr=(
            "saignement minime",
            "pas de douleur intense",
            "pas de fièvre",
            "transit normal",
        ),
        signes_en=("minimal bleeding", "no severe pain", "no fever", "normal bowel habit"),
        antecedents_fr=("constipation", "grossesse récente", "hémorroïdes connues"),
        antecedents_en=("constipation", "recent pregnancy", "known haemorrhoids"),
        delai="semaines",
        justification=(
            "Saignement anal minime d'allure hémorroïdaire, sans retentissement général ni "
            "douleur intense évoquant une thrombose."
        ),
        recommandation=(
            "Régularisation du transit et traitement local, consultation programmée. "
            "Consultation urgente si saignement abondant ou douleur brutale et intense."
        ),
    ),
    Presentation(
        id="bouchon_cerumen",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(10, 85),
        motif_fr="oreille bouchée avec baisse d'audition d'un côté",
        motif_en="blocked ear with reduced hearing on one side",
        signes_fr=("pas de douleur", "pas d'écoulement", "pas de vertige", "pas de fièvre"),
        signes_en=("no pain", "no discharge", "no dizziness", "no fever"),
        antecedents_fr=(
            "utilisation de cotons-tiges",
            "épisodes identiques",
            "port d'aides auditives",
        ),
        antecedents_en=("uses cotton buds", "identical episodes", "wears hearing aids"),
        delai="jours",
        justification=(
            "Obstruction du conduit auditif sans douleur, écoulement, vertige ni fièvre : "
            "situation bénigne relevant d'un soin programmé."
        ),
        recommandation=(
            "Ramollissement du bouchon puis lavage d'oreille en consultation programmée. "
            "Consultation rapide en cas de douleur, d'écoulement ou de vertige."
        ),
    ),
    Presentation(
        id="cystite_simple",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(16, 60),
        motif_fr="brûlures en urinant depuis la veille, sans fièvre",
        motif_en="burning on urination since yesterday, no fever",
        signes_fr=(
            "envies fréquentes",
            "pas de douleur lombaire",
            "pas de frissons",
            "état général conservé",
        ),
        signes_en=(
            "frequent urge to urinate",
            "no flank pain",
            "no shivering",
            "feeling well otherwise",
        ),
        antecedents_fr=("cystites occasionnelles", "pas de grossesse", "pas de diabète"),
        antecedents_en=("occasional cystitis", "not pregnant", "no diabetes"),
        delai="jours",
        justification=(
            "Cystite aiguë simple chez une patiente sans facteur de risque de complication : "
            "absence de fièvre et de douleur lombaire écartant la pyélonéphrite."
        ),
        recommandation=(
            "Bandelette urinaire et antibiothérapie courte en ville, hydratation. Consultation "
            "urgente si fièvre, frissons ou douleur du dos apparaissent."
        ),
    ),
    Presentation(
        id="piqure_insecte_locale",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(3, 75),
        motif_fr="piqûre de moustique gonflée et qui démange",
        motif_en="mosquito bite, swollen and itchy",
        signes_fr=(
            "rougeur locale limitée",
            "pas de gêne respiratoire",
            "pas de fièvre",
            "pas d'extension",
        ),
        signes_en=("limited local redness", "no breathing difficulty", "no fever", "not spreading"),
        antecedents_fr=("aucune allergie connue", "séjour en extérieur", "aucun traitement"),
        antecedents_en=("no known allergy", "time spent outdoors", "no medication"),
        delai="jours",
        justification=(
            "Réaction locale isolée à une piqûre, sans signe général ni respiratoire : aucun "
            "critère d'anaphylaxie ni de surinfection."
        ),
        recommandation=(
            "Soins locaux et antihistaminique si les démangeaisons gênent. Consultation immédiate "
            "en cas de gonflement du visage, de gêne respiratoire ou d'extension rapide."
        ),
    ),
)


def presentations_by_level(level: str) -> list[Presentation]:
    """Renvoie les présentations du niveau demandé."""
    return [p for p in PRESENTATIONS if p.level == level]


# Constantes que le récit de certaines présentations nomme explicitement, avec le
# sens dans lequel elles doivent être dégradées.
#
# Le générateur dégrade une ou deux constantes tirées au sort ; cette table force
# celles que le texte nomme. Sans elle, une vignette dont le motif est « fièvre
# avec frissons » sort apyrétique la plupart du temps, et une pré-éclampsie
# sévère normotendue : la description et le relevé se contredisent dans le même
# exemple d'entraînement, et le modèle apprend au passage que les constantes ne
# veulent rien dire.
#
# `test_clinical_data` vérifie qu'aucune présentation nommant une constante n'est
# oubliée ici.
CONSTANTES_IMPOSEES: dict[str, tuple[tuple[str, str], ...]] = {
    # Fièvre nommée dans le motif ou les signes.
    "sepsis_grave": (("temperature", "haute"),),
    "syndrome_meninge": (("temperature", "haute"),),
    "pneumopathie_communautaire": (("temperature", "haute"),),
    "pyelonephrite": (("temperature", "haute"),),
    "erysipele_jambe": (("temperature", "haute"),),
    "abces_dentaire": (("temperature", "haute"),),
    "otite_moyenne_aigue_enfant": (("temperature", "haute"),),
    "lombalgie_febrile": (("temperature", "haute"),),
    "suspicion_appendicite": (("temperature", "haute"),),
    "crise_drepanocytaire": (("temperature", "haute"),),
    # Détresse respiratoire : la saturation est le signe même du tableau.
    "asthme_aigu_grave": (("spo2", "basse"),),
    "bronchiolite_grave_nourrisson": (("spo2", "basse"),),
    "crise_asthme_moderee": (("spo2", "basse"),),
    # L'hypertension **est** le diagnostic.
    "pre_eclampsie_severe": (("systolic_bp", "haute"),),
}


def constantes_imposees(presentation_id: str) -> tuple[tuple[str, str], ...]:
    """Constantes que le récit de cette présentation impose, éventuellement aucune."""
    return CONSTANTES_IMPOSEES.get(presentation_id, ())


# Probabilité qu'une vignette porte « vigilance altérée », présentation par
# présentation.
#
# La probabilité dépend du tableau clinique, non du profil de constantes : des
# tableaux critiques laissent le patient parfaitement conscient, et un taux
# attaché au profil collerait le libellé à ces présentations-là. Il ferait aussi
# de « vigilance altérée » le signe exclusif de l'urgence vitale dans tout le
# jeu, raccourci qu'un modèle apprend volontiers à la place du triage.
#
# La valeur par défaut est nulle : la conscience reste normale sauf là où le
# tableau clinique l'engage.
VIGILANCE_ALTEREE_PROBABLE: dict[str, float] = {
    # La conscience fait partie du tableau.
    "sepsis_grave": 0.75,
    "traumatisme_cranien_grave": 0.80,
    "crise_convulsive_prolongee": 0.85,
    "intoxication_medicamenteuse_volontaire": 0.70,
    "hypoglycemie_severe": 0.80,
    "deshydratation_severe_nourrisson": 0.60,
    "acidocetose_diabetique": 0.45,
    # Elle peut se dégrader, sans être la règle.
    "choc_anaphylactique": 0.35,
    "rupture_anevrisme_aorte": 0.40,
    "hemorragie_digestive_haute": 0.25,
    "hemorragie_du_post_partum": 0.25,
    "bronchiolite_grave_nourrisson": 0.30,
    # Les autres tableaux critiques laissent le patient conscient : syndrome
    # coronarien, asthme aigu grave, embolie pulmonaire, brûlure étendue, plaie
    # pénétrante. Leur probabilité est nulle, par absence de cette table.
}


def vigilance_alteree_probable(presentation_id: str) -> float:
    """Probabilité que cette présentation s'accompagne d'une vigilance altérée."""
    return VIGILANCE_ALTEREE_PROBABLE.get(presentation_id, 0.0)


def presentation_by_id(presentation_id: str) -> Presentation:
    """Renvoie la présentation portant cet identifiant."""
    for presentation in PRESENTATIONS:
        if presentation.id == presentation_id:
            return presentation
    raise KeyError(f"Présentation inconnue : {presentation_id!r}")
