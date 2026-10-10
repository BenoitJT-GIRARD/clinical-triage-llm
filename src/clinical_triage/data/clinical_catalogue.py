"""The catalogue of clinical presentations that carries the dataset's ground truth.

No public medical corpus is annotated with triage levels. Labelling texts by the mere presence
of keywords produces absurd examples and, worse, a circular evaluation: the model relearns the
rule that manufactured the labels, and the rule alone then scores better than the model.

So this goes the other way round. Each entry describes a **typical presentation** met at an
emergency reception desk, with its reference triage level. The vignette generator
(``case_generator``) then dresses that presentation with an age, a history, vital signs and a
wording; the label comes from the presentation, never from re-reading the produced text.

The mapping to levels follows the FRENCH scale used in French emergency departments: levels 1-2
(immediate care) → ``URGENCE_VITALE``, levels 3-4 (a few hours) → ``URGENCE_MODEREE``, level 5 →
``CONSULTATION_DIFFEREE``.

The catalogue is deliberately written by hand and reviewed case by case: annotation quality
matters more here than volume. It covers adults, children, pregnancy, trauma, psychiatry and
the general-practice complaints that crowd an emergency department.

Acknowledged limit: these presentations were written by an engineer, not by an emergency
physician. They are enough for a prototype; clinical validation is a prerequisite to any real
use.

The clinical text is French and English: it is the data, not prose about the project.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Presentation:
    """A typical patient presentation and its reference triage level."""

    id: str
    level: str
    # Expected vital-sign profile: "critical", "intermediate" or "normal".
    vitals_profile: str
    age_range: tuple[int, int]
    complaint_fr: str
    complaint_en: str
    signs_fr: tuple[str, ...]
    signs_en: tuple[str, ...]
    history_fr: tuple[str, ...]
    history_en: tuple[str, ...]
    # Onset time scale: "minutes", "heures", "jours", "semaines".
    onset: str
    # The justification and the recommendation are always in French: the agent answers in
    # French, whatever the language of the description.
    justification: str
    recommendation: str


PRESENTATIONS: tuple[Presentation, ...] = (
    # ------------------------------------------------------------------
    # URGENCE_VITALE — immediate care
    # ------------------------------------------------------------------
    Presentation(
        id="syndrome_coronarien_aigu",
        level="URGENCE_VITALE",
        vitals_profile="critical",
        age_range=(45, 88),
        complaint_fr="douleur thoracique constrictive",
        complaint_en="crushing chest pain",
        signs_fr=(
            "irradiation dans le bras gauche",
            "sueurs profuses",
            "nausées",
            "angoisse de mort",
        ),
        signs_en=(
            "radiating to the left arm",
            "profuse sweating",
            "nausea",
            "sense of impending doom",
        ),
        history_fr=(
            "hypertension artérielle",
            "tabagisme actif",
            "diabète de type 2",
            "hypercholestérolémie",
        ),
        history_en=("hypertension", "current smoker", "type 2 diabetes", "high cholesterol"),
        onset="minutes",
        justification=(
            "Douleur thoracique constrictive avec irradiation brachiale et sueurs : tableau "
            "évocateur d'un syndrome coronarien aigu, dont le pronostic dépend du délai de "
            "reperfusion."
        ),
        recommendation=(
            "Prise en charge immédiate en salle de déchocage, électrocardiogramme dans les dix "
            "minutes, appel du cardiologue et du 15 (SAMU) si la prise en charge ne peut être "
            "immédiate."
        ),
    ),
    Presentation(
        id="avc_deficit_focal",
        level="URGENCE_VITALE",
        vitals_profile="intermediate",
        age_range=(55, 92),
        complaint_fr="déficit moteur brutal d'un hémicorps",
        complaint_en="sudden weakness on one side of the body",
        signs_fr=(
            "troubles de la parole",
            "déviation de la bouche",
            "perte d'équilibre",
            "vision double",
        ),
        signs_en=("slurred speech", "facial droop", "loss of balance", "double vision"),
        history_fr=(
            "fibrillation auriculaire",
            "hypertension artérielle",
            "accident ischémique transitoire",
        ),
        history_en=(
            "atrial fibrillation",
            "hypertension",
            "previous transient ischaemic attack",
        ),
        onset="minutes",
        justification=(
            "Déficit neurologique focal d'installation brutale : suspicion d'accident vasculaire "
            "cérébral, avec une fenêtre thérapeutique de thrombolyse très courte."
        ),
        recommendation=(
            "Alerte thrombolyse immédiate, imagerie cérébrale en urgence, contact de l'unité "
            "neurovasculaire et du 15 (SAMU). Noter précisément l'heure de début des symptômes."
        ),
    ),
    Presentation(
        id="asthme_aigu_grave",
        level="URGENCE_VITALE",
        vitals_profile="critical",
        age_range=(8, 70),
        complaint_fr="crise d'asthme avec difficulté à terminer ses phrases",
        complaint_en="asthma attack, unable to finish sentences",
        signs_fr=(
            "tirage intercostal",
            "sifflements audibles",
            "cyanose des lèvres",
            "position assise penchée en avant",
        ),
        signs_en=(
            "intercostal retractions",
            "audible wheezing",
            "blue lips",
            "sitting forward to breathe",
        ),
        history_fr=(
            "asthme sévère",
            "hospitalisation antérieure en réanimation",
            "allergie aux acariens",
        ),
        history_en=("severe asthma", "previous intensive care admission", "dust mite allergy"),
        onset="heures",
        justification=(
            "L'impossibilité de terminer une phrase et le tirage signent une crise d'asthme aiguë "
            "grave, avec risque d'épuisement respiratoire à court terme."
        ),
        recommendation=(
            "Oxygénothérapie et bronchodilatateurs nébulisés sans délai, corticothérapie "
            "systémique, surveillance continue et appel du 15 (SAMU) en cas d'aggravation."
        ),
    ),
    Presentation(
        id="choc_anaphylactique",
        level="URGENCE_VITALE",
        vitals_profile="critical",
        age_range=(3, 75),
        complaint_fr="malaise avec gonflement du visage après une piqûre d'insecte",
        complaint_en="collapse with facial swelling after an insect sting",
        signs_fr=(
            "urticaire généralisée",
            "gêne à la déglutition",
            "voix rauque",
            "chute de la tension",
        ),
        signs_en=(
            "generalised hives",
            "difficulty swallowing",
            "hoarse voice",
            "dropping blood pressure",
        ),
        history_fr=(
            "allergie connue aux hyménoptères",
            "asthme",
            "porteur d'un stylo d'adrénaline",
        ),
        history_en=(
            "known wasp venom allergy",
            "asthma",
            "carries an adrenaline auto-injector",
        ),
        onset="minutes",
        justification=(
            "Atteinte cutanée, respiratoire et hémodynamique après exposition à un allergène : "
            "choc anaphylactique, dont le traitement ne souffre aucun délai."
        ),
        recommendation=(
            "Adrénaline intramusculaire immédiate, position allongée jambes surélevées, "
            "oxygénothérapie et appel du 15 (SAMU). Surveillance prolongée du fait du risque de "
            "réaction biphasique."
        ),
    ),
    Presentation(
        id="hemorragie_digestive_haute",
        level="URGENCE_VITALE",
        vitals_profile="critical",
        age_range=(40, 90),
        complaint_fr="vomissements de sang rouge",
        complaint_en="vomiting bright red blood",
        signs_fr=("pâleur marquée", "selles noires", "vertiges en se levant", "soif intense"),
        signs_en=("marked pallor", "black stools", "dizziness on standing", "intense thirst"),
        history_fr=(
            "cirrhose",
            "ulcère gastroduodénal",
            "traitement anti-inflammatoire au long cours",
        ),
        history_en=("liver cirrhosis", "peptic ulcer", "long-term anti-inflammatory treatment"),
        onset="heures",
        justification=(
            "Hématémèse avec signes de mauvaise tolérance : hémorragie digestive haute active, "
            "avec risque de choc hémorragique."
        ),
        recommendation=(
            "Deux voies veineuses de gros calibre, bilan avec groupage et commande de culots "
            "globulaires, avis gastro-entérologique urgent pour endoscopie."
        ),
    ),
    Presentation(
        id="sepsis_grave",
        level="URGENCE_VITALE",
        vitals_profile="critical",
        age_range=(35, 92),
        complaint_fr="fièvre avec frissons et confusion",
        complaint_en="fever with shivering and confusion",
        signs_fr=(
            "marbrures des genoux",
            "extrémités froides",
            "somnolence",
            "diminution des urines",
        ),
        signs_en=("mottled knees", "cold extremities", "drowsiness", "reduced urine output"),
        history_fr=("immunodépression", "chimiothérapie en cours", "sonde urinaire à demeure"),
        history_en=("immunosuppression", "ongoing chemotherapy", "indwelling urinary catheter"),
        onset="heures",
        justification=(
            "Fièvre associée à des signes d'hypoperfusion et à une altération de la vigilance : "
            "sepsis avec défaillance d'organe, dont la mortalité croît de façon horaire."
        ),
        recommendation=(
            "Hémocultures puis antibiothérapie probabiliste dans l'heure, remplissage vasculaire, "
            "mesure du lactate et avis réanimateur immédiat."
        ),
    ),
    Presentation(
        id="syndrome_meninge",
        level="URGENCE_VITALE",
        vitals_profile="intermediate",
        age_range=(1, 45),
        complaint_fr="fièvre élevée avec raideur de la nuque",
        complaint_en="high fever with neck stiffness",
        signs_fr=(
            "céphalées intenses",
            "gêne à la lumière",
            "somnolence",
            "taches violacées sur la peau",
        ),
        signs_en=(
            "severe headache",
            "discomfort in bright light",
            "drowsiness",
            "purple skin blotches",
        ),
        history_fr=("vaccination incomplète", "otite récente", "vie en collectivité"),
        history_en=(
            "incomplete vaccination",
            "recent ear infection",
            "lives in shared accommodation",
        ),
        onset="heures",
        justification=(
            "Association fièvre, raideur de nuque et trouble de la vigilance : syndrome méningé "
            "fébrile, avec suspicion de purpura fulminans en présence de lésions cutanées."
        ),
        recommendation=(
            "Isolement gouttelettes, antibiothérapie sans attendre la ponction lombaire en cas de "
            "purpura, avis infectiologique et réanimateur immédiat, appel du 15 (SAMU)."
        ),
    ),
    Presentation(
        id="traumatisme_cranien_grave",
        level="URGENCE_VITALE",
        vitals_profile="critical",
        age_range=(15, 85),
        complaint_fr="chute d'une hauteur avec perte de connaissance",
        complaint_en="fall from height with loss of consciousness",
        signs_fr=(
            "vomissements répétés",
            "confusion persistante",
            "plaie du cuir chevelu",
            "amnésie des faits",
        ),
        signs_en=(
            "repeated vomiting",
            "persistent confusion",
            "scalp wound",
            "no memory of the fall",
        ),
        history_fr=("traitement anticoagulant", "consommation d'alcool", "chutes à répétition"),
        history_en=("anticoagulant therapy", "alcohol use", "recurrent falls"),
        onset="heures",
        justification=(
            "Traumatisme crânien avec perte de connaissance, vomissements et confusion : risque "
            "d'hématome intracrânien, majoré par le traitement anticoagulant."
        ),
        recommendation=(
            "Immobilisation du rachis cervical, scanner cérébral en urgence, surveillance "
            "neurologique rapprochée et avis neurochirurgical."
        ),
    ),
    Presentation(
        id="crise_convulsive_prolongee",
        level="URGENCE_VITALE",
        vitals_profile="critical",
        age_range=(2, 70),
        complaint_fr="crise convulsive qui ne cède pas",
        complaint_en="a seizure that does not stop",
        signs_fr=(
            "morsure de langue",
            "perte d'urines",
            "respiration bruyante",
            "absence de reprise de conscience",
        ),
        signs_en=(
            "tongue biting",
            "incontinence",
            "noisy breathing",
            "not regaining consciousness",
        ),
        history_fr=("épilepsie connue", "arrêt récent du traitement", "sevrage alcoolique"),
        history_en=("known epilepsy", "recently stopped medication", "alcohol withdrawal"),
        onset="minutes",
        justification=(
            "Crise convulsive prolongée sans reprise de conscience : état de mal épileptique, avec "
            "risque de souffrance cérébrale et d'atteinte des voies aériennes."
        ),
        recommendation=(
            "Protection des voies aériennes, oxygénothérapie, benzodiazépine sans délai, "
            "glycémie capillaire et appel du 15 (SAMU)."
        ),
    ),
    Presentation(
        id="intoxication_medicamenteuse_volontaire",
        level="URGENCE_VITALE",
        vitals_profile="critical",
        age_range=(14, 60),
        complaint_fr="ingestion volontaire d'une boîte de médicaments",
        complaint_en="deliberate ingestion of a box of tablets",
        signs_fr=(
            "somnolence croissante",
            "vomissements",
            "propos incohérents",
            "ralentissement respiratoire",
        ),
        signs_en=("increasing drowsiness", "vomiting", "incoherent speech", "slow breathing"),
        history_fr=("dépression", "tentative de suicide antérieure", "rupture récente"),
        history_en=("depression", "previous suicide attempt", "recent break-up"),
        onset="heures",
        justification=(
            "Intoxication médicamenteuse volontaire récente avec retentissement neurologique : "
            "risque toxique immédiat et risque suicidaire élevé."
        ),
        recommendation=(
            "Prise en charge somatique immédiate, contact du centre antipoison, surveillance "
            "continue et évaluation psychiatrique dès la stabilisation. Ne pas laisser le patient seul."
        ),
    ),
    Presentation(
        id="pre_eclampsie_severe",
        level="URGENCE_VITALE",
        vitals_profile="intermediate",
        age_range=(18, 44),
        complaint_fr="céphalées intenses au troisième trimestre de grossesse",
        complaint_en="severe headache in the third trimester of pregnancy",
        signs_fr=(
            "mouches devant les yeux",
            "douleur en barre sous les côtes",
            "œdèmes du visage",
            "prise de poids rapide",
        ),
        signs_en=(
            "visual floaters",
            "band-like pain under the ribs",
            "facial swelling",
            "rapid weight gain",
        ),
        history_fr=("première grossesse", "hypertension gravidique", "grossesse gémellaire"),
        history_en=("first pregnancy", "pregnancy-induced hypertension", "twin pregnancy"),
        onset="heures",
        justification=(
            "Céphalées, troubles visuels et douleur épigastrique en fin de grossesse : "
            "pré-éclampsie sévère, avec risque d'éclampsie et de souffrance fœtale."
        ),
        recommendation=(
            "Transfert immédiat en maternité de niveau adapté, mesure de la tension et de la "
            "protéinurie, surveillance du rythme cardiaque fœtal, avis obstétrical sans délai."
        ),
    ),
    Presentation(
        id="rupture_anevrisme_aorte",
        level="URGENCE_VITALE",
        vitals_profile="critical",
        age_range=(60, 90),
        complaint_fr="douleur abdominale brutale avec malaise",
        complaint_en="sudden abdominal pain with collapse",
        signs_fr=(
            "masse abdominale battante",
            "douleur irradiant dans le dos",
            "pâleur",
            "extrémités froides",
        ),
        signs_en=(
            "pulsatile abdominal mass",
            "pain radiating to the back",
            "pallor",
            "cold extremities",
        ),
        history_fr=(
            "anévrisme de l'aorte connu",
            "tabagisme ancien",
            "artérite des membres inférieurs",
        ),
        history_en=("known aortic aneurysm", "former smoker", "peripheral artery disease"),
        onset="minutes",
        justification=(
            "Douleur abdominale brutale, masse battante et mauvaise tolérance hémodynamique : "
            "rupture d'anévrisme de l'aorte abdominale jusqu'à preuve du contraire."
        ),
        recommendation=(
            "Urgence chirurgicale vitale : alerter le bloc et le chirurgien vasculaire, transfusion "
            "anticipée, appel du 15 (SAMU) sans délai."
        ),
    ),
    Presentation(
        id="embolie_pulmonaire",
        level="URGENCE_VITALE",
        vitals_profile="critical",
        age_range=(30, 85),
        complaint_fr="essoufflement brutal avec douleur au côté",
        complaint_en="sudden breathlessness with side pain",
        signs_fr=(
            "douleur au mollet",
            "accélération du pouls",
            "malaise à l'effort",
            "crachats sanglants",
        ),
        signs_en=("calf pain", "racing pulse", "collapse on exertion", "coughing up blood"),
        history_fr=(
            "chirurgie récente",
            "immobilisation prolongée",
            "contraception œstroprogestative",
            "cancer en cours de traitement",
        ),
        history_en=(
            "recent surgery",
            "prolonged immobilisation",
            "combined oral contraception",
            "cancer under treatment",
        ),
        onset="heures",
        justification=(
            "Dyspnée brutale avec douleur pleurale et facteur de risque thromboembolique : "
            "suspicion d'embolie pulmonaire, potentiellement grave d'emblée."
        ),
        recommendation=(
            "Oxygénothérapie, score de probabilité clinique, angioscanner en urgence et "
            "anticoagulation dès la suspicion forte, après avis médical."
        ),
    ),
    Presentation(
        id="acidocetose_diabetique",
        level="URGENCE_VITALE",
        vitals_profile="critical",
        age_range=(10, 60),
        complaint_fr="soif intense avec respiration rapide et profonde",
        complaint_en="intense thirst with deep rapid breathing",
        signs_fr=(
            "haleine fruitée",
            "douleurs abdominales",
            "somnolence",
            "urines très abondantes",
        ),
        signs_en=(
            "fruity breath",
            "abdominal pain",
            "drowsiness",
            "passing large amounts of urine",
        ),
        history_fr=(
            "diabète de type 1",
            "arrêt des injections d'insuline",
            "infection récente",
        ),
        history_en=("type 1 diabetes", "stopped insulin injections", "recent infection"),
        onset="heures",
        justification=(
            "Syndrome polyuro-polydipsique avec polypnée et somnolence chez un diabétique : "
            "acidocétose diabétique, urgence métabolique."
        ),
        recommendation=(
            "Glycémie et cétonémie capillaires immédiates, réhydratation intraveineuse, insuline "
            "en continu et surveillance du potassium en unité de soins continus."
        ),
    ),
    Presentation(
        id="hypoglycemie_severe",
        level="URGENCE_VITALE",
        vitals_profile="critical",
        age_range=(20, 88),
        complaint_fr="malaise avec sueurs et propos incohérents",
        complaint_en="collapse with sweating and confused speech",
        signs_fr=(
            "tremblements",
            "pâleur",
            "agressivité inhabituelle",
            "difficulté à se réveiller",
        ),
        signs_en=("tremor", "pallor", "unusual aggression", "difficult to rouse"),
        history_fr=("diabète traité par insuline", "repas sauté", "insuffisance rénale"),
        history_en=("insulin-treated diabetes", "missed meal", "kidney failure"),
        onset="minutes",
        justification=(
            "Trouble de la conscience avec signes adrénergiques chez un patient sous insuline : "
            "hypoglycémie sévère, réversible mais rapidement délétère pour le cerveau."
        ),
        recommendation=(
            "Glycémie capillaire immédiate, resucrage par voie intraveineuse si trouble de la "
            "conscience, surveillance jusqu'à normalisation puis recherche de la cause."
        ),
    ),
    Presentation(
        id="occlusion_intestinale",
        level="URGENCE_VITALE",
        vitals_profile="intermediate",
        age_range=(40, 88),
        complaint_fr="arrêt des gaz et des selles avec ventre distendu",
        complaint_en="no bowel movements or wind with a distended abdomen",
        signs_fr=(
            "vomissements fécaloïdes",
            "douleurs en crampes",
            "ventre tendu",
            "absence de bruits intestinaux",
        ),
        signs_en=("faeculent vomiting", "cramping pain", "rigid abdomen", "absent bowel sounds"),
        history_fr=("chirurgie abdominale antérieure", "hernie non opérée", "cancer colique"),
        history_en=("previous abdominal surgery", "untreated hernia", "colon cancer"),
        onset="heures",
        justification=(
            "Arrêt du transit avec distension et vomissements : occlusion intestinale, avec risque "
            "d'ischémie digestive et de perforation."
        ),
        recommendation=(
            "À jeun strict, sonde nasogastrique en aspiration, correction hydroélectrolytique, "
            "scanner abdominal et avis chirurgical urgent."
        ),
    ),
    Presentation(
        id="brulure_etendue",
        level="URGENCE_VITALE",
        vitals_profile="critical",
        age_range=(5, 70),
        complaint_fr="brûlure étendue du tronc et des bras",
        complaint_en="extensive burns to the torso and arms",
        signs_fr=(
            "peau cartonnée par endroits",
            "suies autour du nez",
            "voix modifiée",
            "douleur majeure",
        ),
        signs_en=(
            "leathery skin in places",
            "soot around the nose",
            "changed voice",
            "severe pain",
        ),
        history_fr=("accident domestique", "incendie en espace clos", "épilepsie"),
        history_en=("domestic accident", "fire in a closed space", "epilepsy"),
        onset="minutes",
        justification=(
            "Brûlure étendue avec signes d'inhalation de fumées : risque d'obstruction des voies "
            "aériennes et de choc hypovolémique dans les heures qui suivent."
        ),
        recommendation=(
            "Refroidissement puis couverture stérile, oxygénothérapie à haut débit, remplissage "
            "vasculaire et transfert vers un centre de traitement des brûlés via le 15 (SAMU)."
        ),
    ),
    Presentation(
        id="hemorragie_du_post_partum",
        level="URGENCE_VITALE",
        vitals_profile="critical",
        age_range=(18, 44),
        complaint_fr="saignement abondant après un accouchement récent",
        complaint_en="heavy bleeding after a recent delivery",
        signs_fr=("caillots volumineux", "vertiges", "pâleur", "pouls filant"),
        signs_en=("large clots", "dizziness", "pallor", "thready pulse"),
        history_fr=(
            "accouchement il y a moins de 48 heures",
            "grossesse gémellaire",
            "césarienne",
        ),
        history_en=("delivery within the last 48 hours", "twin pregnancy", "caesarean section"),
        onset="heures",
        justification=(
            "Saignement abondant du post-partum avec retentissement hémodynamique : hémorragie de "
            "la délivrance, première cause de mortalité maternelle évitable."
        ),
        recommendation=(
            "Appel immédiat de l'équipe obstétricale et anesthésique, deux voies veineuses, "
            "utérotoniques, commande de produits sanguins et transfert en salle de naissance."
        ),
    ),
    Presentation(
        id="bronchiolite_grave_nourrisson",
        level="URGENCE_VITALE",
        vitals_profile="critical",
        age_range=(0, 1),
        complaint_fr="nourrisson qui respire vite et refuse de boire",
        complaint_en="infant breathing fast and refusing to feed",
        signs_fr=(
            "battement des ailes du nez",
            "creusement entre les côtes",
            "pauses respiratoires",
            "lèvres bleutées",
        ),
        signs_en=("nasal flaring", "chest indrawing", "pauses in breathing", "bluish lips"),
        history_fr=(
            "prématurité",
            "épidémie de bronchiolite en cours",
            "frère ou sœur enrhumé",
        ),
        history_en=("premature birth", "ongoing bronchiolitis season", "sibling with a cold"),
        onset="jours",
        justification=(
            "Détresse respiratoire du nourrisson avec signes de lutte, pauses respiratoires et "
            "refus alimentaire : bronchiolite grave, à risque d'épuisement rapide."
        ),
        recommendation=(
            "Oxygénothérapie, désobstruction rhinopharyngée, monitorage continu et hospitalisation "
            "en unité pédiatrique ; appel du 15 (SAMU) en cas d'apnée."
        ),
    ),
    Presentation(
        id="plaie_penetrante_thoracique",
        level="URGENCE_VITALE",
        vitals_profile="critical",
        age_range=(16, 60),
        complaint_fr="plaie par arme blanche au thorax",
        complaint_en="stab wound to the chest",
        signs_fr=(
            "difficulté à respirer",
            "plaie qui aspire l'air",
            "agitation",
            "pâleur extrême",
        ),
        signs_en=("difficulty breathing", "sucking chest wound", "agitation", "extreme pallor"),
        history_fr=("agression", "consommation d'alcool", "aucun antécédent connu"),
        history_en=("assault", "alcohol use", "no known medical history"),
        onset="minutes",
        justification=(
            "Plaie pénétrante thoracique avec détresse respiratoire : suspicion de pneumothorax "
            "compressif ou d'hémothorax, pronostic vital engagé en quelques minutes."
        ),
        recommendation=(
            "Pansement occlusif trois côtés, oxygénothérapie, voies veineuses, alerte "
            "chirurgicale et déchocage immédiat. Signalement judiciaire à prévoir."
        ),
    ),
    Presentation(
        id="deshydratation_severe_nourrisson",
        level="URGENCE_VITALE",
        vitals_profile="critical",
        age_range=(0, 2),
        complaint_fr="diarrhée profuse chez un nourrisson devenu somnolent",
        complaint_en="profuse diarrhoea in an infant who has become drowsy",
        signs_fr=(
            "yeux creux",
            "pli cutané persistant",
            "couches sèches depuis huit heures",
            "pleurs sans larmes",
        ),
        signs_en=(
            "sunken eyes",
            "skin tenting",
            "dry nappies for eight hours",
            "crying without tears",
        ),
        history_fr=(
            "gastro-entérite en crèche",
            "vomissements associés",
            "faible poids de naissance",
        ),
        history_en=("gastroenteritis at nursery", "associated vomiting", "low birth weight"),
        onset="jours",
        justification=(
            "Déshydratation sévère du nourrisson avec somnolence et anurie : risque de collapsus, "
            "la tolérance de l'enfant à la perte hydrique étant très faible."
        ),
        recommendation=(
            "Voie veineuse et réhydratation intraveineuse immédiate, pesée, ionogramme et "
            "hospitalisation en pédiatrie."
        ),
    ),
    Presentation(
        id="crise_suicidaire_aigue",
        level="URGENCE_VITALE",
        vitals_profile="normal",
        age_range=(14, 70),
        complaint_fr="idées suicidaires avec un scénario précis et des moyens à disposition",
        complaint_en="suicidal thoughts with a specific plan and means at hand",
        signs_fr=("propos d'adieu", "isolement récent", "refus de soins", "agitation"),
        signs_en=("saying goodbye", "recent withdrawal", "refusing care", "agitation"),
        history_fr=("trouble dépressif", "tentative récente", "consommation d'alcool"),
        history_en=("depressive disorder", "recent attempt", "alcohol use"),
        onset="heures",
        justification=(
            "Crise suicidaire avec scénario construit et moyens accessibles : le risque de passage "
            "à l'acte est immédiat, même en l'absence de toute anomalie des constantes."
        ),
        recommendation=(
            "Ne jamais laisser le patient seul, retirer les moyens létaux, évaluation "
            "psychiatrique en urgence et hospitalisation si le risque persiste."
        ),
    ),
    # ------------------------------------------------------------------
    # URGENCE_MODEREE — care within a few hours
    # ------------------------------------------------------------------
    Presentation(
        id="suspicion_appendicite",
        level="URGENCE_MODEREE",
        vitals_profile="intermediate",
        age_range=(6, 45),
        complaint_fr="douleur de la fosse iliaque droite",
        complaint_en="pain in the lower right abdomen",
        signs_fr=("nausées", "perte d'appétit", "douleur à la décompression", "fièvre modérée"),
        signs_en=("nausea", "loss of appetite", "rebound tenderness", "moderate fever"),
        history_fr=(
            "aucun antécédent chirurgical",
            "épisodes douloureux similaires",
            "constipation récente",
        ),
        history_en=("no previous surgery", "similar pain episodes", "recent constipation"),
        onset="heures",
        justification=(
            "Douleur de la fosse iliaque droite avec signes d'irritation péritonéale et fièvre : "
            "suspicion d'appendicite aiguë, à confirmer avant qu'elle ne se complique."
        ),
        recommendation=(
            "Patient à jeun, antalgie, bilan biologique et imagerie abdominale, avis chirurgical "
            "dans les heures qui suivent. Réévaluer immédiatement en cas de défense généralisée."
        ),
    ),
    Presentation(
        id="colique_nephretique",
        level="URGENCE_MODEREE",
        vitals_profile="intermediate",
        age_range=(20, 70),
        complaint_fr="douleur lombaire en coup de poignard irradiant vers l'aine",
        complaint_en="stabbing flank pain radiating to the groin",
        signs_fr=(
            "agitation permanente",
            "nausées",
            "envies fréquentes d'uriner",
            "sang dans les urines",
        ),
        signs_en=(
            "constant restlessness",
            "nausea",
            "frequent urge to urinate",
            "blood in the urine",
        ),
        history_fr=(
            "calculs rénaux connus",
            "faible consommation d'eau",
            "épisode identique il y a deux ans",
        ),
        history_en=(
            "known kidney stones",
            "low fluid intake",
            "identical episode two years ago",
        ),
        onset="heures",
        justification=(
            "Douleur lombaire paroxystique irradiant vers l'aine avec hématurie : colique "
            "néphrétique, très douloureuse mais sans détresse vitale en l'absence de fièvre."
        ),
        recommendation=(
            "Antalgie par anti-inflammatoire en l'absence de contre-indication, bandelette "
            "urinaire, imagerie et réévaluation. Retour immédiat en cas de fièvre ou d'anurie."
        ),
    ),
    Presentation(
        id="fracture_poignet_deplacee",
        level="URGENCE_MODEREE",
        vitals_profile="intermediate",
        age_range=(8, 85),
        complaint_fr="poignet déformé après une chute sur la main",
        complaint_en="deformed wrist after falling onto the hand",
        signs_fr=(
            "gonflement important",
            "impossibilité de bouger les doigts sans douleur",
            "hématome",
            "doigts bien colorés",
        ),
        signs_en=(
            "marked swelling",
            "cannot move fingers without pain",
            "bruising",
            "fingers well perfused",
        ),
        history_fr=("ostéoporose", "chute de sa hauteur", "pratique sportive"),
        history_en=("osteoporosis", "fall from standing height", "plays sport"),
        onset="heures",
        justification=(
            "Déformation et impotence fonctionnelle après un traumatisme direct : fracture du "
            "poignet probable, sans atteinte vasculo-nerveuse immédiate."
        ),
        recommendation=(
            "Immobilisation par attelle, antalgie, radiographie puis avis orthopédique pour "
            "réduction. Surveiller la coloration et la sensibilité des doigts."
        ),
    ),
    Presentation(
        id="pyelonephrite",
        level="URGENCE_MODEREE",
        vitals_profile="intermediate",
        age_range=(16, 75),
        complaint_fr="fièvre avec douleur du dos et brûlures urinaires",
        complaint_en="fever with back pain and burning on urination",
        signs_fr=("frissons", "urines troubles", "douleur à la percussion lombaire", "nausées"),
        signs_en=("shivering", "cloudy urine", "tenderness over the kidney", "nausea"),
        history_fr=("infections urinaires à répétition", "grossesse en cours", "diabète"),
        history_en=("recurrent urinary infections", "current pregnancy", "diabetes"),
        onset="jours",
        justification=(
            "Fièvre associée à une douleur lombaire et à des signes urinaires : pyélonéphrite "
            "aiguë, qui impose une antibiothérapie rapide sans être d'emblée une détresse vitale."
        ),
        recommendation=(
            "Bandelette et examen cytobactériologique des urines, hémocultures si frissons, "
            "antibiothérapie adaptée dans les heures qui suivent, hospitalisation si grossesse."
        ),
    ),
    Presentation(
        id="crise_asthme_moderee",
        level="URGENCE_MODEREE",
        vitals_profile="intermediate",
        age_range=(5, 65),
        complaint_fr="gêne respiratoire avec sifflements, phrases complètes possibles",
        complaint_en="wheezy breathing, still able to speak in full sentences",
        signs_fr=(
            "toux sèche",
            "oppression thoracique",
            "amélioration partielle après inhalateur",
            "pas de cyanose",
        ),
        signs_en=("dry cough", "chest tightness", "partial relief after inhaler", "no cyanosis"),
        history_fr=("asthme d'effort", "rhinite allergique", "infection virale en cours"),
        history_en=("exercise-induced asthma", "allergic rhinitis", "current viral infection"),
        onset="heures",
        justification=(
            "Crise d'asthme avec parole conservée et réponse partielle au bronchodilatateur : "
            "exacerbation modérée, à traiter et surveiller sans urgence vitale immédiate."
        ),
        recommendation=(
            "Bronchodilatateurs inhalés répétés, corticothérapie orale courte, réévaluation à "
            "une heure. Consultation immédiate si la parole devient difficile."
        ),
    ),
    Presentation(
        id="plaie_profonde_suturable",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(4, 80),
        complaint_fr="plaie profonde de l'avant-bras nécessitant des points",
        complaint_en="deep forearm wound needing stitches",
        signs_fr=(
            "saignement contrôlé par compression",
            "berges nettes",
            "mobilité conservée",
            "sensibilité normale",
        ),
        signs_en=(
            "bleeding controlled by pressure",
            "clean wound edges",
            "movement preserved",
            "normal sensation",
        ),
        history_fr=(
            "vaccination antitétanique à jour",
            "accident de bricolage",
            "aucun traitement",
        ),
        history_en=("tetanus vaccination up to date", "DIY accident", "no medication"),
        onset="heures",
        justification=(
            "Plaie profonde mais sans saignement actif ni atteinte tendineuse ou nerveuse : "
            "suture nécessaire dans les six heures pour limiter le risque infectieux."
        ),
        recommendation=(
            "Lavage abondant, exploration de la plaie, suture et vérification du statut "
            "antitétanique. Consignes de surveillance de l'infection à la sortie."
        ),
    ),
    Presentation(
        id="entorse_cheville_grave",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(10, 60),
        complaint_fr="cheville très douloureuse après une torsion, appui impossible",
        complaint_en="very painful ankle after twisting it, unable to bear weight",
        signs_fr=(
            "gonflement immédiat",
            "hématome sous la malléole",
            "douleur à la palpation osseuse",
            "pied bien coloré",
        ),
        signs_en=(
            "immediate swelling",
            "bruising below the ankle bone",
            "bony tenderness",
            "foot well perfused",
        ),
        history_fr=("entorses répétées", "sport de pivot", "surpoids"),
        history_en=("repeated sprains", "pivoting sport", "overweight"),
        onset="heures",
        justification=(
            "Impossibilité d'appui avec douleur osseuse après torsion : les critères d'Ottawa "
            "imposent une radiographie pour éliminer une fracture."
        ),
        recommendation=(
            "Glace, immobilisation et surélévation, antalgie, radiographie de cheville puis "
            "orientation vers une consultation orthopédique."
        ),
    ),
    Presentation(
        id="gastro_enterite_deshydratation_moderee",
        level="URGENCE_MODEREE",
        vitals_profile="intermediate",
        age_range=(1, 80),
        complaint_fr="diarrhées et vomissements avec fatigue",
        complaint_en="diarrhoea and vomiting with fatigue",
        signs_fr=(
            "bouche sèche",
            "urines rares et foncées",
            "douleurs abdominales diffuses",
            "vigilance conservée",
        ),
        signs_en=(
            "dry mouth",
            "scant dark urine",
            "diffuse abdominal pain",
            "alert and responsive",
        ),
        history_fr=("repas suspect", "épidémie familiale", "traitement diurétique"),
        history_en=("suspect meal", "family outbreak", "diuretic treatment"),
        onset="jours",
        justification=(
            "Pertes digestives prolongées avec signes de déshydratation modérée mais vigilance "
            "normale : réhydratation nécessaire dans les heures qui suivent."
        ),
        recommendation=(
            "Réhydratation orale fractionnée, ionogramme si terrain fragile, surveillance du "
            "poids et des urines. Retour immédiat en cas de somnolence ou de sang dans les selles."
        ),
    ),
    Presentation(
        id="migraine_severe",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(15, 60),
        complaint_fr="céphalée pulsatile invalidante avec nausées",
        complaint_en="disabling throbbing headache with nausea",
        signs_fr=(
            "gêne à la lumière et au bruit",
            "aura visuelle régressive",
            "examen neurologique normal",
            "épisodes identiques connus",
        ),
        signs_en=(
            "light and noise sensitivity",
            "resolving visual aura",
            "normal neurological examination",
            "known identical episodes",
        ),
        history_fr=(
            "migraine avec aura",
            "antécédents familiaux",
            "traitement de crise inefficace aujourd'hui",
        ),
        history_en=(
            "migraine with aura",
            "family history",
            "usual treatment ineffective today",
        ),
        onset="heures",
        justification=(
            "Céphalée typique d'une migraine déjà connue, avec examen neurologique normal : pas de "
            "critère de gravité, mais douleur invalidante justifiant une prise en charge rapide."
        ),
        recommendation=(
            "Antalgie adaptée au repos dans le calme et l'obscurité, antiémétique si besoin. "
            "Imagerie uniquement si la céphalée change de caractère ou s'installe brutalement."
        ),
    ),
    Presentation(
        id="fibrillation_auriculaire_toleree",
        level="URGENCE_MODEREE",
        vitals_profile="intermediate",
        age_range=(50, 88),
        complaint_fr="palpitations irrégulières",
        complaint_en="irregular palpitations",
        signs_fr=(
            "fatigue à l'effort",
            "pas de douleur thoracique",
            "tension conservée",
            "pouls irrégulier",
        ),
        signs_en=(
            "tiredness on exertion",
            "no chest pain",
            "blood pressure maintained",
            "irregular pulse",
        ),
        history_fr=(
            "hypertension artérielle",
            "apnées du sommeil",
            "consommation récente d'alcool",
        ),
        history_en=("hypertension", "sleep apnoea", "recent alcohol intake"),
        onset="heures",
        justification=(
            "Arythmie bien tolérée sur le plan hémodynamique, sans douleur thoracique ni signe "
            "d'insuffisance cardiaque : évaluation cardiologique nécessaire mais non immédiate."
        ),
        recommendation=(
            "Électrocardiogramme, ionogramme et bilan thyroïdien, évaluation du risque "
            "thromboembolique et avis cardiologique dans la journée."
        ),
    ),
    Presentation(
        id="zona_ophtalmique",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(45, 88),
        complaint_fr="éruption douloureuse sur le front et la paupière",
        complaint_en="painful rash on the forehead and eyelid",
        signs_fr=(
            "vésicules groupées d'un seul côté",
            "œil rouge",
            "larmoiement",
            "douleur à type de brûlure",
        ),
        signs_en=("clustered blisters on one side", "red eye", "watery eye", "burning pain"),
        history_fr=("varicelle dans l'enfance", "immunodépression", "stress récent"),
        history_en=("childhood chickenpox", "immunosuppression", "recent stress"),
        onset="jours",
        justification=(
            "Zona du territoire ophtalmique : le risque de complication cornéenne impose un "
            "traitement antiviral dans les 72 heures et un avis spécialisé rapide."
        ),
        recommendation=(
            "Antiviral par voie orale sans attendre, antalgie, avis ophtalmologique dans les 24 "
            "heures. Éviter tout contact avec des personnes non immunisées."
        ),
    ),
    Presentation(
        id="abces_dentaire",
        level="URGENCE_MODEREE",
        vitals_profile="intermediate",
        age_range=(12, 70),
        complaint_fr="joue gonflée avec douleur dentaire intense",
        complaint_en="swollen cheek with severe toothache",
        signs_fr=(
            "fièvre modérée",
            "difficulté à ouvrir la bouche",
            "ganglion sous la mâchoire",
            "déglutition possible",
        ),
        signs_en=(
            "moderate fever",
            "difficulty opening the mouth",
            "swollen gland under the jaw",
            "able to swallow",
        ),
        history_fr=("carie négligée", "suivi dentaire irrégulier", "diabète"),
        history_en=("neglected cavity", "irregular dental care", "diabetes"),
        onset="jours",
        justification=(
            "Cellulite d'origine dentaire avec fièvre et limitation de l'ouverture buccale : "
            "infection à traiter rapidement, sans signe d'extension cervicale pour l'instant."
        ),
        recommendation=(
            "Antibiothérapie et antalgie, avis stomatologique pour drainage. Consultation "
            "immédiate si la déglutition ou la respiration deviennent difficiles."
        ),
    ),
    Presentation(
        id="otite_moyenne_aigue_enfant",
        level="URGENCE_MODEREE",
        vitals_profile="intermediate",
        age_range=(1, 8),
        complaint_fr="enfant qui pleure en se tenant l'oreille avec de la fièvre",
        complaint_en="child crying and holding the ear, with fever",
        signs_fr=(
            "réveils nocturnes",
            "rhume depuis trois jours",
            "bon contact",
            "alimentation conservée",
        ),
        signs_en=("waking at night", "cold for three days", "interacts normally", "still eating"),
        history_fr=("otites à répétition", "vie en collectivité", "vaccination à jour"),
        history_en=("recurrent ear infections", "attends nursery", "vaccinations up to date"),
        onset="jours",
        justification=(
            "Otalgie fébrile chez un enfant en bon état général : otite moyenne aiguë probable, "
            "à examiner et traiter dans la journée sans critère de gravité."
        ),
        recommendation=(
            "Examen otoscopique, antalgie et antipyrétique, antibiothérapie selon l'âge et "
            "l'aspect du tympan. Réévaluation à 48 heures."
        ),
    ),
    Presentation(
        id="pneumopathie_communautaire",
        level="URGENCE_MODEREE",
        vitals_profile="intermediate",
        age_range=(25, 80),
        complaint_fr="toux grasse et fièvre avec point de côté",
        complaint_en="productive cough and fever with chest pain on breathing",
        signs_fr=("crachats colorés", "essoufflement à l'effort", "frissons", "vigilance normale"),
        signs_en=("coloured sputum", "breathless on exertion", "shivering", "fully alert"),
        history_fr=("tabagisme", "bronchopneumopathie chronique", "grippe récente"),
        history_en=("smoking", "chronic lung disease", "recent influenza"),
        onset="jours",
        justification=(
            "Foyer infectieux pulmonaire probable avec fièvre et douleur pleurale, sans détresse "
            "respiratoire ni trouble de la vigilance : traitement à instaurer dans la journée."
        ),
        recommendation=(
            "Radiographie thoracique, évaluation de la gravité par score clinique, "
            "antibiothérapie probabiliste et réévaluation à 48 heures."
        ),
    ),
    Presentation(
        id="lombalgie_febrile",
        level="URGENCE_MODEREE",
        vitals_profile="intermediate",
        age_range=(30, 80),
        complaint_fr="douleur lombaire avec fièvre",
        complaint_en="low back pain with fever",
        signs_fr=(
            "raideur du dos",
            "sueurs nocturnes",
            "pas de déficit moteur",
            "marche possible",
        ),
        signs_en=("stiff back", "night sweats", "no weakness", "able to walk"),
        history_fr=("injection récente", "toxicomanie intraveineuse", "immunodépression"),
        history_en=("recent injection", "intravenous drug use", "immunosuppression"),
        onset="jours",
        justification=(
            "Lombalgie fébrile : drapeau rouge imposant d'éliminer une spondylodiscite ou un "
            "abcès, même en l'absence de déficit neurologique."
        ),
        recommendation=(
            "Bilan inflammatoire et hémocultures, imagerie du rachis, avis spécialisé dans la "
            "journée. Consultation immédiate en cas de déficit moteur ou de troubles sphinctériens."
        ),
    ),
    Presentation(
        id="traumatisme_cranien_sous_anticoagulant",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(65, 92),
        complaint_fr="chute de sa hauteur avec choc à la tête, sans perte de connaissance",
        complaint_en="fall from standing height with a head knock, no loss of consciousness",
        signs_fr=(
            "bosse frontale",
            "céphalée légère",
            "pas de vomissement",
            "examen neurologique normal",
        ),
        signs_en=(
            "forehead lump",
            "mild headache",
            "no vomiting",
            "normal neurological examination",
        ),
        history_fr=(
            "traitement anticoagulant oral",
            "fibrillation auriculaire",
            "chutes récentes",
        ),
        history_en=("oral anticoagulant therapy", "atrial fibrillation", "recent falls"),
        onset="heures",
        justification=(
            "Traumatisme crânien sans signe de gravité immédiat, mais le traitement anticoagulant "
            "impose un scanner dans l'heure et une surveillance : l'hématome sous-dural du sujet "
            "âgé anticoagulé est justement asymptomatique à la phase initiale."
        ),
        recommendation=(
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
        complaint_fr="brûlure de l'avant-bras par liquide bouillant",
        complaint_en="forearm scald from boiling liquid",
        signs_fr=(
            "phlyctènes",
            "douleur vive",
            "surface inférieure à une paume",
            "pas d'atteinte du visage",
        ),
        signs_en=("blisters", "sharp pain", "smaller than one palm", "face not involved"),
        history_fr=(
            "accident de cuisine",
            "vaccination antitétanique à jour",
            "aucun traitement",
        ),
        history_en=("kitchen accident", "tetanus vaccination up to date", "no medication"),
        onset="heures",
        justification=(
            "Brûlure du deuxième degré de surface limitée, sans atteinte des zones fonctionnelles "
            "ni signe d'inhalation : soins locaux nécessaires dans les heures qui suivent."
        ),
        recommendation=(
            "Refroidissement, antalgie, pansement gras et évaluation de la profondeur à 48 heures. "
            "Consultation spécialisée si la cicatrisation traîne."
        ),
    ),
    Presentation(
        id="erysipele_jambe",
        level="URGENCE_MODEREE",
        vitals_profile="intermediate",
        age_range=(40, 88),
        complaint_fr="jambe rouge, chaude et douloureuse avec fièvre",
        complaint_en="red, hot, painful leg with fever",
        signs_fr=(
            "placard bien limité",
            "ganglion inguinal",
            "frissons",
            "pas de nécrose cutanée",
        ),
        signs_en=(
            "well-demarcated red patch",
            "swollen groin gland",
            "shivering",
            "no skin necrosis",
        ),
        history_fr=("insuffisance veineuse", "mycose entre les orteils", "surpoids"),
        history_en=("venous insufficiency", "athlete's foot", "overweight"),
        onset="jours",
        justification=(
            "Dermohypodermite bactérienne aiguë fébrile sans signe de gravité locale : "
            "antibiothérapie à débuter rapidement, surveillance de l'extension."
        ),
        recommendation=(
            "Antibiothérapie antistreptococcique, repos jambe surélevée, délimitation au crayon "
            "de la zone rouge et réévaluation à 48 heures. Retour immédiat si douleur disproportionnée."
        ),
    ),
    Presentation(
        id="corps_etranger_oculaire",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(16, 65),
        complaint_fr="sensation de corps étranger dans l'œil après meulage",
        complaint_en="foreign body sensation in the eye after grinding metal",
        signs_fr=(
            "œil rouge et larmoyant",
            "gêne à la lumière",
            "vision conservée",
            "clignements incessants",
        ),
        signs_en=("red watery eye", "light sensitivity", "vision preserved", "constant blinking"),
        history_fr=(
            "travail sans lunettes de protection",
            "aucun antécédent ophtalmologique",
            "port de lentilles",
        ),
        history_en=("working without safety glasses", "no eye history", "wears contact lenses"),
        onset="heures",
        justification=(
            "Corps étranger cornéen probable après projection métallique : risque d'abcès ou de "
            "rouille cornéenne, à retirer rapidement, sans menace immédiate pour la vision."
        ),
        recommendation=(
            "Mesure de l'acuité visuelle, examen à la fluorescéine, retrait par un praticien "
            "entraîné et avis ophtalmologique dans les 24 heures."
        ),
    ),
    Presentation(
        id="vertige_rotatoire_recent",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(35, 80),
        complaint_fr="vertiges rotatoires avec vomissements",
        complaint_en="spinning dizziness with vomiting",
        signs_fr=(
            "aggravation aux mouvements de tête",
            "marche instable",
            "pas de déficit moteur",
            "audition normale",
        ),
        signs_en=(
            "worse on head movement",
            "unsteady walking",
            "no limb weakness",
            "normal hearing",
        ),
        history_fr=("épisodes similaires", "migraine", "hypertension artérielle"),
        history_en=("similar episodes", "migraine", "hypertension"),
        onset="heures",
        justification=(
            "Syndrome vestibulaire aigu : l'absence de déficit focal oriente vers une cause "
            "périphérique, mais une origine centrale doit être écartée par l'examen."
        ),
        recommendation=(
            "Examen neurologique et manœuvres vestibulaires, antiémétique, imagerie cérébrale si "
            "le moindre signe central apparaît. Éviter la conduite jusqu'à résolution."
        ),
    ),
    Presentation(
        id="crise_drepanocytaire",
        level="URGENCE_MODEREE",
        vitals_profile="intermediate",
        age_range=(5, 45),
        complaint_fr="douleurs osseuses diffuses chez un patient drépanocytaire",
        complaint_en="widespread bone pain in a patient with sickle cell disease",
        signs_fr=(
            "douleur habituelle mais plus intense",
            "pas de fièvre élevée",
            "respiration normale",
            "hydratation insuffisante",
        ),
        signs_en=(
            "usual pain but more intense",
            "no high fever",
            "normal breathing",
            "poorly hydrated",
        ),
        history_fr=(
            "drépanocytose homozygote",
            "crises vaso-occlusives répétées",
            "épisode déclenché par le froid",
        ),
        history_en=(
            "sickle cell anaemia",
            "recurrent vaso-occlusive crises",
            "triggered by cold",
        ),
        onset="heures",
        justification=(
            "Crise vaso-occlusive typique sans syndrome thoracique aigu ni fièvre élevée : "
            "antalgie urgente mais pas de défaillance d'organe constatée."
        ),
        recommendation=(
            "Antalgie de palier adapté sans délai, hydratation, oxygénothérapie si la saturation "
            "baisse, recherche d'un facteur déclenchant. Surveillance respiratoire rapprochée."
        ),
    ),
    Presentation(
        id="reaction_allergique_cutanee",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(2, 70),
        complaint_fr="plaques d'urticaire étendues après la prise d'un médicament",
        complaint_en="widespread hives after taking a medication",
        signs_fr=(
            "démangeaisons intenses",
            "pas de gêne respiratoire",
            "déglutition normale",
            "tension conservée",
        ),
        signs_en=(
            "intense itching",
            "no breathing difficulty",
            "normal swallowing",
            "blood pressure maintained",
        ),
        history_fr=(
            "antibiotique débuté la veille",
            "terrain allergique",
            "aucune allergie connue",
        ),
        history_en=("antibiotic started yesterday", "allergic background", "no known allergy"),
        onset="heures",
        justification=(
            "Urticaire médicamenteuse étendue sans atteinte respiratoire ni hémodynamique : "
            "réaction allergique à traiter et à surveiller, sans critère d'anaphylaxie."
        ),
        recommendation=(
            "Arrêt du médicament suspect, antihistaminique, surveillance de deux heures. "
            "Consultation immédiate si gonflement du visage, gêne à avaler ou essoufflement."
        ),
    ),
    Presentation(
        id="douleur_thoracique_parietale",
        level="URGENCE_MODEREE",
        vitals_profile="normal",
        age_range=(18, 55),
        complaint_fr="douleur thoracique reproduite à la palpation après un effort de musculation",
        complaint_en="chest pain reproduced by pressing, after weight training",
        signs_fr=(
            "douleur au mouvement du bras",
            "pas de sueurs",
            "pas d'essoufflement",
            "électrocardiogramme normal",
        ),
        signs_en=(
            "pain on arm movement",
            "no sweating",
            "no breathlessness",
            "normal electrocardiogram",
        ),
        history_fr=(
            "aucun facteur de risque cardiovasculaire",
            "sport intensif récent",
            "non-fumeur",
        ),
        history_en=("no cardiovascular risk factors", "recent intensive sport", "non-smoker"),
        onset="jours",
        justification=(
            "Douleur thoracique reproductible à la palpation, d'allure pariétale, chez un sujet "
            "jeune sans facteur de risque : l'origine coronarienne reste à écarter formellement."
        ),
        recommendation=(
            "Électrocardiogramme et dosage de troponine pour éliminer une cause coronarienne, "
            "antalgie simple puis retour à domicile avec consignes de surveillance."
        ),
    ),
    Presentation(
        id="hyperglycemie_sans_cetose",
        level="URGENCE_MODEREE",
        vitals_profile="intermediate",
        age_range=(30, 80),
        complaint_fr="glycémie très élevée au doigt avec soif et fatigue",
        complaint_en="very high finger-prick glucose with thirst and fatigue",
        signs_fr=("urines fréquentes", "vision floue", "pas de cétones", "vigilance normale"),
        signs_en=("frequent urination", "blurred vision", "no ketones", "fully alert"),
        history_fr=("diabète de type 2", "oubli du traitement", "corticothérapie récente"),
        history_en=("type 2 diabetes", "missed medication", "recent steroid treatment"),
        onset="jours",
        justification=(
            "Déséquilibre glycémique franc sans cétose ni trouble de la vigilance : adaptation "
            "thérapeutique nécessaire rapidement, sans urgence vitale."
        ),
        recommendation=(
            "Hydratation, recherche de cétones, ionogramme, adaptation du traitement et avis "
            "diabétologique. Recherche d'un facteur déclenchant infectieux."
        ),
    ),
    Presentation(
        id="agitation_anxieuse",
        level="URGENCE_MODEREE",
        vitals_profile="intermediate",
        age_range=(16, 65),
        complaint_fr="crise d'angoisse avec sensation d'étouffement",
        complaint_en="panic attack with a feeling of suffocation",
        signs_fr=(
            "fourmillements des mains",
            "respiration rapide",
            "peur de mourir",
            "examen clinique normal",
        ),
        signs_en=(
            "tingling hands",
            "rapid breathing",
            "fear of dying",
            "normal physical examination",
        ),
        history_fr=(
            "trouble anxieux",
            "épisodes identiques",
            "consommation de café importante",
        ),
        history_en=("anxiety disorder", "identical episodes", "high caffeine intake"),
        onset="heures",
        justification=(
            "Tableau typique d'attaque de panique déjà connue, avec examen clinique normal : la "
            "prise en charge est rapide mais les causes organiques doivent être écartées."
        ),
        recommendation=(
            "Réassurance dans un endroit calme, contrôle de la respiration, électrocardiogramme "
            "et glycémie pour éliminer une cause organique, orientation vers un suivi psychologique."
        ),
    ),
    # ------------------------------------------------------------------
    # CONSULTATION_DIFFEREE — no immediate severity criterion
    # ------------------------------------------------------------------
    Presentation(
        id="rhinopharyngite",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(2, 70),
        complaint_fr="nez bouché et mal de gorge",
        complaint_en="blocked nose and sore throat",
        signs_fr=("éternuements", "fatigue légère", "pas de fièvre", "appétit conservé"),
        signs_en=("sneezing", "mild tiredness", "no fever", "normal appetite"),
        history_fr=("aucun antécédent notable", "épisodes hivernaux habituels", "non-fumeur"),
        history_en=("no notable history", "usual winter episodes", "non-smoker"),
        onset="jours",
        justification=(
            "Infection virale bénigne des voies aériennes supérieures, sans fièvre ni signe "
            "respiratoire de gravité : aucun critère d'urgence."
        ),
        recommendation=(
            "Traitement symptomatique et lavages de nez, consultation chez le médecin traitant "
            "si les symptômes persistent au-delà de dix jours ou si une fièvre apparaît."
        ),
    ),
    Presentation(
        id="lombalgie_mecanique",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(25, 70),
        complaint_fr="douleur du bas du dos après avoir porté une charge",
        complaint_en="lower back pain after lifting a heavy load",
        signs_fr=(
            "douleur soulagée au repos",
            "pas de fièvre",
            "pas de déficit des jambes",
            "marche possible",
        ),
        signs_en=("pain relieved by rest", "no fever", "no leg weakness", "able to walk"),
        history_fr=("travail physique", "épisodes similaires", "sédentarité"),
        history_en=("physical work", "similar episodes", "sedentary lifestyle"),
        onset="jours",
        justification=(
            "Lombalgie commune sans drapeau rouge : ni fièvre, ni déficit neurologique, ni "
            "traumatisme violent, ni altération de l'état général."
        ),
        recommendation=(
            "Antalgie simple, poursuite d'une activité adaptée plutôt que le repos strict, "
            "consultation du médecin traitant si la douleur dépasse quatre semaines."
        ),
    ),
    Presentation(
        id="renouvellement_ordonnance",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(35, 85),
        complaint_fr="demande de renouvellement d'un traitement chronique",
        complaint_en="request to renew a long-term prescription",
        signs_fr=(
            "aucun symptôme aigu",
            "traitement bien supporté",
            "constantes habituelles",
            "bon état général",
        ),
        signs_en=(
            "no acute symptoms",
            "medication well tolerated",
            "usual observations",
            "good general condition",
        ),
        history_fr=("hypertension équilibrée", "hypothyroïdie substituée", "suivi régulier"),
        history_en=(
            "well-controlled hypertension",
            "treated hypothyroidism",
            "regular follow-up",
        ),
        onset="semaines",
        justification=(
            "Demande administrative sans symptôme aigu : ce motif relève de la médecine de ville "
            "et non du service d'urgence."
        ),
        recommendation=(
            "Orienter vers le médecin traitant ou le pharmacien pour une dispensation de "
            "dépannage, expliquer le circuit adapté hors urgences."
        ),
    ),
    Presentation(
        id="eczema_poussee",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(1, 60),
        complaint_fr="plaques sèches qui démangent aux plis des coudes",
        complaint_en="dry itchy patches in the elbow creases",
        signs_fr=("peau épaissie", "grattage nocturne", "pas de suintement", "pas de fièvre"),
        signs_en=("thickened skin", "scratching at night", "no oozing", "no fever"),
        history_fr=("dermatite atopique", "asthme dans la famille", "arrêt récent des crèmes"),
        history_en=("atopic dermatitis", "asthma in the family", "recently stopped creams"),
        onset="semaines",
        justification=(
            "Poussée d'eczéma chronique sans signe de surinfection ni retentissement général : "
            "situation dermatologique courante, non urgente."
        ),
        recommendation=(
            "Reprise des émollients et d'un dermocorticoïde adapté, consultation programmée chez "
            "le dermatologue ou le médecin traitant."
        ),
    ),
    Presentation(
        id="conjonctivite_simple",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(2, 70),
        complaint_fr="œil rouge collé au réveil",
        complaint_en="red eye, sticky on waking",
        signs_fr=(
            "sécrétions claires",
            "vision normale",
            "pas de douleur profonde",
            "pas de gêne à la lumière",
        ),
        signs_en=("clear discharge", "normal vision", "no deep pain", "no light sensitivity"),
        history_fr=(
            "contage familial",
            "pas de port de lentilles",
            "aucun antécédent oculaire",
        ),
        history_en=("family member affected", "no contact lenses", "no eye history"),
        onset="jours",
        justification=(
            "Conjonctivite banale : vision conservée, absence de douleur profonde et de "
            "photophobie, ce qui écarte les causes ophtalmologiques graves."
        ),
        recommendation=(
            "Lavages oculaires au sérum physiologique et mesures d'hygiène. Consultation rapide "
            "si baisse de vision, douleur intense ou photophobie."
        ),
    ),
    Presentation(
        id="plaie_superficielle_doigt",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(5, 80),
        complaint_fr="petite coupure au doigt avec un couteau de cuisine",
        complaint_en="small finger cut from a kitchen knife",
        signs_fr=(
            "saignement arrêté",
            "plaie superficielle",
            "mobilité et sensibilité normales",
            "berges propres",
        ),
        signs_en=(
            "bleeding stopped",
            "superficial wound",
            "normal movement and sensation",
            "clean edges",
        ),
        history_fr=(
            "vaccination antitétanique à jour",
            "aucun traitement",
            "accident domestique",
        ),
        history_en=("tetanus vaccination up to date", "no medication", "domestic accident"),
        onset="heures",
        justification=(
            "Plaie superficielle sans saignement actif ni atteinte tendineuse ou nerveuse : "
            "aucun geste urgent nécessaire."
        ),
        recommendation=(
            "Nettoyage, désinfection et pansement, surveillance des signes d'infection. "
            "Consultation si rougeur, chaleur ou douleur croissante."
        ),
    ),
    Presentation(
        id="allergie_saisonniere",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(6, 60),
        complaint_fr="éternuements et yeux qui piquent au printemps",
        complaint_en="sneezing and itchy eyes in spring",
        signs_fr=(
            "nez qui coule clair",
            "pas de fièvre",
            "respiration normale",
            "symptômes en extérieur",
        ),
        signs_en=("clear runny nose", "no fever", "normal breathing", "symptoms outdoors"),
        history_fr=(
            "rhinite allergique connue",
            "antécédents familiaux d'allergie",
            "traitement habituel épuisé",
        ),
        history_en=(
            "known allergic rhinitis",
            "family history of allergy",
            "ran out of usual treatment",
        ),
        onset="semaines",
        justification=(
            "Rhinoconjonctivite allergique saisonnière typique, sans signe respiratoire bas ni "
            "retentissement général."
        ),
        recommendation=(
            "Antihistaminique et lavages de nez, consultation programmée pour un bilan "
            "allergologique si les symptômes sont invalidants chaque année."
        ),
    ),
    Presentation(
        id="douleur_dentaire_ancienne",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(15, 75),
        complaint_fr="douleur dentaire intermittente",
        complaint_en="on-and-off toothache",
        signs_fr=(
            "sensibilité au froid",
            "pas de gonflement",
            "pas de fièvre",
            "ouverture de bouche normale",
        ),
        signs_en=("sensitivity to cold", "no swelling", "no fever", "normal mouth opening"),
        history_fr=("carie connue", "rendez-vous dentaire prévu", "suivi irrégulier"),
        history_en=("known cavity", "dental appointment booked", "irregular follow-up"),
        onset="semaines",
        justification=(
            "Douleur dentaire chronique sans signe infectieux local ni général : relève du "
            "chirurgien-dentiste et non du service d'urgence."
        ),
        recommendation=(
            "Antalgie simple et consultation dentaire programmée. Retour aux urgences en cas de "
            "gonflement du visage, de fièvre ou de difficulté à avaler."
        ),
    ),
    Presentation(
        id="constipation_chronique",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(25, 85),
        complaint_fr="constipation avec ballonnements",
        complaint_en="constipation with bloating",
        signs_fr=("ventre souple", "gaz présents", "pas de sang", "poids stable"),
        signs_en=("soft abdomen", "passing wind", "no blood", "stable weight"),
        history_fr=("alimentation pauvre en fibres", "sédentarité", "traitement par opiacés"),
        history_en=("low-fibre diet", "sedentary lifestyle", "opioid treatment"),
        onset="semaines",
        justification=(
            "Constipation chronique avec transit gazeux conservé et abdomen souple : aucun signe "
            "d'occlusion ni d'alerte digestive."
        ),
        recommendation=(
            "Mesures hygiéno-diététiques et laxatif doux, consultation programmée chez le médecin "
            "traitant. Consultation urgente si arrêt des gaz, vomissements ou sang dans les selles."
        ),
    ),
    Presentation(
        id="verrue_plantaire",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(8, 55),
        complaint_fr="petite excroissance douloureuse sous le pied",
        complaint_en="small painful growth under the foot",
        signs_fr=(
            "gêne à la marche prolongée",
            "pas de rougeur",
            "pas de fièvre",
            "aspect stable",
        ),
        signs_en=("discomfort on long walks", "no redness", "no fever", "unchanged appearance"),
        history_fr=("fréquentation de la piscine", "aucun antécédent", "diabète absent"),
        history_en=("swimming pool use", "no medical history", "no diabetes"),
        onset="semaines",
        justification=(
            "Lésion cutanée bénigne d'évolution lente, sans signe infectieux ni terrain à risque : "
            "aucun caractère urgent."
        ),
        recommendation=(
            "Traitement kératolytique en pharmacie, consultation dermatologique programmée si "
            "persistance ou si la lésion se modifie."
        ),
    ),
    Presentation(
        id="fatigue_chronique",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(20, 70),
        complaint_fr="fatigue sans autre symptôme",
        complaint_en="tiredness with no other symptom",
        signs_fr=(
            "sommeil perturbé",
            "pas de perte de poids",
            "pas de fièvre",
            "examen clinique normal",
        ),
        signs_en=("disturbed sleep", "no weight loss", "no fever", "normal physical examination"),
        history_fr=(
            "charge de travail importante",
            "aucun traitement",
            "bilan sanguin ancien normal",
        ),
        history_en=("heavy workload", "no medication", "previous blood tests normal"),
        onset="semaines",
        justification=(
            "Asthénie isolée sans signe d'alerte associé : bilan à organiser en ville, aucun "
            "élément ne justifie un passage aux urgences."
        ),
        recommendation=(
            "Consultation programmée chez le médecin traitant pour un bilan étiologique, "
            "hygiène de sommeil et évaluation de la charge de travail."
        ),
    ),
    Presentation(
        id="tendinite_epaule",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(30, 70),
        complaint_fr="douleur de l'épaule à l'élévation du bras",
        complaint_en="shoulder pain when raising the arm",
        signs_fr=(
            "douleur nocturne modérée",
            "pas de traumatisme",
            "pas de fièvre",
            "force conservée",
        ),
        signs_en=("moderate night pain", "no injury", "no fever", "strength preserved"),
        history_fr=("travail répétitif", "pratique du tennis", "aucun traitement"),
        history_en=("repetitive work", "plays tennis", "no medication"),
        onset="semaines",
        justification=(
            "Tendinopathie d'installation progressive sans traumatisme, sans fièvre et sans "
            "déficit de force : pathologie chronique relevant du suivi ambulatoire."
        ),
        recommendation=(
            "Antalgie, repos relatif et kinésithérapie, consultation programmée avec le médecin "
            "traitant ou un rhumatologue."
        ),
    ),
    Presentation(
        id="acne_inflammatoire",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(13, 30),
        complaint_fr="boutons inflammatoires du visage et du dos",
        complaint_en="inflamed spots on the face and back",
        signs_fr=(
            "évolution progressive",
            "pas de fièvre",
            "retentissement esthétique",
            "pas d'abcès",
        ),
        signs_en=("gradual course", "no fever", "cosmetic impact", "no abscess"),
        history_fr=(
            "acné depuis l'adolescence",
            "traitements locaux inefficaces",
            "aucun antécédent médical",
        ),
        history_en=(
            "acne since adolescence",
            "topical treatments ineffective",
            "no medical history",
        ),
        onset="semaines",
        justification=(
            "Dermatose chronique sans signe infectieux aigu : prise en charge dermatologique "
            "programmée, aucun critère d'urgence."
        ),
        recommendation=(
            "Consultation dermatologique programmée pour adapter le traitement de fond, "
            "poursuite des soins locaux dans l'intervalle."
        ),
    ),
    Presentation(
        id="reflux_gastro_oesophagien",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(25, 70),
        complaint_fr="brûlures remontant derrière le sternum après les repas",
        complaint_en="burning rising behind the breastbone after meals",
        signs_fr=(
            "aggravation en position allongée",
            "pas d'irradiation au bras",
            "pas de sueurs",
            "poids stable",
        ),
        signs_en=("worse when lying down", "no arm radiation", "no sweating", "stable weight"),
        history_fr=("reflux connu", "surpoids", "repas tardifs"),
        history_en=("known reflux", "overweight", "late meals"),
        onset="semaines",
        justification=(
            "Symptomatologie de reflux typique, rythmée par les repas et la position, sans signe "
            "d'alarme digestif ni élément en faveur d'une douleur coronarienne."
        ),
        recommendation=(
            "Mesures hygiéno-diététiques et traitement antiacide d'épreuve, consultation "
            "programmée. Consultation urgente en cas de difficulté à avaler ou d'amaigrissement."
        ),
    ),
    Presentation(
        id="suivi_tension_stable",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(45, 85),
        complaint_fr="tension mesurée un peu élevée à la maison, sans symptôme",
        complaint_en="slightly high blood pressure measured at home, no symptoms",
        signs_fr=(
            "pas de céphalée",
            "pas de trouble visuel",
            "pas de douleur thoracique",
            "tension normale à l'accueil",
        ),
        signs_en=(
            "no headache",
            "no visual disturbance",
            "no chest pain",
            "normal reading on arrival",
        ),
        history_fr=("hypertension traitée", "automesure récente", "suivi régulier"),
        history_en=("treated hypertension", "recent home monitoring", "regular follow-up"),
        onset="semaines",
        justification=(
            "Chiffres tensionnels isolés sans aucun signe de retentissement : il n'existe pas "
            "d'urgence hypertensive en l'absence de symptôme."
        ),
        recommendation=(
            "Poursuite du traitement et de l'automesure, consultation programmée pour adapter le "
            "traitement. Consultation urgente en cas de céphalée intense ou de trouble visuel."
        ),
    ),
    Presentation(
        id="entorse_doigt_benigne",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(10, 60),
        complaint_fr="doigt douloureux après un choc au ballon",
        complaint_en="painful finger after a ball impact",
        signs_fr=(
            "léger gonflement",
            "flexion possible",
            "pas de déformation",
            "pas d'hématome important",
        ),
        signs_en=("slight swelling", "able to bend it", "no deformity", "no significant bruising"),
        history_fr=("sport de ballon", "aucun antécédent", "vaccination à jour"),
        history_en=("ball sport", "no medical history", "vaccinations up to date"),
        onset="heures",
        justification=(
            "Traumatisme digital bénin : mobilité conservée et absence de déformation rendent la "
            "fracture très improbable."
        ),
        recommendation=(
            "Glace, syndactylie et antalgie simple. Consultation si la douleur persiste au-delà "
            "d'une semaine ou si une déformation apparaît."
        ),
    ),
    Presentation(
        id="mycose_ongle",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(35, 80),
        complaint_fr="ongle de pied épaissi et jauni",
        complaint_en="thickened yellow toenail",
        signs_fr=(
            "pas de douleur",
            "pas de rougeur du pourtour",
            "pas de fièvre",
            "évolution très lente",
        ),
        signs_en=("no pain", "no redness around the nail", "no fever", "very slow course"),
        history_fr=(
            "fréquentation de vestiaires collectifs",
            "pas de diabète",
            "chaussures fermées",
        ),
        history_en=("uses shared changing rooms", "no diabetes", "closed shoes"),
        onset="semaines",
        justification=(
            "Onychomycose d'évolution chronique, sans douleur ni signe inflammatoire, chez un "
            "patient sans terrain à risque podologique."
        ),
        recommendation=(
            "Prélèvement mycologique et traitement local en consultation programmée, mesures "
            "d'hygiène des pieds."
        ),
    ),
    Presentation(
        id="certificat_sport",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(8, 60),
        complaint_fr="demande de certificat médical d'aptitude au sport",
        complaint_en="request for a sports fitness certificate",
        signs_fr=(
            "aucun symptôme",
            "activité physique régulière",
            "examen normal",
            "constantes normales",
        ),
        signs_en=(
            "no symptoms",
            "regular physical activity",
            "normal examination",
            "normal observations",
        ),
        history_fr=(
            "aucun antécédent",
            "pas de traitement",
            "pas d'antécédent familial cardiaque",
        ),
        history_en=(
            "no medical history",
            "no medication",
            "no family history of heart disease",
        ),
        onset="semaines",
        justification=(
            "Demande administrative sans plainte médicale : ce motif ne relève pas du service "
            "d'urgence."
        ),
        recommendation=(
            "Orienter vers le médecin traitant pour une consultation dédiée, expliquer que les "
            "urgences ne délivrent pas ce type de certificat."
        ),
    ),
    Presentation(
        id="insomnie_chronique",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(25, 80),
        complaint_fr="difficultés d'endormissement",
        complaint_en="difficulty falling asleep",
        signs_fr=("réveils nocturnes", "fatigue diurne", "pas d'idées noires", "examen normal"),
        signs_en=("night waking", "daytime tiredness", "no dark thoughts", "normal examination"),
        history_fr=("stress professionnel", "écrans le soir", "consommation de café"),
        history_en=("work stress", "screens in the evening", "caffeine intake"),
        onset="semaines",
        justification=(
            "Trouble du sommeil chronique sans souffrance psychiatrique aiguë ni idée suicidaire : "
            "prise en charge programmée en ville."
        ),
        recommendation=(
            "Conseils d'hygiène du sommeil, consultation programmée chez le médecin traitant. "
            "Consultation urgente si apparition d'idées suicidaires."
        ),
    ),
    Presentation(
        id="hemorroides_non_compliquees",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(25, 75),
        complaint_fr="gêne anale avec un peu de sang sur le papier",
        complaint_en="anal discomfort with a little blood on the paper",
        signs_fr=(
            "saignement minime",
            "pas de douleur intense",
            "pas de fièvre",
            "transit normal",
        ),
        signs_en=("minimal bleeding", "no severe pain", "no fever", "normal bowel habit"),
        history_fr=("constipation", "grossesse récente", "hémorroïdes connues"),
        history_en=("constipation", "recent pregnancy", "known haemorrhoids"),
        onset="semaines",
        justification=(
            "Saignement anal minime d'allure hémorroïdaire, sans retentissement général ni "
            "douleur intense évoquant une thrombose."
        ),
        recommendation=(
            "Régularisation du transit et traitement local, consultation programmée. "
            "Consultation urgente si saignement abondant ou douleur brutale et intense."
        ),
    ),
    Presentation(
        id="bouchon_cerumen",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(10, 85),
        complaint_fr="oreille bouchée avec baisse d'audition d'un côté",
        complaint_en="blocked ear with reduced hearing on one side",
        signs_fr=("pas de douleur", "pas d'écoulement", "pas de vertige", "pas de fièvre"),
        signs_en=("no pain", "no discharge", "no dizziness", "no fever"),
        history_fr=(
            "utilisation de cotons-tiges",
            "épisodes identiques",
            "port d'aides auditives",
        ),
        history_en=("uses cotton buds", "identical episodes", "wears hearing aids"),
        onset="jours",
        justification=(
            "Obstruction du conduit auditif sans douleur, écoulement, vertige ni fièvre : "
            "situation bénigne relevant d'un soin programmé."
        ),
        recommendation=(
            "Ramollissement du bouchon puis lavage d'oreille en consultation programmée. "
            "Consultation rapide en cas de douleur, d'écoulement ou de vertige."
        ),
    ),
    Presentation(
        id="cystite_simple",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(16, 60),
        complaint_fr="brûlures en urinant, sans fièvre",
        complaint_en="burning on urination, no fever",
        signs_fr=(
            "envies fréquentes",
            "pas de douleur lombaire",
            "pas de frissons",
            "état général conservé",
        ),
        signs_en=(
            "frequent urge to urinate",
            "no flank pain",
            "no shivering",
            "feeling well otherwise",
        ),
        history_fr=("cystites occasionnelles", "pas de grossesse", "pas de diabète"),
        history_en=("occasional cystitis", "not pregnant", "no diabetes"),
        onset="jours",
        justification=(
            "Cystite aiguë simple chez une patiente sans facteur de risque de complication : "
            "absence de fièvre et de douleur lombaire écartant la pyélonéphrite."
        ),
        recommendation=(
            "Bandelette urinaire et antibiothérapie courte en ville, hydratation. Consultation "
            "urgente si fièvre, frissons ou douleur du dos apparaissent."
        ),
    ),
    Presentation(
        id="piqure_insecte_locale",
        level="CONSULTATION_DIFFEREE",
        vitals_profile="normal",
        age_range=(3, 75),
        complaint_fr="piqûre de moustique gonflée et qui démange",
        complaint_en="mosquito bite, swollen and itchy",
        signs_fr=(
            "rougeur locale limitée",
            "pas de gêne respiratoire",
            "pas de fièvre",
            "pas d'extension",
        ),
        signs_en=("limited local redness", "no breathing difficulty", "no fever", "not spreading"),
        history_fr=("aucune allergie connue", "séjour en extérieur", "aucun traitement"),
        history_en=("no known allergy", "time spent outdoors", "no medication"),
        onset="jours",
        justification=(
            "Réaction locale isolée à une piqûre, sans signe général ni respiratoire : aucun "
            "critère d'anaphylaxie ni de surinfection."
        ),
        recommendation=(
            "Soins locaux et antihistaminique si les démangeaisons gênent. Consultation immédiate "
            "en cas de gonflement du visage, de gêne respiratoire ou d'extension rapide."
        ),
    ),
)


def presentations_by_level(level: str) -> list[Presentation]:
    """Return the presentations of the requested level."""
    return [p for p in PRESENTATIONS if p.level == level]


# Vital signs that the narrative of certain presentations names explicitly, with the direction
# in which they must be degraded.
#
# The generator degrades one or two vital signs drawn at random; this table forces the ones the
# text names. Without it, a vignette whose complaint is "fever with chills" comes out afebrile
# most of the time, and a severe pre-eclampsia normotensive: the description and the reading
# contradict each other inside the same training example, and the model learns along the way
# that vital signs mean nothing.
#
# ``test_clinical_data`` checks that no presentation naming a vital sign is forgotten here.
FORCED_VITALS: dict[str, tuple[tuple[str, str], ...]] = {
    # Fever named in the complaint or in the signs.
    "sepsis_grave": (("temperature", "high"),),
    "syndrome_meninge": (("temperature", "high"),),
    "pneumopathie_communautaire": (("temperature", "high"),),
    "pyelonephrite": (("temperature", "high"),),
    "erysipele_jambe": (("temperature", "high"),),
    "abces_dentaire": (("temperature", "high"),),
    "otite_moyenne_aigue_enfant": (("temperature", "high"),),
    "lombalgie_febrile": (("temperature", "high"),),
    "suspicion_appendicite": (("temperature", "high"),),
    "crise_drepanocytaire": (("temperature", "high"),),
    # Respiratory distress: saturation is the sign of the picture itself.
    "asthme_aigu_grave": (("spo2", "low"),),
    "bronchiolite_grave_nourrisson": (("spo2", "low"),),
    "crise_asthme_moderee": (("spo2", "low"),),
    # Hypertension **is** the diagnosis.
    "pre_eclampsie_severe": (("systolic_bp", "high"),),
}


def forced_vitals(presentation_id: str) -> tuple[tuple[str, str], ...]:
    """Vital signs this presentation's narrative forces, possibly none."""
    return FORCED_VITALS.get(presentation_id, ())


# Probability that a vignette carries "altered consciousness", presentation by presentation.
#
# The probability depends on the clinical picture, not on the vital-sign profile: critical
# pictures leave some patients perfectly conscious, and a rate attached to the profile would
# glue the label onto those presentations. It would also make "altered consciousness" the
# exclusive sign of a life-threatening case across the whole set, a shortcut a model learns
# happily in place of triage.
#
# The default is zero: consciousness stays normal except where the clinical picture involves it.
ALTERED_CONSCIOUSNESS_LIKELIHOOD: dict[str, float] = {
    # Consciousness is part of the picture.
    "sepsis_grave": 0.75,
    "traumatisme_cranien_grave": 0.80,
    "crise_convulsive_prolongee": 0.85,
    "intoxication_medicamenteuse_volontaire": 0.70,
    "hypoglycemie_severe": 0.80,
    "deshydratation_severe_nourrisson": 0.60,
    "acidocetose_diabetique": 0.45,
    # It can degrade, without being the rule.
    "choc_anaphylactique": 0.35,
    "rupture_anevrisme_aorte": 0.40,
    "hemorragie_digestive_haute": 0.25,
    "hemorragie_du_post_partum": 0.25,
    "bronchiolite_grave_nourrisson": 0.30,
    # The other critical pictures leave the patient conscious: acute coronary syndrome, severe
    # asthma, pulmonary embolism, extensive burn, penetrating wound. Their probability is zero,
    # by absence from this table.
}


def altered_consciousness_likely(presentation_id: str) -> float:
    """Probability that this presentation comes with altered consciousness."""
    return ALTERED_CONSCIOUSNESS_LIKELIHOOD.get(presentation_id, 0.0)


def presentation_by_id(presentation_id: str) -> Presentation:
    """Return the presentation carrying this identifier."""
    for presentation in PRESENTATIONS:
        if presentation.id == presentation_id:
            return presentation
    raise KeyError(f"Unknown presentation: {presentation_id!r}")
