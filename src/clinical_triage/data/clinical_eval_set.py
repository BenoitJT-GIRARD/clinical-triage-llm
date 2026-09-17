"""The independent clinical evaluation set, written by hand.

It is the only set the published figures are computed on, and it is built so that they mean
something:

- **it comes neither from the rule nor from the generator.** The training vignettes are
  composed by ``case_generator`` from templates; the cases below are written one by one, with a
  different syntax, a different vocabulary and information out of order, as the narratives
  collected at a reception desk are;
- **it never enters training.** The preparation script removes from the SFT set and from the DPO
  set any user turn identical to one of these cases;
- **it contains traps.** Nearly half the cases are atypical presentations: an emergency that
  looks benign, a spectacular but harmless symptom, a serious sign explicitly denied, vital
  signs that contradict the narrative. A keyword rule gets those cases wrong; that is precisely
  where a language model has to show what it adds.

Each case carries the clinical reason for its label. That note is not given to the model: it
serves auditability and error analysis.

Acknowledged limit: these cases were written by an engineer from the triage literature, not by
an emergency physician. Having a clinician review them is the first item of the roadmap.

The cases are French and English clinical text: they are the data being evaluated on.
"""

from __future__ import annotations

from dataclasses import dataclass

from clinical_triage.data.case_generator import USER_TEMPLATES


@dataclass(frozen=True)
class EvalCase:
    """One clinical evaluation case and its ground truth."""

    id: str
    lang: str
    level: str
    # Nature of the trap, empty when the presentation is direct. Used to measure performance
    # separately on typical and on atypical cases.
    case_type: str
    description: str
    note: str

    @property
    def user_turn(self) -> str:
        """The user turn presented to the model (first template, no variation)."""
        return USER_TEMPLATES[self.lang][0].format(description=self.description)


# (id, language, level, trap, description, reason for the label)
_CASES: tuple[tuple[str, str, str, str, str, str], ...] = (
    # ---------------- URGENCE_VITALE — direct presentations ----------------
    (
        "ev01",
        "fr",
        "URGENCE_VITALE",
        "",
        (
            "Un homme de 67 ans est amené par sa femme. Il serre son poing sur sa poitrine, "
            "transpire abondamment et dit que ça a commencé en montant l'escalier il y a vingt minutes. "
            "Il est hypertendu et fume depuis quarante ans. TA 148/92, FC 102, SpO2 96 %."
        ),
        "Douleur thoracique d'effort avec sueurs chez un patient à haut risque cardiovasculaire.",
    ),
    (
        "ev02",
        "fr",
        "URGENCE_VITALE",
        "",
        (
            "Femme de 74 ans : sa fille l'a trouvée ce matin incapable de parler correctement, "
            "le coin de la bouche tombant et le bras droit sans force. Dernière fois vue normale : hier soir 22 h. "
            "Elle prend un anticoagulant pour une arythmie."
        ),
        "Déficit neurologique focal d'installation récente : filière AVC.",
    ),
    (
        "ev03",
        "en",
        "URGENCE_VITALE",
        "",
        (
            "A 5-year-old boy was stung by a wasp fifteen minutes ago. His lips are swelling, "
            "he is covered in hives and his voice has gone hoarse. He is struggling to swallow his saliva. "
            "SpO2 93%, HR 140."
        ),
        "Atteinte cutanée, respiratoire et ORL après piqûre : anaphylaxie.",
    ),
    (
        "ev04",
        "fr",
        "URGENCE_VITALE",
        "",
        (
            "Patiente de 31 ans, enceinte de 34 semaines, se plaint depuis ce matin de céphalées qui ne cèdent pas, "
            "de points lumineux devant les yeux et d'une barre douloureuse sous les côtes. Ses chevilles ont beaucoup gonflé cette semaine. "
            "TA 168/104."
        ),
        "Céphalées, troubles visuels et douleur épigastrique en fin de grossesse : pré-éclampsie sévère.",
    ),
    (
        "ev05",
        "en",
        "URGENCE_VITALE",
        "",
        (
            "A 3-month-old infant has been feeding poorly for two days and is now breathing very fast, "
            "with the skin sucking in between the ribs. The mother reports two episodes where he seemed to stop breathing. "
            "SpO2 89%, RR 68."
        ),
        "Détresse respiratoire du nourrisson avec apnées et désaturation.",
    ),
    (
        "ev06",
        "fr",
        "URGENCE_VITALE",
        "",
        (
            "Homme de 45 ans, cirrhotique, a vomi deux fois du sang rouge il y a une heure. "
            "Il est très pâle, se sent partir quand il se lève. TA 88/54, FC 118."
        ),
        "Hématémèse avec mauvaise tolérance hémodynamique : hémorragie digestive active.",
    ),
    (
        "ev07",
        "en",
        "URGENCE_VITALE",
        "",
        (
            "A 22-year-old woman was found by her flatmate next to an empty box of paracetamol and a goodbye note. "
            "She took the tablets about two hours ago and is becoming increasingly drowsy."
        ),
        "Intoxication volontaire récente avec altération de la vigilance et risque suicidaire.",
    ),
    (
        "ev08",
        "fr",
        "URGENCE_VITALE",
        "",
        (
            "Enfant de 2 ans, fébrile à 39,8, qui geint, refuse qu'on lui plie la nuque et somnole entre deux pleurs. "
            "Sa mère a remarqué de petites taches violacées sur les jambes qui ne s'effacent pas à la pression."
        ),
        "Syndrome méningé fébrile avec purpura ne s'effaçant pas : urgence absolue.",
    ),
    (
        "ev09",
        "en",
        "URGENCE_VITALE",
        "",
        (
            "A 58-year-old man had a sudden tearing pain in the abdomen radiating to his back while gardening. "
            "He nearly fainted. He is known to have an aortic aneurysm under surveillance. BP 84/50."
        ),
        "Douleur abdominale brutale irradiant dans le dos avec hypotension sur anévrisme connu.",
    ),
    (
        "ev10",
        "fr",
        "URGENCE_VITALE",
        "",
        (
            "Homme de 36 ans, épileptique connu, convulse depuis plus de dix minutes sans reprendre conscience entre les crises. "
            "Il respire bruyamment."
        ),
        "État de mal épileptique : atteinte des voies aériennes et souffrance cérébrale.",
    ),
    # ---------------- URGENCE_VITALE — traps ----------------
    (
        "ev11",
        "fr",
        "URGENCE_VITALE",
        "falsely_reassuring",
        (
            "Femme de 71 ans, diabétique de longue date, se dit seulement « très fatiguée » depuis deux heures, "
            "avec une vague nausée et un essoufflement inhabituel en montant chez elle. Elle insiste sur le fait qu'elle n'a aucune douleur. "
            "FC 112, TA 96/60, SpO2 93 %."
        ),
        "Infarctus silencieux du diabétique : l'absence de douleur thoracique est fréquente et trompeuse.",
    ),
    (
        "ev12",
        "en",
        "URGENCE_VITALE",
        "falsely_reassuring",
        (
            "An 84-year-old nursing home resident is described as 'just not herself' since last night: more confused than usual, "
            "eating nothing. She has no fever. HR 104, BP 92/58, RR 24."
        ),
        "Sepsis du sujet âgé : confusion et hypotension sans fièvre, la fièvre manquant souvent chez la personne âgée.",
    ),
    (
        "ev13",
        "fr",
        "URGENCE_VITALE",
        "discordant_vitals",
        (
            "Homme de 29 ans venu pour « une grosse fatigue » après un match de football. Il parle normalement et marche seul. "
            "Constantes à l'accueil : FC 38, TA 82/48, SpO2 97 %, vigilance normale."
        ),
        "Bradycardie sévère avec hypotension : l'état apparent ne reflète pas la gravité des constantes.",
    ),
    (
        "ev14",
        "en",
        "URGENCE_VITALE",
        "falsely_reassuring",
        (
            "A 26-year-old woman on the contraceptive pill returned from a long-haul flight yesterday. "
            "She feels short of breath when climbing stairs and has a mild ache in her right calf. She looks well at rest. RR 26, HR 116."
        ),
        "Embolie pulmonaire : présentation discrète chez une patiente jeune avec facteurs de risque thromboembolique.",
    ),
    (
        "ev15",
        "fr",
        "URGENCE_VITALE",
        "falsely_reassuring",
        (
            "Adolescent de 16 ans amené pour douleurs abdominales et vomissements depuis la veille. Il dit avoir beaucoup soif "
            "et être allé uriner sans arrêt. Son haleine a une odeur sucrée. Il est diabétique et a arrêté ses injections depuis trois jours."
        ),
        "Acidocétose diabétique se présentant comme un tableau digestif banal.",
    ),
    (
        "ev16",
        "en",
        "URGENCE_VITALE",
        "falsely_reassuring",
        (
            "A 47-year-old builder cut his forearm on a rusty sheet three days ago. The wound now looks only slightly red, "
            "but the pain is out of proportion and he feels feverish and shivery. HR 124, BP 94/56."
        ),
        "Douleur disproportionnée avec instabilité hémodynamique : suspicion de fasciite nécrosante.",
    ),
    (
        "ev17",
        "fr",
        "URGENCE_VITALE",
        "negation",
        (
            "Homme de 62 ans : pas de douleur thoracique, pas de fièvre, pas de traumatisme. "
            "Mais depuis une heure il ne trouve plus ses mots et son bras gauche lui échappe."
        ),
        "Les négations portent sur des signes absents ; le déficit neurologique brutal reste une urgence absolue.",
    ),
    (
        "ev18",
        "fr",
        "URGENCE_VITALE",
        "falsely_reassuring",
        (
            "Femme de 24 ans qui a accouché il y a deux jours et rentre chez elle. Elle signale des « règles un peu abondantes » "
            "et a changé quatre protections en une heure, avec des caillots. Elle se sent partir en se levant. FC 122."
        ),
        "Hémorragie du post-partum minimisée par la patiente : quantification et retentissement font le diagnostic.",
    ),
    (
        "ev19",
        "en",
        "URGENCE_VITALE",
        "discordant_vitals",
        (
            "A 19-year-old with known asthma says his chest feels 'a bit tight' and he is not too worried. "
            "He speaks in short phrases and stops between words. SpO2 90%, RR 32, HR 128."
        ),
        "Crise d'asthme aiguë grave sous-estimée par le patient : la parole hachée et la désaturation tranchent.",
    ),
    (
        "ev20",
        "fr",
        "URGENCE_VITALE",
        "falsely_reassuring",
        (
            "Homme de 52 ans qui consulte pour un problème d'ordonnance. En fin d'entretien, il mentionne qu'il a tout préparé "
            "pour en finir ce week-end et qu'il a acheté une corde. Il refuse d'en dire plus."
        ),
        "Crise suicidaire avec scénario et moyen : risque immédiat malgré un motif de consultation anodin.",
    ),
    # ---------------- URGENCE_MODEREE — direct presentations ----------------
    (
        "ev21",
        "fr",
        "URGENCE_MODEREE",
        "",
        (
            "Jeune femme de 23 ans, douleur du bas-ventre à droite depuis hier soir, qui s'est intensifiée. "
            "Elle n'a pas faim et a vomi une fois. Température 38,1. La palpation réveille la douleur quand on relâche."
        ),
        "Douleur de la fosse iliaque droite fébrile avec défense : appendicite à explorer dans les heures qui suivent.",
    ),
    (
        "ev22",
        "en",
        "URGENCE_MODEREE",
        "",
        (
            "A 41-year-old man has been rolling around in pain for two hours with severe right flank pain going down to the groin. "
            "He cannot find a comfortable position. Urine dipstick shows blood. No fever."
        ),
        "Colique néphrétique non compliquée : douleur intense mais sans fièvre ni anurie.",
    ),
    (
        "ev23",
        "fr",
        "URGENCE_MODEREE",
        "",
        (
            "Garçon de 11 ans tombé de vélo il y a une heure. Le poignet est déformé en dos de fourchette, très gonflé. "
            "Les doigts sont roses, chauds et sensibles. Douleur 7/10."
        ),
        "Fracture déplacée du poignet sans atteinte vasculo-nerveuse : réduction nécessaire mais différable de quelques heures.",
    ),
    (
        "ev24",
        "en",
        "URGENCE_MODEREE",
        "",
        (
            "A 33-year-old woman has had burning on passing urine for three days and now has a fever of 38.6 "
            "with pain over her right lower back and shivering."
        ),
        "Pyélonéphrite : infection urinaire haute fébrile à traiter rapidement, sans signe de choc.",
    ),
    (
        "ev25",
        "fr",
        "URGENCE_MODEREE",
        "",
        (
            "Homme de 68 ans dont la jambe gauche est rouge, chaude et tendue depuis deux jours, avec 38,4 de fièvre et des frissons. "
            "Il a une mycose entre les orteils. La peau n'est pas noire et la douleur est proportionnée."
        ),
        "Érysipèle sans signe de gravité locale : antibiothérapie rapide et surveillance de l'extension.",
    ),
    (
        "ev26",
        "en",
        "URGENCE_MODEREE",
        "",
        (
            "A 55-year-old smoker has had a productive cough with green sputum and fever for four days, "
            "and a sharp pain on the left when breathing in. He is fully alert, RR 22, SpO2 95%."
        ),
        "Pneumopathie communautaire sans critère de gravité : traitement dans la journée.",
    ),
    (
        "ev27",
        "fr",
        "URGENCE_MODEREE",
        "",
        (
            "Femme de 78 ans sous anticoagulant, tombée de sa hauteur ce matin, s'est cognée le front. "
            "Pas de perte de connaissance, pas de vomissement, examen neurologique normal. Une bosse est visible."
        ),
        (
            "Traumatisme crânien sous anticoagulant : scanner dans l'heure et surveillance "
            "imposés malgré l'absence de signe immédiat. Tri 3 de l'échelle FRENCH — prise en "
            "charge médicale sous une heure — et non tri 2 : c'est la borne de délai qui "
            "distingue les deux niveaux, et ce cas est de ceux qu'une relecture par un "
            "urgentiste doit trancher en priorité."
        ),
    ),
    (
        "ev28",
        "en",
        "URGENCE_MODEREE",
        "",
        (
            "A 9-year-old girl has had watery diarrhoea and vomiting for two days. Her mouth is dry and she has passed "
            "only a small amount of dark urine today, but she is alert and drinking when encouraged."
        ),
        "Déshydratation modérée avec vigilance conservée : réhydratation nécessaire sans urgence vitale.",
    ),
    (
        "ev29",
        "fr",
        "URGENCE_MODEREE",
        "",
        (
            "Homme de 34 ans qui meulait sans lunettes et sent depuis une heure un corps étranger dans l'œil droit, "
            "qui pleure et supporte mal la lumière. Il voit normalement."
        ),
        "Corps étranger cornéen probable : retrait rapide nécessaire, vision conservée.",
    ),
    (
        "ev30",
        "en",
        "URGENCE_MODEREE",
        "",
        (
            "A 17-year-old with sickle cell disease has the usual bone pain in both legs since last night, worse than her normal crises. "
            "No fever, no chest pain, breathing normally."
        ),
        "Crise vaso-occlusive sans syndrome thoracique : antalgie urgente sans défaillance d'organe.",
    ),
    # ---------------- URGENCE_MODEREE — traps ----------------
    (
        "ev31",
        "fr",
        "URGENCE_MODEREE",
        "falsely_alarming",
        (
            "Femme de 28 ans amenée par ses collègues : elle dit étouffer, a les mains qui fourmillent et répète qu'elle va mourir. "
            "Elle a déjà eu exactement les mêmes épisodes, suivis pour un trouble anxieux. Examen normal, SpO2 99 %, ECG normal."
        ),
        "Attaque de panique typique et déjà connue, examen et constantes normaux : prise en charge rapide mais sans détresse.",
    ),
    (
        "ev32",
        "en",
        "URGENCE_MODEREE",
        "falsely_alarming",
        (
            "A 14-year-old boy has been bleeding from the left nostril for twenty minutes. There is blood on his shirt and he is frightened. "
            "The bleeding stops with pinching. He is on no medication and has normal observations."
        ),
        "Épistaxis antérieure banale : impressionnante mais contrôlée, sans terrain hémorragique.",
    ),
    (
        "ev33",
        "fr",
        "URGENCE_MODEREE",
        "falsely_alarming",
        (
            "Homme de 30 ans, sportif, très inquiet : il a une douleur dans la poitrine depuis deux jours, "
            "qui se réveille quand on appuie dessus et quand il bouge le bras. Aucun facteur de risque. ECG normal, troponine négative."
        ),
        "Douleur pariétale reproductible avec bilan négatif : origine coronarienne écartée, surveillance simple.",
    ),
    (
        "ev34",
        "en",
        "URGENCE_MODEREE",
        "falsely_reassuring",
        (
            "A 63-year-old woman has a rash of small blisters on her right forehead and upper eyelid that appeared two days ago. "
            "It is painful but she says it is 'just a rash'. Her right eye is slightly red."
        ),
        "Zona ophtalmique : le risque cornéen impose un traitement sous 72 heures malgré une apparence banale.",
    ),
    (
        "ev35",
        "fr",
        "URGENCE_MODEREE",
        "falsely_reassuring",
        (
            "Homme de 44 ans qui consulte pour un mal de dos « comme d'habitude » depuis dix jours. "
            "Il signale en passant des sueurs la nuit et 38,2 ce matin. Pas de déficit, il marche normalement. "
            "Il a été traité pour une infection dentaire le mois dernier."
        ),
        "Lombalgie fébrile : drapeau rouge imposant d'éliminer une spondylodiscite malgré un motif banal.",
    ),
    (
        "ev36",
        "en",
        "URGENCE_MODEREE",
        "negation",
        (
            "A 70-year-old man has no chest pain, no shortness of breath and no dizziness, but his pulse has felt irregular "
            "since this morning and he is more tired than usual on walking. BP 128/76."
        ),
        "Arythmie bien tolérée : les négations écartent la gravité immédiate mais un bilan cardiologique reste nécessaire.",
    ),
    (
        "ev37",
        "fr",
        "URGENCE_MODEREE",
        "falsely_alarming",
        (
            "Femme de 47 ans qui vomit et ne tient pas debout depuis ce matin tant tout tourne. "
            "Les vertiges s'aggravent quand elle bouge la tête. Elle parle bien, bouge les quatre membres, marche en s'appuyant. "
            "Épisode identique il y a deux ans."
        ),
        "Syndrome vestibulaire périphérique : spectaculaire mais sans signe neurologique central.",
    ),
    (
        "ev38",
        "en",
        "URGENCE_MODEREE",
        "falsely_reassuring",
        (
            "A 38-year-old woman started a new antibiotic yesterday and now has hives over her trunk and arms with intense itching. "
            "She is breathing comfortably, swallowing normally, BP 118/72."
        ),
        "Urticaire médicamenteuse sans atteinte respiratoire ni hémodynamique : pas d'anaphylaxie.",
    ),
    (
        "ev39",
        "fr",
        "URGENCE_MODEREE",
        "discordant_vitals",
        (
            "Homme de 59 ans venu pour un renouvellement de traitement. Il ne se plaint de rien. "
            "Constantes systématiques à l'accueil : température 38,7, FC 108, TA 118/72, SpO2 97 %."
        ),
        "Fièvre et tachycardie découvertes fortuitement : cause à rechercher dans la journée, sans signe de gravité.",
    ),
    (
        "ev40",
        "en",
        "URGENCE_MODEREE",
        "falsely_alarming",
        (
            "A 6-year-old has been crying and holding his right ear all night with a temperature of 38.4. "
            "He is playing with his toys in the waiting room, drinking normally and interacting well."
        ),
        "Otite moyenne aiguë chez un enfant en bon état général : examen et traitement dans la journée.",
    ),
    # ---------------- CONSULTATION_DIFFEREE — direct presentations ----------------
    (
        "ev41",
        "fr",
        "CONSULTATION_DIFFEREE",
        "",
        (
            "Homme de 34 ans avec le nez bouché, la gorge irritée et quelques éternuements depuis deux jours. "
            "Pas de fièvre, il mange et dort normalement, il est venu « parce que le cabinet était fermé »."
        ),
        "Rhinopharyngite virale bénigne sans fièvre ni signe respiratoire.",
    ),
    (
        "ev42",
        "en",
        "CONSULTATION_DIFFEREE",
        "",
        (
            "A 45-year-old warehouse worker strained his lower back lifting a box three days ago. "
            "The pain eases when he lies down, he walks without difficulty, no fever, no leg weakness or numbness."
        ),
        "Lombalgie commune sans drapeau rouge.",
    ),
    (
        "ev43",
        "fr",
        "CONSULTATION_DIFFEREE",
        "",
        (
            "Femme de 62 ans venue faire renouveler son traitement pour la thyroïde, parti à la poubelle par erreur. "
            "Elle n'a aucun symptôme et se sent bien."
        ),
        "Demande administrative sans plainte médicale : relève de la médecine de ville.",
    ),
    (
        "ev44",
        "en",
        "CONSULTATION_DIFFEREE",
        "",
        (
            "A 29-year-old has had a red, sticky left eye for two days, worse in the morning. "
            "Vision is normal, there is no deep pain and bright light does not bother her. Her son had the same last week."
        ),
        "Conjonctivite banale : vision conservée, pas de photophobie ni de douleur profonde.",
    ),
    (
        "ev45",
        "fr",
        "CONSULTATION_DIFFEREE",
        "",
        (
            "Homme de 51 ans avec une douleur de l'épaule droite quand il lève le bras, depuis environ un mois. "
            "Pas de chute, pas de fièvre, la force est normale. Il joue au tennis deux fois par semaine."
        ),
        "Tendinopathie chronique sans traumatisme ni déficit.",
    ),
    (
        "ev46",
        "en",
        "CONSULTATION_DIFFEREE",
        "",
        (
            "A 24-year-old cut her index finger on a kitchen knife an hour ago. The bleeding stopped with pressure, "
            "the cut is shallow, she can bend the finger fully and sensation is normal. Tetanus up to date."
        ),
        "Plaie superficielle sans atteinte fonctionnelle.",
    ),
    (
        "ev47",
        "fr",
        "CONSULTATION_DIFFEREE",
        "",
        (
            "Femme de 39 ans qui se plaint de fatigue depuis six semaines, sans amaigrissement, sans fièvre, "
            "sans essoufflement. Elle dort mal depuis une réorganisation au travail. Examen clinique normal."
        ),
        "Asthénie isolée sans signe d'alerte : bilan à organiser en ville.",
    ),
    (
        "ev48",
        "en",
        "CONSULTATION_DIFFEREE",
        "",
        (
            "A 58-year-old man measured his blood pressure at home as 152/90 and came in worried. "
            "He has no headache, no chest pain, no visual problems. Reading on arrival is 138/84."
        ),
        "Chiffres tensionnels isolés sans retentissement : pas d'urgence hypertensive.",
    ),
    (
        "ev49",
        "fr",
        "CONSULTATION_DIFFEREE",
        "",
        (
            "Homme de 27 ans avec des brûlures derrière le sternum après les repas copieux, depuis plusieurs mois, "
            "surtout quand il se couche. Pas d'irradiation au bras, pas de sueur, poids stable, il avale normalement."
        ),
        "Reflux gastro-œsophagien typique sans signe d'alarme digestif ni argument coronarien.",
    ),
    (
        "ev50",
        "en",
        "CONSULTATION_DIFFEREE",
        "",
        (
            "A 36-year-old woman has had burning on urination since yesterday. No fever, no back pain, no shivering, "
            "she feels otherwise well. Not pregnant, no diabetes."
        ),
        "Cystite simple sans critère de complication.",
    ),
    # ---------------- CONSULTATION_DIFFEREE — traps ----------------
    (
        "ev51",
        "fr",
        "CONSULTATION_DIFFEREE",
        "falsely_alarming",
        (
            "Femme de 33 ans, migraineuse connue depuis l'adolescence, avec depuis trois heures sa céphalée habituelle : "
            "pulsatile d'un côté, avec nausées et gêne à la lumière, précédée des mêmes zigzags visuels que d'ordinaire, déjà dissipés. "
            "Examen neurologique strictement normal."
        ),
        "Migraine avec aura identique aux épisodes connus et examen normal : pas de critère d'urgence.",
    ),
    (
        "ev52",
        "en",
        "CONSULTATION_DIFFEREE",
        "falsely_alarming",
        (
            "A 7-year-old fell off the sofa and has a large bruise on his thigh. He cried briefly, is now running around the waiting room, "
            "walks normally and has full movement. No head injury, no loss of consciousness."
        ),
        "Contusion bénigne chez un enfant en parfait état général.",
    ),
    (
        "ev53",
        "fr",
        "CONSULTATION_DIFFEREE",
        "falsely_alarming",
        (
            "Homme de 41 ans inquiet d'avoir vu du sang sur le papier toilette depuis quelques jours, "
            "avec une gêne à la selle. Le saignement est minime et rouge vif, il n'a ni douleur intense, ni fièvre, "
            "ni amaigrissement, et son transit est normal."
        ),
        "Saignement hémorroïdaire minime sans retentissement : bilan programmé en ville.",
    ),
    (
        "ev54",
        "en",
        "CONSULTATION_DIFFEREE",
        "negation",
        (
            "A 52-year-old presents with a blocked right ear and reduced hearing for four days. "
            "No pain, no discharge, no dizziness, no fever. He admits using cotton buds daily."
        ),
        "Bouchon de cérumen : toutes les complications sont explicitement absentes.",
    ),
    (
        "ev55",
        "fr",
        "CONSULTATION_DIFFEREE",
        "falsely_alarming",
        (
            "Mère inquiète pour son fils de 4 ans qui a 38,3 depuis ce matin. L'enfant court dans la salle d'attente, "
            "boit son biberon, joue et rit. Pas de raideur de nuque, pas d'éruption, pas de gêne respiratoire."
        ),
        "Fièvre isolée bien tolérée chez un enfant en excellent état général.",
    ),
    (
        "ev56",
        "en",
        "CONSULTATION_DIFFEREE",
        "",
        (
            "A 30-year-old asks for a medical certificate to join a football club. He has no symptoms, "
            "exercises three times a week, and no family history of heart disease."
        ),
        "Demande administrative sans plainte : hors périmètre des urgences.",
    ),
    (
        "ev57",
        "fr",
        "CONSULTATION_DIFFEREE",
        "falsely_alarming",
        (
            "Femme de 26 ans piquée par un moustique hier, avec une plaque rouge de 4 cm qui gratte beaucoup sur l'avant-bras. "
            "Elle respire normalement, n'a pas de gonflement du visage et la plaque ne s'étend plus depuis ce matin."
        ),
        "Réaction locale isolée sans signe général : pas d'anaphylaxie ni de surinfection.",
    ),
    (
        "ev58",
        "en",
        "CONSULTATION_DIFFEREE",
        "",
        (
            "A 19-year-old has had worsening spots on her face and back for two years. Topical creams from the pharmacy "
            "have not helped. No fever, no abscess, no systemic symptoms."
        ),
        "Acné inflammatoire chronique : prise en charge dermatologique programmée.",
    ),
    (
        "ev59",
        "fr",
        "CONSULTATION_DIFFEREE",
        "falsely_alarming",
        (
            "Homme de 55 ans affolé parce qu'il a une douleur dentaire depuis trois semaines qui le réveille la nuit. "
            "Il n'a pas de gonflement de la joue, pas de fièvre, ouvre la bouche normalement et avale sans difficulté."
        ),
        "Douleur dentaire chronique sans signe infectieux local ou général : relève du chirurgien-dentiste.",
    ),
    (
        "ev60",
        "en",
        "CONSULTATION_DIFFEREE",
        "negation",
        (
            "A 44-year-old reports constipation for six weeks with bloating. He is still passing wind, "
            "there is no blood, no vomiting, no weight loss, and his abdomen is soft."
        ),
        "Constipation chronique avec transit gazeux conservé : aucun signe d'occlusion.",
    ),
)


def eval_cases() -> list[EvalCase]:
    """Return the complete clinical evaluation set."""
    return [
        EvalCase(id=c[0], lang=c[1], level=c[2], case_type=c[3], description=c[4], note=c[5])
        for c in _CASES
    ]


def eval_user_turns() -> set[str]:
    """User turns of the evaluation set, to be excluded from every training set."""
    return {case.user_turn for case in eval_cases()}
