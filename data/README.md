---
license: mit
language:
  - fr
  - en
task_categories:
  - text-generation
tags:
  - medical
  - triage
  - emergency-department
  - french
size_categories:
  - 1K<n<10K
---

# Dataset de triage médical bilingue — the emergency department

Corpus d'entraînement d'un agent d'aide au triage des urgences. À partir d'une
description de patient — motif, symptômes, antécédents, constantes relevées à
l'accueil, en français ou en anglais — le modèle doit produire un **niveau de
priorité**, une **justification clinique** et une **conduite à tenir**, toujours
en français.

> Ce jeu de données est produit pour un prototype pédagogique. Il ne contient
> aucune donnée patient réelle, et il n'a pas été validé par un médecin
> urgentiste. Il ne doit pas servir à entraîner un système utilisé en situation
> clinique réelle.

## Fichiers

| Fichier | Contenu | Usage | Dans le dépôt |
|---|---|---|---|
| `sft_train.jsonl` | paires invite / réponse | entraînement supervisé | non |
| `sft_validation.jsonl` | idem | suivi de la convergence | non |
| `sft_test.jsonl` | idem | test à la même distribution que l'entraînement | non |
| `dpo_train.jsonl` | triplets invite / préférée / rejetée | alignement par préférences | non |
| `clinical_eval.jsonl` | cas rédigés à la main | **évaluation, jamais entraînement** | oui |
| `metadata.json` | schéma, statistiques, provenance, RGPD, contrôles | auditabilité | oui |

Les quatre jeux d'entraînement ne sont pas versionnés : ce sont des dérivés du
code et des corpus publics, que `scripts/build_dataset.py` reconstruit à
l'identique à graine fixe, et qui sont publiés sur le Hub. Les deux autres le
sont : on doit pouvoir lire les cas d'évaluation annotés et le contrôle
d'anonymisation en clonant le dépôt, sans rien télécharger.

## Comment ce corpus a été construit, et pourquoi

Le cahier des charges désigne quatre corpus publics : MediQAl, FrenchMedMCQA,
MedQuAD et UltraMedical-Preference. La première tâche a été de vérifier ce qu'ils
permettent réellement de construire.

**Aucun n'est annoté en niveaux de triage.** Ce sont des jeux de
questions-réponses médicales et de questions d'examen. Les utiliser tels quels —
enrober une question d'examen dans un gabarit de triage et l'étiqueter par
présence de mots-clés — produit des exemples absurdes du type « Un patient se
présente avec : *Levamisole is used as all except -* », et une évaluation
circulaire où le modèle ne fait que réapprendre la règle qui a produit les
étiquettes.

Quatre constats de terrain, vérifiés sur les corpus eux-mêmes :

- **MediQAl est le seul à décrire des patients.** Sa colonne `clinical_case`
  porte 3 075 vignettes françaises distinctes — motif, antécédents, et pour 569
  d'entre elles les constantes relevées à l'entrée. C'est la seule source
  authentique francophone du corpus ;
- **FrenchMedMCQA compte 1 080 questions** sur ses trois découpages, dont six
  seulement sont reconnues comme présentation de patient — et aucune ne porte de
  signe de triage identifiable. C'est un jeu de questions de pharmacie ; il ne peut pas
  porter la moitié francophone d'un dataset de triage ;
- **MedQuAD interroge des pathologies, pas des patients.** Ses fiches de
  symptômes sont retournées en plaintes de patient, ce qui en tire quelques cas,
  mais il ne décrit pas de situations cliniques ;
- **UltraMedical-Preference étiquette 38,6 % de ses paires par la seule longueur
  de la réponse**, mesuré sur 5 000 lignes — un signal qui, utilisé pour un
  alignement, apprend au modèle que « plus long vaut mieux ».

Le corpus repose donc sur trois apports.

### 1. Vignettes cliniques générées — la majorité du corpus

Un **catalogue de présentations types** rencontrées à l'accueil des urgences a
été rédigé pour ce projet : motif, signes associés, antécédents plausibles,
profil de constantes, et **niveau de triage de référence**. Adulte, enfant, femme
enceinte, traumatologie, psychiatrie, et les motifs de médecine générale.

Un générateur habille ensuite une présentation d'un patient : âge, sexe,
antécédents, délai d'installation, relevé de constantes cohérent avec la gravité,
formulation variable. **L'étiquette vient de la présentation d'origine, jamais
d'une relecture du texte produit.** C'est ce point qui rend l'évaluation
honnête : il n'existe aucune règle lexicale à réapprendre.

Niveau de confiance : `haute`.

### 2. Cas extraits des corpus publics

Les vignettes cliniques de MediQAl et de MedMCQA, et les descriptions de
symptômes de MedQuAD, sont conservées après retrait de la question d'examen, puis
étiquetées par la règle de triage explicite du projet.

Cette étiquette est **réexaminée après l'anonymisation** : si le masquage a retiré
le signe clinique qui la justifiait, le cas est écarté plutôt que livré avec une
étiquette que son propre texte ne soutient plus.

Une règle de sécurité encadre cette étiquette : **un cas n'est conservé que si la
règle identifie explicitement un signe**. L'absence de signe détecté ne prouve
pas l'absence de gravité — « suspected pneumoperitoneum » ne contient aucun mot
d'alerte et reste une urgence chirurgicale. Fabriquer une étiquette « non
urgent » à partir du silence d'une règle serait dangereux.

Niveau de confiance : `moyenne`.

### 3. Jeu d'évaluation clinique, écrit à la main

Les cas de `clinical_eval.jsonl` sont rédigés un par un, avec une autre syntaxe
et un autre vocabulaire que les gabarits du générateur. Près de la moitié d'entre
eux sont des **présentations atypiques** : urgence qui se donne l'air bénin, symptôme
spectaculaire mais sans gravité, signe grave explicitement nié, constantes qui
contredisent le récit. Chaque cas porte la raison clinique de son étiquette.

Ce jeu **n'entre jamais dans l'entraînement** : le script de préparation retire du
jeu supervisé et du jeu de préférences tout tour utilisateur identique à l'un
d'eux, et échoue si l'un s'y trouve.

### 4. Paires de préférences pour l'alignement

Le cahier des charges prévoit un alignement DPO fondé sur UltraMedical-Preference.
Ce corpus n'est pas utilisé ici pour l'entraînement : ses réponses sont de longues
dissertations en anglais, quand le contrat de sortie tient en trois lignes en
français, et 38,6 % de ses paires sont étiquetées par la seule longueur de la
réponse. Il est réservé à l'évaluation, comme mesure d'alignement indépendante.

Les paires de `dpo_train.jsonl` sont construites à partir du **seul découpage
d'entraînement** du jeu supervisé. À chaque invite, la réponse de référence est
opposée à une variante dégradée selon l'une de quatre stratégies : niveau
sous-évalué, conduite à tenir qui retarde la prise en charge, diagnostic présenté
comme certain, réponse hors de la langue imposée.

Trois règles encadrent la construction :

1. **même format, même longueur.** Chaque défaut existe en plusieurs longueurs et
   l'on retient celle qui colle au plus près de la réponse préférée. Sans cela, la
   seule différence systématique entre les deux réponses serait la longueur, et
   c'est elle que l'alignement apprendrait ;
2. **jamais de surclassement en réponse rejetée.** La consigne système impose de
   surclasser au moindre doute ; opposer une réponse trop prudente comme mauvais
   exemple enseignerait l'inverse ;
3. **les cas graves pèsent plus**, le sous-triage d'une urgence vitale étant la
   faute la plus coûteuse.

## Schéma

| Champ | Contenu |
|---|---|
| `prompt` | invite ChatML complète, ouvrant le tour assistant |
| `completion` | réponse attendue : niveau, justification, recommendation |
| `user_turn` | tour patient seul, utilisé pour la déduplication et l'audit |
| `level` | `URGENCE_VITALE` · `URGENCE_MODEREE` · `CONSULTATION_DIFFEREE` |
| `lang` | langue de la description du patient (`fr` ou `en`) |
| `source` | origine de l'exemple |
| `confiance` | `haute` (catalogue clinique) ou `moyenne` (règle appliquée à un corpus) |
| `symptomes` | signes cliniques présents dans la description |
| `antecedents` | antécédents mentionnés |
| `constantes` | relevé de constantes vitales, vide si le tri se fait sans mesure |
| `presentation_id` | présentation type d'origine, pour remonter à la décision de référence |

Les deux autres fichiers n'ont pas ces colonnes-là. Le jeu de préférences porte :

| Champ | Contenu |
|---|---|
| `prompt` | invite ChatML, identique à celle du jeu supervisé |
| `chosen` | réponse préférée : bon niveau, format respecté, conduite à tenir sûre |
| `rejected` | réponse rejetée, de même format et de longueur comparable |
| `user_turn` | tour patient seul |
| `level` | niveau de triage de référence du cas |
| `lang` | langue de la description |
| `strategie` | nature du défaut introduit : `sous_triage` · `recommandation_dangereuse` · `diagnostic_affirme` · `reponse_en_anglais` |
| `source` | toujours `preference_securite` |

Et le jeu d'évaluation clinique :

| Champ | Contenu |
|---|---|
| `prompt` | invite ChatML, identique à celle du jeu supervisé |
| `completion` | toujours vide : la réponse attendue n'est pas donnée, seul le niveau l'est |
| `user_turn` | tour patient seul, tel qu'il est soumis au modèle |
| `id` | identifiant du cas |
| `level` | niveau de référence, écrit à la main |
| `lang` | langue de la description |
| `piege` | nature de la difficulté : vide · `faux_rassurant` · `faux_alarmant` · `negation` · `constantes_discordantes` |
| `description` | description du patient seule, sans la consigne qui l'encadre |
| `note_clinique` | raison clinique de l'étiquette, pour l'auditabilité et l'analyse d'erreurs |

Ces trois listes sont celles de `metadata.json`, et un test du dépôt vérifie
qu'elles décrivent exactement les colonnes écrites dans les fichiers.

## Taxonomie

Trois niveaux, rattachés à l'échelle FRENCH utilisée dans les services d'urgence
français :

| Niveau | Délai | Échelle FRENCH |
|---|---|---|
| `URGENCE_VITALE` | prise en charge immédiate | tris 1 et 2 |
| `URGENCE_MODEREE` | quelques heures | tris 3 et 4 |
| `CONSULTATION_DIFFEREE` | consultation programmée | tri 5 |

## Sources et licences

| Source | Langue | Licence | Rôle |
|---|---|---|---|
| Catalogue de présentations du projet | fr + en | MIT | vérité terrain du triage |
| [`ANR-MALADES/MediQAl`](https://huggingface.co/datasets/ANR-MALADES/MediQAl) | fr | CC BY 4.0 | **vignettes cliniques françaises** — le seul corpus imposé qui décrive des patients |
| [`keivalya/MedQuad-MedicalQnADataset`](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) | en | CC BY 4.0 | descriptions authentiques de symptômes |
| [`nthngdy/frenchmedmcqa`](https://huggingface.co/datasets/nthngdy/frenchmedmcqa) | fr | Apache-2.0 | corpus francophone du cahier des charges |
| [`TsinghuaC3I/UltraMedical-Preference`](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference) | en | MIT | jeu de préférences externe |
| [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa) | en | Apache-2.0 | vignettes cliniques d'examen *(hors cahier des charges)* |

Deux de ces dépôts ne déclarent pas de licence sur le Hub : `MedQuad-MedicalQnADataset`
et `frenchmedmcqa` sont des miroirs. Les licences reportées ci-dessus sont celles
de leurs dépôts d'origine, où elles ont été lues — le `LICENSE.txt` de
[`abachaa/MedQuAD`](https://github.com/abachaa/MedQuAD) est le texte CC BY 4.0,
et [`qanastek/frenchmedmcqa`](https://huggingface.co/datasets/qanastek/frenchmedmcqa)
déclare Apache-2.0. On passe par les miroirs parce que le dépôt d'origine de
FrenchMedMCQA n'expose ses données que par un script de chargement, que `datasets`
n'exécute plus.

Rendement mesuré de chaque corpus, après filtrage des cas réellement exploitables
pour du triage :

<!-- rendement:debut — tableau écrit par scripts/build_dataset.py, ne pas modifier à la main -->
| Corpus | Entrées lues | Sans patient décrit | Hors bornes de longueur | Sans signe identifié | Doublons | Cas extraits | Cas livrés | Rendement |
|---|---|---|---|---|---|---|---|---|
| MediQAl | 3 075 | 1 407 | 514 | 792 | 0 | 362 | **313** | **10,2 %** |
| MedQuAD | 16 407 | 15 909 | 0 | 315 | 0 | 183 | 120 | 0,7 % |
| MedMCQA | 182 822 | 171 251 | 372 | 9 038 | 71 | 2 090 | 1 283 | 0,7 % |
| FrenchMedMCQA | 1 080 | 1 074 | 0 | 6 | 0 | 0 | **0** | **0,0 %** |
<!-- rendement:fin -->

Les quatre corpus sont lus **intégralement**, sans plafond de lecture : un
plafond donnerait un rendement qui décrit la limite qu'on s'est fixée et non la
source. Les colonnes de perte s'additionnent avec « cas extraits » pour retrouver
les entrées lues ; « cas livrés » est ce qu'il en reste après plafonnement par
case, anonymisation, ré-examen des étiquettes et déduplication finale.

MediQAl a de loin le meilleur rendement, et c'est attendu : ses entrées *sont*
des cas patients, là où les deux corpus de questions à choix multiples n'en
contiennent qu'incidemment. MedMCQA fournit néanmoins le plus gros volume, par sa
seule taille.

Trois des colonnes de perte relèvent de la forme, et sont révisables : le motif
qui reconnaît une présentation de patient, les bornes de longueur, la
déduplication. La quatrième relève de la sécurité : un cas n'est conservé que si
la règle de triage y identifie explicitement un signe, parce qu'on ne fabrique
pas une étiquette « non urgent » à partir du silence d'une règle à mots-clés.
C'est elle qui rend la classe `CONSULTATION_DIFFEREE` inatteignable depuis les
corpus, et donc entièrement issue du catalogue.

Le zéro de FrenchMedMCQA n'est pas un oubli : ce sont des questions d'examen de
pharmacie, sans patient décrit. Les y forcer réintroduirait exactement les
exemples absurdes que ce dataset a été reconstruit pour éliminer.

MedMCQA ne figure pas au cahier des charges. Il a été ajouté parce que les
trois corpus exploitables qu'il désigne sont soit francophones — MediQAl,
FrenchMedMCQA — soit sans description de patient — MedQuAD. Sans lui, la moitié
anglophone du corpus authentique n'aurait aucune vignette clinique. Cet ajout est
documenté plutôt que passé sous silence.

## Conformité RGPD

**Minimisation par conception.** Aucune donnée patient réelle n'entre dans le
projet : les vignettes sont synthétiques, et les corpus publics sont des jeux de
recherche sans donnée identifiante.

**Anonymisation.** Les textes issus des corpus passent par Presidio, avec un
réglage adapté au texte médical. Le réglage par défaut est inutilisable ici, et
l'écart a été mesuré : sur 400 exemples analysés chacun dans sa langue, les
entités `DATE_TIME`, `LOCATION` et `NRP` masqueraient un fragment de récit dans
81 % des cas. `DATE_TIME` emporte les délais d'évolution et l'âge du patient ;
`LOCATION` emporte `TA`, l'abréviation de la tension artérielle, cent trente-six
fois à elle seule. Délai, âge et constantes décident précisément du niveau de
triage. Trois décisions en découlent :

1. seules les entités réellement identifiantes sont masquées : nom, téléphone,
   adresse électronique, identifiants bancaires, adresse IP, URL, numéro de
   sécurité sociale, date de naissance. Quatre reconnaisseurs absents de Presidio
   ont été ajoutés et enregistrés dans les deux langues : le numéro de sécurité
   sociale, le téléphone au format national comme international, et la date de
   naissance en JJ/MM/AAAA comme en AAAA-MM-JJ ;
2. le vocabulaire clinique du projet est protégé — aucun terme du catalogue ni du
   lexique de triage ne peut être masqué. Cette garde vise les entités *retenues*,
   qui déraillent elles aussi sur du texte médical : sans elle, « inhibiteurs de
   recapture de la sérotonine » devient « inhibiteurs de recapture de la
   `<PERSON>` » ;
3. le contrôle qualité est **indépendant du détecteur** : un jeu d'expressions
   régulières distinct cherche, après masquage, ce qui aurait pu passer. Ses
   résultats sont publiés dans `metadata.json`.

**Auditabilité.** Chaque transformation est dans le code, la graine est fixée, et
la révision du dépôt ayant produit le jeu est inscrite dans `metadata.json`.

**Séparation des jeux.** Vérifiée par le code : le script de préparation échoue
si un tour utilisateur apparaît dans deux découpages, ou si un cas d'évaluation
se retrouve à l'entraînement. Les comptages sont publiés dans `metadata.json`.

## Reproduire

```bash
uv run python scripts/build_dataset.py
```

Graine fixée, sorties déterministes à version de corpus constante.

## Limites connues

- **Vignettes synthétiques** : variées et cliniquement cohérentes, mais sans le
  désordre du langage réel — récits rapportés par un tiers, informations
  contradictoires, patients qui minimisent.
- **Catalogue non validé cliniquement** : rédigé par un ingénieur à partir de la
  littérature de triage. C'est la limite principale du jeu de données.
- **Étiquettes de confiance moyenne** : les cas issus des corpus portent une
  étiquette produite par une règle lexicale, et leur champ `confiance` le dit.
