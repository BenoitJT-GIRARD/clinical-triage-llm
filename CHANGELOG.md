# Journal des versions

Les versions suivent [SemVer](https://semver.org/lang/fr/). Le format du dataset
et le contrat de l'API font partie de l'interface publique du projet : une
rupture sur l'un ou l'autre incrémente la version majeure.

## 1.0.0

Première version complète du prototype.

### Données

- Catalogue de présentations cliniques rédigé pour le projet : il porte désormais
  la vérité terrain du triage, à la place de l'étiquetage par mots-clés.
- Générateur de vignettes cliniques bilingues, avec âge, antécédents, délai
  d'installation et constantes vitales cohérentes avec la gravité.
- Jeu d'évaluation clinique rédigé à la main, dont près de la moitié de
  présentations atypiques, tenu à l'écart de tout entraînement.
- Extraction filtrée des corpus publics : seuls les cas où un signe clinique est
  effectivement identifié sont conservés, avec un niveau de confiance explicite.
- MediQAl intégré : 3 075 vignettes cliniques françaises distinctes, dont 569 avec
  constantes lisibles. C'est le seul corpus du cahier des charges qui décrive des
  patients, et il porte désormais le volume authentique francophone.
- Anonymisation restreinte aux entités réellement identifiantes, avec protection
  du vocabulaire clinique et contrôle qualité indépendant du détecteur.
- Constantes vitales des vignettes pédiatriques : pas de douleur auto-évaluée
  avant six ans, l'échelle visuelle analogique n'existant pas à cet âge.
- Séparation des jeux vérifiée par le code : la construction échoue en cas de fuite.
- Les quatre corpus sont lus intégralement plutôt que jusqu'à un plafond
  d'entrées : 182 822 entrées pour MedMCQA, 16 407 pour MedQuAD, les trois
  découpages de FrenchMedMCQA. Le rendement publié décrit ainsi les sources
  entières, et les corpus fournissent 34 % du jeu livré.
- Le motif qui reconnaît une présentation de patient accepte la syntaxe employée
  par MediQAl — « Homme, 58 ans », « M. Dupont, 30 ans » — et pas seulement
  « homme de 58 ans » : 342 vignettes françaises authentiques étaient écartées
  sans examen.
- Borne haute de longueur à 800 caractères, valeur pour laquelle aucun cas livré
  ne dépasse le budget de description du service. Elle est vérifiée à chaque
  construction contre le tokenizer réel, sur les cas réellement livrés et non sur
  le vivier d'extraction, et la construction échoue si un cas la dépasse.
- Fiches de symptômes coupées à la phrase et non au caractère : 120 des 172 cas
  MedQuAD livrés se terminaient au milieu d'un mot. Le préambule qui les
  introduit compte dans la borne, sans quoi 178 descriptions la dépassaient.
- Question d'examen distinguée du pronom relatif. Supprimer toute phrase
  contenant « which » ou « quel » emportait du récit clinique, et laissait passer
  les familles d'énoncés les plus courantes de MedMCQA, qui n'emploient aucun mot
  interrogatif. Le résidu d'énoncés de QCM tombe à 0,5 % du jeu.
- Déduplication des cas de corpus portée sur la description et non sur le tour
  patient, qui n'en est que l'enveloppe tirée au hasard.
- Un récit déjà terminé par un point d'interrogation ne reçoit pas de point
  supplémentaire.
- Une lecture de corpus interrompue en route arrête la construction. En
  streaming, tout le trafic a lieu pendant l'itération : une coupure produirait
  un corpus amputé dont le rendement publié décrirait une lecture complète.
- Un corpus indisponible fait échouer la construction au lieu de produire
  silencieusement un jeu amputé.
- L'apport des corpus est borné par case niveau × langue avant de l'être
  globalement, sans quoi la case anglophone la mieux fournie déborde le volume
  cible et rompt l'équilibre entre langues.
- `--sft-size` est validé : sous le seuil où les jeux de validation et de test
  sortiraient vides, la construction refuse de commencer.
- La carte du dataset enregistre la commande réellement exécutée, options
  comprises, dit explicitement quand l'anonymisation a été désactivée, et publie
  l'entonnoir d'extraction corpus par corpus.
- Trois diagnostics du jeu produit sont publiés : nombre de réponses attendues
  distinctes, séparabilité en découpage groupé par présentation, part du niveau
  prédictible par les seules métadonnées du générateur.

### Modèle

- Gabarit de dialogue installé par le projet et jeton de fin de séquence corrigé,
  partagés par l'entraînement, l'alignement, l'évaluation et le service.
- Tête de sortie entraînée avec les projections : dans le modèle de base, les
  jetons ChatML sont un vecteur unique jamais entraîné et la tête est liée aux
  embeddings, ce qui rendait le jeton de fin impossible à produire. Les arrêts
  nets passent de 0 % à 100 %, et la réponse de 220 à 97 jetons.
- Fine-tuning porté sur Unsloth : même configuration, 7,23 Go au lieu de 10,03.
- Fusion des adaptateurs corrigée : la tête de sortie est découplée **avant** la
  fusion et le calcul se fait en float32, sans quoi le delta s'écrivait dans la
  matrice d'embeddings partagée. L'export refuse de se terminer si la
  vérification échoue.
- Jeu supervisé exposé en colonnes `prompt` et `completion` : la perte n'est
  calculée que sur la réponse, sans collateur particulier.
- Comparaison de quatre configurations d'hyperparamètres, jugées sur la perte de
  validation et sur l'exactitude de triage, publiées avec leur intervalle de
  confiance : sur soixante cas de validation les quatre se recouvrent, et rejouer
  la comparaison fait changer l'ordre des places intermédiaires. La comparaison
  établit qu'aucune configuration ne dégrade le résultat, pas qu'une l'emporte.
- Alignement DPO avec le modèle supervisé comme référence, perte de préférence
  combinée à un terme supervisé qui retient la dérive, et jeu de validation de
  préférences.
- Paires de préférence de même format et de longueur comparable ; aucun
  surclassement présenté comme contre-exemple.

### Évaluation

- Trois références : classe majoritaire, prudence maximale, règle explicite.
- Aucune estimation publiée nue, et l'estimateur choisi selon la nature de la
  grandeur : Wilson au-delà de trente cas, Clopper-Pearson exact en dessous et
  sur le sous-triage — la seule mesure dont une sous-couverture se paie en
  patients —, rien du tout sous six cas, où la fraction brute remplace un taux
  que l'effectif ne permet pas d'estimer. Les comptages exhaustifs du corpus ne
  portent pas de barre d'erreur : ils sont dénombrés, pas estimés.
- Comparaisons lues sur le bon test plutôt que sur le recouvrement de deux
  intervalles, qui ne prouve rien : McNemar exact quand les systèmes voient les
  mêmes cas — c'est le cas du modèle contre la règle explicite —, intervalle de
  Newcombe sur l'écart quand les deux jeux sont indépendants.
- Figures reprises aux standards de publication : axes et unités sur chaque
  panneau, effectif et estimateur au pied de chaque figure, nombres à la
  française, échelle de couleur commune et barre d'échelle sur les matrices de
  confusion, et une mesure absente qui ne se dessine plus comme une valeur nulle.
- Contrôles de sécurité sur le contenu généré : recommandation incohérente,
  diagnostic affirmé, réponse hors langue, constante inventée ou falsifiée,
  structure de réponse incomplète, niveau annoncé hors de la taxonomie.
- Contrôles de sécurité validés sur les réponses de référence du catalogue : les
  motifs étaient accentués alors que la lecture d'une réponse retire les accents,
  ce qui signalait à tort dix réponses conformes sur soixante-dix. Un test de
  non-régression l'empêche de revenir.
- Comparaison des deux livraisons possibles du modèle — adaptateur appliqué à
  chaud, et modèle fusionné — sur l'exactitude, la latence et le poids à
  télécharger, avec le taux d'accord entre leurs prédictions.
- Matrice de confusion avec colonne dédiée aux réponses hors format.
- Détail par langue et par nature de cas, et table d'erreurs commentée.
- Centiles de latence calculés par rang le plus proche, convention énoncée : le
  95ᵉ centile publié conditionne un critère de mise en service. La mesure porte
  sur `POST /triage` — ce que le service rend, règle, anonymisation et journal
  compris — et non sur le moteur seul, qui n'en donne que la décomposition.
- Règle explicite écartée des références du jeu de test interne : elle sert à
  filtrer le corpus en amont, et s'y comparerait donc à elle-même.
- Règle explicite resserrée sur deux tournures qu'elle sur-déclenchait : « ne
  répond pas » sans complément — celui d'un patient qui ne répond pas au
  traitement — et « hémorragie » nu, qui couvre l'hémorragie sous-conjonctivale.
- Entrées dégradées : celle qui décrit un syndrome coronarien porte le niveau
  qu'on en attend, et un triage différé y est compté non conforme même bien
  formé. Le contrôle ne vérifiait que la forme.
- Quatrième référence : un classifieur classique — n-grammes pondérés et
  séparateur linéaire — entraîné sur les mêmes paires que le modèle, sur le seul
  jeu d'entraînement, et dont la configuration est choisie sur le jeu de
  validation sans consulter le jeu clinique. Battre la règle à mots-clés
  n'établit pas ce que le fine-tuning apporte par-dessus un apprentissage
  ordinaire.
- La comparaison appariée par test de McNemar porte sur les deux références qui
  apprennent, et non sur la seule règle explicite.
- Un nom de modèle inconnu passé à `--models` est refusé au lancement, et non
  après l'évaluation des autres sur un `KeyError` qui laissait le fichier de
  résultats non écrit.

### Service et déploiement

- Clé d'API obligatoire au démarrage, comparée à temps constant, avec quota par
  appelant. L'en-tête ne compte comme identité que lorsqu'il a été vérifié : en
  mode ouvert il ne l'est par personne, et en changer à chaque requête donnait un
  seau de comptage neuf à chaque fois.
- Moteur d'inférence protégé par sa propre clé dès qu'il est joignable autrement
  que par la boucle locale, faute de quoi son adresse suffit à obtenir une
  inférence GPU sans quota, sans anonymisation et sans trace d'audit.
- Sonde de santé qui interroge réellement le moteur d'inférence, et ne publie pas
  son adresse : elle n'exige ni clé ni quota, et le message d'erreur du client
  HTTP porte l'URL interrogée.
- Descriptions bornées à la fenêtre du modèle, et la coupe annoncée. Le contrat
  acceptait quatre mille caractères pour une fenêtre qui en tient bien moins : au
  delà, la génération ne dégradait pas, elle s'interrompait sur une erreur de
  dimension de tenseur. C'est le contrôle de robustesse « copier-coller de deux
  pages » qui l'a trouvé. La réponse et le journal d'audit portent désormais
  `description_tronquee` — une décision de triage prise sur un récit incomplet
  doit laisser une trace, et seul un soignant sait si ce qui manque comptait.
- Même borne sur la mesure de préférences, dont les dissertations dépassaient
  aussi la fenêtre. La comparaison reste juste — même borne des deux côtés, score
  normalisé par la longueur — et le nombre de réponses écrêtées est journalisé.
- Champs de requête bornés, y compris le dictionnaire de réponses du
  questionnaire : tout y est concaténé en une chaîne unique, que la règle de
  triage relit ensuite.
- Questionnaire adaptatif au motif, compilant ses réponses en **phrases** et non
  en questions suivies de leur réponse : la description produite est relue par la
  règle de triage, et « difficulté à respirer ? non » y était lu comme le symptôme
  lui-même. Un rhume dont tout était nié ressortait classé urgence vitale.
- La règle des réponses alarmantes ne s'applique qu'aux questions qui attendent
  un oui ou un non. Comparées sans cette garde, une réponse en texte libre et une
  question sans réponse alarmante valaient toutes deux « rien » : décrire le
  mécanisme d'une entorse arrêtait la collecte sur une détresse vitale.
- Réponse exposant l'avis de la règle explicite et l'accord avec le modèle. Les
  intitulés de réponse libre du questionnaire ne portent aucun terme du lexique
  de gravité : « Idées suicidaires : je ne sais pas » faisait apparaître ce signe
  dans les raisons montrées au soignant, pour un patient qui n'avait rien
  exprimé. Les phrases canoniques, elles, rapportent une réponse réellement
  donnée et gardent le mot juste.
- Journal d'audit anonymisé par le module lui-même — la description reçue **et**
  la réponse rendue, qui reprend le récit du soignant — dans les deux langues,
  l'API acceptant du texte libre et le moteur français ne repérant pas un nom
  dans une syntaxe anglaise. Avec le modèle réellement chargé et une durée de
  conservation.
- Image de service épinglée par empreinte, exécutée sans privilège, dépendances
  épinglées à la version exacte.
- Intégration continue : style, sécurité, audit des dépendances, tests avec seuil
  de couverture, construction de l'image puis démarrage et appel du service.
- Déploiement continu sur Modal : image à chaque poussée, version de modèle figée
  par une étiquette posée sur les dépôts du Hub, redéploiement à cette révision
  et vérification de la santé de l'endpoint qui en résulte. Désarmé par défaut.
- Ports publiés sur l'hôte paramétrables, `TRIAGE_VLLM_PORT` et
  `TRIAGE_API_PORT` : 8000 est un port courant, et une autre pile déjà lancée sur
  la machine empêchait le démarrage.
- Dossier des poids paramétrable, `TRIAGE_MODELS_DIR` : sur un poste dont le
  dépôt est synchronisé dans le nuage, Docker Desktop monte un dossier vide sans
  le signaler, et le conteneur démarre sans voir les poids.
- Image du moteur d'inférence surchargeable par `TRIAGE_VLLM_IMAGE`, la version
  épinglée par empreinte restant celle par défaut. Le moteur V1 de vLLM exige
  l'adressage virtuel unifié de CUDA, que WSL 2 n'expose pas : sur un poste
  Windows, la pile s'arrête sur « UVA is not available » avant de charger les
  poids.
- Journal d'audit écrit dans un volume nommé. Monté depuis un dossier de l'hôte,
  il appartient à root, et le service — qui tourne sans privilège — échouait à
  l'écrire : chaque requête répondait 500 après avoir produit sa décision.
- Le service ouvre son journal d'audit au démarrage et refuse de servir s'il ne
  peut pas l'écrire. Découvrir l'indisponibilité à la première requête revient à
  avoir déjà répondu sans trace.

### Documentation

- Publication du modèle supervisé fusionné sur le Hub : c'est le modèle de base
  de l'adaptateur DPO, qui sans lui serait inchargeable.
- Limites exposées : absence de vérité terrain externe, auteur commun au
  catalogue et au jeu d'évaluation, faible diversité des réponses attendues, part
  du niveau lisible sur la forme des vignettes.
- La longueur médiane des réponses attendues est mesurée à la construction du jeu
  au lieu d'être recopiée : écrite à la main, elle avait dérivé.
- Table des estimateurs levée d'une contradiction : la ligne « moins de six cas,
  aucun intervalle » s'appliquait aussi au sous-triage, que la ligne précédente
  couvre à tout effectif.
- Trois affirmations que les mesures démentaient sont corrigées : le corpus n'est
  plus dit « équilibré à une unité près » quand l'écart est de deux, MedQuAD
  n'est plus dit « sans fournir de cas » quand il en fournit, et le décompte des
  corpus lus ne confond plus les cinq corpus lus avec les quatre du cahier des
  charges.
