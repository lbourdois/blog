---
title: "JEUX DE DONNÉES AUDIO POUR LE FRANÇAIS"
categories:
  - projets
tags:
  - Dataset audio french
  - Dataset audio français
  - Dataset french audio
  - Dataset français audio
excerpt : Audio - Jeux de données pour pré-entraîner un modèle d'audio puis le finetuner sur une tâche en particulier
header :
    overlay_image: "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/NLP_radom_blog.png"
    teaser: "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Audio_datasets/Ortholang.png"
author_profile: false
classes: wide
---


# <span style="color: #FF0000"> **Avant-propos** </span>
Ci-dessous, vous trouverez plusieurs listes de jeux de données afin de pouvoir entraîner vos modèles d'audio.
Seuls ceux ayant un nombre d’heures conséquents sont listés (volume disponible supérieur à la dizaine d’heures). Les « petits » jeux de données non listés sont trouvables sur [Ortholang](https://www.ortolang.fr/market/corpora/cluster/speech_corpora).   
A noter que tous les jeux de données n’étant pas forcément du même format audio et textuel, un nettoyage devra être effectué afin d’uniformiser les formats.
<br><br><br>

# <span style="color: #FF0000"> **Apprentissage autosupervisé** </span>

<br>


|**Nom du jeu de données**|**Heures**|**Lien pour y accéder**|**Informations**|**Licence**|
| :-: | :-: | :-: | :-: | :-: |
|**VoxpopuliV2**|22 800 H |[Cliquer-ici](https://github.com/facebookresearch/voxpopuli)|Enregistrements récoltés au Parlement Européen entre 2009 et 2020.|CC0|
|**Librivox**|1 096 (ou 2 158 H ? => A vérifier)|[Cliquer-ici](https://librivox.org/search?primary_key=2&search_category=language&search_page=1&search_form=get_results)  |996 livres de grands auteurs français tombés dans le domaine public. Note : le jeu de données [M-AILABS French-v0.9](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/) est basé en partie sur Librivox. De même pour [Multilingual LibriSpeech]( https://arxiv.org/abs/2012.03411). |Domaine public|

<br>

Ce sont ainsi au moins de **24 000 heures** d’audio qui sont disponibles pour l’apprentissage autosupervisé.
<br><br><br>

# <span style="color: #FF0000"> ***Finetuning*** </span>

## <span style="color: #FFBF00"> ***Automatic Speech Recognition* (ASR)** </span>

### <span style="color: #51C353"> **Données en libre accès** </span>

<br>


|**Nom du jeu de données**|**Heures**|**Lien pour y accéder**|**Informations**|**Licence**|
| :-: | :-: | :-: | :-: | :-: |
|**Common Voice**|1 113 H|[Cliquer-ici](https://commonvoice.mozilla.org/en/datasets) |Chiffres indiqués pour la version 16, 981H sur les 1 113 ont été validées|CC-0 |
|**Corpus d'Etude pour le Français Contemporain (CEFC)**|450 H| [Cliquer-ici](https://repository.ortolang.fr/api/content/cefc-orfeo/10/documentation/site-orfeo/index.html) ou [Cliquer-ici](https://www.ortolang.fr/market/corpora/cefc-orfeo) | Regroupe 10 corpus sources (CFPP2000,  CLAPI, C-ORAL-ROM, CRFP, FLEURON, FRENCH ORAL NARRATIVE, OFROM, TUFS, Valibel). Possibilité de trier ce que l’on souhaite (tv, radio, téléphone, face à face, etc.)|CC-BY 4.0|
|**ESLO**| 300 H pour ESLO1 400 H pour ESLO2 | [Cliquer-ici](https://ct3.ortolang.fr/data/eslo/) ou [Cliquer-ici](https://www.ortolang.fr/market/corpora/eslo/v1) | ESLO1 contient des entretiens (formels ou informels de type conversation dans une rue) enregistrés entre 1968 et 1974. Les données ne sont pas forcément de bonnes qualités (grésillements).   ESLO2 reprend le même principe que ESLO1 mais porte sur des entretiens datant de 2008 à 2020. |CC-BY 4.0|
|**Conférences Pierre Mendès France**|300 H|[Cliquer-ici](https://www.data.gouv.fr/fr/datasets/transcriptionsxml-audiomp3-mefr-ccpmf-2012-2020-zip/)|Audios au format MP3 et transcriptions au format XML des conférences du centre de conférences Pierre Mendès France du MEFR (2012-2020).|Open Licence version 2.0|
|**VoxpopuliV2**|211 H |[Cliquer-ici](https://github.com/facebookresearch/voxpopuli)| Parti annoté du corpus. Enregistrements annotés récoltés au Parlement Européen entre 2009 et 2020.|CC0|
|**TCOF**|146 H|[Cliquer-ici](https://www.ortolang.fr/market/corpora/tcof) | Des enregistrements d'interactions adultes-enfants (enfants jusque 7 ans) et des enregistrements d'interactions entre adultes. | CC BY-NC-SA 2.0|
|**PFC**|131 H|[Cliquer-ici](https://www.ortolang.fr/market/corpora/pfc) | Le corpus complet contient plus de 50 enquêtes (soit plus de 400 locuteurs). Nous avons ici accès qu’à une sous-partie de ce corpus (16 enquêtes, soit 164 locuteurs) qui a été anonymisée. | CC BY-NC 4.0|
|**SynPaFlex**|87 H|[Cliquer-ici](https://www.ortolang.fr/market/corpora/synpaflex-corpus) |Annotation de 87h de corpus de livres-audios. |CC-BY 2.0|
|**MPF**|78 H|[Cliquer-ici](https://www.ortolang.fr/market/corpora/mpf)| Ce corpus vise à documenter des évolutions en cours dans le français, l’émergence d’un vernaculaire urbain contemporain, ainsi que les effets sur le français du contact avec les langues de l’immigration.  Nécessite un compte (gratuit) sur Ortholang pour télécharger le corpus.|CC-BY 4.0|
|**Lingua Libre**|44 H| [Cliquer-ici](https://lingualibre.org/LanguagesGallery/)|Prononciation de mots| CC BY-SA 4.0 |
|**African Accented French**|22 H| [Cliquer-ici](http://www.openslr.org/57/) |Interviews réalisées par l’armée américaine|Apache 2.0|
|**ALIPE**|15 H| [Cliquer-ici](https://www.ortolang.fr/market/corpora/alipe-000853) |Ce corpus contient la transcription d’environ 15H de conversations informelles entre enfant et parents.| CC-BY-SA 4.0|
|**Fleurs**|13 H| [Cliquer-ici](https://huggingface.co/datasets/google/fleurs) |Lecture de phrases issues du jeu de données [FLoRes]( https://arxiv.org/abs/2106.03193) | CC-BY 4.0|
|**SUMM-RE ASRU**|12,5 H| [Cliquer-ici](https://www.ortolang.fr/market/corpora/summ-re-asru)  |Réunions de 3 à 4 personnes transcrites avec Whisper puis corrigée manuellement| CC-BY-SA 4.0|
|**SIWIS** |~ 10 H|[Cliquer-ici](https://datashare.ed.ac.uk/handle/10283/2353) | Au total, 9750 énoncés provenant de sources diverses telles que des débats parlementaires et des romans. |CC-BY 4.0|

<br>

Ce sont ainsi environ **2 900 heures** d’audio qui sont disponibles librement pour l’apprentissage supervisé de la tâche d’ASR.
<br><br><br>


### <span style="color: #51C353"> **Corpus à accès limités (demandes auprès d’Université / Labo à effectuer)** </span>

<br>

|**Nom du jeu de données**|**Heures**|**Lien pour y accéder**|**Informations**|**Licence**|
| :- | :- | :- | :- | :- |
|**INA**|1200H sont disponibles et 3000H sont indiqués comme "à venir"|[Cliquer-ici](https://dataset.ina.fr/) |Données de l'INA disponibles dans différents sous jeux de données. Pour pouvoir avoir accès aux données il faut remplir un formulaire (cf. le lien). Il est précisé que "seuls sont autorisés à s’inscrire les laboratoires de recherche, les PME innovantes ainsi que toutes autres personnes morales disposant d’un service ou d’une activité de recherche scientifique."| Licence non précisée mais les CGU sont assez restrictives concernant leur utilisation à des fins non universitaires.|
|**Decoda-RATP**|74H|[Cliquer-ici](http://www.lrec-conf.org/proceedings/lrec2012/pdf/684_Paper.pdf) |Appels téléphoniques à la RATP enregistrés et annotés (transcription, NER, etc.)| Non précisé, il faut contacter les auteurs |
|**NCCFr**|35H|[Cliquer-ici](https://mirjamernestus.nl/Ernestus/NCCFr/index.php) |Conversations entre amis annotées par des professionnels| Non précisé, il faut contacter les auteurs | 

Ce sont ainsi environ **1 300 heures** d’audio qui sont disponibles sous condition d’accès aux données pour l’apprentissage supervisé de la tâche d’ASR.

<br>

### <span style="color: #51C353"> **Données payantes** </span>

<br>


|**Nom du jeu de données**|**Heures**|**Lien pour y accéder**|**Informations**|**Licence**|
| :-: | :-: | :-: | :-: | :-: |
|**ESTER**|100 H annotées + 1700 H non annotées|[Cliquer-ici](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0241/) |Corpus d’enregistrements d’émissions radiophoniques.|3 types de licences (avec usage commercial ou non). Cf. le lien pour plus d’informations. |
|**ESTER 2**|~200 H|[Cliquer-ici](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0338/)|Inclus les 100H annotées d'ESTER1 + 100 nouvelles heures annotées. Corpus de transcriptions manuelles d’émissions radiophoniques et de transcriptions manuelles rapides de radios africaines. |2 types de licences (avec usage commercial ou non). Cf. le lien pour plus d’informations.|
|**EPAC**|~100 H|[Cliquer-ici](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0305/)|100H de transcriptions manuelles réalisées à partir des 1 700 heures d'enregistrements non transcrits du jeu de données ESTER. |2 types de licences (avec usage commercial ou non). Cf. le lien pour plus d’informations.|
|**MEDIA**|70 H|[Cliquer-ici](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0272/)|1 258 dialogues transcrits pour 250 locuteurs adultes sur le domaine du tourisme et de la réservation d’hôtel.|2 types de licences (avec usage commercial ou non). Cf. le lien pour plus d’informations.|
|**ETAPE**|30 H|[Cliquer-ici](https://catalogue.elra.info/en-us/repository/browse/ELRA-E0046/)|Environ 30H de radio et TV françaises incluant de la parole non planifiée et une proportion raisonnable de données multi-locuteurs. Des données transcrites soigneusement en incluant l’annotation des entités nommées.|3 types de licences (avec usage commercial ou non). Cf. le lien pour plus d’informations.|

<br>

Ce sont ainsi environ **400 heures** d’audio qui sont disponibles pour l’apprentissage supervisé et 1300 heures qui sont disponibles pour l’apprentissage autosupervisé en achetant ces corpus.

<br><br>


## <span style="color: #FFBF00"> **Audio Classification**</span>

### <span style="color: #51C353"> **Données en libre accès** </span>

<br>

|**Nom du jeu de données**|**Heures**|**Lien pour y accéder**|**Qualité / Source**|**Licence**|
| :- | :- | :- | :- | :- |
|**Voxlingua107**|67 H|[Cliquer-ici](http://bark.phon.ioc.ee/voxlingua107/)|Audios issues de YouTube|CC-BY 4.0|
|**FLEURS-LangID**| 13H pour le français et ~1400H au total pour les 102 langues |[Cliquer-ici](https://huggingface.co/datasets/google/fleurs) |Identifier à quelle langue appartient un audio parmi une liste de 102 langues| CC BY-NC 4.0 |
|**Minds14**|1h15|[Cliquer-ici](https://huggingface.co/datasets/PolyAI/minds14) |Audios à classer parmi 14 classes différentes| CC BY-NC 4.0 |

<br>

Ce sont ainsi environ **80 heures** d’audio qui sont disponibles pour la tâche d’identification d’une langue (en pratique nettement plus si on inclus les jeux de données pour la traduction de la section suivante) et **1h15** pour la classification d’intentions.
<br><br>

### <span style="color: #51C353"> **Corpus à accès limités (demandes auprès d’Université / Labo à effectuer)** </span>

<br>

|**Nom du jeu de données**|**Heures**|**Lien pour y accéder**|**Qualité / Source**|**Licence**|
| :- | :- | :- | :- | :- |
|**Allosat**|~37H|[Cliquer-ici](https://aclanthology.org/2020.lrec-1.197.pdf) |Appels enregistrés à un centre d'appel dont les conversations portent sur des thèmes de type : énergie, agence de voyage, agence immobilière et assurances. Les données ont aussi été retranscrites mais automatiquement à l'aide de Kaldi| Non précisé, il faut contacter les auteurs |
|**Cemo**|20H|[Cliquer-ici](https://arxiv.org/pdf/2110.14957.pdf) | Appels aux urgences annotées. Il semble également que les données ont aussi été retranscrites d'après la conclusion du papier.| Non précisé, il faut contacter les auteurs | 
|**RECOLA**|9,5H|[Cliquer-ici](https://diuf.unifr.ch/main/diva/recola/download.html) | Enregistrements audio, visuels et physiologiques (électrocardiogramme et activité électrodermale) d'interactions dyadiques en ligne entre 46 participants francophones, qui résolvaient une tâche en collaboration. | [EULA](https://diuf.unifr.ch/main/diva/recola/data/eula_recola_database.pdf) |
|**mGEMEP**|0,9H|[Cliquer-ici](https://www.unige.ch/cisa/gemep) |Données provenant d'acteurs| Non précisé, il faut contacter les auteurs |

<br>
Ce sont ainsi environ **120 heures** d’audio qui sont disponibles sous condition d’accès aux données afin d’entraîner un modèle de classification d’audio de type reconnaissance d’émotions.

<br><br>



## <span style="color: #FFBF00"> ***Automatic Speech Translation* (AST)** </span>

### <span style="color: #51C353"> **Données en libre accès** </span>

<br>

|**Nom du jeu de données**|**Heures**|**Lien pour y accéder**|**Qualité / Source**|**Licence**|
| :- | :- | :- | :- | :- |
|**Europarl-ST (fr->x) et (x->fr)**| 176H de fr->x et 179H de x->fr soit 355H au total|[Cliquer-ici](https://mllp.upv.es/europarl-st/) | Corpus multilingue (français, anglais, allemand, italien, espagnol, portugais, polonais, roumain, néerlandais) construits à partir des débats menés au Parlement européen entre 2008 et 2012. | CC BY-NC 4.0 |
|**MuST-C (en->fr)**|236H|[Cliquer-ici](https://mt.fbk.eu/must-c/) | Provient de TEDs en anglais |CC BY-NC-ND 4.0 |
|**Covost2 (fr->en)**|225H|[Cliquer-ici](https://huggingface.co/datasets/covost2) |Données basées sur Common Voice 4.0 | CC0 |
|**mTEDx (fr->x)**|25H à 50H en fonction de la langue cible, 189H au total|[Cliquer-ici](http://www.openslr.org/100) |Données issues des conférences TED. Les langues disponibles étant le français, l’espagnol, l’allemand, l’italien, le russe, le portugais, le grec, l’arabe et l’anglais |CC BY-NC-ND 4.0 |

<br>

Ce sont ainsi environ **1000 heures** d’audio qui sont disponibles afin d’entraîner un modèle de traduction d’audio incluant à partir ou à destination du français.


<br><br><br>


# <span style="color: #FF0000"> **Références** <span>

-	[VoxPopuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation]( https://arxiv.org/abs/2101.00390) de Wang et al. (2021)
-	[Common Voice: A Massively-Multilingual Speech Corpus](https://arxiv.org/abs/1912.06670) d’ Ardila et al. (2020)
-	 [Le projet ORFÉO : un corpus d'études pour le français contemporain.](https://journals.openedition.org/corpus/2936) de Benzitoun et al.  (2016). 
-	[Discours sur la ville. Corpus de Français Parlé Parisien des années 2000 (CFPP2000)]( https://cocoon.huma-num.fr/exist/crdo/meta/cocoon-8bc96a4e-9899-30e4-99be-c72d216eb38b) de Branca-Rosoff et al. (2020)
-	[CLAPI, une base de données multimodale pour la parole en interaction : apports et dilemmes](https://journals.openedition.org/corpus/2991) de Baldauf-Quilliatre et al. (2016)
-	[The C-ORAL-ROM CORPUS. A Multilingual Resource of Spontaneous Speech for Romance Languages](https://aclanthology.org/L04-1200/) de Cresti et al. (2004)
-	[Corpus de référence du français parlé](https://shs.hal.science/halshs-01388193) de Delic et al. (2004)
-	[De l’archive de parole au corpus de référence : la base de données orales du français de Suisse romande](https://journals.openedition.org/corpus/3060) d’Avanzi et al. (2016)
-	[Disfluences et vieillissement langagier. De la base de données VALIBEL aux corpus outillés en français parlé](https://journals.openedition.org/corpus/3006) de Bolly et al. (2016)
-	[Un grand corpus oral disponible : le Corpus d’Orléans 1968-2012 [A Large available oral corpus: Orleans corpus 1968-2012]]( https://aclanthology.org/2011.tal-3.2/)  d’Eshkol-Taravella et al. (2012)
-	[Traitement de Corpus Oraux en Français](https://journals.openedition.org/pratiques/1597) d’André et Canut (2010)
-	[Le projet PFC: une source de données primaires structurées](https://www.researchgate.net/publication/49135381_Le_projet_PFC_Phonologie_du_Francais_Contemporain_une_source_de_donnees_primaires_structurees) de Durand et al. (2009)
-	[SynPaFlex-Corpus: An Expressive French Audiobooks Corpus dedicated to expressive speech synthesis.](https://aclanthology.org/L18-1677/) de Sini et al (2018)
-	[Les parlers jeunes dans l’Île-de-France multiculturelle](https://journals.openedition.org/lidil/4854) de Gadet et al. (2017)
-	[ALIPE (Acquisition de la Liaison et Interactions Parents Enfants)]( https://lrl.uca.fr/projet_du_labo/alipe-fr/)  de Chabanal et al. (2013)
-	[Transcribing And Aligning Conversational Speech: A Hybrid Pipeline Applied To French Conversations] de Yamasaki et al. (2023)
-	[The SIWIS French Speech Synthesis Database](https://datashare.ed.ac.uk/handle/10283/2353) de Yamagishi et al. (2017)
-	[Enhancing The RATP-DECODA Corpus With Linguistic Annotations For Performing A Large Range Of NLP Tasks](https://aclanthology.org/L16-1166/) de Lailler et al. (2016)
-	[Nijmegen Corpus of Casual French](https://mirjamernestus.nl/Ernestus/NCCFr/index.php) de Torreira et al. (2010)
-	[Multilingual and Cross-Lingual Intent Detection from Spoken Data]( https://arxiv.org/abs/2104.08524) de Gerz, Su et al. (2021)
-	[FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech](https://arxiv.org/abs/2205.12446) de Conneau et al. (2022)  
-	[On the use of Self-supervised Pre-trained Acoustic and Linguistic Features for Continuous Speech Emotion Recognition]( https://aclanthology.org/2020.lrec-1.197.pdf) de Macary et al. (2020)
-	[End-to-End Speech Emotion Recognition: Challenges of Real-Life Emergency Call Centers Data Recordings](https://arxiv.org/pdf/2110.14957.pdf) de Deschamps-Berger et al. (2021)
-	[Introducing the RECOLA Multimodal Corpus of Remote Collaborative and  Affective Interactions](https://drive.google.com/file/d/0B2V_I9XKBODhNENKUnZWNFdVXzQ/view?resourcekey=0-pkUwtWY7x82Gw5zurnQNag) de Ringeval et al. (2013)
-	[Introducing the Geneva Multimodal expression corpus for experimental research on emotion perception](https://pubmed.ncbi.nlm.nih.gov/22081890/) de Bänziger  et al. (2012)
-	[Europarl-ST: A Multilingual Corpus For Speech Translation Of Parliamentary Debates]( https://arxiv.org/abs/1911.03167) d’Iranzo-Sánchez et al. (2019)
-	[MuST-C: a Multilingual Speech Translation Corpus](https://aclanthology.org/N19-1202/) de Di Gangi et al. (2019)
-	[CoVoST 2: A Massively Multilingual Speech-to-Text Translation Corpus](https://arxiv.org/abs/2007.10310) de Wang, Wu et Pino (2020)
-	[The Multilingual TEDx Corpus for Speech Recognition and Translation](https://arxiv.org/abs/2102.01757) de Salesky et al. (2021)
-	[VoxLingua107: a Dataset for Spoken Language Recognition](https://arxiv.org/abs/2011.12998) de Valk et Alumäe (2020)

<br><br><br>
# <span style="color: #FF0000"> **Citation** <span>
Si vous venez à utiliser des éléments de cet article, veillez s'il vous plait en créditer les auteurs en utilisant par exemple comme suit :<br>
“*Jeux De Données Audio Pour Le Français* par Loïck BOURDOIS (https://lbourdois.github.io/blog/projets/dataset_audio_fr/)”<br>
Merci :)
