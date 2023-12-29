---
title: "JEUX DE DONNÉES AUDIO POUR LE FRANÇAIS"
categories:
  - projets
tags:
  - Introduction aux SSM 
excerpt : Audio - Jeux de données pour entraîner un modèle d'audio générique et pour le finetuner ensuite sur une tâche en particulier
header :
    overlay_image: "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/NLP_radom_blog.png"
    teaser: "https://github.com/lbourdois/blog/assets/58078086/cb2dca34-9a3e-481a-8773-2360a1ceaa1c"
author_profile: false
classes: wide
---

WIP  
(Il s'agit ci-dessous d'une bibliographie que j'ai effectué en 2022, elle doit être actualisée pour intégrer les jeux de données apparus depuis).  
Voir notamment : https://huggingface.co/datasets?task_categories=task_categories:automatic-speech-recognition&language=language:fr&sort=trending
<br><br><br>

# <span style="color: #FF0000"> **Avant-propos** </span>
Ci-dessous, vous trouverez plusieurs listes de jeux de données afin de pouvoir entraîner vos modèles d'audio.
Seuls ceux ayant un nombre d’heures conséquents sont listés (volume disponible supérieur à la dizaine d’heures). 
De même, uniquement les données possédant une licence permettant une réutilisation sont indiquées. 
A noter que tous les jeux de données n’étant pas forcément du même format audio et textuel, un nettoyage devra être effectué afin d’uniformiser les formats.
<br><br><br>

# <span style="color: #FF0000"> **Apprentissage autosupervisé** </span>

<br>

|**Nom du jeu de données**|**Heures**|**Lien pour y accéder**|**Informations**|**Licence**|
| :-: | :-: | :-: | :-: | :-: |
|**VoxpopuliV2**|22 800 H |[Cliquer-ici](https://github.com/facebookresearch/voxpopuli)|Enregistrements récoltés au Parlement Européen entre 2009 et 2020.|[CC0](https://creativecommons.org/share-your-work/public-domain/cc0/) |
|**Librivox**|1 096 (ou 2 158 H ? => A vérifier)|[Cliquer-ici](https://librivox.org/search?primary_key=2&search_category=language&search_page=1&search_form=get_results)  |996 livres de grands auteurs français tombés dans le domaine public.|Domaine public|
|**M-AILABS French-v0.9 Corpus**|190,5 H|[Cliquer-ici](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/)|Textes publiés entre 1884 et 1964 tombés dans le domaine public.|[M-AILABS LICENSE](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/)|
|**mTEDx (fr->x)**|189 H|[Cliquer-ici](http://www.openslr.org/100)|Données issues des conférences TED. Normalement ce jeu de données sert à la tâche d’Automatic Speech Translation (AST) c’est-à-dire la traduction d’audios. Si ce n’est pas notre tâche finale, ce jeu de données peut servir pour la phase d’apprentissage autosupervisé.|CC-BY 4.0|
|**CFPP2000**|150 H|[Cliquer-ici](https://www.ortolang.fr/market/corpora/cfpp2000)|Ensemble d'interviews sur les quartiers de Paris et de la proche banlieue. |CC-BY 3.0|
|**Voxlingua107**|67 H|[Cliquer-ici](http://bark.phon.ioc.ee/voxlingua107/)|Audios issues de YouTube|CC-BY 4.0|

<br>

Ce sont ainsi plus de **24 500 heures** d’audio qui sont disponibles pour l’apprentissage autosupervisé.
<br><br><br>

# <span style="color: #FF0000"> ***Finetuning*** </span>

## <span style="color: #FFBF00"> ***Automatic Speech Recognition* (ASR)** </span>

### <span style="color: #51C353"> **Données en libre accès** </span>

<br>

|**Nom du jeu de données**|**Heures**|**Lien pour y accéder**|**Informations**|**Licence**|
| :-: | :-: | :-: | :-: | :-: |
|**Common Voice**|1 113 H|[Cliquer-ici](https://commonvoice.mozilla.org/en/datasets) |Chiffres indiqués pour la version 16, 981H sur les 1 113 ont été validées|CC-0 |
|**Corpus d'Etude pour le Français Contemporain (CEFC)**|450 H| [Cliquer-ici](https://repository.ortolang.fr/api/content/cefc-orfeo/10/documentation/site-orfeo/index.html) ou [Cliquer-ici](https://www.ortolang.fr/market/corpora/cefc-orfeo) |Possibilité de trier ce que l’on souhaite (tv, radio, téléphone, face à face, etc.)|CC-BY 4.0|
|**ESLO**| 300 H pour ESLO1 & 400 H pour ESLO2 | [Cliquer-ici](https://ct3.ortolang.fr/data/eslo/) ou Cliquer-ici](https://www.ortolang.fr/market/corpora/eslo/v1) | ESLO1 contient des entretiens (formels ou informels de type conversation dans une rue) enregistrés entre 1968 et 1974. Les données ne sont pas forcément de bonnes qualités (grésillements).   ESLO2 reprend le même principe que ESLO1 mais porte sur des entretiens datant de 2008 à 2020. Les données sont de bonnes qualités.|CC-BY 4.0|
|**Conférences Pierre Mendès France**|300 H|[Cliquer-ici](https://www.data.gouv.fr/fr/datasets/transcriptionsxml-audiomp3-mefr-ccpmf-2012-2020-zip/)|Audios au format MP3 et transcriptions au format XML des conférences du centre de conférences Pierre Mendès France du MEFR (2012-2020).|Open Licence version 2.0|
|**VoxpopuliV2**|211 H |[Cliquer-ici](https://github.com/facebookresearch/voxpopuli)|Enregistrements annotés récoltés au Parlement Européen entre 2009 et 2020.|[CC0](https://creativecommons.org/share-your-work/public-domain/cc0/) |
|**Traitement de Corpus Oraux en Français (TCOF)**|146 H|[Cliquer-ici](https://www.ortolang.fr/market/corpora/tcof)| Corpus oraux collectés dans les années 80-90 entre 5 et 45 minutes. |CC-BY 2.0|
|**SynPaFlex**|87 H|[Cliquer-ici](https://www.ortolang.fr/market/corpora/synpaflex-corpus) |Annotation de 87h de corpus de livres-audios. |CC-BY 2.0|
|**MPF**|78 H|[Cliquer-ici](https://www.ortolang.fr/market/corpora/mpf)| Ce corpus vise à documenter des évolutions en cours dans le français, l’émergence d’un vernaculaire urbain contemporain, ainsi que les effets sur le français du contact avec les langues de l’immigration.</p><p>Nécessite un compte (gratuit) sur Ortholang pour télécharger le corpus.|CC-BY 4.0|
|<h1>**CLAPI**</h1>|46 H|[Cliquer-ici](https://www.ortolang.fr/market/corpora/clapi)|La base comprend des corpus oraux composés d'enregistrements audio et vidéo de situations naturelles d'interactions, principalement entre natifs dans des situations sociales très variées : professionnelles, privées, commerciales, institutionnelles, médicales.|CC-BY 4.0|
|**African Accented French**|22 H| [Cliquer-ici](http://www.openslr.org/57/) |Interviews réalisées par l’armée américaine|Apache 2.0|
|**SIWIS** |~ 10 H|[Cliquer-ici](https://datashare.ed.ac.uk/handle/10283/2353) ||CC-BY 4.0|

<br>

Ce sont ainsi environ **2900 heures** d’audio qui sont disponibles pour l’apprentissage supervisé.
<br><br><br>

### <span style="color: #51C353"> **Corpus à accès limités (demandes auprès d’Université / Labo à effectuer)** </span>

<br>

|**Nom du jeu de données**|**Heures**|**Lien pour y accéder**|**Qualité / Source**|**Licence**|
| :- | :- | :- | :- | :- |
|**INA**|1200H sont disponibles et 3000H sont indiqués comme "à venir"|[Cliquer-ici](https://dataset.ina.fr/) |Données de l'INA disponibles dans différents sous jeux de données. Pour pouvoir avoir accès aux données il faut remplir un formulaire (cf. le lien). Il est précisé que "seuls sont autorisés à s’inscrire les laboratoires de recherche, les PME innovantes ainsi que toutes autres personnes morales disposant d’un service ou d’une activité de recherche scientifique."| Licence non précisée mais les CGU sont assez restrictives concernant leur utilisation à des fins non universitaires.|
|**Decoda-RATP**|74H|[Cliquer-ici](http://www.lrec-conf.org/proceedings/lrec2012/pdf/684_Paper.pdf) |Appels téléphoniques à la RATP enregistrés et annotés (transcription, NER, etc.)|?|
|**NCCFr**|35H|[Cliquer-ici](https://mirjamernestus.nl/Ernestus/NCCFr/index.php) |Conversations entre amis annotées par des professionnels|?| 


### <span style="color: #51C353"> **Données payantes** </span>

<br>

|**Nom du jeu de données**|**Heures**|**Lien pour y accéder**|**Informations**|**Licence**|
| :-: | :-: | :-: | :-: | :-: |
|**ESTER**|100 H annotées + 1700 H non annotées|[Cliquer-ici](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0241/) |Corpus d’enregistrements d’émissions radiophoniques.|3 types de licences (avec usage commercial ou non). Cf.  le lien pour plus d’informations. |
|**ESTER 2**|~200 H|[Cliquer-ici](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0338/)|Inclus les 100H annotées d'ESTER1 + 100 nouvelles heures annotées. Corpus de transcriptions manuelles d’émissions radiophoniques et de transcriptions manuelles rapides de radios africaines. |2 types de licences (avec usage commercial ou non). Cf.  le lien pour plus d’informations.|
|**EPAC**|~100 H|[Cliquer-ici](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0305/)|100H de transcriptions manuelles réalisées à partir des 1 700 heures d'enregistrements non transcrits du jeu de données ESTER. |2 types de licences (avec usage commercial ou non). Cf.  le lien pour plus d’informations.|
|**MEDIA**|70 H|[Cliquer-ici](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0272/)|1 258 dialogues transcrits pour 250 locuteurs adultes sur le domaine du tourisme et de la réservation d’hôtel.|2 types de licences (avec usage commercial ou non). Cf.  le lien pour plus d’informations.|
|**ETAPE**|30 H|[Cliquer-ici](https://catalogue.elra.info/en-us/repository/browse/ELRA-E0046/)|Environ 30H de radio et TV françaises incluant de la parole non planifiée et une proportion raisonnable de données multi-locuteurs. Des données transcrites soigneusement en incluant l’annotation des entités nommées.|3 types de licences (avec usage commercial ou non). Cf.  le lien pour plus d’informations.|

<br>

Ce sont ainsi environ **400 heures** d’audio qui sont disponibles pour l’apprentissage supervisé et 1300 heures qui sont disponibles pour l’apprentissage autosupervisé en achetant ces corpus.


## <span style="color: #FFBF00"> ***Automatic Emotion Recognition* (AER)** </span>

### <span style="color: #51C353"> **Données en libre accès** </span>
Il ne semble pas y avoir de données en accès libre concernant cette tâche.

### <span style="color: #51C353"> **Corpus à accès limités (demandes auprès d’Université / Labo à effectuer)** </span>

|**Nom du jeu de données**|**Heures**|**Lien pour y accéder**|**Qualité / Source**|**Licence**|
| :- | :- | :- | :- | :- |
|**Allosat**|~37H|[Cliquer-ici](https://aclanthology.org/2020.lrec-1.197.pdf) |Appels enregistrés à un centre d'appel dont les conversations portent sur des thèmes de type : énergie, agence de voyage, agence immobilière et assurances. Les données ont aussi été retranscrites mais automatiquement à l'aide de Kaldi|?|
|**Cemo**|20H|[Cliquer-ici](https://arxiv.org/pdf/2110.14957.pdf) |Appels aux urgences annotées. Il semble aussi que les données ont aussi été retranscrites d'après la conclusion du papier.|?| 
|**RECOLA**|9,5H|[Cliquer-ici](https://diuf.unifr.ch/main/diva/recola/download.html) | | ? |
|**mGEMEP**|0,9H|[Cliquer-ici](https://www.unige.ch/cisa/gemep) |Données provenant d'acteurs|? |


## <span style="color: #FFBF00"> ***Automatic Speech Translation* (AST)** </span>

### <span style="color: #51C353"> **Données en libre accès** </span>

|**Nom du jeu de données**|**Heures**|**Lien pour y accéder**|**Qualité / Source**|**Licence**|
| :- | :- | :- | :- | :- |
|**MuST-C (en->fr)**|236H|[Cliquer-ici](https://ict.fbk.eu/must-c/) | Provient de TEDs en anglais|CC BY-NC-ND 4.0 |
|**mTEDx (fr->x)**|25H-50H (en fonction de la langue cible)|[Cliquer-ici](http://www.openslr.org/100) |Données issues des conférences TED|CC BY-NC-ND 4.0 |
