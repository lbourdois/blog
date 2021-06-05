---
title: "TACHES ET JEUX DE DONNÉES FREQUEMMENT UTILISÉS DANS LES PUBLICATIONS DE NLP"
categories:
  - NLP
tags:
  - Jeux de données NLP
  - Taches NLP
  - FQuAD NLP
  - CamemBERT NLP
  - FlauBERT NLP
  - FLUE NLP
excerpt : "NLP - Taches et jeux de données fréquemments utilisés dans les publications de NLP"
header :
    overlay_image: "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/NLP_radom_blog.png"
    teaser: "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Taches-et-BDD/Questions-R%C3%A9ponses.png"
toc: true
toc_sticky: true
author_profile: false
sidebar:
    nav: sidebar-sample

---


# <span style="color: #FF0000"> **Avant-propos** </span>
Les publications scientifiques étant exclusivement en anglais, je fais en sorte de laisser les mots clés en anglais et de donner des explications en français (je ne sais même pas si certains mots ont une traduction en français). Le but étant que vous puissiez après lire de vous même n’importe quelle publication ou article de vulgarisation en anglais sur le sujet et les comprendre.<br>
Le domaine du NLP évoluant actuellement très rapidement il m’est impossible de présenter tous les nouveaux modèles par manque de temps. D’où là encore la nécessité que vous alliez consulter certaines sources anglophones par vous même.<br>
Un élément qui me semble important pour pouvoir comprendre n’importe quelle publication est de connaître les tâches qui servent à comparer les performances des différents modèles ainsi que les jeux de données test en lien avec ces tâches.<br>
La présentation des jeux de données anglophones se base sur l’[article de Lilian Weng](https://www.topbots.com/generalized-language-models-tasks-datasets/) (travaillant chez OpenAI). Merci à elle de m'avoir autorisé à effectuer cette traduction.<br>
Celle des jeux de données francophones se base sur la [publication de l’équipe ayant développé le modèle CamemBERT](https://arxiv.org/pdf/1911.03894.pdf), sur la [publication de l’équipe ayant développé le modèle FlauBERT](https://arxiv.org/abs/1912.05372), sur la [publication de l’équipe ayant développé FQuAD](https://arxiv.org/abs/2002.06071) et mes propres recherches.
<br><br><br>


# <span style="color: #FF0000"> **La librairie Dataset** </span>

En 2020, l'entreprise Hugging Face a dévoilé sa librairie [*Dataset*](https://github.com/huggingface/datasets) (anciennement *nlp*).
<center>
<figure class="image">
  <img src="https://huggingface.co/landing/assets/transformers-docs/huggingface_logo.svg" width="100px">
</figure>
</center>
Grâce à celle-ci vous pouvez charger en une ligne de code l'un des 900 jeux de données (français et autres langues) actuellement disponible sur la librairie (chiffre datant de mai 2021). Vous trouverez sur cette librairie, la plupart des jeux de données présentés dans les paragraphes qui suivent.
Pour connaître les jeux de données disponibles, vous pouvez consulter leur [application Streamlit](https://huggingface.co/nlp/viewer/), et pour un tutoriel d'utilisation, vous pouvez consulter le [Collab suivant](https://colab.research.google.com/github/huggingface/nlp/blob/master/notebooks/Overview.ipynb).
<br><br>


# <span style="color: #FF0000"> **Question-Answering (Questions/Réponses)** </span>
Le modèle doit répondre à une série de questions en lien avec un jeu de données test. Certaines questions peuvent ne pas avoir de réponses. Exemple : 
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Taches-et-BDD/Questions-R%C3%A9ponses.png">
  <figcaption>
 Exemple de questions provenant de la base de données SquAD. Cet exemple est en lien avec les Normands mais de nombreux autres thèmes sont disponibles. De plus, seul le premier paragraphe est montré ici. Le texte intégral en comprend 38 autres.</figcaption>
</figure>
</center>
 
 
Les jeux de données anglophones communs : 
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (Stanford Question Answering Dataset) de [Rajpurkar et al.](https://arxiv.org/abs/1606.05250) pour la version 1 et [Rajpurkar et al.](https://arxiv.org/abs/1806.03822) pour la version 2. Il s'agit d'un jeu de données sur la compréhension de la lecture, composé de questions posées sur un ensemble d’articles de Wikipédia, où la réponse à chaque question est un intervalle de texte.
- [RACE](http://www.qizhexie.com/data/RACE_leaderboard) (ReAding Comprehension from Examinations) de [Lai, Xie et al.](https://arxiv.org/pdf/1704.04683.pdf). Il s'agit d'un jeu de données sur la compréhension de la lecture comprenant plus de 28 000 passages et près de 100 000 questions. Le dataset provient d’examens d’anglais en Chine, qui sont conçus pour les élèves du collège et du lycée.
<br><br>

Pour le français :
- [FQuAD](https://fquad.illuin.tech/) de l’entreprise Illuin Technology (plus particulièrement [Hoffschmidt et al.](https://arxiv.org/pdf/2002.06071.pdf)) qui se base sur la méthodologie de SQUAD 1.0.  Elle contient plus de 25.000 questions/réponses basées sur des [articles de qualité de Wikipédia](https://fr.wikipedia.org/wiki/Cat%C3%A9gorie:Article_de_qualit%C3%A9). Pour plus d'informations concernant la répartition des types de questions (qui ? quoi ? où ? quand ?, etc...), la répartion des entités (noms communs, personnes, lieu, etc...) et les résultats des premiers benchmarks basés sur CamemBERT et FlauBERT, je vous invite à lire la [publication de l'équipe d'Illuin Technology](https://arxiv.org/abs/2002.06071).<br>
- le projet [PIAF](https://piaf.etalab.studio/) (Pour une IA Francophone) porté par [Etalab](https://www.etalab.gouv.fr/) et plus particulièrement [Keraron et al.](https://www.aclweb.org/anthology/2020.lrec-1.673/). Les données de PIAF sont accessibles librement [ici](https://www.data.gouv.fr/fr/datasets/piaf-le-dataset-francophone-de-questions-reponses/).

Les jeux de données en français se basent sur la méthodologie de SQUAD 1.0. Ainsi la base de données a été conçue de telle sorte qu’à chaque question posée, la réponse est trouvable dans le texte. SQUAD 2.0 introduit des questions dont la réponse ne se trouve pas dans le texte. Cela permet au modèle d’apprendre également la possibilité : « la réponse n’est pas dans le texte » ou bien « je ne sais pas » au lieu de vouloir coute que coute répondre quelque chose.  

- [Mkqa](https://github.com/apple/ml-mkqa/) est un jeu de données multilingues avec une partie en français. Il est proposé par [Longpre et al](https://arxiv.org/abs/2007.15207).
<br><br>


Jeu de données de questions/anwsering où la réponse peut être à choix multiples :
- Base à cheval entre les questions/réponses et le raisonnement : [disaster_response_messages](https://huggingface.co/datasets/disaster_response_messages). Dans cette base multilingue ayant une partie en français, un texte est fourni avec des questions associées. Les réponses à ces questions étant exclusivement « oui » ou « non ».
- La base [Exams](https://github.com/mhardalov/exams-qa) de [Hardalov et al.](https://arxiv.org/abs/2011.03080) est une base multilingue ayant une partie en français, un texte est fourni avec des questions associées. Cinq réponses sont proposées à ces questions. Les textes étant des QCM pour des lycéens.
<br><br><br>



# <span style="color: #FF0000"> **Commonsense Reasoning (Raisonnement)** </span>
Parmi un choix de propositions (souvent entre 2 et 4), le modèle doit choisir laquelle est la plus vraisemblable étant donné le texte fournit en entrée. Exemple :

| Texte  | Proposition 1 (la plus vraisemblable) | Proposition 2  | 
| Gina a égaré son téléphone chez ses grands-parents. Il n’était nulle part dans le salon. Elle a réalisé qu’elle était dans la voiture avant. Elle a pris les clés de son père et est sortie en courant.|	Elle a trouvé son téléphone dans la voiture. | Elle ne voulait plus son téléphone.|

<br>
Les jeux de données anglophones communs : 
- [Story Cloze Test](https://www.cs.rochester.edu/nlp/rocstories/) de [Mostafazadeh et al.](https://arxiv.org/abs/1604.01696) : sert à évaluer le raisonnement au niveau de la compréhension et la génération des histoires. Le test exige qu’à partir de deux options, un système choisisse la fin correcte à des histoires développées sur plusieurs phrases.
- [SWAG](https://rowanzellers.com/swag/) (Situations With Adversarial Generations) de [Zellers et al.](https://arxiv.org/abs/1808.05326) : choix multiples ; contient 113 000 exemples de paires de phrases qui évaluent les inférences fondées sur le bon sens.
<br><br>

Pour le français :
- Possibilité d’utiliser la partie en français de la base multilingues [Conceptnet5](https://github.com/commonsense/conceptnet5/wiki) de [Speers et al.](https://arxiv.org/abs/1612.03975) (fait peu commun, la base en français est plus grande que l’anglaise).
<br><br><br>



# <span style="color: #FF0000"> **Natural Language Inference (NLI)** </span>
Peut être également rencontré dans la littérature sous le nom de Text Entailment. Il s’agit d’un exercice pour discerner en logique si une phrase peut être déduite d’une autre. 
<br>
Les jeux de données anglophones communs : 
- [RTE](https://aclweb.org/aclwiki/Textual_Entailment_Resource_Pool) (Recognizing Textual Entailment)  : un ensemble de jeux de données initiés par des défis de Text Entailment.
- [SNLI](https://nlp.stanford.edu/projects/snli/) (Stanford Natural Language Inference) de [Young et al.](https://transacl.org/ojs/index.php/tacl/article/view/229) : une collection de 570 000 de phrases en anglais écrites par l’homme à la main. Elles sont étiquetées à la main pour une classification équilibrée des labels entailment, contradiction, et neutral.
-	[MNLI](https://cims.nyu.edu/~sbowman/multinli/) (Multi-Genre NLI) de [Williams et al.](http://aclweb.org/anthology/N18-1101) : semblable à SNLI, mais avec une plus grande variété de styles de texte et de sujets, recueillis à partir de transcriptions de discours, de fictions populaires et de rapports gouvernementaux.
-	[QNLI](https://gluebenchmark.com/tasks) (Question NLI) : convertion de SQuAD en une tâche de classification binaire des paires de la forme (question, phrase).
-	[SciTail](http://data.allenai.org/scitail/) de [Khot et al](https://www.semanticscholar.org/paper/SciTaiL%3A-A-Textual-Entailment-Dataset-from-Science-Khot-Sabharwal/cf8c493079702ec420ab4fc9c0fabb56b2a16c84) : un jeu de données créé à partir d’examens scientifiques à choix multiples et de phrases sur le Web.
<br><br>

Pour le français :
- vous pouvez utiliser la partie francophone du jeu de données [XNLI](http://www.nyu.edu/projects/bowman/xnli/) (même auteurs que SNLI),
- ou bien, utiliser [FLUE](https://github.com/getalp/Flaubert/tree/master/flue) (French Language Understand-ing Evaluation).
FLUE est l'équivalent francophone de GLUE (cf. Benchmark multi-tâches de l'article). Il a été crée par Le et al. les auteurs de [FlauBERT](https://arxiv.org/pdf/1912.05372.pdf). En pratique, la partie NLI de FLUE est la partie francophone du jeu de données XNLI évoqué au point précédent. Utiliser l'un ou l'autre revient donc au même. L'intérêt de FLUE est qu'il regroupe plusieurs tâches.
<br><br><br>


# <span style="color: #FF0000"> **Named Entity Recognition (NER) (Reconnaissance d'entités nommées)** </span>
Etiquette les séquences de mots d’un texte qui sont des noms de choses (personnes, sociétés, gènes, protéines, etc.).
<br>
Les jeux de données anglophones communs : 
-	[CoNLL 2003 NER task](https://www.clips.uantwerpen.be/conll2003/) : consiste en un flux d’informations émanant de Reuters, se concentrant sur quatre types d’entités nommées : les personnes, les lieux, les organisations et les noms d’entités diverses.
-	[OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) de Weischedel et al. : ce corpus contient des texte en anglais, arabe et chinois, avec quatre types d’entités différents (personne, lieux, organisation, noms d’entités diverses).
-	[Reuters Corpus](https://trec.nist.gov/data/reuters/reuters.html) de [Lewis et al.](https://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf) : une grande collection d’articles de Reuters.
<br><br>

Pour le français :
- le FTB (French Treebank) crée par les équipes de l'Université Paris-Diderot (laboratoire LLF et notamment [Abeillé et al](http://www.lrec-conf.org/proceedings/lrec2000/pdf/230.pdf)) contenant plus de 21 550 phrases provenant d’articles du journal *Le Monde* publiés entre 1989 et 1995. L'accès à cette base de données est cependant restreinte. Pour y avoir accès, il faut en effectuer la [demande](http://ftb.linguist.univ-paris-diderot.fr/telecharger.php).
- la partie en français de la base Wikiner disponible [ici](https://github.com/dice-group/FOX/tree/master/input/Wikiner) de [Nothman et al.](http://dx.doi.org/10.1016/j.artint.2012.03.006).
- [Wikiann](https://github.com/afshinrahimi/mmner) de [Rahimi et al.](https://arxiv.org/abs/1902.00193) basé sur [Pan, Xiaoman, et al.](https://www.aclweb.org/anthology/P19-1015/) qui permet de faire du transfert de NER entre plusieurs langues.
<br><br><br>


# <span style="color: #FF0000"> **Sentiment Analysis (Analyse de sentiments)** </span>
Le modèle doit classer correctement un texte (positif, négatif, etc…). 
<br>
Les jeux de données anglophones communs : 
-	[SST](https://nlp.stanford.edu/sentiment/index.html) (Stanford Sentiment Treebank) de [Socher et al.](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) : contient 215 154 phrases labellisées basées sur 11 855 phrases de critiques de films.
-	[IMDb](http://ai.stanford.edu/~amaas/data/sentiment/) de [Maas et al.](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf) : un grand jeu de données de critiques de films avec des étiquettes binaires de classification des sentiments.
<br>

Pour le français :
- [FLUE](https://github.com/getalp/Flaubert/tree/master/flue) met à disposition le jeu de données CLS-FR. Celui-ci est composé d'avis d'utilisateurs de trois types de produits proposés sur Amazon : livres, musiques et DVD (4000 avis pour chacun des types de produits). Il permet de faire de la classification multi-classes.
- dans la même logique que le point précédent, le corpus [The Multilingual Amazon Reviews Corpus](https://registry.opendata.aws/amazon-reviews-ml/) de [Keung et al.](https://arxiv.org/abs/2010.02573) propose des avis de client issue d’Amazon. Cependant ce corpus multilingue propose beaucoup plus d’avis pour le français (environ 200 000) et sur des données plus récentes que celles de FLUE.
- Google propose une base multilingue binaire intégrant du français (il faut dire si une phrase est négative ou positive) : [Senti_lex](https://sites.google.com/site/datascienceslab/projects/multilingualsentiment) 
- [WiLI_2018](https://zenodo.org/record/841984) de [Martin Thoma](https://arxiv.org/pdf/1801.07779.pdf) propose 1000 phrases à classer pour les 235 langues proposées.
- [Aspect-Based Sentiment Analysis in French](http://www.lrec-conf.org/proceedings/lrec2016/summaries/61.html) de Apidianaki et al. contient 457 avis de restaurants (2365 phrases) et 162 de musées (655 phrases).
- Un jeu de données binaire scrapé sur [AlloCiné](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert/blob/master/allocine_dataset/data.tar.bz2) par [Blard](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert) propose  plus de 200.000 critiques de films à classer en « positive » ou « négative ».
<br><br><br>


# <span style="color: #FF0000"> **Semantic Role Labeling (SRL)** </span>
Modélise la structure prévisible d’un argument d’une phrase. Peut être vu comme une réponse à la question « Qui a fait quoi à qui ».
<br
En anglais :
-	[CoNLL-2004 & CoNLL-2005](https://www.cs.upc.edu/~srlconll/)

En français :  
Pas d’équivalent en français à l’heure actuelle à ma connaissance.
<br><br><br>



# <span style="color: #FF0000"> **Sentence similarity** </span>
Cette tâche consiste à déterminer dans quelle mesure deux textes sont similaires. Cela peut se faire en attribuant une note de 1 à 5. Les tâches connexes sont la paraphrase ou l’identification des doublons. 
<br>
En anglais :
-	[MRPC (MicRosoft Paraphrase Corpus)](https://www.microsoft.com/en-us/download/details.aspx?id=52398) : contient des paires de phrases extraites de sources d’information sur le Web, avec des annotations indiquant si chaque paire est sémantiquement équivalente.
-	[QQP (Quora Question Pairs) STS Benchmark](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) de [Iyer et al.](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) : consiste en plus de 400 000 paires de questions issues du site Quora.
<br>

Pour le français :
- [FLUE](https://github.com/getalp/Flaubert/tree/master/flue) met à disposition la partie en français de [PAWS-X](https://ai.googleblog.com/2019/10/releasing-paws-and-paws-x-two-new.html) de [Zhang et al.](https://arxiv.org/pdf/1904.01130.pdf), soit environ 49 000 données d'apprentissage et 2 000 de test.
- [TaPaCo](https://zenodo.org/record/3707949#.X9Dh0cYza3I) de [Scherrer](https://www.aclweb.org/anthology/2020.lrec-1.848.pdf) est une base de paraphrases portant sur 73 langues incluant une partie en français.
- La base de données [REFreSD](https://github.com/Elbria/xling-SemDiv/tree/master/REFreSD) de [Briakou et Carpuat](https://www.aclweb.org/anthology/2020.emnlp-main.121/) est de la sentence similarity mais entre deux langues : l’anglais et le français. Une phrase est donnée en anglais puis une autre en français. Le modèle doit dire si la phrase en français est liée à l’anglais ou pas.
<br><br><br>


# <span style="color: #FF0000"> **Sentence Acceptability** </span>
Annotation des phrases pour qu’elles soient grammaticalement acceptables. Les jeux de données anglophones communs :<br>

En anglais :
-	[CoLA](https://nyu-mll.github.io/CoLA/) (Corpus of Linguistic Acceptability) de [Warstadt et al.](https://arxiv.org/abs/1805.12471) : classification binaire de phrases.

En français :
Pas d’équivalent en français à l’heure actuelle à ma connaissance
<br><br><br>


# <span style="color: #FF0000"> **Part-of-Speech (POS) Tagging** </span>
Consiste à attribuer à chaque mot sa catégorie grammaticale correspondante. L’analyse des dépendances consiste à prédire l’arbre syntaxique capturant les relations syntaxiques entre les mots.
<br>
En anglais :
- [Universal Dependencies (UD)](https://universaldependencies.org/) ([les contributeurs](https://universaldependencies.org/contributors.html)) : contient plus de 100 banques d'arbres dans plus de 60 langues
- [Ritter et al.](https://www.aclweb.org/anthology/D11-1141/) : issues de l'analyse de "social medias" anglais.
<br>

En français :<br>
Huit banques d’arbres sont disponibles gratuitement dans [UD v2.2](https://universaldependencies.org/) : 
-	[GSD](https://universaldependencies.org/treebanks/fr_gsd/index.html) (données provenant de blogs, d’articles de presse, de critiques et de Wikipedia) de De Marneffe et al.,
-	[Sequoia](https://universaldependencies.org/treebanks/fr_sequoia/index.html) de [Candito et al.](https://hal.inria.fr/hal-00969191v2/document) (contient plus de 3000 phrases provenant du journal régional L’Est Républicain, Wikipédia et des documents de l’agence européenne de la médecine),
-	[Spoken](https://universaldependencies.org/treebanks/fr_spoken/index.html) de [Lacheret et al.](https://hal.sorbonne-universite.fr/hal-00968959/document),
-	[ParTUT](https://universaldependencies.org/treebanks/fr_partut/) (conversion de données multilingues émanant de l’Université de Turin) de [Bosco et Sanguinetti](http://www.di.unito.it/~tutreeb/publications.html).
<br><br><br>



# <span style="color: #FF0000"> **La simplification de textes** </span>
Les modèles de simplification de texte permettent de conserver le sens de la phrase mais avec une syntaxe différente et souvent plus courte. Deux approches sont envisageables. La première où le texte original est paraphrasé. La deuxième consiste à faire un résumé du texte original.

## <span style="color: #FFBF00"> **Simplification par paraphrase** </span>
En anglais :
- [Asset](https://github.com/facebookresearch/asset) de [Alva-Manchego, Martin et al.](https://www.aclweb.org/anthology/2020.acl-main.424/) permet de fine-tuner les modèles de simplification de texte.

En français :
- Le jeu de données [ALECTOR](https://alectorsite.wordpress.com/corpus/) de [Gala et al.](https://hal.archives-ouvertes.fr/hal-02503986/document) contient des extraits de sites proposant du matériel pédagogique pour les niveaux CE1, CE2 et CM1 de l’école primaire. Chaque texte original a été adapté (simplifié) au niveau du lexique (vocabulaire), de la morpho-syntaxe (catégories grammaticales, structures de phrase) et du discours (co-référence).	
- [MUSS](https://github.com/facebookresearch/muss) de [Martin et al.](https://arxiv.org/abs/2005.00352) porte sur un outil permettant une simplification de phrases multilingues.
<br><br>


## <span style="color: #FFBF00"> **Simplification par résumé** </span>

Dans cette approche, nous avons le texte original en entrée et un résumé de ce texte en sortie.<br>

En anglais :
- [XSum](https://github.com/EdinburghNLP/XSum) de [Narayan et al.](https://arxiv.org/abs/1808.08745)
- [CNN/DM](https://github.com/deepmind/rc-data/) de [Hermann  et al.](https://papers.nips.cc/paper/2015/file/afdec7005cc9f14302cd0474fd0f3c96-Paper.pdf)

En français :
- [Orange_sum](https://github.com/Tixierae/OrangeSum) de [Eddine et al.](https://arxiv.org/abs/2010.12321) introduite avec leur modèle BARThez, consistant en des résumés d’articles du site orange news.
- la partie en français de la base de données multilingues [MLSUM](https://github.com/recitalAI/MLSUM) de [Scialom et al.](https://arxiv.org/pdf/2004.14900.pdf).
- [WikiLingua](https://github.com/esdurmus/Wikilingua) de [Ladhak et al.](https://arxiv.org/abs/2010.03093) propose une base multilingue contenant une partie en français consistant à faire des résumés d’articles de WikiHow.
- Cette [base](https://webhose.io/free-datasets/french-news-articles/) des résumés d'articles de journaux français 
<br><br><br>


# <span style="color: #FF0000"> **Machine Translation (Traduction automatique)** </span> 
Tâche qui consiste à traduire une phrase dans une langue donnée dans une autre langue.<br>

Le site [Manythings](http://www.manythings.org/anki/) contient plus de 80 paires de langues de la forme anglais/seconde_langue. Le couple Anglais/Français contient pour plus de 175 623 paires de phrases.  
Il existe d’autres corpus Anglais/Français. On peut par exemple citer :  le [WMT14](http://www.statmt.org/wmt14/translation-task.html), le WMT20, les données de [Pytorch](https://download.pytorch.org/tutorial/data.zip) ou encore celles de [ParaPat](https://figshare.com/articles/dataset/ParaPat_The_Multi-Million_Sentences_Parallel_Corpus_of_Patents_Abstracts/12627632) de [Soares et al.](https://www.aclweb.org/anthology/2020.lrec-1.465/)
<br><br>
Pour des corpus français/Seconde_langue avec Seconde_langue différente de l’Anglais, il y a énormément (+ de 50) de jeux de données européens disponibles à l’adresse suivante : [http://opus.nlpl.eu/index.php](http://opus.nlpl.eu/index.php). On accède aux différents jeux de données en cliquant sur les liens tout en haut de la page.
Quelques exemples de textes trouvables :  
- Ecb : textes traduits en plusieurs langues de rapports de la banque centrale européenne
- Emea : textes traduits en plusieurs langues de rapports de l’agence européenne du médicament
- Euronews : textes traduits en plusieurs langues de la chaine d’information Euronews
Et pleins d’autres choses comme des traductions de livres, de la constitution, de la déclaration des droits de l’homme, des sous-titres de TED, etc…
<br><br>
Microsoft propose également un jeu de données de textes techniques pouvant être utilisée pour développer des versions localisées d'applications qui s'intègrent aux produits Microsoft. Elle peut également être utilisée pour intégrer la terminologie Microsoft dans d'autres collections terminologiques ou servir de glossaire informatique de base pour le développement linguistique dans les quelque [100 langues disponibles](https://www.microsoft.com/en-us/language/terminology).
<br><br><br>



# <span style="color: #FF0000"> **Coreference Resolution** </span>
Associe les parties d’un texte qui se réfèrent aux mêmes notions.
Exemple :

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Taches-et-BDD/Id%C3%A9fix.png">
</figure>
</center>

En anglais :
-	[CoNLL-2012](http://conll.cemantix.org/2012/data.html)

En français :
- La partie francophone par [Candito el al.](https://gitlab.com/parseme/sharedtask-data/-/tree/master/1.2/FR) de la base multilingues [PARSEME](https://gitlab.com/parseme/sharedtask-data/-/tree/master/1.2).
<br><br><br>



# <span style="color: #FF0000"> **Long-range Dependency** </span>
Les jeux de données anglophones communs :
-	[LAMBADA](https://wiki.cimec.unitn.it/tiki-index.php?page=CLIC) (LAnguage Modeling Broadened to Account for Discourse Aspects) de [Paperno et al.](https://www.aclweb.org/anthology/P16-1144/). C'est une collection de passages narratifs extraits de BookCorpus (voir section suivante). La tâche est de prédire le dernier mot (ce qui nécessite au moins 50 tokens de contexte pour qu’un humain puisse prédire avec succès).
-	[Children’s Book Test](https://research.fb.com/downloads/babi/) de [Wetson et al.](https://arxiv.org/abs/1502.05698) est construit à partir de livres qui sont librement disponibles dans le [Projet Gutenberg](https://www.gutenberg.org). La tâche consiste à prédire le mot manquant parmi 10 candidats.<br>

En français :
Pas d’équivalent en français à l’heure actuelle à ma connaissance.
<br><br><br>



# <span style="color: #FF0000"> **Jeux de données pour l'entraînement** </span>
Les jeux de données anglophones communs : 
-	[Books corpus](https://googlebooks.byu.edu) : Le corpus contient « plus de 7 000 livres non publiés de genres variés, dont l’Aventure, la Fantaisie et la Romance ».
-	[1B Word Language Model Benchmark](http://www.statmt.org/lm-benchmark/)
-	[English Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Database_download#English-language_Wikipedia) : ~2500M words
<br><br><br>

Les jeux de données utilisés par CamemBERT et FlauBERT : 
- [OSCAR](https://traces1.inria.fr/oscar/) d'[Ortiz Suárez et al.](https://www.aclweb.org/anthology/2020.acl-main.156/)
- [Common Crawl](https://commoncrawl.org/)
- la [page 8 de la publication de FlauBERT](https://arxiv.org/pdf/1912.05372.pdf) pour connaître les jeux de données qu'ils ont utilisés.
<br><br><br>



# <span style="color: #FF0000"> **Benchmark multi-tâches** </span>
Les benchmark anglophones : 
-	[GLUE multi-task benchmark](https://gluebenchmark.com/)  de [Wang et al.](https://openreview.net/pdf?id=rJ4km2R5t7)
-	[decaNLP benchmark](https://decanlp.com/) de [Mcann et al.](https://arxiv.org/abs/1806.08730)
<br><br>
Les benchmark francophones : 
- [FLUE](https://github.com/getalp/Flaubert/tree/master/flue#4-constituency-parsing)
- [XGLUE](https://microsoft.github.io/XGLUE/) version multilingue de GLUE ayant une partie en français de [Liang et al.](https://arxiv.org/abs/2004.01401)
<br><br><br>



# <span style="color: #FF0000"> **Références** <span>
 - [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) de Rajpurkar et al. (2016)
 - [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1606.05250) de Rajpurkar et al. (2018)
- [RACE: Large-scale ReAding Comprehension Dataset From Examinations](https://arxiv.org/abs/1704.04683)
  de Lai, Xie et al. (2017)
- [FQuAD: French Question Answering Dataset](https://arxiv.org/pdf/2002.06071.pdf) de Hoffschmidt et al. (2020)   
- [Project PIAF: Building a Native French Question-Answering Dataset](https://www.aclweb.org/anthology/2020.lrec-1.673/) de Keraron et al. (2020)
- [MKQA: Multilingual Knowledge Questions & Answers](https://arxiv.org/abs/2007.15207) de Longpre et al. (2020)
- [EXAMS: A Multi-subject High School Examinations Dataset for Cross-lingual and Multilingual Question Answering](https://arxiv.org/abs/2011.03080) de Hardalov et al. (2020) 
- [A Corpus and Evaluation Framework for Deeper Understanding of Commonsense Stories](https://arxiv.org/abs/1604.01696) de Mostafazadeh et al. (2016) 
- [SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference](https://arxiv.org/abs/1808.05326) de  Zellers et al. (2018) 
- [ConceptNet 5.5: An Open Multilingual Graph of General Knowledge](https://arxiv.org/abs/1612.03975) de  Speers et al. (2016) 

- [Khot et al](https://www.semanticscholar.org/paper/SciTaiL%3A-A-Textual-Entailment-Dataset-from-Science-Khot-Sabharwal/cf8c493079702ec420ab4fc9c0fabb56b2a16c84)
- [Williams et al.](http://aclweb.org/anthology/N18-1101)
- [Young et al.](https://transacl.org/ojs/index.php/tacl/article/view/229)
- [Le et al.](https://arxiv.org/pdf/1912.05372.pdf)
  
- [Lewis et al.](https://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf)
- [Weischedel et al](https://catalog.ldc.upenn.edu/LDC2013T19)
- [Abeillé et al](http://www.lrec-conf.org/proceedings/lrec2000/pdf/230.pdf)
- [Nothman et al.](http://dx.doi.org/10.1016/j.artint.2012.03.006).
- [Rahimi et al.](https://arxiv.org/abs/1902.00193)
- [Pan, Xiaoman, et al.](https://www.aclweb.org/anthology/P19-1015/) 
  
- [Socher et al.](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)
- [Maas et al.](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)
- [Keung et al.](https://arxiv.org/abs/2010.02573)
- [Apidianaki et al.](http://www.lrec-conf.org/proceedings/lrec2016/summaries/61.html)
- [Martin Thoma](https://arxiv.org/pdf/1801.07779.pdf) 
- [Blard](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert)

- [Iyer et al.](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)
- [Zhang et al.](https://arxiv.org/pdf/1904.01130.pdf)
- [Scherrer](https://www.aclweb.org/anthology/2020.lrec-1.848.pdf)
- [Briakou et Carpuat](https://www.aclweb.org/anthology/2020.emnlp-main.121/)

- [Warstadt et al.](https://arxiv.org/abs/1805.12471)

- [Lacheret et al.](https://hal.sorbonne-universite.fr/hal-00968959/document)
- [Ritter et al.](https://www.aclweb.org/anthology/D11-1141/)
- [Candito et al.](https://github.com/UniversalDependencies/UD_French-Sequoia/blob/master/README.md)
- [Bosco et Sanguinetti](http://www.di.unito.it/~tutreeb/publications.html)
- [Hermann  et al.](https://papers.nips.cc/paper/2015/file/afdec7005cc9f14302cd0474fd0f3c96-Paper.pdf)
- [Alva-Manchego, Martin et al.](https://www.aclweb.org/anthology/2020.acl-main.424/) 
- [Gala et al.](https://hal.archives-ouvertes.fr/hal-02503986/document)
- [Martin et al.](https://arxiv.org/abs/2005.00352)
- [Narayan et al.](https://arxiv.org/abs/1808.08745)
- [Eddine et al.](https://arxiv.org/abs/2010.12321)
- [Scialom et al.](https://arxiv.org/pdf/2004.14900.pdf)
- [Ladhak et al.](https://arxiv.org/abs/2010.03093)
  
- [Paperno et al.](https://www.aclweb.org/anthology/P16-1144/)
- [Wetson et al.](https://arxiv.org/abs/1502.05698)
  
- [Ortiz Suárez et al.](https://www.aclweb.org/anthology/2020.acl-main.156/) 
  
- [Wang et al.](https://openreview.net/pdf?id=rJ4km2R5t7)
- [Mcann et al.](https://arxiv.org/abs/1806.08730)
- [Liang et al.](https://arxiv.org/abs/2004.01401)
  
<br><br><br>
  
  
# <span style="color: #FF0000"> **Citation** <span>
Si vous venez à utiliser des éléments de cet article, veillez s'il vous plait en créditer les auteurs en utilisant par exemple comme suit :<br>
“*Tâches et jeux de données fréquemment utilisés dans les publications de NLP* par Loïck BOURDOIS (https://lbourdois.github.io/blog/nlp/Taches-et-jeux-de-donnees-en-NLP/) (2020)”<br>
Merci :)
