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


# <span style="color: #FF0000"> **La librairie NLP** </span>

En 2020, l'entreprise Hugging Face a dévoilé sa librairie *[nlp](https://github.com/huggingface/nlp)*.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/huggingface/nlp/master/docs/source/imgs/nlp_logo_name.png" width="100px">
</figure>
</center>
Grâce à celle-ci vous pouvez charger en une ligne de code l'un des 120 jeux de données (français et autres langues) actuellement disponible sur la librairie. Vous trouverez sur cette librairie, la plupart des jeux de données présentés dans les paragraphes qui suivent.
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
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (Stanford Question Answering Dataset): Un jeu de données sur la compréhension de la lecture, composé de questions posées sur un ensemble d’articles de Wikipédia, où la réponse à chaque question est un intervalle de texte.
- [RACE](http://www.qizhexie.com/data/RACE_leaderboard) (ReAding Comprehension from Examinations): Un jeu de données sur la compréhension de la lecture comprenant plus de 28 000 passages et près de 100 000 questions. Le dataset provient d’examens d’anglais en Chine, qui sont conçus pour les élèves du collège et du lycée.
<br><br>

Pour le français, vous pouvez utiliser la base de données mise à disposition par l'entreprise Illuin Technology : [FQuAD](https://fquad.illuin.tech/). Elle contient plus de 25.000 questions/réponses basées sur des [articles de qualité de Wikipédia](https://fr.wikipedia.org/wiki/Cat%C3%A9gorie:Article_de_qualit%C3%A9). Pour plus d'informations concernant la répartition des types de questions (qui ? quoi ? où ? quand ?, etc...), la répartion des entités (noms communs, personnes, lieu, etc...) et les résultats des premiers benchmarks basés sur CamemBERT et FlauBERT, je vous invite à lire la [publication de l'équipe d'Illuin Technology](https://arxiv.org/abs/2002.06071).<br>

A noter qu'il existe également le projet [PIAF](https://piaf.etalab.studio/) (Pour une IA Francophone) porté par [Etalab](https://www.etalab.gouv.fr/) qui a pour objectif de créer lui aussi une base de Question/Réponse. Ce projet à vocation open-data est ouvert à tous. Si vous souhaitez contribuer (ne serais-ce qu'annoter 1 ou 2 textes pour voir de vous même la méthodologie appliquée), je vous invite à vous rentre sur le site du projet :)<br>
Les données de PIAF sont accessibles librement [ici](https://piaf.etalab.studio/dataset/).
<br><br><br>



# <span style="color: #FF0000"> **Commonsense Reasoning (Raisonnement)** </span>
Parmi un choix de propositions (souvent entre 2 et 4), le modèle doit choisir laquelle est la plus vraisemblable étant donné le texte fournit en entrée. Exemple :

| Texte  | Proposition 1 (la plus vraisemblable) | Proposition 2  | 
| Gina a égaré son téléphone chez ses grands-parents. Il n’était nulle part dans le salon. Elle a réalisé qu’elle était dans la voiture avant. Elle a pris les clés de son père et est sortie en courant.|	Elle a trouvé son téléphone dans la voiture. | Elle ne voulait plus son téléphone.|

<br>
Les jeux de données anglophones communs : 
- [Story Cloze Test](https://www.cs.rochester.edu/nlp/rocstories/) : Sert à évaluer le raisonnement au niveau de la compréhension et la génération des histoires. Le test exige qu’à partir de deux options, un système choisisse la fin correcte à des histoires développées sur plusieurs phrases.
- [SWAG](https://rowanzellers.com/swag/) (Situations With Adversarial Generations): choix multiples ; contient 113 000 exemples de paires de phrases qui évaluent les inférences fondées sur le bon sens.
<br><br><br>



# <span style="color: #FF0000"> **Natural Language Inference (NLI)** </span>
Peut être également rencontré dans la littérature sous le nom de Text Entailment. Il s’agit d’un exercice pour discerner en logique si une phrase peut être déduite d’une autre.
<br>
Les jeux de données anglophones communs : 
- [RTE](https://aclweb.org/aclwiki/Textual_Entailment_Resource_Pool) (Recognizing Textual Entailment) : Un ensemble de jeux de données initiés par des défis de Text Entailment.
- [SNLI](https://nlp.stanford.edu/projects/snli/) (Stanford Natural Language Inference) : Une collection de 570 000 de phrases en anglais écrites par l’homme à la main. Elles sont étiquetées à la main pour une classification équilibrée des labels entailment, contradiction, et neutral.
-	[MNLI](https://www.nyu.edu/projects/bowman/multinli/) (Multi-Genre NLI) : Semblable à SNLI, mais avec une plus grande variété de styles de texte et de sujets, recueillis à partir de transcriptions de discours, de fictions populaires et de rapports gouvernementaux.
-	[QNLI](https://gluebenchmark.com/tasks) (Question NLI) : Convertion de SQuAD en une tâche de classification binaire des paires de la forme (question, phrase).
-	[SciTail](http://data.allenai.org/scitail/) : Un entailment dataset créé à partir d’examens scientifiques à choix multiples et de phrases sur le Web.

<br>
Pour le français, deux options s'offrent à vous :
- vous pouvez utiliser la partie francophone du jeu de données [XNLI](http://www.nyu.edu/projects/bowman/xnli/),
- ou bien, utiliser [FLUE](https://github.com/getalp/Flaubert/tree/master/flue) (French Language Understand-ing Evaluation).
FLUE est l'équivalent francophone de GLUE (cf. Benchmark multi-tâches de l'article). Il a été crée par les auteurs de [FlauBERT](https://arxiv.org/pdf/1912.05372.pdf). En pratique, la partie NLI de FLUE est la partie francophone du jeu de données XNLI évoqué au point précédent. Utiliser l'un ou l'autre revient donc au même. L'intérêt de FLUE est qu'il regroupe plusieurs tâches.
<br><br><br>



# <span style="color: #FF0000"> **Named Entity Recognition (NER) (Reconnaissance d'entités nommées)** </span>
Labélise les séquences de mots d’un texte qui sont des noms de choses (personnes, sociétés, gènes et protéines).
<br>
Les jeux de données anglophones communs : 
-	[CoNLL 2003 NER task](https://www.clips.uantwerpen.be/conll2003/) : consiste en un flux d’informations émanant de Reuters, se concentrant sur quatre types d’entités nommées : les personnes, les lieux, les organisations et les noms d’entités diverses.
-	[OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) : Ce corpus contient des texte en anglais, arabe et chinois, avec quatre types d’entités différents (personne, lieux, organisation, noms d’entités diverses).
-	[Reuters Corpus](https://trec.nist.gov/data/reuters/reuters.html) : Une grande collection d’articles de Reuters.
<br>

Pour le français deux choix s'offrent à vous :
- le FTB (French Treebank) crée par les équipes de l'Université Paris-Diderot (laboratoire LLF) contenant plus de 21 550 phrases provenant d’articles du journal *Le Monde* publiés entre 1989 et 1995. L'accès à cette base de données est cependant restreinte. Pour y avoir accès, il faut en effectuer la [demande](http://ftb.linguist.univ-paris-diderot.fr/telecharger.php).
- la partie en français de la base Wikiner disponible [ici](https://github.com/dice-group/FOX/tree/master/input/Wikiner).
<br><br><br>



# <span style="color: #FF0000"> **Sentiment Analysis (Analyse de sentiments)** </span>
Le modèle doit classer correctement un texte (positif, négatif, etc…). Un exemple avec la base SST et le modèle BERT est disponible ici.
<br>
Les jeux de données anglophones communs : 
-	[SST](https://nlp.stanford.edu/sentiment/index.html) (Stanford Sentiment Treebank) : contient 215 154 phrases labellisées basées sur 11 855 phrases de critiques de films.
-	[IMDb](http://ai.stanford.edu/~amaas/data/sentiment/) : Un grand jeu de données de critiques de films avec des étiquettes binaires de classification des sentiments.
<br>

Pour le français, [FLUE](https://github.com/getalp/Flaubert/tree/master/flue) met à disposition le jeu de données CLS-FR. Celui-ci est composé d'avis d'utilisateurs de trois types de produits proposés sur Amazon : livres, musiques et DVD (4000 avis pour chacun des types de produits). Il permet de faire de la classification multi-classes.
<br><br>
En 2020, Theophile Blard a scrapé le site [AlloCiné](http://www.allocine.fr/) afin de créer un nouveau jeu de données pour l'analyse de sentiments. Ce jeu de données contient plus de 200.000 critiques de films. Néanmoins il se limite à de la classification binaire. La méthodologie appliquée est disponible [ici](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert/tree/master/allocine_dataset) et le jeu de données [ici](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert/blob/master/allocine_dataset/data.tar.bz2)
<br><br><br>


# <span style="color: #FF0000"> **Semantic Role Labeling (SRL)** </span>
Modélise la structure prévisible d’un argument d’une phrase. Peut être vu comme une réponse à la question « Qui a fait quoi à qui ».
<br>
-	[CoNLL-2004 & CoNLL-2005](www.lsi.upc.edu/~srlconll/)
<br><br><br>



# <span style="color: #FF0000"> **Sentence similarity** </span>
Cette tâche consiste à déterminer dans quelle mesure deux textes sont similaires. Cela peut se faire en attribuant une note de 1 à 5. Les tâches connexes sont la paraphrase ou l'identification des doublons.
<br>
Les jeux de données anglophones communs : 
-	[MRPC (MicRosoft Paraphrase Corpus)](https://www.microsoft.com/en-us/download/details.aspx?id=52398) : contient des paires de phrases extraites de sources d’information sur le Web, avec des annotations indiquant si chaque paire est sémantiquement équivalente.
-	[QQP (Quora Question Pairs) STS Benchmark](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs): consiste en plus de 400 000 paires de questions issues du site Quora.
<br>

Pour le français, [FLUE](https://github.com/getalp/Flaubert/tree/master/flue) met à disposition la partie en français de [PAWS-X](https://ai.googleblog.com/2019/10/releasing-paws-and-paws-x-two-new.html) de Google, soit environ 49 000 données d'apprentissage et 2 000 de test.



# <span style="color: #FF0000"> **Sentence Acceptability** </span>
Annotation des phrases pour qu’elles soient grammaticalement acceptables.
Les jeux de données anglophones communs : 
-	[CoLA](https://nyu-mll.github.io/CoLA/) (Corpus of Linguistic Acceptability) : classification binaire de phrases.
<br><br><br>


# <span style="color: #FF0000"> **Part-of-Speech (POS) Tagging** </span>
Consiste à attribuer à chaque mot sa catégorie grammaticale correspondante. L’analyse des dépendances consiste à prédire l’arbre syntaxique capturant les relations syntaxiques entre les mots.
<br>

Les jeux de données anglophones communs : 
- [Universal Dependencies (UD)](https://universaldependencies.org/) : contient plus de 100 banques d'arbres dans plus de 60 langues
- [Ritter](https://www.aclweb.org/anthology/D11-1141/) : issues de l'analyse de "social medias" anglais.
<br>

Pour le français, quatre banques d’arbres sont disponibles gratuitement dans UD v2.2 : 
-	GSD (données provenant de blogs, d’articles de presse, de critiques et de Wikipedia),
-	[Sequoia](https://deep-sequoia.inria.fr/) de l'Inria (contient plus de 3000 phrases provenant du journal régional L’Est Républicain, Wikipédia et des documents de l’agence européenne de la médecine),
-	Spoken (corpus converti automatiquement depuis la treebank [Projet Rhapsodie](https://www.projet-rhapsodie.fr/)) ,
-	ParTUT (conversion de données multilingues émanant de l’Université de Turin).
<br><br><br>



# <span style="color: #FF0000"> **Machine Translation** </span> 
Tache qui consiste à traduire une phrase dans une langue donnée dans une autre langue.

Pour l'anglais, le site de [Standard NLP](https://nlp.stanford.edu/projects/nmt/) contient des paires de phrases en Anglais/Tchèque, en Anglais/Allemand et en Anglais/Vietnamien.
Le site [Manythings.org](http://www.manythings.org/anki/) contient quant à lui plus de 80 paires de langues de la forme Anglais/Seconde_langue. Le couple Anglais/Français contient pour ça part plus de 175 623 paires de phrases.
Il existe d'autres corpus Anglais/Français. On peut par exemple citer : le [WMT14](http://www.statmt.org/wmt14/translation-task.html) ou encore les données de [Pytorch](https://download.pytorch.org/tutorial/data.zip).
Des corpus massifs Français/Seconde_langue avec Seconde_langue différent de l'Anglais sont beaucoup plus difficiles à trouver.
<br><br><br>



# <span style="color: #FF0000"> **Coreference Resolution** </span>
Associe les parties d’un texte qui se réfèrent aux mêmes notions.
Exemple :

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Taches-et-BDD/Id%C3%A9fix.png">
</figure>
</center>

-	[CoNLL-2012](http://conll.cemantix.org/2012/data.html)
<br><br><br>



# <span style="color: #FF0000"> **Long-range Dependency** </span>
Les jeux de données anglophones communs :
-	[LAMBADA](https://wiki.cimec.unitn.it/tiki-index.php?page=CLIC) (LAnguage Modeling Broadened to Account for Discourse Aspects) : Une collection de passages narratifs extraits de BookCorpus (voir section suivante). La tâche est de prédire le dernier mot (ce qui nécessite au moins 50 tokens de contexte pour qu’un humain puisse prédire avec succès).
-	[Children’s Book Test](https://research.fb.com/downloads/babi/) : est construit à partir de livres qui sont librement disponibles dans le [Projet Gutenberg](https://www.gutenberg.org). La tâche consiste à prédire le mot manquant parmi 10 candidats.
<br><br><br>



# <span style="color: #FF0000"> **Jeux de données pour le pré-entraînement non supervisé** </span>
Les jeux de données anglophones communs : 
-	[Books corpus](https://googlebooks.byu.edu) : Le corpus contient « plus de 7 000 livres non publiés de genres variés, dont l’Aventure, la Fantaisie et la Romance ».
-	[1B Word Language Model Benchmark](http://www.statmt.org/lm-benchmark/)
-	[English Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Database_download#English-language_Wikipedia) : ~2500M words
<br><br><br>

Les jeux de données utilisés par CamemBERT et FlauBERT : 
- [OSCAR](https://traces1.inria.fr/oscar/) : données de [Common Crawl](https://commoncrawl.org/) récupérées par l'INRIA (utilisé par CamemBERT)
- voir la [page 8 de la publication de FlauBERT](https://arxiv.org/pdf/1912.05372.pdf) pour connaître les jeux de données qu'ils ont utilisés.
<br><br><br>



# <span style="color: #FF0000"> **Benchmark multi-tâches** </span>
Les benchmark anglophones : 
-	[GLUE multi-task benchmark](https://gluebenchmark.com/) 
-	[decaNLP benchmark](https://decanlp.com/) 


Les benchmark francophones : 
- [FLUE](https://github.com/getalp/Flaubert/tree/master/flue#4-constituency-parsing)
<br><br><br>



# <span style="color: #FF0000"> **Citation** <span>
Si vous venez à utiliser des éléments de cet article, veillez s'il vous plait en créditer les auteurs en utilisant par exemple comme suit :<br>
“*Tâches et jeux de données fréquemment utilisés dans les publications de NLP* par Loïck BOURDOIS (https://lbourdois.github.io/blog/nlp/Taches-et-jeux-de-donnees-en-NLP/), d’après Lilian WENG, *Generalized Language Models : Common Tasks & Datasets* (https://www.topbots.com/generalized-language-models-tasks-datasets/)”<br>
Merci :)
