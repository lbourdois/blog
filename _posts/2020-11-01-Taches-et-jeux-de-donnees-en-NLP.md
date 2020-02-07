---
title: "TACHES ET JEUX DE DONNÉES TEST FREQUEMMENT UTILISÉS DANS LES PUBLICATIONS DE NLP"
categories:
  - NLP
tags:
  - Jeux de données NLP
  - Taches NLP
excerpt : "NLP"
header :
    overlay_image: "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/NLP_radom_blog.png"
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
Celle des jeux de données francophones se base sur la [publication de l’équipe ayant développé le modèle CamemBERT](https://arxiv.org/pdf/1911.03894.pdf) et sur la [publication de l’équipe ayant développé le modèle FlauBERT](https://arxiv.org/abs/1912.05372).
<br><br><br>



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

Pour le français, il n'existe pas pour le moment une base de données pour le Question-Answering.<br>
Le projet [PIAF](https://piaf.etalab.studio/) (Pour une IA Francophone) porté par [Etalab](https://www.etalab.gouv.fr/) a pour objectif de créer une telle base. Ce projet à vocation open-data est ouvert à tous. Si vous souhaitez contribuer (ne serais-ce qu'annoter 1 ou 2 textes pour voir de vous même la méthodologie appliquée), je vous invite à vous rentre sur le site du projet :)
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

Pour le français, il existe le FTB (French Treebank) crée par les équipes de l'Université Paris-Diderot (laboratoire LLF) contenant plus de 21 550 phrases provenant d’articles du journal *Le Monde* publiés entre 1989 et 1995. L'accès à cette base de données est cependant restreinte. Pour y avoir accès, il faut en effectuer la [demande](http://ftb.linguist.univ-paris-diderot.fr/telecharger.php).
<br><br><br>



# <span style="color: #FF0000"> **Sentiment Analysis (Analyse de sentiments)** </span>
Le modèle doit classer correctement un texte (positif, négatif, etc…). Un exemple avec la base SST et le modèle BERT est disponible ici.
<br>
Les jeux de données anglophones communs : 
-	[SST](https://nlp.stanford.edu/sentiment/index.html) (Stanford Sentiment Treebank)
-	[IMDb](http://ai.stanford.edu/~amaas/data/sentiment/) : Un grand jeu de données de critiques de films avec des étiquettes binaires de classification des sentiments.
<br>

Pour le français, [FLUE](https://github.com/getalp/Flaubert/tree/master/flue) met à disposition le jeu de données CLS-FR. Celui-ci est composé d'avis d'utilisateurs de trois types de produits proposés sur Amazon : livres, musiques et DVD (4000 avis pour chacun des types de produits).
<br><br><br>



# <span style="color: #FF0000"> **Semantic Role Labeling (SRL)** </span>
Modélise la structure prévisible d’un argument d’une phrase. Peut être vu comme une réponse à la question « Qui a fait quoi à qui ».
<br>
-	[CoNLL-2004 & CoNLL-2005](www.lsi.upc.edu/~srlconll/)
<br><br><br>



# <span style="color: #FF0000"> **Sentence similarity** </span>
Aussi connu sous le terme « paraphrase detection ».
<br>
Les jeux de données anglophones communs : 
-	[MRPC (MicRosoft Paraphrase Corpus)](https://www.microsoft.com/en-us/download/details.aspx?id=52398) : contient des paires de phrases extraites de sources d’information sur le Web, avec des annotations indiquant si chaque paire est sémantiquement équivalente.
-	[QQP (Quora Question Pairs) STS Benchmark](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs): Semantic Textual Similarity
<br>

Pour le français, [FLUE](https://github.com/getalp/Flaubert/tree/master/flue) met à disposition la partie en français de [PAWS-X](https://ai.googleblog.com/2019/10/releasing-paws-and-paws-x-two-new.html) de Google, soit environ 49 000 données d'apprentissage et 2 000 de test.
<br><br><br>




# <span style="color: #FF0000"> **Sentence Acceptability** </span>
Annotation des phrases pour qu’elles soient grammaticalement acceptables.
Les jeux de données anglophones communs : 
-	[CoLA](https://nyu-mll.github.io/CoLA/) (Corpus of Linguistic Acceptability) : classification binaire de phrases.
<br><br><br>



# <span style="color: #FF0000"> **Text Chunking** </span>
Pour diviser un en parties syntaxiquement corrélées.
Les jeux de données anglophones communs : 
-	[CoNLL-2000](https://www.clips.uantwerpen.be/conll2000/chunking/)
<br><br><br>



# <span style="color: #FF0000"> **Part-of-Speech (POS) Tagging** </span>
Consiste à attribuer à chaque mot sa catégorie grammaticale correspondante. L’analyse des dépendances consiste à prédire l’arbre syntaxique capturant les relations syntaxiques entre les mots.<br>

Pour le français, quatre banques d’arbres sont disponibles gratuitement dans UD v2.2 UD v2.2 (UD pour Universal Dependencies) : 
-	GSD (données provenant de blogs, d’articles de presse, de critiques et de Wikipedia),
-	[Sequoia](https://deep-sequoia.inria.fr/) de l'Inria (contient plus de 3000 phrases provenant du journal régional L’Est Républicain, Wikipédia et des documents de l’agence européenne de la médecine),
-	Spoken (corpus converti automatiquement depuis la treebank [Projet Rhapsodie](https://www.projet-rhapsodie.fr/)) ,
-	ParTUT (conversion de données multilingues émanant de l’Université de Turin).
<br><br><br>



# <span style="color: #FF0000"> **Machine Translation** </span> 
Voir le site de [Standard NLP](https://nlp.stanford.edu/projects/nmt/) qui contient différents jeux de données :
-	WMT 2015 English-Tchèque (Large)
-	WMT 2014 English-Allemand (Moyen)
-	IWSLT 2015 English-Vietnamien (Petit)
<br><br><br>



# <span style="color: #FF0000"> **Coreference Resolution** </span>
Associe les parties d’un texte qui se réfèrent aux mêmes notions.
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



# <span style="color: #FF0000"> **Benchmark multi-tâches** </span>
Les benchmark anglophones : 
-	[GLUE multi-task benchmark](https://gluebenchmark.com/) 
-	[decaNLP benmark](https://decanlp.com/) 



Les benchmark francophones : 
- [FLUE](https://github.com/getalp/Flaubert/tree/master/flue#4-constituency-parsing)
