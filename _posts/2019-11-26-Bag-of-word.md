---
title: "ILLUSTRATION DU BAG-OF-WORDS"
categories:
  - NLP
tags:
  - bag of word expliqué en français
  - bag of word NLP en français
excerpt : "NLP - Illustration de la technique du bag-of-words"
header :
    overlay_image: "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/NLP_radom_blog.png"
author_profile: false
sidebar:
    nav: sidebar-sample
classes: wide
---

# <span style="color: #FF0000"> **Avant-propos** </span>
 Cet article est le premier de la série consacrée au traitement du langage naturel (NLP) sur le blog.<br>
 Il est court car cette technique est maintenant ancienne et la littérature à son sujet est assez abondante en français.
 Je ne vois donc pas l'intérêt d'effectuer un doublon.<br>
 Je vais me limiter à énoncer très succinctement le principe de la méthode et vous renvoyer vers d'autres sites internet pour plus de détails en ce qui concerne le codage par exemple.
<br><br><br>


# <span style="color: #FF0000"> **Rapide descriptif de la méthode** </span>
Le *bag-of-words* (sac de mots en français) est la technique qui était majoritairement utilisée avant l’apparition des modèles basés sur les réseaux de neurones tel que les RNNs (eux-mêmes moins utilisés aujourd’hui au profit des *transformers*).
Elle consiste principalement à nettoyer les données textuelles à notre disposition d’une certaine manière afin de pouvoir ensuite les fournir en entrée de modèles d'apprentissage automatique 
(qu'ils soient classiques sous-entendus non basés sur des réseaux de neurones ou de d'apprentissage profond).<br>
Bien que son usage ait diminué, cette technique possède de nombreux atouts qui font qu’elle peut vous être tout de même utile. En effet, le *bag-of-words* est :
* très simple à comprendre,
*	peut être utilisé sur un faible échantillon de données alors que les techniques basées sur les réseaux de neurones ont en besoin d’énormément,
*	ne nécessite pas de capacités de calcul importantes (un CPU suffit alors que les techniques de réseaux de neurones nécessitent des GPUs ou des TPUs qui coûtent extrêmement chers).
<br><br>


Les algorithmes d'apprentissage automatique ne peuvent pas fonctionner directement avec du texte brut. 
Le texte doit être converti en chiffres et notamment en vecteur.
Un *bag-of-words* est une représentation du texte qui décrit la présence de mots dans un document. Cela implique deux choses :
*	un vocabulaire de mots connus,
* une mesure de la présence des mots connus.
<br>

Il s’agit d’un « sac » de mots, car toute information sur l’ordre ou la structure des mots dans le document est rejetée.
Le modèle se préoccupe seulement de savoir si des mots connus se trouvent dans le document et non pas où ils se trouvent dans le document.
Ainsi, **le *bag-of-words* élimine toute l’information relative à l’ordre des mots** et met l’accent sur l’occurrence des mots dans un document. 
L’intuition est que les documents sont similaires s’ils ont un contenu similaire.
De plus, du seul contenu nous pouvons apprendre quelque chose sur la signification du document.<br>
Le sac de mots peut être aussi simple ou complexe que vous le souhaitez. 
La complexité vient à la fois de la façon de concevoir le vocabulaire des mots connus (ou *tokens*) et de la façon d’évaluer la présence des mots connus.
<br><br><br>



# <span style="color: #FF0000"> **Aller plus loin** <span>
Comme indiqué plus haut, je ne vais pas entrer dans les détails des différentes étapes de la méthode à savoir :
* créer le vocabulaire de mots connus,
* établir une mesure de la présence des mots connus.
<br>
  
Enonçons simplement les mots clés que vous pourrez ensuite chercher par vous même si les sites que je vous propose un peu plus loin n'ont pas répondu à toutes vos interrogations.
<br>
  
Le premier point lié au vocabulaire repose sur des étapes de : 
* normalisation du texte
* *tokenization*
* suppression des *stopwords*
* *lemmatization* (ou *stemming*)
<br>

Le second point consacrée à la mesure peut être calculé de plusieurs manières différentes :
*	le comptage (compter le nombre de fois que chaque mot apparaît dans un document),
* les fréquences (calculer la fréquence à laquelle chaque mot apparaît dans un document parmi tous les mots du document. La métrique la plus connue étant [TF-IDF](https://fr.wikipedia.org/wiki/TF-IDF),
*	le hachage.
<br><br>


Ces différentes techniques sont expliquées sur les sites suivants :
- le tutoriel de [Victor Bigand sur Actuia.com](https://www.actuia.com/contribution/victorbigand/tutoriel-tal-pour-les-debutants-classification-de-texte/
https://www.actuia.com/contribution/victorbigand/tutoriel-tal-pour-les-debutants-classification-de-texte/) 
explique bien le premier point TFIDF et a l'avantage de proposer la syntaxe Python associée.
- la [documentation de Scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
) explique l'ensemble des techniques énoncées ainsi que les fonctions Python utiles à la mise en place du *bag-of-words*. Elle est cependant en anglais.
<br>
En plus de Scikit-learn évoqué à l'instant, d'autres librairies Python sont également fréquemment utilisées :
[Nltk](https://www.nltk.org/) (utilisée par Victor Bigand dans son tutoriel) et [SpaCy](https://spacy.io/).
<br><br><br>



# <span style="color: #FF0000"> **Conclusion** <span>
Le *bag-of-words* est très simple à comprendre et à mettre en œuvre.
<br>
Cette méthode souffre de néanmoins certaines lacunes :
- le vocabulaire : pour être efficace, cette méthode exige que le vocabulaire soit conçu de manière extrêmement rigoureuse (les fautes d'orthographe par exemple peuvent poser problème),
- le sens : ne pas tenir compte de l’ordre des mots ne permet pas de prendre en compte le contexte et par conséquent le sens des mots dans le document (sémantique). 
<br>
Les méthodes présentées dans les autres articles du blog, permettent de prendre en compte ce problème.
<br><br><br>




# <span style="color: #FF0000"> **Citation** <span>
> @inproceedings{bag_og_words_blog_post,  
  author    = {Loïck BOURDOIS},  
  title     = {Illustration du Bag of words},  
  year      = {2019},  
  url = {https://lbourdois.github.io/blog/nlp/Bag-of-word/}  
}
