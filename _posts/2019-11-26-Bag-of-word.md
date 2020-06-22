---
title: "ILLUSTRATION DU BAG-OF-WORD"
categories:
  - NLP
tags:
  - bag of word expliqué en français
  - bag of word NLP en français
excerpt : "NLP - Illustration de la technique du bag of word"
header :
    overlay_image: "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/NLP_radom_blog.png"
toc: true
toc_sticky: true
author_profile: false
sidebar:
    nav: sidebar-sample
---

# <span style="color: #FF0000"> **Avant-propos** </span>
 Cet article est le premier de la série consacrée au NLP sur le blog.<br>
 Il est court car cette technique est maintenant ancienne et la litérarrure à son sujet est assez abondante en français.
 Je ne vois donc pas l'intérêt d'effectuer un doublon.<br>
 Je vais me limiter à énoncer très succintement le principe de la méthode et vous renvoyer vers d'autres sites internet pour plus de détails en ce qui concerne le codage par exemple.
<br><br><br>


# <span style="color: #FF0000"> **Rapide descriptif de la méthode** </span>
Le bag-of-word (sac de mots en français) est la technique qui était majoritairement utilisée avant l’apparition des modèles basés sur les réseaux de neurones tel que les RNN (eux mêmes moins utilisés aujourd’hui au profit des Transformers).
Elle consiste principalement à nettoyer les données textuelles à notre disposition d’une certaine manière, afin de pouvoir ensuite les fournir en entrée de modèles de machine learning 
(qu'ils soient classiques sous entendus non basés sur des réseaux de neurones ou de deep learning).<br>
Bien que son usage est diminué, cette technique possède de nombreux atouts qui font qu’elle peut vous être tout de même utile. En effet, le bag-of-word est :
* très simple à comprendre
*	peut être utilisé sur un faible échantillon de données alors que les techniques basées sur les réseaux de neurones ont en besoin d’énormément
*	ne nécessite pas de capacités de calcul importantes (un CPU suffit alors que les techniques de réseaux de neurones nécessitent des GPU voir des TGU qui coûtent extrêmement cheres)
<br><br>


Les algorithmes de machine learning ne peuvent pas fonctionner directement avec du texte brut. 
Le texte doit être converti en chiffres et notamment en vecteur.
Un bag-of-words (BoW) est une représentation du texte qui décrit la présence de mots dans un document. Cela implique deux choses :
*	un vocabulaire de mots connus.
* une mesure de la présence des mots connus.
<br>

Il s’agit d’un « sac » de mots, car toute information sur l’ordre ou la structure des mots dans le document est rejetée.
Le modèle se préoccupe seulement de savoir si des mots connus se trouvent dans le document, et non pas où ils se trouvent dans le document.
Ainsi, **le BoW élimine toute l’information relative à l’ordre des mots** et met l’accent sur l’occurrence des mots dans un document. 
L’intuition est que les documents sont similaires s’ils ont un contenu similaire.
De plus, du seul contenu nous pouvons apprendre quelque chose sur la signification du document.<br>
Le sac de mots peut être aussi simple ou complexe que vous le souhaitez. 
La complexité vient à la fois de la façon de concevoir le vocabulaire des mots connus (ou tokens) et de la façon d’évaluer la présence des mots connus.
<br><br><br>



# <span style="color: #FF0000"> **Aller plus loin** <span>
Comme indiqué plus haut, je ne vais pas entrer dans les détails des différentes étapes de la méthode à savoir :
* créer le vocabulaire de mots connus
* établir une mesure de la présence des mots connus.
<br>
  
Enonçons simplement les mots clés que vous pourrez ensuite chercher par vous même si les sites que je vous propose un peu plus loin n'ont pas répondu à toutes vos interogations.
<br>
  
Le premier point lié au vocabulaire repose sur des étapes de : 
* Normalisation du texte
* Tokénization
* Suppression des stopwords
* Lemmatization (ou Stemming)
<br>

Le second point consacrée à la mesure peut être calculé de plusieurs manières différentes :
*	Le comptage (compter le nombre de fois que chaque mot apparaît dans un document).
* Les fréquences (calculer la fréquence à laquelle chaque mot apparaît dans un document parmi tous les mots du document; la métrique la plus connue étant [TFIDF](https://fr.wikipedia.org/wiki/TF-IDF))
*	Le hachage.
<br><br>


Ces différentes techniques sont expliquées sur les sites suivants :
- le tutoriel de [Victor Bigand sur Actuia.com](https://www.actuia.com/contribution/victorbigand/tutoriel-tal-pour-les-debutants-classification-de-texte/
https://www.actuia.com/contribution/victorbigand/tutoriel-tal-pour-les-debutants-classification-de-texte/) 
explique bien le premier point TFIDF et a l'avantage de proposer la syntaxe python associée.
- la [documentation de Scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
) explique l'ensemble des techniques énoncées ainsi que les fonctions python utiles à la mise en place du BOW. Elle est cependant en anglais.
<br>
En plus de Scikit-learn évoqué à l'instant, d'autres librairies Python sont également fréquemment utilisées :
[Nltk](https://www.nltk.org/) (utilisée par Victor Bigand dans son tutoriel) et [SpaCy](https://spacy.io/).
<br><br><br>



# <span style="color: #FF0000"> **Conclusion** <span>
Le bag-of-words model est très simple à comprendre et à mettre en œuvre.
<br>
Cette méthode souffre de néanmoins certaines lacunes :
- le vocabulaire : pour être efficace, cette méthode exige que le vocabulaire soit conçut de manière extrêmement rigoureuse (les fautes d'orthographe par exemple peuvent poser problème)
- le sens : ne pas tenir compte de l’ordre des mots ne permet pas de prendre en compte le contexte et par conséquent le sens des mots dans le document (sémantique). 
<br>
Les méthodes présentées dans les autres articles du blog, permettent de prendre en compte ce problème.
<br><br><br>




# <span style="color: #FF0000"> **Citation** <span>
Si vous venez à utiliser des éléments de cet article, veillez s'il vous plait me créditer en utilisant par exemple comme suit :<br>
*Illustration du Bag of word* par Loïck BOURDOIS (https://lbourdois.github.io/blog/nlp/Bag-of-word/), 2019.<br>
Merci :)
