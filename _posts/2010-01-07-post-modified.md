---
title: "ILLUSTRATION DU WORD EMBEDDING ET DU WORD2VEC"
categories:
  - NLP
tags:
  - word embedding en français
---



# Avant-propos

Cet article est une traduction de l’article de Jay Alammar : [The illustrated word2vec](https://jalammar.github.io/illustrated-word2vec/).   Ainsi si dans l’article vous lisez des « Je » ou des « Jay », cela provient d’exemple de l’article original que je n’ai pas pu traduire autrement. 


# 1. Exemple d’introduction

Sur une échelle de 0 à 100, dans quelle mesure êtes-vous introverti/extraverti (où 0 est le plus introverti et 100 le plus extraverti) ?   Avez-vous déjà passé un test de personnalité comme le MBTI – ou mieux encore, le test des cinq grands traits de personnalité ? Si vous ne l’avez pas fait, ce sont des tests qui vous posent une liste de questions, puis vous notent sur un certain nombre d’axes, l’introversion/extraversion étant l’un d’eux.

| [big-five-personality-traits-score](https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-personality-traits-score.png) | 
|:--:| 
| *Exemple de résultat d’un test des cinq grands traits de personnalité. Il aurait une capacité prédictive de réussite scolaire, personnelle, et professionnelle. Pour effectuer ce test : https://projects.fivethirtyeight.com/personality-quiz/. * |




Imaginez avoir obtenu 38/100 comme score d’introversion/extraversion. Nous pouvons tracer cela de cette façon :  



On se ramène à une échelle comprise entre -1 à 1 :

Dans quelle mesure avez-vous l’impression de connaître une personne en ne connaissant que cette seule information à son sujet ? Pas grand-chose. Les gens sont complexes. Ajoutons donc une autre dimension : le score d’un autre trait du test.
Nous pouvons représenter les deux dimensions comme un point sur le graphique, ou mieux encore, comme un vecteur de l’origine à ce point.

Nous n’affichons pas tous les traits que nous prenons en compte. L’objectif est de s’habituer au fait de ne pas savoir ce que chaque dimension représente.

On peut maintenant dire que ce vecteur représente partiellement ma personnalité. L’utilité d’une telle représentation apparaît quand on veut comparer deux autres personnes à moi. Disons que je me fais renverser par un bus et que j’ai besoin d’être remplacé par quelqu’un avec une personnalité similaire. Dans la figure suivante, laquelle des deux personnes me ressemble le plus ?

Lorsqu’il s’agit de vecteurs, un moyen courant de calculer un score de similarité est le Cosinus :
 Person #1 me ressemble le plus.  Les vecteurs pointant dans la même direction (la longueur joue également un rôle) ont un score de similitude cosinus plus élevé.

Encore une fois, deux dimensions ne suffisent pas pour saisir suffisamment d’information sur les différences entre les gens. Des décennies de recherche en psychologie ont mené à cinq traits principaux (et beaucoup de sous-traits). Utilisons donc les cinq dimensions dans notre comparaison :

Le problème avec les cinq dimensions est que nous perdons la capacité de dessiner des flèches nettes comme en deux dimensions. C’est un défi commun en machine learning où nous devons souvent penser dans un espace plus vaste. Ce qui est bien, c’est que le cosinus_similarité fonctionne toujours, avec n’importe quel nombre de dimensions :
Les scores obtenus représentent plus fidèlement la personnalité que ceux obtenus à deux dimensions.

Deux idées centrales se dégagent de ce premier exemple :

    nous pouvons représenter les gens (et les choses) comme des vecteurs de nombres.
    nous pouvons facilement calculer à quel point les vecteurs sont similaires les uns aux autres. 


