---
title: "ILLUSTRATION DU TRANSFORMER"
categories:
  - NLP
tags:
  - Transformer NLP expliqué en français
  - Illustration du Transformer en français NLP
  - Transformer NLP en français
excerpt : "NLP"
header :
    overlay_image: "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/word2vec-blog.png"
toc: true
toc_sticky: true
author_profile: false
sidebar:
    nav: sidebar-sample
---
<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script> 

# <span style="color: #FF0000"> **Avant-propos** </span>
 
Cet article est en grande partie une traduction de l’article de Jay Alammar : [The illustrated transformer](https://jalammar.github.io/illustrated-transformer/). 
J’ai ajouté des éléments supplémentaires quand j’estimais que cela été pertinent.
<br><br><br>



# <span style="color: #FF0000"> **Introduction** </span>
Dans l’article [Le Seq2seq et le processus d’attention](https://lbourdois.github.io/blog/nlp/Seq2seq-et-attention/) nous nous avons introduit l’attention, une méthode omniprésente dans les modèles modernes d’apprentissage profond. 
L’attention est un concept qui a permis d’améliorer les performances des applications de traduction automatique. 
Dans celui-ci, nous nous intéresserons au Transformer, un modèle qui utilise l’attention pour augmenter la vitesse à laquelle 
ces modèles peuvent être entraînés. Le Transformer surpasse le modèle de traduction automatique de Google dans des tâches 
spécifiques. Le plus grand avantage, vient de la façon dont le Transformer se prête à la parallélisation.
C’est en effet la recommandation de Google Cloud d’utiliser le Transformer comme modèle de référence pour utiliser leur offre 
[Cloud TPU](https://cloud.google.com/tpu/). Essayons donc de décomposer le modèle et de voir comment il fonctionne.
<br>

Le Transformer a été proposé dans le document [Attention is All You Need](https://arxiv.org/abs/1706.03762) de A. Vaswani et al. 
Une implémentation de TensorFlow est disponible dans le cadre du package [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor).
Le groupe NLP de Harvard a créé un [guide](http://nlp.seas.harvard.edu/2018/04/03/attention.html) avec l’implémentation en PyTorch. 
Dans cet article, nous tenterons de simplifier les choses et d’introduire les concepts un par un pour, espérons-le,
faciliter la compréhension des gens qui n’ont pas une connaissance approfondie du sujet.
<br><br><br>



# <span style="color: #FF0000"> **1. Un look de haut niveau** </span>
Commençons par considérer le modèle comme une boîte noire. 
Dans une application de traduction automatique, il prendrait une phrase dans une langue et la traduirait dans une autre.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-personality-traits-score.png">
 </figure>
 </center>

En ouvrant la boite, nous voyons un composant d’encodage, un composant de décodage et des connexions entre eux.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-personality-traits-score.png">
 </figure>
 </center>
 

Le composant d’encodage est une pile d’encoders (l’article empile six encodeurs les uns sur les autres – il n’y a rien de magique avec le numéro six, on peut certainement expérimenter avec d’autres arrangements).
Le composant de décodage est une pile de decoders du même nombre.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-personality-traits-score.png">
 </figure>
 </center>
 

Les encoders sont tous identiques mais ils ne partagent pas leurs poids. Chacun est divisé en deux sous-couches :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-personality-traits-score.png">
 </figure>
 </center>
 

Les entrées de l’encoder passent d’abord par une couche d’auto-attention : une couche qui aide l’encoder à regarder les autres mots dans la phrase d’entrée lorsqu’il code un mot spécifique.
Nous examinerons de plus près la question de l’auto-attention plus loin dans l'article.

Les sorties de la couche d’auto-attention sont transmises à un réseau feed-forward.
Le même réseau feed-forward est appliqué indépendamment à chaque encoder.

Le decoder possède ces deux couches, mais entre elles se trouve une couche d’attention qui aide le decoder à se concentrer sur les parties pertinentes de la phrase d’entrée (comme dans les modèles seq2seq).
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-personality-traits-score.png">
 </figure>
 </center>
<br><br><br>



# <span style="color: #FF0000"> **2. Les tenseurs** </span>
Maintenant que nous avons vu les principales composantes du modèle, commençons à examiner les différents vecteurs/tenseurs et la façon dont ils circulent entre ces composantes pour transformer l’entrée d’un modèle entrainé en sortie.

Comme c’est le cas dans les applications NLP en général, nous commençons par transformer chaque mot d’entrée en vecteur à l’aide d’un algorithme d’embedding.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-personality-traits-score.png">
  <figcaption>
  Chaque mot est mis dans un vecteur de taille 512. Nous représentons ces vecteurs avec ces simples boîtes.
  </figcaption>
</figure>
</center>


L’embedding n’a lieu que dans l’encoder inférieur. Le point commun à tous les encoders est qu’ils reçoivent une liste de vecteurs de la taille 512. Dans l’encoder du bas cela serait le word embeddings, mais dans les autres encoders, ce serait la sortie de l’encoder qui serait juste en dessous.

La taille de la liste est un hyperparamètre que nous pouvons définir. Il s’agirait essentiellement de la longueur de la phrase la plus longue dans notre ensemble de données d’entraînement.

Après avoir intégré les mots dans notre séquence d’entrée, chacun d’entre eux traverse chacune des deux couches de l’encoder.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-personality-traits-score.png">
</figure>
</center>
 
Nous commençons à voir une propriété clé du Transformer : dans chacune des positions, le mot circule à travers son propre chemin dans l’encoder. Il y a des dépendances entre ces chemins dans la couche d’auto-attention.

La couche feed-forward n’a pas ces dépendances et donc les différents chemins peuvent être exécutés en parallèle lors de cette couche.

Ensuite, nous allons commuter l’exemple sur une phrase plus courte et regarder ce qui se passe dans chaque sous-couche de l’encoder.
<br><br><br>



# <span style="color: #FF0000"> **3. L'encodage** </span>
Comme nous l’avons déjà mentionné, un encoder reçoit une liste de vecteurs en entrée. Il traite cette liste en passant ces vecteurs dans une couche d’auto-attention, puis dans un réseau feed-forward, et enfin envoie la sortie vers le haut au codeur suivant.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-personality-traits-score.png">
  <figcaption>
  Le mot à chaque position passe par un processus d’auto-attention. Ensuite, chacun d’eux passe par un réseau feed-forward (le même réseau réseau feed-forward pour chaque vecteur mais chacun le traverse séparément).
  </figcaption>
</figure>
</center>
<br><br><br>



# <span style="color: #FF0000"> **4. Introduction à l'auto-attention** </span>
Ne vous laissez pas berner par le mot « d’auto-attention » que je lance comme si c’était un concept que tout le monde devrait connaître. Regardons son fonctionnement.

Disons que la phrase suivante est une phrase d’entrée que nous voulons traduire : 
« The animal didn't cross the street because it was too tired ».

A quoi se réfère « it » dans cette phrase ? Est-ce qu’il fait référence à la rue ou à l’animal ? C’est une question simple pour un humain, mais pas pour un algorithme.

Lorsque le modèle traite le mot « it », l’auto-attention lui permet d’associer « it » à « animal ».

Au fur et à mesure que le modèle traite chaque mot (chaque position dans la séquence d’entrée, l’auto-attention lui permet d’examiner d’autres positions dans la séquence d’entrée à la recherche d’indices qui peuvent aider à un meilleur codage pour ce mot.

L’auto-attention est la méthode que le Transformer utilise pour améliorer la compréhension du mot qu’il est en train de traiter en fonction des autres mots pertinents.

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-personality-traits-score.png">
  <figcaption>
  Comme nous codons le mot « it » dans l’encodeur#5 (le codeur supérieur de la pile), une partie du mécanisme d’attention se concentrait sur « The Animal ».
  </figcaption>
</figure>
</center>
Vous pouvez jouer avec la visualisation interactive en consultant le [notebook Tensor2Tensor](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb).
<br><br><br>



# <span style="color: #FF0000"> **5. L'auto-attention en détail** </span>
Voyons d’abord comment calculer l’auto-attention à l’aide de vecteurs, puis comment elle est réellement mise en œuvre à l’aide de matrices.

La première étape du calcul de l’auto-attention consiste à créer trois vecteurs à partir de chacun des vecteurs d’entrée \(x_{i}\) de l’encoder (dans ce cas, l’embedding de chaque mot).

Chaque vecteur d’entrée \(x_{i}\) est utilisé de trois manières différentes dans l’opération d’auto-attention :
*  Il est comparé à tous les autres vecteurs pour établir les pondérations pour sa propre production \(y_{i}\). Cela forme le vecteur de requête (Query en anglais et dans les figures suivantes).
*  Il est comparé à tous les autres vecteurs pour établir les pondérations pour la sortie du j-ème vecteur \(y_{j}\). Cela forme le vecteur de clé (Key).
*  Il est utilisé comme partie de la somme pondérée pour calculer chaque vecteur de sortie une fois que les pondérations ont été établies. Cela forme le vecteur de valeur (Value). 


Ces vecteurs sont créés en multipliant l’embedding par trois matrices que nous avons formées pendant le processus d’entraînement.
On a donc : $$q_{i} = W^{q} x_{i} ,  k_{i} = W^{k} x_{i} et v_{i} = W^{v} x_{i}$$

Notez que ces nouveaux vecteurs sont de plus petite dimension que le vecteur d’embedding (64 contre 512). 
Ils n’ont pas besoin d’être plus petits. C’est un choix d’architecture pour rendre la computation des têtes d’attentions constante.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-personality-traits-score.png">
  <figcaption>
  Les nombres indiqués en rouge correspondent aux dimensions de chaque vecteur/matrices
  </figcaption>
</figure>
</center>


La deuxième étape du calcul de l’auto-attention consiste à calculer un score.
Disons que nous calculons l’auto-attention pour le premier mot de cet exemple,   « Thinking » . 
Nous devons noter chaque mot de la phrase d’entrée par rapport à ce mot. 
Le score détermine le degré de concentration à placer sur les autres parties de la phrase d’entrée au fur et à mesure que nous codons un mot à une certaine position.

Le score est calculé en prenant le produit scalaire du vecteur de requête avec le vecteur clé du mot que nous évaluons.
Donc, si nous traitons l’auto-attention pour le mot en position #1, le premier score serait le produit scalaire de \(q_{1}\) et \(k_{1}\).
Le deuxième score serait le produit scalaire de \(q_{1}\) et \(k_{2}\).
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-personality-traits-score.png">
</figure>
</center>

Les troisièmes et quatrièmes étapes consistent à diviser les scores par la racine carrée de la dimension des vecteurs clés utilisés (ici on divise donc par 8) . Cela permet d’obtenir des gradients plus stables.

En effet, la fonction softmax que nous appliquons ensuite peut être sensible à de très grandes valeurs d’entrée. Ceux-ci tuent le gradient et ralentissent l’apprentissage, ou l’arrêtent complètement. Puisque la valeur moyenne du produit scalaire augmente avec la dimension de l’embedding, il est utile de redimensionner un peu le produit scalaire pour empêcher les entrées de la fonction softmax de devenir trop grandes.

Il pourrait y avoir d’autres valeurs possibles que la racine carrée de la dimension, mais c’est la valeur par défaut.

L’ application de la fonction Softmax permet de normaliser les scores pour qu’ils soient tous positifs et somment à 1.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-personality-traits-score.png">
</figure>
</center>
Ce score softmax détermine à quel point chaque mot sera exprimé à sa position. Il est donc logique que le mot à sa position aura le score softmax le plus élevé, mais le score des autres mots permet de déterminer leur pertinence par rapport au mot traité.
<br><br>

La cinquième étape consiste à multiplier chaque vecteur de valeur par le score softmax (en vue de les additionner). L’intuition ici est de garder intactes les valeurs du ou des mots sur lesquels nous voulons nous concentrer, et de noyer les mots non pertinents (en les multipliant par de petits nombres comme 0,001, par exemple).
<br><br>

La sixième étape consiste à résumer les vecteurs de valeurs pondérées. Ceci produit la sortie de la couche d’auto-attention à cette position (ici pour le premier mot).
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-personality-traits-score.png">
</figure>
</center>
<br><br>

Résumons toutes ces étapes sous la forme d’un tableau et itérons le processus sur plusieurs mots :

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-personality-traits-score.png">
  <figcaption>
  Tableau pour le traitement du mot « Bonjour »
  </figcaption>
</figure>
</center>

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-personality-traits-score.png">
  <figcaption>
  Tableau pour le traitement du mot « je »
  </figcaption>
</figure>
</center>

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-personality-traits-score.png">
  <figcaption>
  Tableau pour le traitement du mot « suis »
  </figcaption>
</figure>
</center>

Voilà qui conclut le calcul de l’auto-attention. Les vecteurs zi résultants peuvent être envoyés au réseau feed-forward. En pratique cependant, ce calcul est effectué sous forme de matrice pour un traitement plus rapide. Regardons donc comment cela se déroule, maintenant que nous avons vu l’intuition du calcul au niveau d’un vecteur.
<br><br><br>

