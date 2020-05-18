---
title: "ILLUSTRATION DU WORD EMBEDDING ET DU WORD2VEC"
categories:
  - NLP
tags:
  - word embedding en français
  - word embedding expliqué en français
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
 

Cet article est une traduction de l’article de Jay Alammar : [The illustrated word2vec](https://jalammar.github.io/illustrated-word2vec/). Ainsi si dans l’article vous lisez des « Je » ou des « Jay », cela provient d’exemple de l’article original que je n’ai pas pu traduire autrement. Merci à lui de m'avoir autorisé à effectuer cette traduction.
<br><br><br>




# <span style="color: #FF0000"> **1. Exemple d’introduction** </span>

Sur une échelle de 0 à 100, dans quelle mesure êtes-vous introverti/extraverti (où 0 est le plus introverti et 100 le plus extraverti) ?   Avez-vous déjà passé un test de personnalité comme le MBTI – ou mieux encore, le test des cinq grands traits de personnalité ? Si vous ne l’avez pas fait, ce sont des tests qui vous posent une liste de questions, puis vous notent sur un certain nombre d’axes, l’introversion/extraversion étant l’un d’eux.

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-personality-traits-score.png">
  <figcaption>Exemple de résultat d’un test des cinq grands traits de personnalité. Il aurait une capacité prédictive de réussite scolaire, personnelle, et professionnelle. Pour effectuer ce test : https://projects.fivethirtyeight.com/personality-quiz/.</figcaption>
</figure>                                                                                                                                             </center>


Imaginez avoir obtenu 38/100 comme score d’introversion/extraversion. Nous pouvons tracer cela de cette façon :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/introversion-extraversion-100.png">
</figure>   
</center>


On se ramène à une échelle comprise entre -1 à 1 :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/introversion-extraversion-1.png">                                                                                                                                      </figure>   
</center>


Dans quelle mesure avez-vous l’impression de connaître une personne en ne connaissant que cette seule information à son sujet ? Pas grand-chose. Les gens sont complexes. Ajoutons donc une autre dimension : le score d’un autre trait du test.
Nous pouvons représenter les deux dimensions comme un point sur le graphique, ou mieux encore, comme un vecteur de l’origine à ce point.

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/two-traits-vector.png">
  <figcaption>Nous n’affichons pas tous les traits que nous prenons en compte. L’objectif est de s’habituer au fait de ne pas savoir ce que chaque dimension représente.</figcaption>
</figure>                                                                                                                                             </center>


On peut maintenant dire que ce vecteur représente partiellement ma personnalité. L’utilité d’une telle représentation apparaît quand on veut comparer deux autres personnes à moi. Disons que je me fais renverser par un bus et que j’ai besoin d’être remplacé par quelqu’un avec une personnalité similaire. Dans la figure suivante, laquelle des deux personnes me ressemble le plus ?

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/personality-two-persons.png">                                                                                                                                      </figure>   
</center>


Lorsqu’il s’agit de vecteurs, un moyen courant de calculer un score de similarité est le [Cosinus](https://fr.wikipedia.org/wiki/Similarit%C3%A9_cosinus) :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/cosine-similarity.png">
  <figcaption>Person #1 me ressemble le plus.  Les vecteurs pointant dans la même direction (la longueur joue également un rôle) ont un score de similitude cosinus plus élevé.</figcaption>
</figure>                                                                                                                                             </center>

 
Encore une fois, deux dimensions ne suffisent pas pour saisir suffisamment d’information sur les différences entre les gens. Des décennies de recherche en psychologie ont mené à cinq traits principaux (et beaucoup de sous-traits). Utilisons donc les cinq dimensions dans notre comparaison :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/big-five-vectors.png">                                                                                                                                      </figure>   
</center>
  

Le problème avec les cinq dimensions est que nous perdons la capacité de dessiner des flèches nettes comme en deux dimensions. C’est un défi commun en machine learning où nous devons souvent penser dans un espace plus vaste. Ce qui est bien, c’est que le cosinus_similarité fonctionne toujours, avec n’importe quel nombre de dimensions :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/embeddings-cosine-personality.png">
  <figcaption> Les scores obtenus représentent plus fidèlement la personnalité que ceux obtenus à deux dimensions.</figcaption>
</figure>                                                                                                                                             </center>


Deux idées centrales se dégagent de ce premier exemple :
 -   nous pouvons représenter les gens (et les choses) comme des vecteurs de nombres.
 -   nous pouvons facilement calculer à quel point les vecteurs sont similaires les uns aux autres. 
<br><br><br>




# <span style="color: #FF0000"> **2. Word Embeddings** </span>

Avec cette introduction, nous pouvons regarder des exemples de vecteurs de mots entraînés (aussi appelés word embeddings) et commencer à examiner certaines de leurs propriétés intéressantes.

Voici un word embeddings  pour le mot « king » (via GloVe vector qui est entraîné sur Wikipedia) :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/king-wikipedia.png">                                                                                                                                      </figure>   
</center>


C’est une liste de 50 numéros. Nous ne pouvons pas dire grand-chose en regardant les valeurs. Mais regardons-le un peu pour pouvoir le comparer à d’autres vecteurs de mots. Mettons tous ces chiffres sur une ligne :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/king-white-embedding.png">                                                                                                                                      </figure>   
</center>

Colorons ensuite les cellules en fonction de leurs valeurs (rouge si elles sont proches de 2, blanc si elles sont proches de 0, bleu si elles sont proches de -2) :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/king-colored-embedding.png">                                                                                                                                      </figure>   
</center>


Jusqu’à la fin de cette partie, continuons en ignorant les chiffres et en ne regardant que les couleurs pour indiquer les valeurs des cellules.
<br>


Comparons maintenant « King » à d’autres mots :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/king-man-woman-embedding.png">                                                                                                                                      </figure>   
</center>

Vous voyez comme les mots « Man » et « Woman » sont beaucoup plus semblables l’un à l’autre qu’ils ne le sont à « King » ? Ces représentations vectorielles saisissent une bonne partie de l’information, du sens et des associations de ces mots.

Voici une autre liste d’exemples (comparez en regardant verticalement à la recherche de colonnes de couleurs similaires) :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/queen-woman-girl-embeddings.png">                                                                                                                                      </figure>   
</center>
<br>


Quelques points à signaler :
-    Il y a une colonne rouge traversant tous les mots. Ils sont donc similaires le long de cette dimension (mains ne savons pas à quoi chaque dimension correspond).
-    Vous pouvez voir comment « woman » et « girl » sont semblables l’une à l’autre dans beaucoup d’endroits. Il en va de même pour « man » et « boy ».
-    « boy » et « girl » ont aussi des endroits où ils se ressemblent, mais où ils sont différents de « woman » ou « man ». Serait-ce le codage d’une vague conception de la jeunesse ? possible.
-    Tous les mots sauf le dernier sont des mots qui représentent les gens. J’ai ajouté un objet (water) pour montrer les différences entre les catégories. On peut voir, par exemple, une colonne bleue descendre jusqu’en bas et s’arrêter une fois arriver à l’embedding de « water ».
-    Il y a des endroits clairs où « king » et « queen » sont semblables l’un à l’autre et distincts de tous les autres. Cela pourrait-il s’agir d’un vague concept de royauté ?
<br><br><br>



# <span style="color: #FF0000"> **3. Analogies** </span>

Les exemples célèbres qui montrent une incroyable propriété d’embedding est le concept d’analogies. Nous pouvons ajouter et soustraire des insertions de mots et arriver à des résultats intéressants.  
L’exemple le plus célèbre est la formule : « king » – « man » + »woman » :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/king-man%2Bwoman-gensim.png">
  <figcaption>En utilisant la bibliothèque Gensim en Python nous pouvons ajouter et soustraire des vecteurs de mots. La fonction indique aussi les mots les plus similaires au vecteur résultant avec la valeur de similitude via le cosinus.</figcaption>
</figure>                                                                                                                                             </center>
<br><br><br>



# <span style="color: #FF0000"> **4. Language modeling** </span> 

Maintenant que nous nous sommes penchés sur les embeddings entrainés, nous allons en apprendre davantage sur le processus d’entraînement. Mais avant d’en arriver à Word2vec, nous devons nous pencher sur un parent conceptuel aux word embeddings : le neural language model.

Si l’on voulait donner un exemple d’une application de NLP, l’un des meilleurs exemples serait la fonction de prédiction du mot suivant d’un clavier de smartphone. C’est une caractéristique que des milliards de gens utilisent des centaines de fois par jour.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/swiftkey-keyboard.png">
  <figcaption>«Thou shalt» (Tu feras)</figcaption>
</figure>                                                                                                                                             </center>



Cette tâche peut être traitée par un modèle linguistique. Un modèle linguistique peut prendre une liste de mots (disons deux mots) et tenter de prédire le mot qui les suit.

Dans la capture d’écran ci-dessus, nous pouvons penser que le modèle est celui qui a tenu compte des deux mots et qui a retourné une liste de suggestions (« not » étant celui avec la plus forte probabilité) :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/thou-shalt-_.png">
</figure>   
</center>

On peut penser que le modèle ressemble à cette boîte noire :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/language_model_blackbox.png">
</figure>   
</center>


Mais dans la pratique, le modèle ne produit pas un seul mot. Il produit en fait un score de probabilité pour tous les mots qu’il connaît (le « vocabulaire » du modèle, qui peut varier de quelques milliers à plus d’un million de mots). L’application doit alors trouver les mots avec les scores les plus élevés et les présenter à l’utilisateur.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/language_model_blackbox_output_vector.png">
</figure>   
</center>

Après avoir été entraînés, les premiers modèles de langage neuronal ([Bengio 2003](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)) calculent une prédiction en trois étapes :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/skipgram-language-model-training-2.png">
</figure>   
</center>

La première étape est la plus pertinente pour nous lorsque nous discutons de d’embeddings.

L’un des résultats du processus d’entraînement est une matrice qui contient un embedding pour chaque mot de notre vocabulaire. Pendant le temps de prédiction, nous recherchons simplement les embeddings du mot d’entrée, et nous les utilisons pour calculer la prédiction :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/neural-language-model-embedding.png">
</figure>   
</center>

Passons à présent au processus d’entraînement pour en savoir plus sur la façon dont la matrice d’embeddings a été créé.
<br><br><br>



# <span style="color: #FF0000"> **5. Language Model Training** </span> 

Les modèles linguistiques ont un avantage énorme sur la plupart des autres modèles de machine learning. Cet avantage est que nous sommes en mesure de les entraîner sur des textes courants dont nous disposons en abondance. Pensez à tous les livres, articles, contenus Wikipédia et autres formes de données textuelles que nous avons. Comparez ceci aux autres modèles qui ont besoin d’annotations faites à la main et de données spécialement collectées…

Nous attribuons aux mots leurs embeddings en regardant les autres mots à côté desquels ils ont tendance à apparaître. La mécanique est la suivante :
- Nous recevons beaucoup de données textuelles (disons, tous les articles de Wikipedia, par exemple). 
- Nous avons une fenêtre (disons de trois mots) que nous faisons glisser sur tout ce texte.
- La fenêtre coulissante génère des exemples de formation pour notre modèle.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/wikipedia-sliding-window.png">
</figure>   
</center>


Lorsque la fenêtre glisse sur le texte, nous générons (virtuellement) un ensemble de données que nous utilisons pour entraîner un modèle. Pour voir exactement comment cela se fait, voyons comment la fenêtre coulissante traite cette phrase : « Thou shalt not make a machine in the likeness of a human mind » (Tu ne feras pas une machine à l’image d’un esprit humain).

Lorsque nous commençons, la fenêtre est sur les trois premiers mots de la phrase :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/lm-sliding-window.png">
</figure>   
</center>


Nous prenons les deux premiers mots pour des features et le troisième mot pour un label :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/lm-sliding-window-2.png">
</figure>   
</center>


Nous glissons ensuite notre fenêtre à la position suivante et créons un deuxième échantillon :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/lm-sliding-window-3.png">
</figure>   
</center>


Et très rapidement, nous disposons d’un ensemble de données plus vaste dont les mots ont tendance à apparaître après différentes paires de mots :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/lm-sliding-window-4.png">
</figure>   
</center>


Dans la pratique, les modèles ont tendance à être entraînés pendant que nous glissons la fenêtre. Mais je trouve plus clair de séparer logiquement la phase « génération du jeu de données » de la phase d’entraînement. Outre les approches de modélisation du langage fondées sur les réseaux neuronaux, une technique appelée N-grams a été couramment utilisée pour former des modèles de langage. Pour voir comment le passage des N-grammes aux modèles neuronaux se reflète sur les produits du monde réel vous pouvez lire l’article suivant : [https://blog.swiftkey.com/neural-networks-a-meaningful-leap-for-mobile-typing/](https://blog.swiftkey.com/neural-networks-a-meaningful-leap-for-mobile-typing/) (en anglais). Cet exemple montre comment les propriétés algorithmiques des incorporations peuvent être décrites dans un discours marketing.
<br><br><br>



# <span style="color: #FF0000"> **6. Regarder des deux côtés** </span> 

Sachant ce que vous avez lu plus tôt dans l’exemple d’introduction, remplissez le blanc :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/jay_was_hit_by_a_.png">
</figure>   
</center>

Je suis sûr que la plupart des gens devineraient que le mot bus va dans le vide. Mais si je vous donnais une autre information, un mot après le blanc, cela changerait-il votre réponse ?
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/jay_was_hit_by_a_bus.png">
</figure>   
</center>

Cela change complètement ce qui devrait aller dans le blanc. Le mot « red » est maintenant le plus susceptible d’aller dans le blanc. Ce que nous apprenons de ceci est que les mots avant et après un mot spécifique ont une valeur informationnelle. Il s’avère que la prise en compte des deux sens (mots à gauche et à droite du mot que l’on devine) conduit à un meilleur word embeddings.

Voyons comment nous pouvons ajuster la façon dont nous entraînons le modèle pour tenir compte de cela.
<br><br><br>


# <span style="color: #FF0000"> **7. Skipgram** </span> 
Au lieu de regarder seulement deux mots avant le mot cible, nous pouvons aussi regarder deux mots après lui.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/continuous-bag-of-words-example.png">
</figure>   
</center>

Si nous faisons cela, l’ensemble de données que nous construisons et entrainons virtuellement ressemblerait à ceci :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/continuous-bag-of-words-dataset.png">
</figure>   
</center>

C’est ce qu’on appelle une architecture de **continuous bag of words**.


Une autre architecture qui a aussi tendance à donner de bons résultats fait les choses un peu différemment. Au lieu de deviner un mot en fonction de son contexte (les mots avant et après), cette autre architecture essaie de deviner les mots voisins en utilisant le mot courant. On peut penser à la fenêtre qu’il glisse sur le texte d’entraînement comme ressemblant à ceci :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/skipgram-sliding-window.png">
  <figcaption>Le mot dans la fente beige serait le mot d’entrée, chaque case rose serait une sortie possible.</figcaption>
</figure>   
</center>


Les cases roses sont de différentes nuances parce que cette fenêtre coulissante crée en fait quatre échantillons distincts dans notre ensemble de données d’entraînement :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/skipgram-sliding-window-2.png">
</figure>   
</center>

Cette méthode s’appelle l’architecture **skipgram**.


Nous pouvons visualiser la fenêtre coulissante comme suit :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/skipgram-sliding-window-1.png">
</figure>   
</center>

Cela ajouterait ces quatre échantillons à notre ensemble de données d’entraînement :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/skipgram-sliding-window-2.png">
</figure>   
</center>


Nous glissons ensuite notre fenêtre vers la position suivante :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/skipgram-sliding-window-3.png">
</figure>   
</center>

Ce qui génère nos quatre exemples suivants :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/skipgram-sliding-window-4.png">
</figure>   
</center>

Quelques positions plus tard, nous avons beaucoup d’autres exemples :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/skipgram-sliding-window-5.png">
</figure>   
</center>
<br><br><br>



# <span style="color: #FF0000"> **8. Revisiting the training process** </span> 

Maintenant que nous avons notre ensemble de données d’entraînement de Skipgram que nous avons extrait d’un texte, voyons comment nous l’utilisons pour former un modèle de langage neuronal qui prédit le mot voisin.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/skipgram-language-model-training.png">
</figure>   
</center>


Nous commençons par le premier échantillon de notre ensemble de données. Nous saisissons la caractéristique et l’envoyons au modèle non entrainé en lui demandant de prédire un mot voisin approprié.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/skipgram-language-model-training-2.png">
</figure>   
</center>


Le modèle conduit les trois étapes et produit un vecteur de prédiction (avec une probabilité assignée à chaque mot de son vocabulaire). Comme le modèle n’est pas entrainé, sa prédiction est certainement erronée à ce stade. Mais ce n’est pas grave. Nous savons quel mot il aurait dû deviner :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/skipgram-language-model-training-3.png">
</figure>   
</center>

Nous soustrayons les deux vecteurs pour obtenir un vecteur d’erreur :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/skipgram-language-model-training-4.png">
</figure>   
</center>

Ce vecteur d’erreur peut maintenant être utilisé pour mettre à jour le modèle de sorte que la prochaine fois, il est un peu plus susceptible de deviner «thou» quand il a «not» en entrée.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/skipgram-language-model-training-5.png">
</figure>   
</center>

Voilà qui conclut la première étape d’entraînement. Nous procédons de la même façon avec le prochain échantillon de notre ensemble de données, puis le suivant, jusqu’à ce que nous ayons couvert tous les échantillons de l’ensemble de données. Cela conclut une « epoch » d’entraînement. Nous recommençons pendant un certain nombre d’époques, et nous obtenons alors notre modèle entraîné. Nous pouvons en extraire la matrice d’embedding et l’utiliser pour toute autre application.

Bien que cela élargisse notre compréhension du processus, ce n’est pas encore la façon dont word2vec est réellement formé. Il nous manque quelques idées clés.
<br><br><br>



# <span style="color: #FF0000"> **9. Negative Samplings** </span> 
Rappelez-vous les trois étapes de la façon dont ce modèle de langage neuronal calcule sa prédiction :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/language-model-expensive.png">
</figure>   
</center>

La troisième étape est très coûteuse d’un point de vue informatique (computationnel). Nous devons faire quelque chose pour améliorer nos performances.

Une façon de faire est de diviser notre cible en deux étapes :
1) Générer des word embeddings de haute qualité (pas besoin de prédire le mot suivant).                                        
2) Utiliser ces word embeddings de haute qualité pour entraîner un modèle de langage et faire la prédiction du mot suivant.


Nous nous concentrerons sur l’étape 1.

Pour générer des embeddings de haute qualité, nous pouvons passer d’un modèle de la prédiction d’un mot voisin à un modèle qui prend le mot d’entrée et le mot de sortie, et sort un score indiquant s’ils sont voisins ou non (0 pour « non voisin », 1 pour « voisin »).
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/predict-neighboring-word.png">
</figure>   
</center>

Ce simple changement transforme le modèle dont nous avons besoin. Nous passons d’un réseau neuronal à un modèle de régression logistique qui est ainsi beaucoup plus simple et beaucoup plus rapide à calculer.

Ce changement nécessite de changer la structure de notre ensemble de données : le label est maintenant une nouvelle colonne avec les valeurs 0 ou 1. Ils seront tous à 1 puisque tous les mots que nous avons ajoutés sont voisins.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/word2vec-training-dataset.png">
</figure>   
</center>


Le temps d’exécution est considérablement réduit, pouvant maintenant traiter des millions d’exemples en quelques minutes. Mais il y a une faille à combler. Si tous nos exemples sont positifs (cible : 1), nous nous ouvrons à la possibilité d’un modèle qui renvoie toujours 1 – atteignant 100% de précision, mais n’apprenant rien et générant des embeddings de déchets.

Pour y remédier, nous devons introduire des échantillons négatifs dans notre ensemble de données, c’est à dire des échantillons de mots qui ne sont pas voisins. Notre modèle doit retourner 0 pour ces échantillons. C’est un défi que le modèle doit relever avec acharnement, mais toujours à une vitesse fulgurante.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/word2vec-negative-sampling.png">
</figure>   
</center>



Nous assignons ensuite comme mot de sortie des mots pris au hasard dans notre vocabulaire.

Cette idée s’inspire de [Noise-contrastive estimation](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf). Nous comparons le signal réel (exemples positifs de mots voisins) avec le bruit (mots choisis au hasard qui ne sont pas voisins). Il en résulte un grand compromis entre l’efficacité informatique et l’efficacité statistique.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/word2vec-negative-sampling-2.png">
</figure>   
</center>
<br><br><br>



# <span style="color: #FF0000"> **10. Skipgram with Negative Sampling (SGNS)** </span> 
Nous avons maintenant couvert deux des idées centrales de Word2vec. Associées, elles s’appellent Skipgram with Negative Sampling (« skipgram avec un échantillonnage négatif »).
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/skipgram-with-negative-sampling.png">
</figure>   
</center>
<br><br><br>



# <span style="color: #FF0000"> **11. Processus d'entraînement de Word2vec** </span> 
Maintenant que nous avons établi les deux idées centrales du SGNS, nous pouvons examiner de plus près le processus d’entraînement de word2vec.  

Avant le début du processus d’entraînement, nous prétraitons le texte sur lequel nous formons le modèle. Dans cette étape, nous déterminons la taille de notre vocabulaire (nous l’appellerons vocab_size, disons, 10 000) et quels mots lui appartiennent.

Au début de la phase d’entraînement, nous créons deux matrices – une Embedding matrix et une Context matrix. Ces deux matrices ont un embedding pour chaque mot de notre vocabulaire (vocab_size est donc une de leurs dimensions). La seconde dimension est la longueur que nous voulons que chaque vecteur d’embedding soit (une valeur généralement utilisée de embedding_size est 300, mais nous avons regardé un exemple de 50 plus tôt dans ce post).
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/word2vec-embedding-context-matrix.png">
</figure>   
</center>


Au début de l’entraînement, nous initialisons ces matrices avec des valeurs aléatoires. Ensuite, nous commençons le processus. A chaque étape de l’entraînement, nous prenons un exemple positif et les exemples négatifs qui y sont associés. Prenons notre premier groupe :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/word2vec-training-example.png">
</figure>   
</center>

Maintenant nous avons quatre mots : le mot d’entrée not et les mots de sortie/contexte : thou (le voisin actuel), aaron et taco (les exemples négatifs).

Nous procédons à la recherche de leurs embeddings. Pour le mot d’entrée, nous regardons dans l’Embedding matrix. Pour les mots de contexte, nous regardons dans la Context matrix.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/word2vec-lookup-embeddings.png">
</figure>   
</center>

Ensuite, nous effectuons le produit scalaire de l’embeddings d’entrée avec chacun des embeddings de contexte. Dans chaque cas, cela donnerait un nombre, ce nombre indique la similarité entre les embeddings d’entrée et de contexte.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/word2vec-training-dot-product.png">
</figure>   
</center>


Nous devons maintenant trouver un moyen de transformer ces scores en quelque chose qui ressemble à des probabilités. Nous avons besoin qu’ils soient tous positifs et qu’ils aient des valeurs entre zéro et un. Pour cela, nous utilisons  la fonction [sigmoïde](https://fr.wikipedia.org/wiki/Sigmo%C3%AFde_(math%C3%A9matiques)).
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/word2vec-training-dot-product-sigmoid.png">
</figure>   
</center>

Vous pouvez voir que taco a le score le plus élevé et qu’aaron a toujours le score le plus bas avant et après les opérations sigmoïdes.

Maintenant que le modèle non entraîné a fait une prédiction, et vu que nous avons un label auquel la comparer, calculons l’erreur dans la prédiction du modèle. Pour ce faire, il suffit de soustraire les scores sigmoïdes des labels (targets).
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/word2vec-training-error.png">
  <figcaption>error = target – sigmoid_scores</figcaption>
</figure>   
</center>


Nous pouvons maintenant utiliser ce score d’erreur pour ajuster les embeddings de not, thou, aaron, et taco afin que la prochaine fois que nous ferons ce calcul, le résultat soit plus proche des scores cibles.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/word2vec-training-update.png">
</figure>   
</center>

L’étape d’entraînement est terminée. Nous en ressortons avec des embeddings légèrement meilleurs pour les mots impliqués dans cette étape (not, thou, aaron, and taco). Nous passons alors à l’échantillon positif (et les échantillons négatifs associés) suivant et recommençons le même processus.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/word2vec-training-example-2.png">
</figure>   
</center>


L’embeddings continue d’être amélioré pendant que nous parcourons l’ensemble de nos données un certain nombre de fois. Nous pouvons alors arrêter le processus d’entraînement, abandonnons la Context matrix et utilisons l’Embedding matrix comme pré-entrainées pour la tâche suivante.
<br><br><br>



# <span style="color: #FF0000"> **12. Window size and number of negative samples** </span> 

La taille de la fenêtre et le nombre d’échantillons négatifs sont deux hyperparamètres clés dans le processus d’entraînement de word2vec.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/word2vec-window-size.png">
</figure>   
</center>

La taille de la fenêtre la plus adaptée varie en fonction de la tâche à effectuer.    

Une heuristique est que des fenêtres de petite taille (2-15) conduisent à des embeddings avec des scores de similarité élevés entre deux embeddings. Cela signifie que les mots sont interchangeables (remarquez que les antonymes sont souvent interchangeables si l’on considère seulement les mots environnants. Par exemple, bon et mauvais apparaissent souvent dans des contextes similaires).

Des fenêtres de plus grande taille (15-50, ou même plus) mènent à des embeddings où la similarité donne une indication sur la parenté des mots. Dans la pratique, vous devrez souvent fournir des annotations qui guident le processus d’embeddings menant à une similarité utile pour votre tâche. La taille par défaut de la fenêtre de Gensim est 5 (deux mots avant et deux mots après le mot entré, en plus du mot entré lui-même).

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/word_embeddings/word2vec-negative-samples.png">
</figure>   
</center>

Le nombre d’échantillons négatifs est un autre facteur du processus d’entraînement. L’article original prescrit 5-20 comme étant un bon nombre d’échantillons négatifs. Il indique également que 2-5 semble être suffisant quand vous avez un ensemble de données assez grand. La valeur par défaut de Gensim est de 5 échantillons négatifs.
 <br><br><br>



# <span style="color: #FF0000"> **Conclusion** </span> 
J’espère pouvoir ajouter un tutoriel sous forme de notebook plus tard si j’ai le temps.
<br><br><br>



# <span style="color: #FF0000"> **Citation** <span>
Si vous venez à utiliser des éléments de cet article, veillez s'il vous plait en créditer les auteurs en utilisant par exemple comme suit :<br>
“*Illustration du Word Embedding et du Word2vec* par Loïck BOURDOIS (https://lbourdois.github.io/blog/nlp/word_embedding/), d’après Jay ALAMMAR, *The Illustrated Word2vec* (https://jalammar.github.io/illustrated-word2vec/)”<br>
Merci :)
