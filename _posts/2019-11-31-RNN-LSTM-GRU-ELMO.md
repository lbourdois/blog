---
title: "LES RNN, LES LSTM, LES GRU ET ELMO"
categories:
  - NLP
tags:
  - ELMO expliqué en français
  - RNN expliqué en français
  - LSTM expliqué en français
  - GRU expliqué en français
  - ELMO french
  - RNN french
  - LSTM french
  - GRU french
excerpt : "NLP - Illustration des réseaux de neurones récurrents, des Long short-term memory, des Gated Recurrent Unit et de ELMo."
header :
    overlay_image: "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/RNN-LSTM-GRU-ELMO/ELMO_blog.png"
toc: true
toc_sticky: true
author_profile: false
sidebar:
    nav: sidebar-sample   
---


# <span style="color: #FF0000"> **Avant-propos** </span>
 
Les techniques évoquées dans cet article apparaissent maintenant comme anciennes par rapport aux diverses architectures basées sur le Transformer qui sont très populaires depuis environ deux ans.
Ainsi beaucoup de documentation est déjà disponible en français à leur sujet.

Je ne compte donc pas faire un article extrêmement détaillé qui ferait doublon par rapport à ce qui existe a déjà été fait. 
Je me contente d’énoncer les grandes idées et vous renvoie vers d’autres articles / vidéos pour plus de détails (formules mathématiques, exemples d’applications, etc…).
<br><br><br>



# <span style="color: #FF0000"> **Les RNN, les LSTM et les GRU** </span>
Les **RNN** (**recurrent neural network** ou réseaux de neurones récurrents en français) sont des réseaux de neurones qui ont jusqu’à encore environ deux ans, été majoritairement utilisé dans le cadre de problème de NLP.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/RNN-LSTM-GRU-ELMO/RNN.png">
  <figcaption>h_{ini} est un paramètre que vous devez choisir (par exemple la matrice nulle).<br>  
U,  V et  W sont trois matrices de poids (avec notamment V la matrice des poids récurrents).  
    Ces trois matrices sont les mêmes pour chaque étape <i>t</i>.  
Les valeurs de ces matrices sont apprises lors de la phase d’entraînement du réseau.  
</figcaption>
</figure>                                                                                                                              
</center>

Cette architecture possède un problème. 
Lorsque la séquence à traiter est trop longue, la rétropropagation du gradient de l’erreur peut soit devenir beaucoup trop grande et exploser, soit au contraire devenir beaucoup trop petite. 
Le réseau ne fait alors plus la différence entre une information qu’il doit prendre en compte ou non. 
Il se trouve ainsi dans l’incapacité d’apprendre à long terme. 
<br><br>



Les **LSTM** (**Long short-term memory**) proposent une solution à ce problème en prenant en compte un vecteur mémoire via un système de portes (gates) et d’états.
Les portes sont au nombre de 3 et les états au nombre de 2 :
- Forget Gate (capacité à oublier de l’information, quand celle-ci est inutile)
- Input Gate (capacité à prendre en compte de nouvelles informations utiles)
- Output Gate (quel est l’état de la cellule à l’instant t sachant la forget et la input gate)
- Hidden state (état caché)
- Cell state (état de la cellule) 

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/RNN-LSTM-GRU-ELMO/LSTM%20architechture.png">
  <figcaption>Image provenant de l’article Medium de Shi Yan (https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714) que nous avons traduit et légèrement modifiée
</figcaption>
</figure>                                                                                                                              
</center>
<br><br>



Une variante des LSTM sont les **GRU** (**Gated Recurrent Unit**). 
Cette structure est plus simple que les LSTM au sens où moins de paramètres entrent en jeu.
Le nombre de portes passe à 2 et celui d’état à 1 :
- Reset gate (porte de reset)
- Update gate (porte de mise à jour)
- Cell state (état de la cellule)


<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/RNN-LSTM-GRU-ELMO/GRU%20architechture.png">
  <figcaption>Image que nous avons conçu en reprenant la charte graphique adopté par Shi Yan dans son article</figcaption>
</figure>                                                                                                                              
</center>
<br>

En pratique, les GRU et les LSTM permettent d’obtenir des résultats comparables.
L’intérêt des GRU par rapport aux LSTM étant le temps d’exécution qui est plus rapide puisque moins de paramètres doivent être calculés.

Pour plus de détails, je vous invite à lire ou visionner selon votre choix les sources suivantes :
- l’[article Medium de Charles Crouspeyre](https://medium.com/@CharlesCrouspeyre/comment-les-r%C3%A9seaux-de-neurones-%C3%A0-convolution-fonctionnent-c25041d45921) qui est une traduction de la [vidéo de Brandon Rohrer](https://www.youtube.com/watch?v=WCUNPb-5EYI) (ex-senior Datascientist chez Facebook). Il est illustré d’exemples mais n’évoque pas les mathématiques sous-jacentes.
- l’[article Medium de Youcef Messaoud](https://medium.com/smileinnovation/lstm-intelligence-artificielle-9d302c723eda) qui explique les LSTM avec des illustrations.
- les deux vidéos de Thibault Neveu : celle sur les [RNN](https://www.youtube.com/watch?v=EL439RMv3Xc) et celle sur les [LSTM](https://www.youtube.com/watch?v=3xgYxrNyE54).
- la [page Wikipédia](https://fr.wikipedia.org/wiki/R%C3%A9seau_de_neurones_r%C3%A9currents) consacré au sujet (explication courte avec les formules mathématiques).
<br><br>

Une vidéo réalisée par un doctorant dans le domaine qui récapitule l’ensemble de l’article (principes généraux mais aussi mathématiques) est disponible un peu plus loin dans la conclusion.
<br><br><br>



# <span style="color: #FF0000"> **ELMo (l’importance du contexte)** </span>

Pour apporter un peu de valeur ajoutée par rapport aux articles cités dans la partie précédente, j’évoque dans celle-ci le modèle **ELMo** (**Embeddings from Language Models**) qui est basé sur un LSTM bidirectionnel.
Pour cette partie, je me base sur un article du blog de Jay Alammar : [The illustrated BERT, ELMo, and co. (How NLP cracked transfer learning)](https://jalammar.github.io/illustrated-bert/). 
Entrons dans le vif du sujet. 
<br>

Si par exemple nous utilisons la représentation GloVe du mot « stick », alors ce mot est représenté par un unique vecteur, peu importe le contexte.

« Attendez une minute », disent plusieurs chercheurs ([Peters et. al., 2017](https://arxiv.org/abs/1705.00108), [McCann et. al., 2017](https://arxiv.org/abs/1708.00107), et encore une fois [Peters et. al., 2018](https://arxiv.org/pdf/1802.05365.pdf) dans le papier d'ELMo), « stick » a plusieurs sens selon la manière dont où il est utilisé (cf. toutes les définitions de traduction proposées par le [Larousse](https://www.larousse.fr/dictionnaires/anglais-francais/stick/614911)). Pourquoi ne pas lui donner un embedding basé sur le contexte dans lequel il est utilisé ? A la fois pour capturer le sens du mot dans ce contexte ainsi que d’autres informations contextuelles. C’est ainsi qu’ont vu le jour les **contextualized word-embeddings**.

Au lieu d’utiliser un embedding fixe pour chaque mot, ELMo examine l’ensemble de la phrase avant d’assigner une embedding à chaque mot qu’elle contient. Il utilise un LSTM bidirectionnel formé sur une tâche spécifique pour pouvoir créer ces embedding.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/RNN-LSTM-GRU-ELMO/elmo-word-embedding.png">
</figure>                                                                                                                              
</center>


ELMo a constitué un pas important vers le pré-entraînement dans le contexte du NLP. En effet, nous pouvons l’entraîner sur un ensemble massif de données dans la langue de notre ensemble de données, et ensuite nous pouvons l’utiliser comme un composant dans d’autres modèles qui ont besoin de traiter le langage.

Plus précisément, ELMo est entraîné à prédire le mot suivant dans une séquence de mots – une tâche appelée modélisation du langage (Language Modeling). C’est pratique car nous disposons d’une grande quantité de données textuelles dont un tel modèle peut s’inspirer sans avoir besoin de labellisation.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/RNN-LSTM-GRU-ELMO/Bert-language-modeling.png">
  <figcaption> Une étape dans le processus de pré-entraînement d’ELMo : ayant « Let’s stick to » comme entrée, prédisons le mot suivant le plus probable. Lorsqu’il est formé sur un grand ensemble de données, le modèle commence à se familiariser avec les modèles linguistiques. Il est peu probable qu’il puisse deviner avec précision le mot suivant dans cet exemple. De façon plus réaliste, après un mot comme « hang », il attribuera une probabilité plus élevée à un mot comme « out » (pour épeler « hang out ») qu’à « camera ». </figcaption>
</figure>                                                                                                                              
</center>


ELMo va même plus loin et forme un LSTM bidirectionnel – de sorte que son modèle linguistique n’a pas seulement le sens du mot suivant, mais aussi du mot précédent.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/RNN-LSTM-GRU-ELMO/elmo-forward-backward-language-model-embedding.png">
</figure>                                                                                                                              
</center>


ELMo propose l’embedding contextualisé en regroupant les états cachés (et l’embedding initial) d’une certaine manière (concaténation suivie d’une sommation pondérée).
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/RNN-LSTM-GRU-ELMO/elmo-embedding.png">
</figure>                                                                                                                              
</center>
<br><br>

En mai 2020, [Ortiz Suárez et al.](https://www.aclweb.org/anthology/2020.acl-main.156.pdf) ont dévoilé six ELMo différents entraînés respectivement sur du français, du bulgare, du catalan, du danois, du finnois et de l'indonésien. Ces modèles sont disponibles à la [page suivante](https://oscar-corpus.com/) dans la partie "Models".
<br><br><br>


# <span style="color: #FF0000"> **Conclusion** </span>

Si vous préférez regarder des vidéos plutôt que de lire, je vous conseil très fortement de regarder celle effectué par [César Laurent](https://mila.quebec/personne/cesar-laurent/).
Il résume en un peu plus d’une heure tout ce qu’il faut savoir sur le sujet. 
César Laurent étant doctorant au MILA sous la co-direction de [Pascal Vincent](https://mila.quebec/personne/pascal-vincent/) et [Yoshua Bengio](https://mila.quebec/personne/bengio-yoshua/) : [https://www.youtube.com/watch?v=dOpgDv88UOo](https://www.youtube.com/watch?v=dOpgDv88UOo) 
<br><br>

Enfin, même si ces techniques ne sont plus à la mode dans la communauté NLP depuis environ deux ans du fait de l’apparition des différentes architectures liées au Transformer, il ne faut pas les enterrer.
En effet, Stephen Merity a proposé en novembre 2019 une nouvelle architecture intitulée SHA-RNN pour Single Headed Attention RNN (la [publication](https://arxiv.org/abs/1911.11423), le [GitHub](https://github.com/Smerity/sha-rnn) du code).

Le SHA-RNN permet d’obtenir de bons résultats avec une optimisation hyper-paramétrique quasi inexistante. 
Même si cette architecture ne s’impose pas, elle montre que la différence entre les performances de son modèle et les modèles populaires du moment n’est pas aussi claire qu’on aurait pu le supposer.
Elle pourrait servir de base à de futurs modèles.
<br><br><br>



# <span style="color: #FF0000"> **Citation** <span>
Si vous venez à utiliser des éléments de cet article, veillez s'il vous plait en créditer les auteurs en utilisant par exemple comme suit :<br>
“*Les RNN, les LSTM, les GRU et ELMO* par Loïck BOURDOIS (https://lbourdois.github.io/blog/nlp/RNN-LSTM-GRU-ELMO/), 2019”<br>
Merci :)
