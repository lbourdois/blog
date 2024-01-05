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
    teaser : "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/RNN-LSTM-GRU-ELMO/GRU%20architechture.png"
author_profile: false
sidebar:
    nav: sidebar-sample   
classes: wide
---


# <span style="color: #FF0000"> **Avant-propos** </span>
 
Les techniques évoquées dans cet article apparaissent maintenant comme anciennes par rapport aux diverses architectures basées sur le *transformer* qui sont très populaires depuis fin 2017/ début 2018.
Ainsi beaucoup de documentation est déjà disponible en français à leur sujet.

Je ne compte donc pas faire un article extrêmement détaillé qui ferait doublon par rapport à ce qui existe a déjà été fait. 
Je me contente d’énoncer les grandes idées et vous renvoie vers d’autres articles / vidéos pour plus de détails (formules mathématiques, exemples d’applications, etc.).
<br><br><br>



# <span style="color: #FF0000"> **Les RNN, les LSTM et les GRU** </span>
Les **RNNs** (***recurrent neural network*** ou réseaux de neurones récurrents en français) sont des réseaux de neurones qui ont jusqu’à encore 2017/2018, été majoritairement utilisé dans le cadre de problème de traitement du langage naturel.
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



Les [**LSTMs** (***Long Short-Term Memory***)](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory) de Hochreiter et Schmidhuber proposent une solution à ce problème en prenant en compte un vecteur mémoire via un système de portes (*gates*) et d’états.
Les portes sont au nombre de 3 et les états au nombre de 2 :
- *Forget Gate* (capacité à oublier de l’information, quand celle-ci est inutile)
- *Input Gate* (capacité à prendre en compte de nouvelles informations utiles)
- *Output Gate* (quel est l’état de la cellule à l’instant t sachant la *forget gate* et la *input gate*)
- *Hidden state* (état caché)
- *Cell state* (état de la cellule) 

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/RNN-LSTM-GRU-ELMO/LSTM%20architechture.png">
  <figcaption>Image provenant de l’article Medium de Shi Yan (https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714) que nous avons traduit et légèrement modifiée
</figcaption>
</figure>                                                                                                                              
</center>
<br><br>



Une variante des LSTMs sont les [**GRUs** (***Gated Recurrent Unit***)](https://arxiv.org/pdf/1406.1078v3.pdf) développées par Cho et al. 
Cette structure est plus simple que les LSTMs au sens où moins de paramètres entrent en jeu.
Le nombre de portes passe à 2 et celui d’état à 1 :
- *Reset gate* (porte de réinitialisation)
- *Update gate* (porte de mise à jour)
- *Cell state* (état de la cellule)


<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/RNN-LSTM-GRU-ELMO/GRU%20architechture.png">
  <figcaption>Image que nous avons conçu en reprenant la charte graphique adopté par Shi Yan dans son article</figcaption>
</figure>                                                                                                                              
</center>
<br>

En pratique, les GRUs et les LSTMs permettent d’obtenir des résultats comparables.
L’intérêt des GRUs par rapport aux LSTMs étant le temps d’exécution qui est plus rapide puisque moins de paramètres doivent être calculés.

Pour plus de détails, je vous invite à lire ou visionner selon votre choix les sources suivantes :
- l’[article Medium de Charles Crouspeyre](https://medium.com/@CharlesCrouspeyre/comment-les-r%C3%A9seaux-de-neurones-%C3%A0-convolution-fonctionnent-c25041d45921) qui est une traduction de la [vidéo de Brandon Rohrer](https://www.youtube.com/watch?v=WCUNPb-5EYI) (ex-senior Datascientist chez Facebook). Il est illustré d’exemples mais n’évoque pas les mathématiques sous-jacentes.
- l’[article Medium de Youcef Messaoud](https://medium.com/smileinnovation/lstm-intelligence-artificielle-9d302c723eda) qui explique les LSTMs avec des illustrations.
- les deux vidéos de Thibault Neveu : celle sur les [RNNs](https://www.youtube.com/watch?v=EL439RMv3Xc) et celle sur les [LSTMs](https://www.youtube.com/watch?v=3xgYxrNyE54).
- la [page Wikipédia](https://fr.wikipedia.org/wiki/R%C3%A9seau_de_neurones_r%C3%A9currents) consacré au sujet (explication courte avec les formules mathématiques).
<br><br>

Une vidéo réalisée par un doctorant dans le domaine qui récapitule l’ensemble de l’article (principes généraux mais aussi mathématiques) est disponible un peu plus loin dans la conclusion.
<br><br><br>



# <span style="color: #FF0000"> **ELMo (l’importance du contexte)** </span>

Pour apporter un peu de valeur ajoutée par rapport aux articles cités dans la partie précédente, j’évoque dans celle-ci le modèle **ELMo** (**Embeddings from Language Models**) qui est basé sur une LSTM bidirectionnelle.
Pour cette partie, je me base sur un article du blog de Jay Alammar : [The illustrated BERT, ELMo, and co. (How NLP cracked transfer learning)](https://jalammar.github.io/illustrated-bert/). 
Entrons dans le vif du sujet. 
<br>

Si par exemple nous utilisons la représentation GloVe du mot « *stick* », alors ce mot est représenté par un unique vecteur, peu importe le contexte.

« Attendez une minute ! » disent plusieurs chercheurs ([Peters et. al., 2017](https://arxiv.org/abs/1705.00108), [McCann et. al., 2017](https://arxiv.org/abs/1708.00107), et encore une fois [Peters et. al., 2018](https://arxiv.org/pdf/1802.05365.pdf) dans le papier d'ELMo), « stick » a plusieurs sens selon la manière dont où il est utilisé (cf. toutes les définitions de traduction proposées par le [Larousse](https://www.larousse.fr/dictionnaires/anglais-francais/stick/614911)). Pourquoi ne pas lui donner un enchâssement basé sur le contexte dans lequel il est utilisé ? A la fois pour capturer le sens du mot dans ce contexte ainsi que d’autres informations contextuelles. C’est ainsi qu’ont vu le jour les **contextualized word-embeddings** (enchâssement de mot contextualisés).

Au lieu d’utiliser un enchâssement fixe pour chaque mot, ELMo examine l’ensemble de la phrase avant d’assigner un enchâssement à chaque mot qu’elle contient. Il utilise une LSTM bidirectionnelle entraînée sur une tâche spécifique pour pouvoir créer ces enchâssements.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/RNN-LSTM-GRU-ELMO/elmo-word-embedding.png">
</figure>                                                                                                                              
</center>


ELMo a constitué un pas important vers le pré-entraînement dans le contexte du NLP. En effet, nous pouvons l’entraîner sur un ensemble massif de données dans la langue de notre choix, et ensuite nous pouvons l’utiliser comme un composant dans d’autres modèles qui ont besoin de traiter le langage.

Plus précisément, ELMo est entraîné à prédire le mot suivant dans une séquence de mots, une tâche appelée modélisation du langage (*language modeling*). C’est pratique car nous disposons d’une grande quantité de données textuelles dont un tel modèle peut s’inspirer sans avoir besoin de labellisation.

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/RNN-LSTM-GRU-ELMO/Bert-language-modeling.png">
  <figcaption> Une étape dans le processus de pré-entraînement d’ELMo : ayant « Let’s stick to » comme entrée, prédisons le mot suivant le plus probable. Lorsqu’il est entraîné sur un grand jeu de données, le modèle commence à se familiariser avec les modèles linguistiques. Il est peu probable qu’il puisse deviner avec précision le mot suivant dans cet exemple. De façon plus réaliste, après un mot comme « hang », il attribuera une probabilité plus élevée à un mot comme « out » (pour épeler « hang out ») qu’à « camera ». </figcaption>
</figure>                                                                                                                              
</center>


ELMo va même plus loin avec sa LSTM bidirectionnelle puisque son modèle de langage n’a pas seulement le sens du mot suivant, mais aussi du mot précédent.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/RNN-LSTM-GRU-ELMO/elmo-forward-backward-language-model-embedding.png">
</figure>                                                                                                                              
</center>


ELMo propose l’enchâssement contextualisé en regroupant les états cachés (et l’enchâssement initial) d’une certaine manière (concaténation suivie d’une sommation pondérée).
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/RNN-LSTM-GRU-ELMO/elmo-embedding.png">
</figure>                                                                                                                              
</center>
<br><br>

En mai 2020, [Ortiz Suárez et al.](https://www.aclweb.org/anthology/2020.acl-main.156.pdf) ont dévoilé six ELMo différents entraînés respectivement sur du français, du bulgare, du catalan, du danois, du finnois et de l'indonésien. Ces modèles sont disponibles à la [page suivante](https://oscar-corpus.com/) dans la partie « Models ».
<br><br><br>


# <span style="color: #FF0000"> **Conclusion** </span>

Si vous préférez regarder des vidéos plutôt que de lire, je vous conseille très fortement de regarder celle effectué par [César Laurent](https://mila.quebec/personne/cesar-laurent/).
Il résume en un peu plus d’une heure tout ce qu’il faut savoir sur le sujet. 
César Laurent étant doctorant au MILA sous la co-direction de [Pascal Vincent](https://mila.quebec/personne/pascal-vincent/) et [Yoshua Bengio](https://mila.quebec/personne/bengio-yoshua/) : [https://www.youtube.com/watch?v=dOpgDv88UOo](https://www.youtube.com/watch?v=dOpgDv88UOo) 
<br><br>

Enfin, même si ces techniques ne sont plus à la mode dans la communauté NLP depuis 2017/2018 du fait de l’apparition des différentes architectures liées au *transformer*, il ne faut pas les enterrer.
En effet, la recherche avance très vite. Par exemple en 2020, Katharopoulos et al. ont montré dans leur papier [Transformers are RNNs : Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/pdf/2006.16236.pdf) que les *transformers* pouvaient être considérés comme des RNNs.

<!--
Stephen Merity a proposé en novembre 2019 une nouvelle architecture intitulée SHA-RNN pour Single Headed Attention RNN (la [publication](https://arxiv.org/abs/1911.11423), le [GitHub](https://github.com/Smerity/sha-rnn) du code).
Le SHA-RNN permet d’obtenir de bons résultats avec une optimisation hyper-paramétrique quasi inexistante. 
Même si cette architecture ne s’impose pas, elle montre que la différence entre les performances de son modèle et les modèles populaires du moment n’est pas aussi claire qu’on aurait pu le supposer.
Elle pourrait servir de base à de futurs modèles.
-->
<br><br><br>


# <span style="color: #FF0000"> **Références** </span> 
- [Long Short-term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory) de Hochreiter et Schmidhuber (1997)
- [Learning Phrase Representations using RNN Encoder–Decoderfor Statistical Machine Translation](https://arxiv.org/pdf/1406.1078v3.pdf) de Cho et al. (2014)
- [Transformers are RNNs:Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/pdf/2006.16236.pdf) de Katharopoulos et al. (2020)
- [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](https://jalammar.github.io/illustrated-bert/) de Jay Alammar (2018)
- [Comment les Réseaux de neurones récurrents et Long Short-Term Memory fonctionnent](https://medium.com/@CharlesCrouspeyre/comment-les-r%C3%A9seaux-de-neurones-%C3%A0-convolution-fonctionnent-c25041d45921) de Charles Crouspeyre (2017)
- [EH2018-10 - Modèle : Réseaux récurrents (Partie 1)](https://www.youtube.com/watch?v=dOpgDv88UOo) de César Laurent (2018)
- [Learned in Translation: Contextualized Word Vectors](https://arxiv.org/abs/1708.00107) de McCann et al. (2017)
- [Single Headed Attention RNN: Stop Thinking With Your Head](https://arxiv.org/abs/1911.11423) de Stephen Merity (2019)
- [LSTM, Intelligence artificielle sur des données chronologiques](https://medium.com/smileinnovation/lstm-intelligence-artificielle-9d302c723eda ) de Youcef Messaoud (2018)
- [Comprendre les LSTM - Réseaux de neurones récurrents](https://www.youtube.com/watch?v=3xgYxrNyE54) de Thibault Neveu (2019)
- [Comprendre les réseaux de neurones récurrents (RNN)](https://www.youtube.com/watch?v=EL439RMv3Xc) de Thibault Neveu (2019)
- [A Monolingual Approach to Contextualized Word Embeddingsfor Mid-Resource Languages](https://www.aclweb.org/anthology/2020.acl-main.156.pdf) de Ortiz Suárez et al. (2020)
- [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf) de Peters et al. (2018)
- [Semi-supervised sequence tagging with bidirectional language models](https://arxiv.org/abs/1705.00108) de Peters et al. (2017)
- [Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM)](https://www.youtube.com/watch?v=WCUNPb-5EYI) de Brandon Rohrer (2017)
- [Understanding LSTM and its diagrams](https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714) de Shi Yan (2016)
<br><br><br>


# <span style="color: #FF0000"> **Citation** <span>
> @inproceedings{rnn_blog_post,  
  author    = {Loïck BOURDOIS},  
  title     = {Les RNN, les LSTM, les GRU et ELMO},  
  year      = {2019},  
  url = {https://lbourdois.github.io/blog/nlp/RNN-LSTM-GRU-ELMO/}  
}
