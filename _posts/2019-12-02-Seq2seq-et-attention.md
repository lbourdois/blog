---
title: "LE SEQ2SEQ ET LE PROCESSUS D’ATTENTION"
categories:
  - NLP
tags:
  - Modèles séquenciels expliqués en français
  - Processus d'attention en NLP
  - Seq2seq french
  - Seq2seq français
excerpt : "NLP - Illustration du principe de séquence vers séquence et du processus d'attention en traitement du langage naturel"
header :
    overlay_image: "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/NLP_radom_blog.png"
    teaser : "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Seq2seq-attention/seq2seq_1.mp4"
toc: true
toc_sticky: true
author_profile: false
sidebar:
    nav: sidebar-sample

---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


# <span style="color: #FF0000"> **Avant-propos** </span>
 
Cet article est une traduction de l’article de Jay Alammar : [Visualizing neural machine translation mechanics of seq2seq models with attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/).<br>
Merci à lui de m'avoir autorisé à effectuer cette traduction.
<br><br><br>


# <span style="color: #FF0000"> **1. Introduction** </span>

Les modèles de séquence à séquence sont des modèles d'apprentissage profond qui ont connu beaucoup de succès dans des tâches comme la traduction automatique, le résumé de texte et le sous-titrage d’images.
Google Translate a commencé à utiliser un tel modèle à partir de fin 2016.
Ces modèles sont expliqués dans les deux études pionnières ([Sutskever et al.](https://papers.nips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf), 2014, [Cho et al.](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf), 2014).

Pour bien comprendre le modèle et le mettre en œuvre, il faut démêler une série de concepts qui s’imbriquent les uns dans les autres.
L'objectif de cet article est qu'il puisse être un compagnon utile à la lecture des documents mentionnés ci-dessus (et des documents sur l’attention exposés plus loin dans l’article).
<br><br>


Un modèle de séquence à séquence est un modèle qui prend une séquence d’éléments (mots, lettres, caractéristiques d’une image, etc.) et en sort une autre séquence. 
Un modèle entraîné fonctionnerait comme ça :
<video width="100%" height="auto" loop autoplay controls>
  <source src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Seq2seq-attention/seq2seq_1.mp4" type="video/mp4">
</video>
<br>

En traduction automatique, une séquence est une série de mots, traités les uns après les autres.
Le résultat est donc aussi une série de mots :
<video width="100%" height="auto" loop autoplay controls>
  <source src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Seq2seq-attention/seq2seq_2.mp4" type="video/mp4">
</video>
<br><br><br>



# <span style="color: #FF0000"> **2. Regardons sous le capot** </span>
Sous le capot, le modèle est composé d’un encodeur et d’un décodeur.

L’encodeur traite chaque élément de la séquence d’entrée. 
Il compile les informations qu’il capture dans un vecteur (appelé *context*).
Après avoir traité toute la séquence d’entrée, l’encodeur envoie le contexte au décodeur, qui commence à produire la séquence de sortie item par item. 
<video width="100%" height="auto" loop autoplay controls>
  <source src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Seq2seq-attention/seq2seq_3.mp4" type="video/mp4">
</video>
<br>

La même logique est appliquée dans le cas de la traduction automatique.
<video width="100%" height="auto" loop autoplay controls>
  <source src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Seq2seq-attention/seq2seq_4.mp4" type="video/mp4">
</video>
<br>

Le contexte est un vecteur (un tableau de nombres essentiellement) dans le cas de la traduction automatique. 
L’encodeur et le décodeur ont tendance à être tous deux des réseaux de neurones récurrents.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Seq2seq-attention/context.png">
  <figcaption>
  Le context est un vecteur de flottants. Plus loin, nous allons visualiser les vecteurs en couleur en assignant des couleurs plus claires aux cellules avec des valeurs plus élevées.
  </figcaption>
</figure>
</center>
<br>

Vous pouvez définir la taille du vecteur de *context* lorsque vous configurez votre modèle.
C’est essentiellement le nombre d’unités cachées dans l’encodeur RNN.
Ces visualisations montrent un vecteur de taille 4, mais dans les applications du monde réel le vecteur de contexte serait de taille 256, 512, ou 1024.
<br><br><br>


Par conception, un RNN prend deux entrées à chaque pas de temps : une entrée (dans le cas de l’encodeur, un mot de la phrase d’entrée), et un état caché.
Le mot, cependant, doit être représenté par un vecteur.
Pour transformer un mot en vecteur, nous nous tournons vers les méthodes d'enchâssement de mots  (cf. [l'article sur le word embedding du blog](https://lbourdois.github.io/blog/nlp/word_embedding/)).
Ils transforment les mots en espaces vectoriels qui capturent une grande partie de l’information sémantique des mots (ex : roi – homme + femme = reine).
<center>
<figure class="image">
   <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Seq2seq-attention/embedding.png">
  <figcaption>
  Nous devons transformer les mots d’entrée en vecteurs avant de les traiter. Cette transformation se fait à l’aide d’un algorithme de word embedding. Nous pouvons utiliser des enchâssements pré-entrainés ou entrainer nos propres enchâssements sur notre jeu de données.
  Typiquement un enchâssement est de taille 200 ou 300. Nous montrons ici un vecteur de taille quatre pour plus de simplicité.
  </figcaption>
</figure>
</center>
<br>

Maintenant que nous avons présenté nos principaux vecteurs, récapitulons la mécanique d’un RNN et établissons un visuel pour décrire ces modèles :
<video width="100%" height="auto" loop autoplay controls>
  <source src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Seq2seq-attention/RNN_1.mp4" type="video/mp4">
</video>
<center>
<figure class="image">
  <figcaption>
L’étape RNN suivante prend le deuxième vecteur d’entrée et l’état caché #1 pour créer la sortie de cette étape.
</figcaption>
</figure>
</center>
<br>


Dans la visualisation suivante, chaque « impulsion » pour l’encodeur et le décodeur représente le traitement des entrées par le RNN et la génération de la sortie.
Comme l’encodeur et le décodeur sont tous deux des RNNs, chaque fois que l’une des deux étapes effectue un certain traitement, elle met à jour son état caché en fonction de ses entrées et des entrées précédentes qu’elle a vues.

Regardons les états cachés de l’encodeur.
Remarquez comment le dernier état caché est en fait le contexte que nous transmettons au décodeur :
<video width="100%" height="auto" loop autoplay controls>
  <source src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Seq2seq-attention/seq2seq_5.mp4" type="video/mp4">
</video>
<br>


Le décodeur maintient également un état caché qu’il passe d’une étape à l’autre.
Nous ne l’avons tout simplement pas visualisé dans ce graphique car nous nous préoccupons des principales parties du modèle pour le moment.

Voyons maintenant une autre façon de visualiser un modèle de séquence à séquence.
Cette animation facilitera la compréhension des graphiques statiques qui décrivent ces modèles. 
C’est ce qu’on appelle une vue « déroulée » où au lieu de montrer le décodeur, on en montre une copie pour chaque pas de temps.
De cette façon, nous pouvons examiner les entrées et les sorties de chaque étape.
<video width="100%" height="auto" loop autoplay controls>
  <source src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Seq2seq-attention/seq2seq_6.mp4" type="video/mp4">
</video>
<br><br><br>



# <span style="color: #FF0000"> **3. L'attention** </span>
Le vecteur de contexte s’est avéré être un goulot d’étranglement pour ces types de modèles.
Il était donc difficile pour les modèles de composer avec de longues phrases.
Une solution a été proposée dans [Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473) et [Luong et al., 2015](https://arxiv.org/abs/1508.04025).
Ces articles introduisirent et affinèrent une technique appelée « attention », qui améliora considérablement la qualité des systèmes de traduction automatique.
L’attention permet au modèle de se concentrer sur les parties pertinentes de la séquence d’entrée si nécessaire. 
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Seq2seq-attention/attention.png">
</figure>
</center>
<br>

A l’étape 7, le mécanisme d’attention permet au décodeur de se concentrer sur le mot « étudiant » avant de générer la traduction anglaise.
Cette capacité d’amplifier le signal de la partie pertinente de la séquence d’entrée permet aux modèles d’attention de produire de meilleurs résultats que les modèles sans attention.

Continuons à regarder les modèles d’attention à ce haut niveau d’abstraction.
Un modèle d’attention diffère d’un modèle de sequence à sequence classique de deux façons principales :

* Tout d’abord, l’encodeur transmet beaucoup plus de données au decodeur. Au lieu de passer le dernier état caché de l’étape d’encodage, l’encodeur passe tous les états cachés au decodeur :
<video width="100%" height="auto" loop autoplay controls>
  <source src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Seq2seq-attention/seq2seq_7.mp4" type="video/mp4">
</video>
*    Deuxièmement, un décodeur d’attention fait une étape supplémentaire avant de produire sa sortie. Afin de se concentrer sur les parties de l’entrée qui sont pertinentes, le décodeur fait ce qui suit :

  *    Il regarde l’ensemble des états cachés de l’encodeur qu’il a reçu (chaque état caché de l’encodeur  est le plus souvent associé à un certain mot dans la phrase d’entrée).
  *    Il donne un score à chaque état caché (on passe l’étape de comment le *scoring* se fait pour le moment)
  *    Il multiplie chaque état caché par son score attribué via softmax (amplifiant ainsi les états cachés avec des scores élevés, et noyant les états cachés avec des scores faibles) 
  *    
<video width="100%" height="auto" loop autoplay controls>
  <source src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Seq2seq-attention/attention_process.mp4" type="video/mp4">
</video>
<br>
Le « scorage » se fait à chaque pas de temps (nouveau mot) du côté du décodeur.

<br>
Regardons maintenant comment fonctionne le processus de l’attention et regroupons le tout dans la visualisation qui suit :

1) Le décodeur d’attention prend en entrée l’enchâssement du *token* <END> ainsi qu’un état caché initial (\(h_{init}\)).<br>
2) Le RNN traite ces entrées, produisant une sortie et un nouveau vecteur d’état caché (\(h_{4}\)). La sortie est supprimée.<br>
3) L’étape d’attention : nous utilisons les états cachés de l’encoder et le vecteur (\(h_{4}\)) pour calculer un vecteur de contexte (\(C_{4}\)) pour cette étape.<br>
4) Nous concaténons \(h_{4}\) et \(C_{4}\) en un seul vecteur.<br>
5) Nous faisons passer ce vecteur par un réseau neuronal *feedforward* (un réseau entraîné conjointement avec le modèle).<br>
6) La sortie du réseau *feedforward* indique le mot de sortie de ce pas de temps (= la traduction du mot en entrée).<br>
7) On répète les étapes précédentes pour chaque mots. L’état caché fournit en entrée n’étant plus l’initial mais celui de la couche précédente (\(h_{4}\) dans notre exemple) et l’enchâssement n’est plus celui du *token* <END> mais celui obtenu pour la traduction du premier mot (sortie de l’étape 6).<br> 
<video width="100%" height="auto" loop autoplay controls>
  <source src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Seq2seq-attention/attention_tensor_dance.mp4" type="video/mp4">
</video>


Une autre façon de voir à quelle partie de la phrase d’entrée nous prêtons attention à chaque étape de décodage : 
<video width="100%" height="auto" loop autoplay controls>
  <source src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Seq2seq-attention/seq2seq_9.mp4" type="video/mp4">
</video>


Notons que le modèle n’associe pas seulement le premier mot de la sortie avec le premier mot de l’entrée.
En fait, il a appris pendant la phase d’entrainement la façon dont sont liés les mots dans cette paire de langues (le français et l’anglais dans notre exemple).
Un exemple de ce mécanisme est donné dans l’article de [Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473) :
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Seq2seq-attention/attention_sentence.png">
  <figcaption>
Nous pouvons voir comment le modèle a porté son attention pour la sortie de « *European Economic Area* ». En français, l’ordre de ces mots est inversé (« zone économique européenne ») par rapport à l’anglais.
Tous les autres mots de la phrase sont dans le même ordre.
</figcaption>
</figure>
</center>

Un tutoriel sur l’implémentation en Python (TensorFlow) d’un tel modèle est disponible sur ce Github : [https://github.com/tensorflow/nmt](https://github.com/tensorflow/nmt).
<br><br><br>



# <span style="color: #FF0000"> **Conclusion** </span>


Cet article est une introduction au mécanisme d’attention. 
Il se veut assez général car il existe en réalité plusieurs variantes de ce mécanisme. 
Ainsi si vous n’avez pas forcément tout compris, nous revenons plus en détails sur ces variantes dans les articles consacrés au *transformer*, au modèle BERT et au modèle GPT-2.

Si vous préférez les vidéos aux textes, vous pouvez regarder la vidéo de présentation du phénomène d’attention par [César Laurent](https://mila.quebec/personne/cesar-laurent/) (doctorant au MILA sous la co-direction de [Pascal Vincent](https://mila.quebec/personne/pascal-vincent/) et [Yoshua Bengio](https://mila.quebec/personne/bengio-yoshua/)) : [https://www.youtube.com/watch?v=stD_mY2OYj4](https://www.youtube.com/watch?v=stD_mY2OYj4).<br>
Il donne des exemples d’application tel que la traduction ou encore la génération de légende à partir d’un texte ou d’une vidéo.
<br><br><br>


# <span style="color: #FF0000"> **Références** </span> 
- [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/ ) de Jay Alammar (2018)
- [Neural Machine Translation by Jointly Learning to Align and Translate]( https://arxiv.org/pdf/1409.0473.pdf) de Bahdanau et al. (2014)
- [EH2018-11 - Modèle : Réseaux récurrents (Partie 2)](https://www.youtube.com/watch?v=stD_mY2OYj4 ) de César Laurent (2018) 
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf) de Luong et al. (2015) 
- [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt) de Luong et al. (2017) 
<br><br><br>


# <span style="color: #FF0000"> **Citation** <span>
Si vous venez à utiliser des éléments de cet article, veillez s'il vous plait en créditer les auteurs en utilisant par exemple comme suit :<br>
“*Les architectures issues du Transformer* par Loïck BOURDOIS (https://lbourdois.github.io/blog/nlp/Les-architectures-transformers/), 2020”<br>
Merci :)
