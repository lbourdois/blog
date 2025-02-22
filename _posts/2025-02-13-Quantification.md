---
title: "UN GUIDE VISUEL SUR LA QUANTIFICATION"
tags:
  - NLP
  - Quantization
  - Quantification
  - FP32
  - BF16
  - FP16
  - INT8
excerpt : "Divers - Démystifier la compression des grands modèles de langage"
header :
    overlay_color: "#1C2A4D"
    teaser : "https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_1.png"
author_profile: false
sidebar:
    nav: sidebar-misc
classes: wide
---

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script> 

# <span style="color: #FF0000"> **Avant-propos** </span>
Cet article est une traduction de celui de Maarten GROOTENDORST : [A Visual Guide to Quantization](https://substack.com/home/post/p-145531349).<br>
Maarten est co-auteur du livre [*Hands-On Large Language Models*](https://www.llm-book.com/).
<br><br><br>

# <span style="color: #FF0000"> **Introduction** </span>
Comme leur nom l'indique, les grands modèles de langage (LLM pour *Large Language Models*) sont souvent trop volumineux pour être exécutés sur du matériel grand public. Ces modèles peuvent dépasser des milliards de paramètres et ont généralement besoin de GPU avec de grandes quantités de VRAM pour l'inférence.  
À ce titre, de plus en plus de recherches ont été axées sur la réduction de la taille de ces modèles grâce à une amélioration de l‘entraînement,  les adapteurs, etc.  
L'une des principales techniques dans ce domaine s'appelle la quantification.

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_1.png">
</figure>
</center>

Dans cet article, nous allons introduire le concept de la quantification dans le contexte de la modélisation du langage et explorer les concepts un par un pour en comprendre l’intuition. Nous explorerons diverses méthodologies, des cas d'utilisation et les principes qui sous-tendent la quantification, via plus de $$50$$ visuels.
<br><br><br><br>

# <span style="color: #FF0000"> **Partie 1 : Le « problème » des LLM** </span>
Les LLM tirent leur nom du nombre de paramètres qu'ils contiennent. De nos jours, ces modèles ont généralement des milliards de paramètres (principalement des poids) qui peuvent être assez coûteux à stocker.  
Durant l'inférence, les activations sont créées en tant que produit de l'entrée et des poids, qui peuvent également être assez importants.

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_2.png">
</figure>
</center>

Par conséquent, nous aimerions représenter des milliards de valeurs aussi efficacement que possible, en minimisant la quantité d'espace dont nous avons besoin pour stocker une valeur donnée.  
Commençons en explorant comment les valeurs numériques sont représentées avant de voir comment les optimiser. 
<br><br>

## <span style="color: #FFBF00"> **Comment représenter des valeurs numériques** </span>
Une valeur donnée est souvent représentée par un nombre à [virgule flottante]( https://fr.wikipedia.org/wiki/Virgule_flottante) (*float* en anglais) : un nombre positif ou négatif avec une virgule.  
Ces valeurs sont représentées par des « bits », ou chiffres binaires. La [norme IEEE-754](https://fr.wikipedia.org/wiki/IEEE_754) décrit comment les bits peuvent représenter la valeur via trois fonctions : le *signe*, l'*[exposant]( https://fr.wikipedia.org/wiki/Exposant_(math%C3%A9matiques))* et la *fraction* (ou [*mantisse*]( https://fr.wikipedia.org/wiki/Mantisse)).

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_3.png">
</figure>
</center>

Ensemble, ces trois aspects peuvent être utilisés pour calculer une valeur à partir d'un certain ensemble de valeurs binaires :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_4.png">
</figure>
</center>

Plus nous utilisons de bits pour représenter une valeur, plus elle est généralement précise :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_5.png">
</figure>
</center>
<br>

## <span style="color: #FFBF00"> **Contraintes de mémoire** </span>
Plus nous avons de bits disponibles, plus la plage de valeurs qui peut être représentée est grande.

<center>
<table>
    <tr>
        <td><b>Original</b></td>
        <td><b>75505.0</b></td>
        <td><b>1.8e-42</b></td>
    </tr>
    <tr>
        <td>64-bits</td>
        <td>75505.0</td>
        <td>1.8e-42</td>
    </tr>
    <tr>
        <td>32-bits</td>
        <td>75505.0</td>
        <td>1.80066e-42</td>
    </tr>
    <tr>
        <td>16-bits</td>
        <td>inf</td>
        <td>0.0</td>
    </tr>
</table>
</center>

<br>
L'intervalle de nombres qu'une représentation donnée peut prendre est appelé la **plage dynamique** (*dynamic range* en anglais) alors que la distance entre deux valeurs voisines est appelée la **précision**.

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_6.png">
</figure>
</center>

Une caractéristique astucieuse de ces bits est que nous pouvons calculer la quantité de mémoire dont votre machine a besoin pour stocker une valeur donnée. Comme il y a $$8$$ bits dans un octet, nous pouvons créer une formule basique pour la plupart des représentations en virgule flottante :  
mémoire = nombre de paramètres × (nombre de bits) / $$8$$.  
En pratique, c’est un peu plus complexe. La quantité de (V)RAM nécessaire pour l’inférence, dépend aussi de la taille de contexte et de l'architecture.  

Appliquons cette formule. Supposons que nous ayons un modèle de $$70$$ milliards de paramètres. La plupart des modèles sont représentés nativement avec en FP32 (souvent appelé « pleine précision » ou *full-precision*), ce qui nécessiterait $$280$$ Go de mémoire juste pour charger le modèle. En effet :  
-	**64 bits** = 70Mds × 64/8 ≈ **560** GB   
- **32 bits** = 70Mds × 32/8 ≈ **280** GB     
-	**16 bits** = 70Mds × 16/8 ≈ **140** GB  

De ce fait, c’est très intéressant de pouvoir minimiser le nombre de bits pour représenter les paramètres de votre modèle (ainsi que pendant l'entraînement !). Cependant, à mesure que la précision diminue, l'*accuracy* des modèles décroit généralement aussi.  
Nous voulons réduire le nombre de bits représentant des valeurs tout en conservant l'*accuracy*... C'est là qu'intervient la quantification !
<br><br><br>

# <span style="color: #FF0000"> **Partie 2 : Introduction à la quantification** </span>
La quantification vise à réduire la précision des paramètres d'un modèle en passant de largeurs de bits supérieures (comme la virgule flottante $$32$$ bits) à des largeurs de bits inférieures (comme les entiers $$8$$ bits).

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_7.png">
</figure>
</center>

Il y a souvent une certaine perte de précision (granularité) lors de la réduction du nombre de bits pour représenter les paramètres d'origine.  
Pour illustrer cet effet, nous pouvons prendre n'importe quelle image et n'utiliser que $$8$$ couleurs pour la représenter :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_8.jpg">
<figcaption>Image adaptée de celle de <a href="https://pixabay.com/users/slava_web-designer-39623293/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=8668140">Slava Sidorov</a></figcaption>
</figure>
</center>

Remarquez que la partie zoomée semble plus « granuleuse » que l'originale puisque nous utilisons moins de couleurs pour la représenter.  
L'objectif principal de la quantification est de réduire le nombre de bits (couleurs) nécessaires pour représenter les paramètres d'origine tout en préservant au mieux la précision des paramètres d'origine. 
<br><br>

## <span style="color: #FFBF00"> **Types de données courants** </span>
Tout d'abord, examinons les types de données courants et l'impact de leur utilisation plutôt que des représentations en $$32$$ bits (appelées FP32 pour *full-precision* soit la *pleine précision*).
<br>

### <span style="color: #51C353"> **FP16** </span>
Prenons l'exemple d'un passage de $$32$$ à $$16$$ bits (appelé FP16 ou *half precision* soit la *demi-précision*) :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_9.png">
</figure>
</center>

Remarquez que la plage de valeurs que la FP16 peut prendre est un peu plus petite que celle de la FP32. 
<br>

### <span style="color: #51C353"> **BF16** </span>
Pour obtenir une plage de valeurs similaire à celle de FP32, la précision *bfloat 16* a été introduite par Google (le « B » se réfaire au *Brain* de [Google Brain](https://fr.wikipedia.org/wiki/Google_Brain) qui a introduit la méthode) comme un type de « FP32 tronqué » :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_10.png">
</figure>
</center>

BF16 utilise la même quantité de bits que FP16, mais peut prendre une gamme plus large de valeurs et est souvent utilisé dans les applications d'apprentissage profond. 
<br>

### <span style="color: #51C353"> **INT8** </span>
Lorsque nous réduisons encore davantage le nombre de bits, nous nous rapprochons du domaine de représentations basées sur des entiers plutôt que des représentations en virgule flottante. À titre d'exemple, le passage de FP32 à INT8, qui n'a que $$8$$ bits, donne un quart du nombre de bits d'origine :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_11.png">
</figure>
</center>

Selon le matériel, les calculs basés sur des nombres entiers peuvent être plus rapides que les calculs en virgule flottante, mais ce n'est pas toujours le cas. Cependant, les calculs sont généralement plus rapides lorsque l'on utilise moins de bits.  
Pour chaque réduction de bits, une correspondance est effectuée pour « comprimer » les représentations initiales de FP32 dans les bits inférieurs.  
En pratique, nous n'avons pas besoin de faire correspondance toute la plage FP32 [$$-3.4e38,3.4e38$$] dans INT8. Nous avons simplement besoin de trouver un moyen de faire correspondre la plage de nos données (les paramètres du modèle) dans INT8.  
Les méthodes courantes de compression/correspondance sont la *quantification symétrique* et *asymétrique* qui sont des formes de correspondance linéaire.  
Explorons ces méthodes pour quantifier de FP32 à INT8.
<br><br>

## <span style="color: #FFBF00"> **Quantification symétrique** </span>
Dans la quantification symétrique, la plage de valeurs d'origine est mise en correspondance avec une plage symétrique autour de $$0$$ dans l'espace quantifié. Dans les exemples précédents, remarquez que les plages avant et après la quantification restent centrées autour de $$0$$.  
Cela signifie que la valeur $$0$$ dans l'espace à virgule flottante est aussi $$0$$ dans l'espace quantifié.

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_12.png">
</figure>
</center>

Un bon exemple d'une forme de quantification symétrique est la quantification via maximum absolu (absmax).  
Étant donné une liste de valeurs, nous prenons la valeur absolue la plus élevée (<span style="color:#02E1FF;">**α**</span>) comme plage pour effectuer les correspondances linéaires.

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_13.png">
<figcaption>Notez que la plage de valeurs [-127,127] représente la plage restreinte. La plage non restreinte est [-128,127] et dépend de la méthode de quantification.</figcaption>
</figure>
</center>

Comme il s'agit d'une correspondance linéaire centrée autour de $$0$$, la formule est simple.  
Nous calculons d'abord un facteur d'échelle <span style="color:#D79515;">$$s$$</span> en utilisant :  
• **$$b$$**, le nombre d'octets que l'on veut quantifier à ($$8$$),  
• <span style="color:#02E1FF;">**$$α$$**</span>, la valeur absolue la plus élevée,  
Ensuite, nous utilisons le <span style="color:#D79515;">**$$s$$**</span> pour quantifier l'entrée <span style="color:#8C00FF;">**$$x$$**</span> :  

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_14.png">
</figure>
</center>

En renseignant les valeurs, on obtient alors ce qui suit :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_15.png">
</figure>
</center>

Pour retrouver les valeurs FP32 originales, nous pouvons utiliser le facteur d'échelle (<span style="color:#D79515;">**$$s$$**</span>) calculé précédemment pour déquantifier les valeurs quantifiées.

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_16.png">
</figure>
</center>

L'application du processus de quantification puis de déquantification pour récupérer l'original se présente comme suit :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_17.png">
</figure>
</center>

Vous pouvez voir que certaines valeurs, telles que $$3.08$$ et $$3.02$$, sont assignées à l'INT8, c'est-à-dire $$36$$ dans le graphique. Lorsque vous déquantifiez les valeurs pour revenir à FP32, elles perdent de la précision et ne sont plus distinguables.  
C'est ce que l'on appelle souvent l'*erreur de quantification*, que l'on peut calculer en déterminant la différence entre la valeur originale et la valeur déquantifiée.

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_18.png">
</figure>
</center>

En général, plus le nombre de bits est faible, plus nous avons tendance à avoir d'erreurs de quantification.
<br><br>

## <span style="color: #FFBF00"> **Quantification asymétrique** </span>
La quantification asymétrique, en revanche, n'est pas symétrique autour de $$0$$. Au lieu de cela, elle fait correspondre les valeurs minimales (<span style="color:#17AB53;">**$$β$$**</span>) et maximales (<span style="color:#02E1FF;">**$$α$$**</span>) de la plage flottante aux valeurs minimales et maximales de la plage quantifiée.  
La méthode que nous allons explorer s'appelle la *quantification du point $$0$$*.

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_19.png">
</figure>
</center>

Vous avez remarqué que le $$0$$ a changé de position ? C'est pourquoi on parle de quantification asymétrique. Les valeurs min/max ont des distances différentes par rapport à $$0$$ dans la plage [$$-7.59,10.8$$].  
En raison de sa position décalée, nous devons calculer le $$0$$ pour la plage INT8 afin d'effectuer la correspondance linéaire. Comme précédemment, nous devons également calculer un facteur d'échelle (<span style="color:#D79515;">**$$s$$**</span>), mais en utilisant la différence de la plage INT8 à la place [$$-128,127$$].  
<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_20.png">
</figure>
</center>

Remarquez que c'est un peu plus compliqué en raison de la nécessité de calculer le point du $$0$$ (<span style="color:#1B89D8;">**$$z$$**</span>) pour la plage en INT8 afin de modifier les poids.  
Comme précédemment, remplissons la formule :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_21.png">
</figure>
</center>

Pour déquantifier de INT8 à FP32, nous devrons utiliser le facteur d'échelle (<span style="color:#D79515;">**$$s$$**</span>) et point $$0$$ (<span style="color:#1B89D8;">**$$z$$**</span>).
En dehors de cela, la déquantification est simple :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_22.png">
</figure>
</center>

Lorsque nous mettons la quantification symétrique et asymétrique côte à côte, nous pouvons rapidement voir la différence entre les méthodes :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_23.png">
</figure>
</center>
<br><br>

## <span style="color: #FFBF00"> **Elagage de la plage** </span>
Dans nos exemples précédents, nous avons exploré comment la plage de valeurs d'un vecteur donné pouvait être associée à une représentation avec moins de bits. Bien que cela permette de faire correspondre toute la gamme des valeurs vectorielles, cela présente un inconvénient majeur, à savoir les valeurs aberrantes.  
Imaginez que vous ayez un vecteur avec les valeurs suivantes :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_24.png">
</figure>
</center>

On observe qu'une valeur est beaucoup plus grande que toutes les autres et pourrait être considérée comme une valeur aberrante. Si nous devions représenter l’entièreté de la plage de ce vecteur, toutes les petites valeurs seraient représentées sur le même bit et perdraient leur facteur de différenciation :  

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_25.png">
<figcaption>Il s'agit de la méthode absmax que nous avons utilisée précédemment. Notez que le même comportement se produit avec la quantification asymétrique si nous n'appliquons pas d’élagage.</figcaption>
</figure>
</center>

Au lieu de cela, nous pouvons choisir d'élaguer certaines valeurs. L’élagage implique la définition d'une plage dynamique différente des valeurs d'origine, de sorte que toutes les valeurs aberrantes obtiennent la même valeur.  
Dans l'exemple ci-dessous, si nous devions définir manuellement la plage dynamique à [$$- 5,5$$], toutes les valeurs en dehors seront soit associées à $$-127$$, soit à $$127$$, quelle que soit leur valeur :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_26.png">
</figure>
</center>

L'avantage majeur est que l'erreur de quantification des valeurs non aberrantes est considérablement réduite. Cependant, celle des non aberrantes augmente.
<br><br>

## <span style="color: #FFBF00"> **Étalonnage** </span>
Dans l'exemple précédant,  nous avons montré une méthode naïve consistant à choisir une plage arbitraire de [$$- 5,5$$]. Le processus de sélection de cette plage est connu sous le nom d'étalonnage, où le but est de trouver une plage qui comprend les plus de valeurs possibles tout en minimisant l'erreur de quantification.  
La réalisation de cette étape n'est pas la même pour tous les types de paramètres. 
<br>

### <span style="color: #51C353"> **Poids (et biais)** </span>
Nous pouvons considérer les poids et les biais d'un LLM comme des valeurs statiques puisqu'ils sont connus avant l'exécution du modèle. Par exemple, le [fichier de ~20GB](https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main) de [Llama 3](https://arxiv.org/abs/2407.21783) de Meta (2024) est composé principalement de ces derniers.

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_27.png">
</figure>
</center>

Comme il y a beaucoup moins de biais (millions) que de poids (milliards), les biais sont souvent maintenus avec une plus grande précision (en INT16 par exemple), et l'effort principal de quantification est consacré aux poids.  
Pour les poids, qui sont statiques et connus, les techniques d'étalonnage permettant de choisir l'intervalle sont les suivantes :  
- Choix manuel d'un centile de la plage d'entrée  
- Optimiser l'erreur quadratique moyenne (MSE pour *mean squared error*) entre les poids originaux et quantifiés  
- Minimiser l'entropie (divergence de Kullback-Leibler) entre les valeurs originales et quantifiées

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_28.png">
</figure>
</center>

Choisir un centile, par exemple, conduirait à un comportement d'élagage similaire à celui que nous avons observé précédemment.

<br>
### <span style="color: #51C353"> **Activations**</span>

L'entrée qui est continuellement mise à jour tout au long du LLM est généralement appelée « activations ».

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_29.png">
</figure>
</center>

Notez que ces valeurs sont appelées activations car elles passent souvent par une fonction d'activation, comme une sigmoïde ou une ReLU.  
Contrairement aux poids, les activations varient avec chaque donnée d'entrée introduite dans le modèle pendant l'inférence, ce qui rend difficile leur quantification précise.   
Étant donné que ces valeurs sont mises à jour après chaque couche cachée, nous ne pouvons connaître leur valeur exacte qu'au moment de l'inférence, lorsque les données d'entrée passent par le modèle.

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_30.png">
</figure>
</center>

 D'une manière générale, il existe deux méthodes pour calibrer la méthode de quantification des poids et des activations :  
•	Quantification post-entraînement (PTQ pour *Post-Training Quantization*)   
•	Quantification pendant l’entraînement (QAT pour *Quantization Aware Training*) 
<br><br><br>

# <span style="color: #FF0000"> **Partie 3 : Quantification post-entraînement** </span>
L'une des techniques de quantification les plus populaires est la quantification post-entraînement (PTQ). Il s'agit de quantifier les paramètres d'un modèle (poids et activations) après l'entraînement du modèle.  
La quantification des poids est effectuée à l'aide d'une quantification symétrique ou asymétrique.  
La quantification des activations, quant à elle, nécessite l'inférence du modèle pour obtenir leur distribution potentielle puisque nous ne connaissons pas leur plage.  
Il existe deux formes de quantification des activations :  
•	Quantification dynamique   
•	Quantification statique 
<br><br>

## <span style="color: #FFBF00"> **Quantification dynamique** </span>
Une fois que les données passent par une couche cachée, ses activations sont collectées :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_31.png">
</figure>
</center>

Cette distribution des activations est ensuite utilisée pour calculer le point $$0$$ (<span style="color:#1B89D8;">**$$z$$**</span>) et le facteur d'échelle (<span style="color:#D79515;">**$$s$$**</span>) valeurs nécessaires pour quantifier la sortie :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_32.png">
</figure>
</center>

Le processus est répété chaque fois que les données passent par une nouvelle couche. Par conséquent, chaque couche a ses propres valeurs <span style="color:#D79515;">**$$s$$**</span> et <span style="color:#1B89D8;">**$$z$$**</span> et donc des schémas de quantification différents.
<br><br>

## <span style="color: #FFBF00"> **Quantification statique** </span>
Contrairement à la quantification dynamique, la quantification statique ne calcule pas le point $$0$$ (<span style="color:#1B89D8;">**$$z$$**</span>) et le facteur d'échelle (<span style="color:#D79515;">**$$s$$**</span>) pendant l'inférence, mais avant.  
Pour trouver ces valeurs, un jeu de données d'étalonnage est utilisé et donné au modèle pour recueillir ces distributions potentielles.

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_33.png">
</figure>
</center>

Une fois ces valeurs collectées, nous pouvons calculer les valeurs <span style="color:#D79515;">**$$s$$**</span> et <span style="color:#1B89D8;">**$$z$$**</span> nécessaires pour effectuer la quantification pendant l'inférence.  
Lors de l'inférence, les valeurs <span style="color:#D79515;">**$$s$$**</span> et <span style="color:#1B89D8;">**$$z$$**</span> ne sont pas recalculées, mais sont utilisées globalement pour toutes les activations afin de les quantifier.  
En général, la quantification dynamique est un peu plus précise car elle ne tente de calculer les valeurs <span style="color:#D79515;">**$$s$$**</span> et <span style="color:#1B89D8;">**$$z$$**</span> que pour chaque couche cachée. Nénamoins, elle peut entraîner une augmentation du temps de calcul car ces valeurs doivent être calculées pour chaque couche cachée.  
Au contraire, la quantification statique est moins précise mais plus rapide car elle connaît déjà les valeurs <span style="color:#D79515;">**$$s$$**</span> et <span style="color:#1B89D8;">**$$z$$**</span> utilisées pour la quantification.
<br><br>

## <span style="color: #FFBF00"> **Le royaume de la quantification 4 bits** </span>
Descendre en dessous de la quantification à $$8$$ bits s'est avéré être une tâche difficile car l'erreur de quantification augmente à chaque perte de bit. Heureusement, il existe plusieurs façons intelligentes de réduire les bits à $$6$$, $$4$$ et même $$2$$ bits (bien qu'il ne soit généralement pas conseillé de descendre en dessous de $$4$$ bits en utilisant ces méthodes).   
Nous allons explorer deux méthodes qui sont couramment partagées sur Hugging Face :  
•	[GPTQ](https://arxiv.org/abs/2210.17323) de FRANTAR et al. (2022) (modèle entier sur GPU)  
•	[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) de GERGANOV (2023) (possibilité de décharger les couches sur le CPU)  
<br>

### <span style="color: #51C353"> **GPTQ** </span>
GPTQ est sans doute l'une des méthodes les plus connues utilisées dans la pratique pour la quantification à $$4$$ bits.  
Elle utilise la quantification asymétrique et le fait couche par couche de sorte que chacune est traitée indépendamment avant de passer à la suivante :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_34.png">
</figure>
</center>

Au cours de ce processus de quantification par couche, on convertit d'abord les poids de la couche de la hessienne inverse. Il s'agit d'une dérivée seconde de la fonction de perte du modèle, qui nous indique à quel point la sortie du modèle est sensible aux variations de chaque poids.  
En simplifiant, cela démontre l'importance (inverse) de chaque poids dans une couche.  
Les poids associés à des valeurs plus petites dans la matrice hessienne sont plus cruciaux car de petites modifications de ces poids peuvent entraîner des changements significatifs dans les performances du modèle.

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_35.png">
<figcaption>Dans la hessienne inverse, les valeurs les plus faibles indiquent des poids plus « importants ». </figcaption>
</figure>
</center>
 
Ensuite, nous quantifions et déquantifions le poids de la première ligne de notre matrice de poids :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_36.png">
</figure>
</center>

Ce processus nous permet de calculer l'erreur de quantification (<span style="color:#950000;">**$$q$$**</span>) que nous pouvons pondérer à l'aide de la hessienne inverse (<span style="color:#C5E0E3;">**$$h_1$$**</span>) que nous avons calculée au préalable.  
En somme, nous créons une erreur de quantification pondérée en fonction de l'importance des poids :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_37.png">
</figure>
</center>

Puis, nous redistribuons cette erreur de quantification pondérée sur les autres poids de la ligne. Cela permet de maintenir la fonction globale et la sortie du réseau.  
Par exemple, si nous procédons ainsi pour le deuxième poids, à savoir $$0,3$$ (<span style="color:#666814;">**$$x_2$$**</span>), nous ajoutons l'erreur de quantification (<span style="color:#950000;">**$$q$$**</span>) multipliée par la hessienne inverse du deuxième poids (<span style="color:#8EBFC2;">**$$h_2$$**</span>).

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_38.png">
</figure>
</center>

Nous pouvons faire le même processus sur le troisième poids de la ligne :
<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_39.png">
</figure>
</center>

Nous itérons sur ce processus de redistribution de l'erreur de quantification pondérée jusqu'à ce que toutes les valeurs soient quantifiées.  
Cela fonctionne très bien car les poids sont généralement liés les uns aux autres. Ainsi, lorsqu'un poids présente une erreur de quantification, les poids connexes sont mis à jour en conséquence (par l'intermédiaire de la hessienne inverse).  

**Remarque** : Les auteurs ont utilisé plusieurs astuces pour accélérer les calculs et améliorer les performances, comme l'ajout d'un facteur d'amortissement à la hessienne, le « *lazy batching* » et le pré-calcul d'informations à l'aide de la [méthode de Cholesky](https://fr.wikipedia.org/wiki/Factorisation_de_Cholesky). Nous conseillons au lecteur de visionner [cette vidéo YouTube](https://www.youtube.com/watch?v=mii-xFaPCrA) sur le sujet.  

Notez que vous pouvez utiliser la librairie [EXLlama2](https://github.com/turboderp-org/exllamav2) si vous souhaitez une méthode de quantification visant à optimiser les performances et à améliorer la vitesse d'inférence.
<br>

### <span style="color: #51C353"> **GGUF** </span>
Bien que GPTQ soit une excellente méthode de quantification pour exécuter votre LLM en entier sur un GPU, il se peut que vous ne disposiez pas toujours de la capacité nécessaire. Au lieu de cela, nous pouvons utiliser GGUF pour décharger n'importe quelle couche du LLM sur le CPU.  
Cela permet d'utiliser à la fois le CPU et le GPU lorsque la VRAM est insuffisante.  
La méthode de quantification GGUF est fréquemment mise à jour et peut dépendre du niveau de quantification souhaité. Toutefois, le principe général est le suivant.  
Tout d'abord, les poids d'une couche donnée sont divisés en super-blocs contenant chacun un ensemble de sous-blocs. De ces blocs, nous extrayons le facteur d'échelle (<span style="color:#D79515;">**$$s$$**</span>) et l'alpha (<span style="color:#02E1FF;">**$$α$$**</span>) :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_40.png">
</figure>
</center>

Pour quantifier un sous-bloc donné, nous pouvons utiliser la quantification absmax que nous avons utilisée précédemment. Rappelons qu'elle multiplie un poids donné par le facteur d'échelle (<span style="color:#D79515;">**$$s$$**</span>) :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_41.png">
</figure>
</center>

Le facteur d'échelle est calculé à l'aide des informations du sous-bloc, mais la quantification est réalisée en utilisant les informations du super-bloc qui possède son propre facteur d'échelle :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_42.png">
</figure>
</center>

Cette quantification par bloc utilise le facteur d'échelle (<span style="color:#D79515;">**$$s_{super}$$**</span>) du super-bloc pour quantifier le facteur d'échelle (<span style="color:#D79515;">**$$s_{sous}$$**</span>) du sous-bloc.  
Le niveau de quantification de chaque facteur d'échelle peut être différent, le super-bloc ayant généralement une précision plus élevée que le facteur d'échelle du sous-bloc.  
Pour illustrer notre propos, examinons quelques quantifications de niveaux différents ($$2$$ bits, $$4$$ bits et $$6$$ bits) :  

<table>
    <tr>
        <td>Nom</td>
        <td>Quantification du poids</td>
        <td>Echelle de quantification (<font color="#D79617">$$s_{super}$$</font>)</td>
        <td>Echelle de quantification (<font color="#D79617">$$s_{sous}$$</font>)</td>
        <td>Bits par poids (<font color="#B352FF">w</font>)</td>
        <td># Sous blocs</td>
        <td>Poids par bloc</td>
    </tr>
    <tr>
        <td>Q2_K</td>
        <td>2 bits</td>
        <td>4 bits</td>
        <td>2 bits</td>
        <td>2,5625</td>
        <td>16</td>
        <td>16</td>
    </tr>
    <tr>
        <td>Q4_K</td>
        <td>4 bits</td>
        <td>6 bits</td>
        <td>4 bits</td>
        <td>4,5</td>
        <td>8</td>
        <td>32</td>
    </tr>
    <tr>
        <td>Q6_K</td>
        <td>6 bits</td>
        <td>8 bits</td>
        <td>6 bits</td>
        <td>6,5625</td>
        <td>16</td>
        <td>16</td>
    </tr>
</table>

<br>
Note : Selon le type de quantification, une valeur minimale supplémentaire ($$m$$) est nécessaire pour ajuster le point $$0$$.  

Consultez la [*pull request*](https://github.com/ggerganov/llama.cpp/pull/1684) pour obtenir une vue d'ensemble de tous les niveaux de quantification. Consultez également [celle-ci](https://github.com/ggerganov/llama.cpp/pull/4861) pour plus d'informations sur la quantification à l'aide de matrices d'importance.
<br><br><br>

# <span style="color: #FF0000"> **Partie 4 : Quantification pendant l’entraînement** </span>
Dans la troisième partie, nous avons vu comment quantifier un modèle après l'entraînement. L'inconvénient de cette approche est que la quantification ne prend pas en compte le processus d'entraînement.  
C'est là qu'intervient la quantification pendant l'entraînement (QAT pour *Quantization Aware Training*).

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_43.png">
</figure>
</center>

La QAT a tendance à être plus précis que la PTQ puisque la quantification a déjà été prise en compte pendant l'entraînement. Cela fonctionne de la façon suivante.  
Au cours de l'entraînement, des « fausses » quantifications sont introduites. Il s'agit par exemple de quantifier les poids en INT4 puis à les déquantifier en FP32 :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_44.png">
</figure>
</center>

Cette approche permet au modèle de prendre en compte le processus de quantification durant l'entraînement, le calcul de la perte et les mises à jour des poids.  
QAT explore la perte pour trouver des minima « larges » afin de minimiser les erreurs de quantification, car les minima « étroits » ont tendance à en entraîner de plus importantes.

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_45.png">
</figure>
</center>

Par exemple, imaginons que nous ne tenions pas compte de la quantification lors de la passe arrière. Nous choisissons le poids avec la plus petite perte selon la méthode de la descente du gradient. Cependant, cela introduirait une erreur de quantification plus importante s'il se trouve dans un minimum « étroit ».  
En revanche, si nous tenons compte de la quantification, un poids actualisé différent sera sélectionné dans un minimum « large » avec une erreur de quantification beaucoup plus faible.

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_46.png">
</figure>
</center>

Ainsi, bien que PTQ ait une perte plus faible en haute précision (par exemple, FP32), QAT entraîne une perte plus faible en basse précision (par exemple, INT4), ce qui est notre objectif.
<br><br>

## <span style="color: #FFBF00"> **L'ère des LLM 1 bit : BitNet** </span>
Passer à $$4$$ bits, comme nous l'avons vu précédemment, est déjà grosse réduction, mais que se passerait-il si nous réduisions encore plus ?  
C'est là qu'intervient [BitNet](https://arxiv.org/abs/2310.11453) de WANG, MA et al. (2023), qui représente les poids d'un modèle avec $$1$$ bit, en utilisant soit $$-1$$, soit $$1$$ pour un poids donné.  
Pour ce faire, le processus de quantification directement injecté dans l'architecture du [Transformer](https://lbourdois.github.io/blog/nlp/Transformer/).  
Pour rappel, l'architecture Transformer est utilisée comme base de la plupart des LLM et est composée de calculs qui impliquent des couches linéaires :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_47.png">
</figure>

</center>
Ces couches linéaires sont généralement représentées avec une grande précision, par exemple FP16, et c'est là que résident la plupart des poids.  
BitNet remplace ces couches linéaires par quelque chose que les auteurs appellent « BitLlinear » :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_48.png">
</figure>
</center>

Une couche BitLinear fonctionne de la même manière qu'une couche linéaire standard et calcule la sortie sur la base des poids multipliés par l'activation.  
En revanche, une couche bit-linéaire représente les poids d'un modèle sur 1 bit et les activations sur INT8 :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_49.png">
</figure>
</center>

Une couche BitLineary, comme l’approche QAT, effectue une forme de « fausse » quantification pendant l'entraînement pour analyser l'effet de la quantification des poids et des activations :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_50.png">
<figcaption>Dans le papier les auteurs utilisent γ au lieu de α mais puisque nous avons utilisé α tout au long de nos exemples, nous poursuivons avec cette notation. De même, β n'est pas identique à ce que nous avons utilisée pour la quantification du point 0 mais la valeur absolue moyenne. </figcaption>
</figure>
</center>

<br>

Passons en revue le BitLinear étape par étape.
<br>

### <span style="color: #51C353"> **Quantification des poids** </span>
Pendant l'entraînement, les poids sont stockés en INT8, puis quantifiés à $$1$$ bit à l'aide d'une stratégie simple, appelée fonction [signe](https://fr.wikipedia.org/wiki/Fonction_signe).  
En substance, on déplace la distribution des poids pour qu'elle soit centrée autour de $$0$$, puis on attribue tout ce qui est à gauche de $$0$$ à  $$-1$$ et tout ce qui est à droite à $$+1$$ :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_51.png">
</figure>
</center>

De plus, on traque une valeur <span style="color:#8BE8F3;">**$$β$$**</span> (valeur absolue moyenne) que nous utiliserons plus tard pour la déquantification. 
<br>

### <span style="color: #51C353"> **Quantification des activations** </span>
Pour quantifier les activations, BitLinear utilise la quantification absmax pour convertir les activations FP16 en INT8 car elles doivent être plus précises pour la multiplication matricielle. 

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_52.png">
</figure>
</center>

De plus, on traque <span style="color:#08D6F5;">**$$α$$**</span> (valeur absolue maximale) que nous utiliserons plus tard pour la déquantification.
<br>

### <span style="color: #51C353"> **Déquantification** </span>
Nous avons traqué <span style="color:#08D6F5;">**$$α$$**</span> (valeur absolue la plus élevée des activations) et <span style="color:#8BE8F3;">**$$β$$**</span> (valeur absolue moyenne des poids), car ces valeurs nous aideront à déquantifier en FP16 les activations.  
Les activations de sortie sont redimensionnées avec {<span style="color:#08D6F5;">**$$α$$**</span>, <span style="color:#D81B60;">**$$γ$$**</span>} pour les déquantifier à la précision d'origine :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_53.png">
</figure>
</center>

Et c'est tout ! Cette procédure est relativement simple et permet de représenter les modèles avec seulement deux valeurs : $$-1$$ ou $$1$$.  
En utilisant cette procédure, les auteurs ont observé que plus la taille du modèle augmente, plus l'écart de performance entre un modèle $$1$$-bit et un modèle entraîné en FP16 se réduit.  
Cependant, cela ne concerne que les modèles de grande taille (>$$30$$B paramètres) et l'écart avec les modèles plus petits reste assez important.
<br><br>

## <span style="color: #FFBF00"> **Tous les LLM sont en 1,58 bits**</span>
[BitNet 1.58b](https://arxiv.org/abs/2402.17764) de MA, WANG et al. (2024) a été introduit pour améliorer le problème de passage à l'échelle mentionné précédemment.  
Dans cette nouvelle méthode, chaque poids du modèle n'est pas seulement $$-1$$ ou $$1$$, mais peut désormais également prendre $$0$$ comme valeur, ce qui le rend ternaire. Il est intéressant de noter que l'ajout du $$0$$ améliore considérablement le BitNet et permet un calcul beaucoup plus rapide.
<br>

### <span style="color: #51C353"> **Le pouvoir du 0** </span>
Alors, pourquoi l'ajout de $$0$$ est-il une amélioration si majeure ?  
Tout est lié à la multiplication matricielle !  
Tout d'abord, voyons comment fonctionne la multiplication matricielle en général. Lors du calcul de la sortie, nous multiplions une matrice de poids par un vecteur d'entrée. Ci-dessous, la première multiplication de la première couche d'une matrice de poids est visualisée :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_54.png">
</figure>
</center>

Notons que cette multiplication implique deux actions : **multiplier** les poids individuels avec l'entrée, puis les **additionner**.  
BitNet 1.58b, en revanche, parvient à se passer de la multiplication puisque les poids ternaires vous indiquent essentiellement ce qui suit :  
•	$$1$$ : Je veux ajouter cette valeur  
•	$$0$$ : Je ne veux pas cette valeur  
•	$$-1$$ : Je veux soustraire cette valeur  
Par conséquent, vous n'avez besoin d'effectuer une addition que si vos poids sont quantifiés à 1,58 bit :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_55.png">
</figure>
</center>

Cela permet non seulement d'accélérer considérablement les calculs, mais aussi de **filtrer les caractéristiques**.  
En mettant un poids donné à $$0$$, il est possible de l'ignorer au lieu d'ajouter ou de soustraire les poids comme c'est le cas avec les représentations sur $$1$$ bit.
<br>

### <span style="color: #51C353"> **Quantification** </span>
Pour effectuer la quantification des poids, BitNet 1.58b utilise la quantification *absmean*  qui est une variante de la quantification *absmax* que nous avons vue précédemment.  
Il compresse simplement la distribution des poids et utilise la moyenne absolue (<span style="color:#02E1FF;">**$$α$$**</span>) pour quantifier les valeurs. Ils sont ensuite arrondis à $$-1$$, $$0$$ ou $$1$$ :

<center>
<figure class="image">
<img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Quantification/image_56.png">
</figure>
</center>

Par rapport à BitNet, la quantification de l'activation est la même à l'exception d'une chose. Au lieu d’étalonner les activations sur une plage  [$$0,2b⁻¹$$], elles le sont sur [$$-2b⁻¹,2b⁻¹$$] du fait de la quantification *absmax*.  
Et c'est tout ! La quantification 1,58 bit nécessitait (principalement) deux astuces :  
•	Ajouter l’option $$0$$ pour créer des représentations ternaires [$$-1,0,1$$]  
•	la quantification *absmean* pour les poids  

> « Le BitNet 13B 1.58b est plus efficace, en termes de latence, d'utilisation de la mémoire et de consommation d'énergie qu'un LLM 3B FP16 »

En conséquence, nous obtenons des modèles légers car nous n'avons que 1,58 bits ce qui est efficace en termes de calcul !
<br><br><br>

# <span style="color: #FF0000"> **Conclusion** </span>
Ainsi se termine notre voyage dans la quantification ! J'espère que cet article vous donnera une meilleure compréhension du potentiel de la quantification, de GPTQ, GGUF et du BitNet. Qui sait à quel point les modèles seront petits à l'avenir ?  

Si vous voulez aller plus loin, je vous suggère les ressources suivantes :  
• Les articles de [blog d'Hugging Face](hf.co/blog) et notamment :  
&nbsp;&nbsp;&nbsp;•	Cet [article](https://hf.co/blog/hf-bitsandbytes-integration) sur la méthode de quantification [LLM.int8()](https://arxiv.org/abs/2208.07339).   
&nbsp;&nbsp;&nbsp;•	Cet [article](https://hf.co/blog/embedding-quantization) sur la quantification des *embeddings*.  
&nbsp;&nbsp;&nbsp;•	Cet [article](https://huggingface.co/blog/1_58_llm_extreme_quantization) consacré au BitNet.  
• Hugging Face a également un [cours sur les fondamentaux de la quantification](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/) sur DeepLearning.AI.  
•	Un [article de blog d’Eleuther.ai](https://blog.eleuther.ai/transformer-math/) décrivant les mathématiques de base liées au calcul et à l'utilisation de la mémoire pour les Transformers.  
•	Cette [application](https://hf.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator) et [celle-ci](https://vram.asmirnov.xyz/) sont deux bonnes ressources pour calculer la (V)RAM dont vous avez besoin pour un modèle donné.  
•	Le papier sur le [QLoRA](https://arxiv.org/abs/2305.14314) pour de la quantification pour les méthodes PEFT.  
•	Une [vidéo YouTube](https://www.youtube.com/watch?v=mii-xFaPCrA) sur GPTQ expliquée de manière incroyablement intuitive. 

Nous vous invitions également à jeter un œil à des librairies pour dédiées au sujet comme :  
•	[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (vous pouvez consulter [cet article](https://huggingface.co/blog/hf-bitsandbytes-integration) et [cet article](https://huggingface.co/blog/4bit-transformers-bitsandbytes))  
•	[AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) (vous pouvez consulter également [cet article](https://huggingface.co/blog/gptq-integration))  
•	[transfomers](https://github.com/huggingface/transformers) intègre également sous le capot ces deux librairies (vous pouvez lire [cet article de blog](https://huggingface.co/blog/overview-quantization-transformers) sur le sujet, et surtout le [documentation officielle](https://huggingface.co/docs/transformers/v4.49.0/quantization/overview))


<br><br><br>
# <span style="color: #FF0000"> **Références** </span>
- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) de Maarten GROOTENDORST (2024)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) de Meta (2024)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) d’Elias FRANTAR, Saleh ASHKBOOS, Torsten HOEFLER et Dan ALISTARH (2022)
- [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) de Georgi GERGANOV (2023)
- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453) de Hongyu WANG, Shuming MA, Li DONG, Shaohan HUANG, Huaijie WANG, Lingxiao MA, Fan YANG, Ruiping WANG, U, Furu WEI (2023)
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764) de Shuming MA, Hongyu WANG, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong XUE, Furu WEI (2024)

<br><br><br>
# <span style="color: #FF0000"> **Citation** </span>
```
@inproceedings{quantification_blog_post,
author = {Loïck BOURDOIS},
title = {Un guide visuel sur la quantification},
year = {2025},
url = {https://lbourdois.github.io/blog/Quantification/}
}
```
