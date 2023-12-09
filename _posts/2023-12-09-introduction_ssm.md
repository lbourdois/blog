---
title: "INTRODUCTION AUX STATE SPACE MODELS (SSM)"
categories:
  - SSM
tags:
  - Introduction aux SSM 
excerpt : SSM - Bases des SSM en apprentissage profond
header :
    overlay_image: "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/NLP_radom_blog.png"
    teaser: "https://github.com/lbourdois/blog/assets/58078086/cb2dca34-9a3e-481a-8773-2360a1ceaa1c"
author_profile: false
classes: wide
---

# <span style="color: #FF0000"> **Avant-propos** </span>

7 octobre 2021, alors que je me demandais si [AK](https://hf.co/akhaliq) était un bot ou un humain, je vis passer l'un de ses [tweets](https://twitter.com/_akhaliq/status/1445931206030282756).
Un lien vers une publication sur [open-review.net](https://openreview.net/forum?id=uYLFoz1vlAC) accompagné de l'image suivante :
![image/png](https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/QMpNVGwdQV2jRw-jYalxa.png)

Intrigué par les résultats annoncés, je suis allé lire en quoi consistait ce modèle S3 qui sera renommé moins d'un mois plus tard en [S4](https://twitter.com/_albertgu/status/1456031299194470407) ([lien](https://github.com/lbourdois/blog/blob/master/assets/efficiently_modeling_long_sequences_s3.pdf) de la version de quand il s'appelait encore S3 pour les intéressés).  
C'est l'unique article scientifique qui m'a donné la chair de poule en le lisant tellement je l'ai trouvé beau. A l'époque, j'étais persuadé que les State Space Models (SSM) allaient remplacer dans les transformers dans les mois qui allés suivre. Deux ans plus tard, force est de constater que je me suis complètement trompé devant le ras-de-marrée de LLM en pagaille qui font l'actualité en NLP.  

Néanmoins ce lundi 4 décembre 2023, l'annonce de Mamba par [Albert Gu](https://twitter.com/_albertgu/status/1731727672286294400) et [Tri Dao](https://twitter.com/tri_dao/status/1731728602230890895) a suscité quelques intérêts. Phénomène accentué 4 jours plus tard avec l’annonce de [StripedHyena](https://twitter.com/togethercompute/status/1733213267185762411) par Together AI.  
Une bonne occasion pour moi d'écrire quelques mots sur les développements des SSM en l'espace de ces deux années.  
Je prévois trois articles pour commencer. Celui-ci définissant ce que sont que les SSM. Puis un sur les évolutions de ces modèles en 2022, et un dernier sur les évolutions de ces modèles en 2023. 
Le but étant donc de commencer avec une vue d'ensemble des évolutions. J'espère dans un second temps, si le temps me le permet, d'entrer dans les détails des architectures de certains SSM spécifique avec des animations ✨.


# <span style="color: #FF0000"> **Qu’est-ce qu’un SSM ?** </span>

Les *States Spaces Models* ou représentation d'état en français, sont utilisés traditionnellement en théorie du contrôle afin de modéliser un système dynamique via des variables d'état.  
Wikipedia indiquant qu'ils sont également utilisés en automatique, il n'est pas exclu qu'ils soient utilisés dans d'autres domaines et/ou sous la forme d'un autre nom.  
Dans le cadre de l'apprentissage profond, lorsque l'on parle de SSM, on se réfère en réalité qu'à un sous-ensemble des représentations existantes, à savoir les systèmes linéaires invariants (ou stationnaires).  

## <span style="color: #FFBF00"> **Définition de la formule générique** </span>

Utilisons l'image ci-dessus afin de définir un SSM :
| ![image/png](https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/G7icfkYoxIqHZcJGHM7UD.png) |
|:--:|
| *Vue d'un SSM continu et invariant dans le temps, image tirée de [https://en.wikipedia.org/wiki/State-space_representation](https://en.wikipedia.org/wiki/State-space_representation)*|

On peut observer qu'un SSM repose sur trois variables dépendant du temps t :
- $x(t) \in \mathbb {R} ^{n}$ représente les n variables d'état,
- $u(t) \in \mathbb {R} ^{m}$ représente les m entrées d'état,
- $y(t) \in \mathbb {R} ^{p}$ représente les p sorties,

On peut aussi observer qu'il est composé de quatre matrices pouvant être apprises : $\mathbf A, \mathbf B, \mathbf C$ et $\mathbf D$.
- $\mathbf A \in \mathbb {R} ^{m \times n}$ est la matrice d'état (contrôlant l'état lattent x),
- $\mathbf B \in \mathbb {R} ^{n \times m}$ est la matrice de contrôle,
- $\mathbf C \in \mathbb {R} ^{p \times n}$ est la matrice de sortie,
- $\mathbf D \in \mathbb {R} ^{p \times m}$ est la matrice de commande,

Nous modélisons le signal de sortie à l'aide de l'équation suivante :

$$
  \begin{aligned}
    x'(t) &= \mathbf{A}x(t) + \mathbf{B}u(t) \\
    y(t) &= \mathbf{C}x(t) + \mathbf{D}u(t)
  \end{aligned}
$$

Note : j'utilise ici la notation x' pour désigner la dérivée de x. Il n'est pas exclu que vous rencontriez à la place la notation ẋ dans la littérature. J'ai pour ma part simplement privilégié la notation ne nécessitant pas de LaTex ;)

Pour alléger les notations, l'équation précédente est généralement écrite sous la forme suivante, puisqu'il est implicite que les variables dépendent du temps :

$$
  \begin{aligned}
    x' &= \mathbf{A}x + \mathbf{B}u \\
    y &= \mathbf{C}x + \mathbf{D}u
  \end{aligned}
$$

Spécifions aussi que $\mathbf{D}u = 0$ dans les SSM en apprentissage profond car vue comme une skip connexion.  

Le système que nous avons ici est continu. Il n'est pas possible de le donner tel quel à un ordinateur. Nous devons pour cela le discrétiser.

## <span style="color: #FFBF00"> **Discrétisation** </span>

La discrétisation est l’un, voire le point le plus important dans les SSM. Toute la beauté de cette architecture est dans cette étape puis qu’elle permet d’aboutir à deux vues différentes d’un modèle en temps continu : la vue récurrente et la vue convolutive.  
Si vous devez retenir quelque chose de cet article, c’est ce point-ci.

| ![image](https://github.com/lbourdois/blog/assets/58078086/cb2dca34-9a3e-481a-8773-2360a1ceaa1c) |
|:--:|
| *Image provenant de l’article [Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers](https://arxiv.org/pdf/2110.13985.pdf), prédécesseur du S4*|


En effet, la première vue permet d’avoir un mécanisme efficace lors de l’inférence. La seconde permet un entraînement rapide car parallélisable.
Nous verrons aussi dans les prochains articles qu’il existe plusieurs discrétisations possibles. C’est d’ailleurs l’une des différences principales entre les différentes architectures de SSM existantes.  
Pour ce premier article, je vous propose d’appliquer la discrétisation appliquée dans le S4 pour illustrer ce point des deux vues possibles pour un SSM. Faisons donc (un peu) de mathématiques.

## <span style="color: #FFBF00"> **Vue récursive d’un SSM** </span>

Pour discrétiser le cas continu, utilisons la méthode des trapèzes.  
Posons $x’(t) = f(t,x)$. On a alors :  $x_{n+1} = x_n + \frac{1}{2}\Delta(f(t_n,y_n) + f(t_{n+1},y_{n+1}))$ avec $\Delta = t_{n+1} - t_n$.  
Si $x'_n = \mathbf{A}x_n + \mathbf{B} u_n$ (première ligne de l’équation d’un SSM), correspond à $f$, alors :  

$$
\begin{align}
x_{n+1} & = x_n + \frac{\Delta}{2} (\mathbf{A}x_n + \mathbf{B} u_n + \mathbf{A}x_{n+1} + \mathbf{B} u_{n+1}) \\
\Longleftrightarrow  x_{n+1} - \frac{\Delta}{2}\mathbf{A}x_{n+1} & = x_n + \frac{\Delta}{2}\mathbf{A}x_{n} + \frac{\Delta}{2}\mathbf{B}(u_{n+1} + u_n) \\
& \text{On pose $u_{n+1} = u_n$ car le vecteur de contrôle est supposé constant sur un petit $\Delta$} :\\
\Longleftrightarrow  (\mathbf{I}  - \frac{\Delta}{2} \mathbf{A}) x_{n+1} & = (\mathbf{I}  + \frac{\Delta}{2} \mathbf{A}) x_{n} + \Delta \mathbf{B} u_{n+1} \\
\Longleftrightarrow  x_{n+1} & = (\mathbf{I}  - \frac{\Delta}{2} \mathbf{A})^{-1} (\mathbf{I}  + \frac{\Delta}{2} \mathbf{A}) x_n + (\mathbf{I}  - \frac{\Delta}{2} \mathbf{A})^{-1} \Delta \mathbf{B} u_{n+1}
\end{align}
$$

Nous venons d’obtenir notre SSM discret !  
Pour que cela soit complètement explicite, posons :

$$
\begin{aligned}
  \mathbf{\bar{A}} &= (\mathbf {I} - \frac{\Delta}{2} \mathbf{A})^{-1}(\mathbf {I} + \frac{\Delta}{2} \mathbf{A}) \\
  \mathbf {\bar{B}} &= (\mathbf{I} - \frac{\Delta}{2}  \mathbf {A})^{-1} \Delta \mathbf{B} \\
  \mathbf {\bar{C}} &= \mathbf{C}\\
\end{aligned}
$$

On a alors

$$
  \begin{aligned}
    x_k &= \mathbf{\bar{A}}x_{k-1} + \mathbf{\bar{B}}u_k \\
    y_k &= \mathbf{\bar{C}}x_k
  \end{aligned}
$$

La notation des matrices avec une barre a été introduite dans le S4 pour désigner les matrices dans le cas discret. Elle semble depuis être devenue une convention dans le domaine des SSM appliqués à l’apprentissage profond.  
Dans la version précédente, nous venons d’écrire une version enroulée d’un réseau de neurones récurrents. La version déroulée de la récurrence étant la suivante :  
[IMAGE]  
Vous pouvez observer ici que $\mathbf{\bar{A}}$ est nécessaire pour initialiser le réseau.   
Son choix est très important puisqu’un mauvais $\mathbf{\bar{A}}$ (techniquement $\mathbf{A}$ car $\mathbf{\bar{A}}$ est juste une réécriture de la matrice) peut aboutir à un SSM donnant de très mauvais résultats.    
Dans le cadre de cet article, nous donnerons plus bas la formule de la matrice $\mathbf{A}$ utilisée dans le S4.  

## <span style="color: #FFBF00"> **Vue convolutive d’un SSM** </span>

La récurrence que nous venons d’écrire peut s’écrire sous la forme d’une convolution. Pour cela, il suffit d’itérer la formule 

$$
  \begin{aligned}
    x_k &= \mathbf{\bar{A}}x_{k-1} + \mathbf{\bar{B}}u_k \\
    y_k &= \mathbf{\bar{C}}x_k
  \end{aligned}
$$

Commençons par la première ligne du système :  
Etape 0 : $x_0 = \mathbf{\bar{B}}  u_0$  
Etape 1 : $x_1 = \mathbf{\bar{A}}x_{0} + \mathbf{\bar{B}}u_1  = \mathbf{\bar{A}} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{B}}u_1$  
Etape 2 : $x_2 = \mathbf{\bar{A}}x_{1} + \mathbf{\bar{B}}u_2  = \mathbf{\bar{A}} (\mathbf{\bar{A}} \mathbf{\bar{B}}  u_0) + \mathbf{\bar{B}}u_2 = \mathbf{\bar{A}}^{2} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{A}} \mathbf{\bar{B}}  u_1 + \mathbf{\bar{B}}u_2$  
D’où $x_k$ peut s’écrire sous la forme d’une fonction $f$ paramétrée par $u_0, u_1, … u_k$.  

Passons ensuite à la seconde ligne du système où il est à présent possible d’injecter les valeurs x_k calculées à l’instant :  
Etape 0 : $y_0 = \mathbf{\bar{C}} x_0  = \mathbf{\bar{C}}  \mathbf{\bar{B}}  u_0$  
Etape 1 : $y_1 = \mathbf{\bar{C}} x_1  =  \mathbf{\bar{C}} ( \mathbf{\bar{A}} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{B}}u_1) =  \mathbf{\bar{C}} \mathbf{\bar{A}} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{C}} \mathbf{\bar{B}}u_1$  
Etape 2 : $y_2 = \mathbf{\bar{C}} x_2 =  \mathbf{\bar{C}}(\mathbf{\bar{A}}^{2} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{A}} \mathbf{\bar{B}}  u_1 + \mathbf{\bar{B}}u_2 ) = \mathbf{\bar{C}}\mathbf{\bar{A}}^{2} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{C}}\mathbf{\bar{A}} \mathbf{\bar{B}}  u_1 + \mathbf{\bar{C}}\mathbf{\bar{B}}u_2$  
On peut observer le noyau de convolution $\mathbf{\bar{K}} _k = (\mathbf{\bar{C}}  \mathbf{\bar{B}}, \mathbf{\bar{C}} \mathbf{\bar{A}}  \mathbf{\bar{B}}, …, \mathbf{\bar{C}}  \mathbf{\bar{A}}^{k} \mathbf{\bar{B}})$ applicable aux $u_k$, d’où $K \ast u$.  

Comme pour les matrices, nous nous appliquons une barre sur le $\mathbf{\bar{K}}$  pour spécifier qu’il s’agit du noyau de convolution obtenu pour la discrétisation. Il est généralement appelé noyau de convolution SSM dans la littérature et sa taille est équivalente à l’entièreté de la séquence d’entrée.  
Ce noyau de convolution est calculé par FFT mais nous détaillerons cela dans les prochains articles (vous aimez la flash attention des transformers, vous adorerez la flash convolution que nous verrons dans le troisième article de blog).

## <span style="color: #FFBF00"> **Apprentissage des matrices A, B et C** </span>

Dans le noyau de convolution développé à l’instant, $\mathbf{\bar{C}}$ et $\mathbf{\bar{B}}$, sont des scalaires apprenables.   
Concernant $\mathbf{\bar{A}}$, dans notre noyau de convolution, elle s’exprime comme une puissance de $k$ au temps $k$.   
Cela peut être très long à calculer c’est pourquoi, on cherche à avoir $\mathbf{\bar{A}}$ fixe, et assez facilement, il apparait que la meilleure option est de la rendre diagonale.  
La façon d’avoir une matrice diagonale n’est pas évidente au premier abord.   
Ainsi, en plus du choix de la discrétisation citée ci-dessus, la manière de définir (et initier) $\mathbf{\bar{A}}$ est l’un des points qui différencient les différentes architectures de SSM développées dans la littérature que nous développerons dans le prochain article de blog.  
Pour les personnes intéressées, vous voir ci-dessous quelle est la matrice $\mathbf{\bar{A}}$ utilisée dans le papier du S4.   
A noter qu’il s’agit de mathématiques pas forcément triviales et que si vous ne les comprenez pas, ce n’est pas forcément important car il ne s’agit plus d’une matrice utilisée (nous en utilisons des beaucoup plus simples).   
Donc si vous aimez les maths, vous pouvez continuer à lire cette section, sinon vous pouvez passer directement à la suivante consacrée aux résultats obtenables avec les SSM. 

A TERMINER  
 $$
     \mathbf{A_{nk}} = -
    \begin{cases}
        (2n + 1)^{1/2} (2k + 1)^{1/2} & \text{if $n > k$,} \\
        n + 1                         & \text{if $n = k$,} \\
        0                             & \text{if $n < k$.} \\
    \end{cases}
 $$
 
# <span style="color: #FF0000"> **Quelques Resultats** <span>
AJOUTER UNE SECTION SUR LES RESULTATS POUR ILLUSTRER QUE CA SERT POUR TOUT.

# <span style="color: #FF0000"> **Conclusion** <span>
Les SSM sont des modèles possédant trois vues. Une vision continue, et lorsque nous la discrétisons sur le temps, une vue récurrente ainsi que convolutive.  
La vue convolutive sert à entraîner efficacement le modèle, et la vue récurrente permet de réaliser une inférence potentiellement infinie (pas de limite de taille de contexte ou encore de positionnal encoding comparés aux transformers).  
Ce type de modèle est très versatile puisqu’il est applicable pour les tâches de texte, de vision, d’audio, de séries temporelles ou encore de graphes !  
Nous verrons dans les prochains articles que les principales différences entre les diverses architectures de SSM existantes viennent principalement de la façon de discrétiser l’équation de base des SSM ou encore de définir la matrice A. 

## Remerciements
Je tiens à remercier Boris ALBAR, Pierre BEDU et Nicolas PREVOT d’avoir acceptés de monter un groupe de travail sur le sujet des SSM.

## Références
