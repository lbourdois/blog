---
title: "INTRODUCTION AUX STATE SPACE MODELS (SSM) ET AU S4"
categories:
  - SSM
tags:
  - Introduction aux SSM 
excerpt : SSM - Bases des SSM en apprentissage profond via le modèle S4
header :
    overlay_image: "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/NLP_radom_blog.png"
    teaser: "https://github.com/lbourdois/blog/assets/58078086/cb2dca34-9a3e-481a-8773-2360a1ceaa1c"
author_profile: false
sidebar:
    nav: sidebar-ssm
classes: wide
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# <span style="color: #FF0000"> **Avant-Propos** </span> 
Je tiens à remercier chaleureusement Boris ALBAR, Pierre BEDU et Nicolas PREVOT d’avoir accepté de monter un groupe de travail sur le sujet des SSM et de m'avoir ainsi accompagné dans la découverte de ce type de modèle. Un remerciement supplémentaire au premier pour avoir pris le temps de relire cet article de blog.
<br><br><br>

# <span style="color: #FF0000"> **Introduction** </span>
Les ***States Spaces Models*** (ou Modèles en Espace d'Etat en français) sont utilisés traditionnellement en théorie du contrôle afin de modéliser un système dynamique via des variables d'état.  

Dans le cadre de l'apprentissage profond, lorsque l'on parle de SSM, on ne se réfère en réalité qu'à un sous-ensemble des représentations existantes, à savoir les systèmes linéaires invariants (ou stationnaires).  
Ces modèles ont montré des performances impressionnantes dès octobre 2021 avec l'article « [*Efficiently Modeling Long Sequences with Structured State Spaces*](https://arxiv.org/abs/2111.00396) » d'Albert GU et al., au point de se positionner comme une alternative aux *transformers*.  
Dans cet article, nous allons définir les bases d'un SSM en apprentissage profond en nous appuyant sur le S4. A l'image du papier « [*Attention is all you need*](https://arxiv.org/abs/1706.03762) » d'Ashish VASWANI et al. (2017) pour les *transformers*, le S4 est le fondement d'un nouveau type d'architecture de réseau de neurones qui se doit d'être connu, mais ce n'est pas un modèle qui est utilisé tel quel en pratique (d'autres SSM plus performants ou plus faciles à implémenter étant maintenant disponibles). Sorti une semaine plus tôt que le S4, le [LSSL](https://arxiv.org/abs/2110.13985), par les mêmes auteurs, est également une source importante d'informations sur le sujet. Nous verrons les différentes évolutions qui découlent du S4 dans un prochain article de blog. Plongeons nous auparavant dans les bases des SSM.
<br><br><br>

# <span style="color: #FF0000"> **Définition d'un SSM en apprentissage profond** </span>

Utilisons l'image ci-dessous afin de définir un SSM :

| ![image/png](https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/G7icfkYoxIqHZcJGHM7UD.png) |
|:--:|
| Figure 1 : *Vue d'un SSM continu et invariant dans le temps (Source : [https://en.wikipedia.org/wiki/State-space_representation](https://en.wikipedia.org/wiki/State-space_representation))* |

<br>

On peut observer qu'un SSM repose sur trois variables dépendant du temps $$t$$ :
- $$x(t) \in \mathbb {C}^{n}$$ représente les $$n$$ variables d'état,
- $$u(t) \in \mathbb {C}^{m}$$ représente les $$m$$ entrées d'état,
- $$y(t) \in \mathbb {C}^{p}$$ représente les $$p$$ sorties,

On peut aussi observer qu'il est composé de quatre matrices pouvant être apprises : $$\mathbf A, \mathbf B, \mathbf C$$ et $$\mathbf D$$.
- $$\mathbf A \in \mathbb {C}^{m \times n}$$ est la matrice d'état (contrôlant l'état lattent $$x$$),
- $$\mathbf B \in \mathbb {C}^{n \times m}$$ est la matrice de contrôle,
- $$\mathbf C \in \mathbb {C}^{p \times n}$$ est la matrice de sortie,
- $$\mathbf D \in \mathbb {C}^{p \times m}$$ est la matrice de commande,

Il est possible de ramener l'image ci-dessus au système d'équations suivant :

$$
  \begin{aligned}
    x'(t) &= \mathbf{A}x(t) + \mathbf{B}u(t) \\
    y(t) &= \mathbf{C}x(t) + \mathbf{D}u(t)
  \end{aligned}
$$

Note : nous utilisons ici la notation $$x'$$ pour désigner la dérivée de $$x$$. Il n'est pas exclu de rencontrer à la place la notation $$ẋ$$ dans la littérature.  

De même, puisqu'il est implicite que les variables dépendent du temps, l'équation précédente est généralement écrite sous la forme suivante par souci d'allègement :

$$
  \begin{aligned}
    x' &= \mathbf{A}x + \mathbf{B}u \\
    y &= \mathbf{C}x + \mathbf{D}u
  \end{aligned}
$$

Ce système peut s'alléger même davantage, car dans les SSM en apprentissage profond, $$\mathbf{D}u = 0$$ est vue comme une *skip connexion* facilement calculable.  

$$
  \begin{aligned}
    x' &= \mathbf{A}x + \mathbf{B}u \\
    y &= \mathbf{C}x
  \end{aligned}
$$

Ce système est continu. Il doit donc d'abord être discrétisé afin de pouvoir être fourni à un ordinateur.
<br><br><br>

# <span style="color: #FF0000"> **Discrétisation** </span>

La discrétisation est l’un, voire le point le plus important dans les SSM. Toute l'efficacité de cette architecture réside dans cette étape puisqu’elle permet de passer de la vue continue du SSM à ses deux autres vues : la **vue récursive** et la **vue convolutive**.  
S’il n’y a qu’une chose à retenir de cet article, c’est bien celle-ci.

| ![image](https://github.com/lbourdois/blog/assets/58078086/12bbe1cf-3911-4bad-9a3b-3f427bc6bc82)|
|:--:|
| Figure 2 : *Image provenant de l'article de blog « [Structured State Spaces: Combining Continuous-Time, Recurrent, and Convolutional Models](https://hazyresearch.stanford.edu/blog/2022-01-14-s4-3) » d'Albert GU et al. (2022)*|

<br>

Nous verrons dans les prochains articles qu’il existe plusieurs discrétisations possibles. Ce point forme l’une des différences principales entre les diverses architectures de SSM existantes.  
Pour ce premier article, appliquons la discrétisation « originale » proposée dans le S4 afin d'illustrer les deux vues supplémentaires d'un SSM.
<br><br><br>

# <span style="color: #FF0000"> **Vue récursive d’un SSM** </span>

Pour discrétiser le cas continu, utilisons la [méthode des trapèzes](https://fr.wikipedia.org/wiki/Transformation_bilin%C3%A9aire#Approximation_trap%C3%A9zo%C3%AFdale) où le principe est d'assimiler la région sous la courbe représentative d'une fonction $$f$$ définie sur un segment $$[t_n , t_{n+1}]$$ à un trapèze et d'en calculer l'aire $$T$$ : $$T=(t_{n+1} - t_n){\frac  {f(t_n)+f(t_{n+1})}{2}}$$.  

On a alors :  $$x_{n+1} - x_n = \frac{1}{2}\Delta(f(t_n) + f(t_{n+1}))$$ avec $$\Delta = t_{n+1} - t_n$$.  
Si $$x'_n = \mathbf{A}x_n + \mathbf{B} u_n$$ (première ligne de l’équation d’un SSM), correspond à $$f$$, alors :  

$$
\begin{align}
x_{n+1} & = x_n + \frac{\Delta}{2} (\mathbf{A}x_n + \mathbf{B} u_n + \mathbf{A}x_{n+1} + \mathbf{B} u_{n+1}) \\
\Longleftrightarrow  x_{n+1} - \frac{\Delta}{2}\mathbf{A}x_{n+1} & = x_n + \frac{\Delta}{2}\mathbf{A}x_{n} + \frac{\Delta}{2}\mathbf{B}(u_{n+1} + u_n) \\
(*) \Longleftrightarrow  (\mathbf{I}  - \frac{\Delta}{2} \mathbf{A}) x_{n+1} & = (\mathbf{I}  + \frac{\Delta}{2} \mathbf{A}) x_{n} + \Delta \mathbf{B} u_{n+1}\\
\Longleftrightarrow  x_{n+1} & = (\mathbf{I}  - \frac{\Delta}{2} \mathbf{A})^{-1} (\mathbf{I}  + \frac{\Delta}{2} \mathbf{A}) x_n + (\mathbf{I}  - \frac{\Delta}{2} \mathbf{A})^{-1} \Delta \mathbf{B} u_{n+1}
\end{align}
$$

(*) $$u_{n+1} \overset{\Delta}{\simeq} u_n$$ (le vecteur de contrôle est supposé constant sur un petit $$\Delta$$).

Nous venons d’obtenir notre SSM discrétisé !  
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

La notation des matrices avec une barre a été introduite dans le S4 pour désigner les matrices dans le cas discret et est devenue depuis une convention dans le domaine des SSM appliqués à l’apprentissage profond.  
<br><br><br>

# <span style="color: #FF0000"> **Vue convolutive d’un SSM** </span>

Cette récurrence peut s’écrire sous la forme d’une convolution. Pour cela, il suffit d’itérer les équations du système 

$$
  \begin{aligned}
    x_k &= \mathbf{\bar{A}}x_{k-1} + \mathbf{\bar{B}}u_k \\
    y_k &= \mathbf{\bar{C}}x_k
  \end{aligned}
$$

Commençons par la première ligne du système :  
Etape 0 : $$x_0 = \mathbf{\bar{B}}  u_0$$  
Etape 1 : $$x_1 = \mathbf{\bar{A}}x_{0} + \mathbf{\bar{B}}u_1  = \mathbf{\bar{A}} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{B}}u_1$$  
Etape 2 : $$x_2 = \mathbf{\bar{A}}x_{1} + \mathbf{\bar{B}}u_2  = \mathbf{\bar{A}} (\mathbf{\bar{A}} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{B}}u_1) + \mathbf{\bar{B}}u_2 = \mathbf{\bar{A}}^{2} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{A}} \mathbf{\bar{B}}  u_1 + \mathbf{\bar{B}}u_2$$  
Nous avons $$x_k$$ qui peut s’écrire sous la forme d’une fonction $$f$$ paramétrée par $$u_0, u_1, … u_k$$.  

Passons ensuite à la seconde ligne du système où il est à présent possible d’injecter les valeurs $$x_k$$ calculées à l’instant :  
Etape 0 : $$y_0 = \mathbf{\bar{C}} x_0  = \mathbf{\bar{C}}  \mathbf{\bar{B}}  u_0$$  
Etape 1 : $$y_1 = \mathbf{\bar{C}} x_1  =  \mathbf{\bar{C}} ( \mathbf{\bar{A}} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{B}}u_1) =  \mathbf{\bar{C}} \mathbf{\bar{A}} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{C}} \mathbf{\bar{B}}u_1$$  
Etape 2 : $$y_2 = \mathbf{\bar{C}} x_2 =  \mathbf{\bar{C}}(\mathbf{\bar{A}}^{2} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{A}} \mathbf{\bar{B}}  u_1 + \mathbf{\bar{B}}u_2 ) = \mathbf{\bar{C}}\mathbf{\bar{A}}^{2} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{C}}\mathbf{\bar{A}} \mathbf{\bar{B}}  u_1 + \mathbf{\bar{C}}\mathbf{\bar{B}}u_2$$  
On peut observer le noyau de convolution $$\mathbf{\bar{K}} _k = (\mathbf{\bar{C}}  \mathbf{\bar{B}}, \mathbf{\bar{C}} \mathbf{\bar{A}}  \mathbf{\bar{B}}, …, \mathbf{\bar{C}}  \mathbf{\bar{A}}^{k} \mathbf{\bar{B}})$$ applicable aux $$u_k$$, d’où $$K \ast u$$.  

Comme pour les matrices, nous appliquons une barre sur le $$\mathbf{\bar{K}}$$  pour spécifier qu’il s’agit du noyau de convolution obtenu après discrétisation. Il est généralement appelé **noyau de convolution SSM** dans la littérature et sa taille est équivalente à l’entièreté de la séquence d’entrée.  
Ce noyau de convolution est calculé par [Transformation de Fourier Rapide](https://fr.wikipedia.org/wiki/Transformation_de_Fourier_rapide) (FFT) et sera explicité dans les prochains articles (vous aimez la Flash Attention des *transformers* ? Vous adorerez la Flash FFT Convolution que nous verrons dans le troisième article de blog).
<br><br><br>


# <span style="color: #FF0000"> **Avantages et limites de chacune des trois vues** </span>

| ![image](https://github.com/lbourdois/blog/assets/58078086/cb2dca34-9a3e-481a-8773-2360a1ceaa1c) |
|:--:|
| Figure 3 : *Image provenant du papier « [Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers](https://arxiv.org/abs/2110.13985) » d'Albert GU et al., sorti à une semaine d'intervalle du S4*|

<br>

Les vues différentes du SSM ont chacunes des avantages et des inconvénients, détaillons-les.

Pour la <u><b>vue continue</b></u>, les avantages et inconvénients sont les suivants :  
✓ Gère automatiquement les données continues (signaux audio, séries temporelles, par exemple). Cela représente énorme avantage pratique pour traiter des données à échantillonnage irrégulier ou décalé dans le temps.  
✓ Analyse mathématiquement réalisable, par exemple en calculant des trajectoires exactes ou en construisant des systèmes de mémorisation (HiPPO).  
✗ Extrêmement lent à la fois pour la formation et l'inférence.  

Pour la <u><b>vue récursive</b></u>, il s'agit ici des avantages et inconvénients bien connus des réseaux de neurones récurrents (voir l'[article](https://lbourdois.github.io/blog/nlp/RNN-LSTM-GRU-ELMO/) qui leur est consacré sur le blog) à savoir :  
✓ Un biais inductif naturel pour les données séquentielles, et en principe un contexte non borné.  
✓ Une inférence efficace (mises à jour d'état en temps constant).  
✗ Un apprentissage lent (manque de parallélisme).  
✗ Une disparition ou explosion du gradient lors de l'entraînement de séquence trop longues.  

Pour la <u><b>vue convolutive</b></u>, il s'agit ici des avantages et inconvénients bien connus des réseaux de neurones convolutifs (nous sommes ici dans le cadre de leur version unidimensionnelle), à savoir :  
✓ Caractéristiques locales et interprétables.  
✓ Entraînement efficace (parallélisable).  
✗ Lenteur dans les contextes en ligne ou autorégressifs (doit recalculer l'ensemble de l'entrée pour chaque nouveau point de données).  
✗ Taille de contexte fixe.    


Ainsi, en fonction de l'étape du processus (entraînement ou inférence) ou du type de données à notre disposition, il est possible de passer d'une vue à une autre afin de retomber sur un cadre favorable permettant de tirer le meilleur parti du modèle.   
Nous priviliègierons la vue convolutive pour l'entraînement pour un entraînement rapide via la parallélisation, la vue récursive pour une inférence efficace, et la vue continue pour traiter des données continues.  
<br><br><br>


# <span style="color: #FF0000"> **Apprentissage des matrices** </span>

Dans le noyau de convolution développé plus haut, $$\mathbf{\bar{C}}$$ et $$\mathbf{\bar{B}}$$, sont des scalaires apprenables.   
Concernant $$\mathbf{\bar{A}}$$, nous avons vu que dans notre noyau de convolution, elle s’exprime comme une puissance de $$k$$ au temps $$k$$. Cela peut être très long à calculer c’est pourquoi, on cherche à avoir $$\mathbf{\bar{A}}$$ fixe. Pour cela, la meilleure option est de l'avoir diagonale :

$$
\mathbf{A} =  \begin{bmatrix} 
\lambda_{1} & 0 & \cdots & 0 \\ 
0 & \lambda_{2} & \cdots & 0 \\ 
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_{n} 
\end{bmatrix} 

\Rightarrow

\mathbf{A^k} =  \begin{bmatrix} 
\lambda_{1}^k & 0 & \cdots & 0 \\ 
0 & \lambda_{2}^k & \cdots & 0 \\ 
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_{n}^k 
\end{bmatrix} 
$$

Par le [théorème spectral](https://fr.wikipedia.org/wiki/Th%C3%A9or%C3%A8me_spectral) de l'algèbre linéaire, il s'agit exactement de la classe des [matrices normales](https://fr.wikipedia.org/wiki/Matrice_normale).    
En plus du choix de la discrétisation citée ci-dessus, la manière de définir et initier $$\mathbf{\bar{A}}$$ est l’un des points qui différencient les diverses architectures de SSM développées dans la littérature que nous développerons dans le prochain article de blog. En effet, empiriquement, il apparait qu'un SSM initialisé avec une matrice $$\mathbf{A}$$ aléatoire conduit à de mauvais résultats alors qu'une initialisation effectuée à partir de la matrice $$HiPPO$$ (pour *High-Order Polynomial Projection Operator*) donne des résultats très bons (passage de de 60% à 98% sur le benchmark MNIST sequential).    

La matrice $$HiPPO$$ a été introduite par les auteurs du S4 dans un précédent [papier](https://arxiv.org/abs/2008.07669) (2020). Elle est reprise dans le [papier LSSL](https://arxiv.org/abs/2110.13985) (2021), aussi par les auteurs du S4, ainsi que dans l'appendix du S4.
Sa formule est la suivante :  

$$
\mathbf{A} =
\begin{bmatrix}
  1 \\
  -1 & 2 \\
  1 & -3 & 3 \\
  -1 & 3 & -5 & 4 \\
  1 & -3 & 5 & -7 & 5 \\
  -1 & 3 & -5 & 7 & -9 & 6 \\
  1 & -3 & 5 & -7 & 9 & -11 & 7 \\
  -1 & 3 & -5 & 7 & -9 & 11 & -13 & 8 \\
  \vdots & & & & & & & & \ddots \\
\end{bmatrix}
\\
\Rightarrow
\mathbf{A}_{nk} =
\begin{cases}%
  (-1)^{n-k} (2k+1) & n > k \\
  k+1 & n=k \\
  0 & n<k
\end{cases}
$$

Cette matrice n'est pas normale mais elle peut être décomposée sous la forme d'une matrice normale plus une matrice de rang inférieur (résumé dans le papier par NPLR pour *Normal Plus Low Rank*). Les auteurs prouvent dans leur papier que ce type de matrice peut être calculé efficacement via trois techniques (voir l'algorithme 1 dans le papier) : [série génératrice tronquée](https://fr.wikipedia.org/wiki/S%C3%A9rie_g%C3%A9n%C3%A9ratrice), [noyaux de Cauchy](https://en.wikipedia.org/wiki/Cauchy_matrix) et [identité de Woodbury](https://en.wikipedia.org/wiki/Woodbury_matrix_identity).   

<!--
Elle permet, in fine, de réécrire la vue réccurente sous la forme suivante :

$$
\begin{aligned}
  x_{k} &= \mathbf{\overline{A}} x_{k-1} + \mathbf{\overline{B}} u_k \\
  &= \mathbf{A_1} \mathbf{A_0} x_{k-1} + 2 \mathbf{A_1} \mathbf{B} u_k \\
  y_k &= \mathbf{C} x_k
\end{aligned}
$$

avec $$ A_0 = frac{2}{\Lambda}\mathbf{I} + (\mathbf{\Lambda} - \mathbf{P} \mathbf{Q}^*)$$ et $$A_1 = (\frac{2}{\Delta}-\mathbf{\Lambda})^{-1} - (\frac{2}{\Delta}-\mathbf{\Lambda})^{-1} \mathbf{p} (\mathbf{I} + \mathbf{q}^* (\frac{2}{\Delta}-\mathbf{\Lambda} )^{-1} \mathbf{p})^{-1} \mathbf{q}^*( \frac{2}{\Delta}-\mathbf{\Lambda})^{-1}$$ où $$\Lambda$$ est une matrice diagonale, $$p$$ et $$q$$ des vecteurs $$\in ℂ^N\times1$$  
-->

<!--
Les détails de la démonstration montrant qu'une matrice NPLR peut être calculée efficacement comme une matrice diagonale est basée sur des mathématiques pas forcément triviales mais très élégentes quand on prend le temps de la refaire. Le point principal est qu'elle s'étend sur plus de 8 pages dans l'appendix (voir la partie B et C) du papier. La reprendre entièrement rendrait cet article de blog trop long alors qu'il se veut être une introduction aux SSM. Je ne vais donc pas rentrer dans les détails de cette matrice.  
-->

Les détails de la démonstration montrant qu'une matrice NPLR peut être calculée efficacement comme une matrice diagonale peuvent être consultés dans l'appendix (voir la partie B et C) du papier.  
Les auteurs du S4 ont par la suite apporté des modifications à la matrice $$HiPPO$$ (sur la manière de l'initier) dans leur papier « [*How to Train Your HiPPO*](https://arxiv.org/abs/2206.12037v2) » (2022). Le modèle résultant de ce papier est généralement appelé « S4 V2 » ou « S4 updated » dans la littérature à opposer au « S4 original » ou « S4 V1 ».  
Nous verrons dans le prochain article, que d'autres auteurs (notamment [Ankit GUPTA](https://sites.google.com/view/ag1988/home)) ont proposé d'utiliser une matrice diagonale au lieu d'une matrice NPRL, approche qui est à présent privilégiée car plus simple à implémenter. 

<!--
Ainsi, la compréhenstion des mathématiques sous-jacentes à la matrice $$HiPPO$$ n'est pas un pré-requis obligatoire car elle n'est plus utilisée en pratique.
-->

<br><br><br>

# <span style="color: #FF0000"> **Résultats des expérimentations** </span>

Terminons cet article de blog en analysant une sélection des résultats du S4 sur diverses tâches et benchmarks afin de nous rendre compte du potentiel des SSM.

Commençons avec une tâche d'audio et le benchmark [*Speech Commands*](https://arxiv.org/abs/1804.03209v1) de WARDEN (2018).

|![image](https://github.com/lbourdois/blog/assets/58078086/1a902a38-a499-47ef-b015-0644cf2cebc4)|
|:--:|
| *Figure 4 : Image provenant du papier « [On the Parameterization and Initialization of Diagonal State Space Models](https://arxiv.org/abs/2206.11893) » d'Albert GU et al. (2022), aussi connu sous le nom de S4D parru après le S4 mais qui reprend sous une forme plus structurée les résultats du S4 pour ce benchmark (les résultats du S4D ayant été supprimés de l'image pour ne pas spoiler le prochain article ;)*|

<br>

On peut observer plusieurs choses sur ce tableau.  
Premièrement qu'à nombre de paramètres plus ou moins équivalent, le S4 fait beaucoup mieux (au moins + 13%) que les autres modèles, ici de type ConvNet.  
Deuxièmement, pour obtenir des performances équivalentes, un ConvNet nécessite 85 fois plus de paramètres.  
Troisièmement, un ConvNet entraîné sur du 16K Hz donne de très mauvais résultats quand il est ensuite appliqué sur des données 8K Hz. A contrario, le S4 conserve 95% de sa performance sur ce ré-échantillonage. Cela s'explique par la vue continue du SSM où il a suffit de diviser par deux la valeur de $$\Delta$$ au moment de la phase de test.  

<br>

Continuons avec une tâche de séries temporelles (introduite dans une révision du S4).

|![image](https://github.com/lbourdois/blog/assets/58078086/92b4b1aa-d3ab-4efb-a1a0-2fab1afdafa8)|
|:--:|
| *Figure 5 : Image provenant de l'appendix du S4*|

<br>

Les auteurs du papier reprennent la méthodologie du modèle « [Informer](https://arxiv.org/abs/2012.07436) » de ZHOU et al. (2020) et montrent que leur modèle surpasse ce *transformer* sur 40 des 50 configurations. Les résultats du tableau sont montrés dans un cadre univarié mais la même chose est observable pour un cadre multivarié (table 14 dans l'appendix).

<br>

Poursuivons avec une tâche de vision et le benchmark [*sCIFAR-10*](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) de KRIZHESKY (2009).

|![image](https://github.com/lbourdois/blog/assets/58078086/0334101e-fc91-426a-a845-5b08448ad08c)|
|:--:|
| *Figure 6 : Image provenant de l'appendix du S4* |

<br>

Le S4 établit le SoTA sur sCIFAR-10 avec seulement 100 000 paramètres (les auteurs ne précisant pas leur nombre pour les autres méthodes).

<br>

Concluons avec une tâche textuelle et le benchmark [*Long Range Arena* (LRA)](https://arxiv.org/abs/2011.04006) de TAY et al. (2020).

|![image](https://github.com/lbourdois/blog/assets/58078086/a9eab9ac-6753-4242-8788-c0f28616ccb0)|
|:--:|
| *Figure 7 : Image provenant de l'appendix du S4*|

<br>

Le LRA est composé de 6 tâches dont Path-X d'une longueur de 16K tokens pour laquelle le S4 est le premier modèle à la réussir démontrant ses performances sur des tâches de très longues séquences.  
Il faudra plus de 2 ans pour qu'AMOS et al. montre dans leur papier « [*Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors*](https://arxiv.org/abs/2310.02980.) » (2023) que les *transformers* (non hybridés avec un SSM) peuvent aussi résoudre cette tâche. Ils n'arrivent cependant pas à passer le PathX-256 d'une longueur de 65K tokens contrairement aux SSM.      

A noter néanmoins un point négatif concernant le texte pour le S4 : il obtient une perplexité plus élevée par rapport à celle d'un *transformer* (standard, des versions plus optimisées ayant une perplexité encore plus faible) sur [WikiText-103](https://arxiv.org/abs/1609.07843v1) de MERITY et al. (2016).  

|![image](https://github.com/lbourdois/blog/assets/58078086/b2339a9a-415a-453a-8341-96ce2bd1a61d)|
|:--:|
| *Figure 8 : Image provenant de l'appendix du S4*|

<br>

Cela s'explique probablement par la nature non continue du texte (il n'a pas été échantillonné à partir d'un processus physique sous-jacent comme la parole ou les séries temporelles). Nous verrons dans l'article consacré aux évolutions des SSM en 2023 que ce point a fait l'objet de beaucoup de travaux et que les SSM ont aujourd'hui réussi à combler cet écart.
<br><br><br>

# <span style="color: #FF0000"> **Conclusion** <span>
Les SSM sont des modèles possédant trois vues. Une vue continue, et lorsque nous la discrétisons, une vue récurrente ainsi que convolutive.  
Tout l'enjeu de ce type d'architecture consiste à savoir quand privilégier une vue plutôt qu'une autre en fonction de l'étape du processus (entraînement ou inférence) et du type de données traitées.  
Ce type de modèle est très versatile puisqu’il est applicable pour les tâches de texte, de vision, d’audio, de séries temporelles (ou encore aux graphes).    
Un de ses atouts est d'être capable de gérer de très longues séquences pour généralement un nombre de paramètres inférieurs aux autres modèles (ConvNet ou *transformers*) tout en étant très rapide.  
Nous verrons dans les prochains articles que les principales différences entre les diverses architectures de SSM existantes viennent principalement de la façon de discrétiser l’équation de base des SSM ou encore de définir la matrice $$\mathbf A$$. 
<br><br><br>

# <span style="color: #FF0000"> **Pour aller plus loin** <span>
Concernant le S4, vous pouvez consulter les ressources suivantes (toutes en anglais) :
* Vidéos : 
  -	[Efficiently Modeling Long Sequences with Structured State Spaces - Albert Gu - Stanford MLSys #46](https://www.youtube.com/watch?v=EvQ3ncuriCM) par Albert GU
  -	[MedAI #41: Efficiently Modeling Long Sequences with Structured State Spaces](https://www.youtube.com/watch?v=luCBXCErkCs) par Albert GU (un peu plus longue car montre plus d’exemples traités)
  -	[JAX Talk: Generating Extremely Long Sequences with S4](https://www.youtube.com/watch?v=GqwhkbrWDOI) par Sasha RUSH + les [slides](https://srush.github.io/annotated-s4/slides.html#22 ) utilisées dans la vidéo  
* Codes :
  - [The Annotated S4 ](https://srush.github.io/annotated-s4/ ) (en Jax) par Sasha RUSH et Sidd KARAMCHETI
  - [Le GitHub de l'implémentation officielle du S4](https://github.com/state-spaces/s4) (en PyTorch)  
* Articles de blog :
   - Les articles sur le S4 issus du blog Hazy Research qui est le groupe de recherche de Stanford où Albert GU a fait son doctorat ; [partie 1](https://hazyresearch.stanford.edu/blog/2022-01-14-s4-1), [partie 2](https://hazyresearch.stanford.edu/blog/2022-01-14-s4-2) et [partie 3](https://hazyresearch.stanford.edu/blog/2022-01-14-s4-3).  
* Papier :
  - Le prédécesseur du S4 est le [Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks](https://proceedings.neurips.cc/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf) (LMU) de VOELKER et al. (2019)
    
Concernant la matrice $$HiPPO$$, vous pouvez consulter les ressources suivantes (toutes en anglais) :
-	L'[article du blog Hazy Research](https://hazyresearch.stanford.edu/blog/2020-12-05-hippo) consacré au sujet
-	Le papier [*How to Train Your HiPPO: State Space Models with Generalized Orthogonal Basis Projections*](https://arxiv.org/abs/2206.12037) d'Albert GU et al. (2022)  

Concernant les SSM, vous pouvez regarder :
- le cours (en français) sur les [systèmes dynamiques](https://www.youtube.com/watch?v=sDD13PI89hA&list=PLImFpdng6y55wxYZgt7hxbocWkeMWHtCN) d'Ion HAZYUK, Maitre de Conferences à l'INSA de Toulouse (la partie les [modèles en espace d'état](https://www.youtube.com/watch?v=XGhDvhHKjiY&list=PLImFpdng6y55wxYZgt7hxbocWkeMWHtCN&index=45) débutent à partie de la section 5.2)
- la [thèse de doctorat](https://searchworks.stanford.edu/view/14784021) (en anglais) d'Albert GU
<br><br><br>

# <span style="color: #FF0000"> **Références** </span> 
- [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) d'Alex KRIZHESKY (2009)
- [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843v1) de Stephen MERITY, Caiming XIONG, James BRADBURY, Richard SOCHER (2016)
- [*Attention is all you need*](https://arxiv.org/abs/1706.03762) de Ashish VASWANI, Noam SHAZEER, Niki PARMAR, Jakob USZKOREIT, Llion JONES, Aidan N. GOMEZ, Lukasz KAISER, Illia POLOSUKHIN (2017)
- [Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition](https://arxiv.org/abs/1804.03209v1) de Pete WARDEN (2018)
- [Long Range Arena: A Benchmark for Efficient Transformers](https://arxiv.org/abs/2011.04006) de Yi TAY, Mostafa DEHGHANI, Samira ABNAR, Yikang SHEN, Dara BAHRI, Philip PHAM, Jinfeng RAO, Liu YANG, Sebastian RUDER, Donald METZLER (2020)
- [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436) d'Haoyi ZHOU, Shanghang ZHANG, Jieqi peng, Shuai ZHANG, Jianxin LI, Hui XIONG, Wancai ZHANG (2020)
- [HiPPO: Recurrent Memory with Optimal Polynomial Projections](https://arxiv.org/abs/2008.07669) d'Albert GU, Tri DAO, Stefano ERMON, Atri RUDRA, Christopher RÉ (2020)
- [Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers](https://arxiv.org/abs/2110.13985) d'Albert GU, Isys JOHNSON, Karan GOEL, Khaled SAAB, Tri DAO, Atri RUDRA, Christopher RÉ (2021)
- [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) d'Albert GU, Karan GOEL, et Christopher RÉ (2021)
- [On the Parameterization and Initialization of Diagonal State Space Models](https://arxiv.org/abs/2206.11893) d'Albert GU,  Ankit GUPTA, Karan GOEL, Christopher RÉ (2022)
- [Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors](https://arxiv.org/abs/2310.02980) d'Ido AMOS, Jonathan BERANT, Ankit GUPTA (2023)
<br><br><br>


# <span style="color: #FF0000"> **Citation** <span>
> @inproceedings{ssm_introduction_blog_post,  
  author    = {Loïck BOURDOIS},  
  title     = {Introduction aux State Space Models (SSM) et au S4},  
  year      = {2023},  
  url = {https://lbourdois.github.io/blog/ssm/introduction_ssm}  
}
