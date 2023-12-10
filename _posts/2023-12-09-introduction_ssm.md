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
classes: wide
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# <span style="color: #FF0000"> **Introduction** </span>
TODO

# <span style="color: #FF0000"> **Qu’est-ce qu’un SSM ?** </span>

Les *States Spaces Models* ou représentation d'état en français, sont utilisés traditionnellement en théorie du contrôle afin de modéliser un système dynamique via des variables d'état.  
[Wikipedia](https://fr.wikipedia.org/wiki/Repr%C3%A9sentation_d%27%C3%A9tat) indiquant qu'ils sont également présents en automatique, il n'est pas exclu qu'ils soient utilisés dans d'autres domaines et/ou sous la forme d'un autre nom.  
Dans le cadre de l'apprentissage profond, lorsque l'on parle de SSM, on se réfère en réalité qu'à un sous-ensemble des représentations existantes, à savoir les systèmes linéaires invariants (ou stationnaires).  

## <span style="color: #FFBF00"> **Définition d'un SSM en apprentissage profond** </span>

Utilisons l'image ci-dessous afin de définir un SSM :

| ![image/png](https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/G7icfkYoxIqHZcJGHM7UD.png) |
|:--:|
| *Vue d'un SSM continu et invariant dans le temps, image tirée de [https://en.wikipedia.org/wiki/State-space_representation](https://en.wikipedia.org/wiki/State-space_representation)* |

<br>

On peut observer qu'un SSM repose sur trois variables dépendant du temps $$t$$ :
- $$x(t) \in \mathbb {R}^{n}$$ représente les $$n$$ variables d'état,
- $$u(t) \in \mathbb {R}^{m}$$ représente les $$m$$ entrées d'état,
- $$y(t) \in \mathbb {R}^{p}$$ représente les $$p$$ sorties,

On peut aussi observer qu'il est composé de quatre matrices pouvant être apprises : $$\mathbf A, \mathbf B, \mathbf C$$ et $$\mathbf D$$.
- $$\mathbf A \in \mathbb {R}^{m \times n}$$ est la matrice d'état (contrôlant l'état lattent $$x$$),
- $$\mathbf B \in \mathbb {R}^{n \times m}$$ est la matrice de contrôle,
- $$\mathbf C \in \mathbb {R}^{p \times n}$$ est la matrice de sortie,
- $$\mathbf D \in \mathbb {R}^{p \times m}$$ est la matrice de commande,

Il est possible de ramener l'image ci-dessus au système d'équations suivant :

$$
  \begin{aligned}
    x'(t) &= \mathbf{A}x(t) + \mathbf{B}u(t) \\
    y(t) &= \mathbf{C}x(t) + \mathbf{D}u(t)
  \end{aligned}
$$

Note : j'utilise ici la notation $$x'$$ pour désigner la dérivée de $$x$$. Il n'est pas exclu que vous rencontriez à la place la notation $$ẋ$$ dans la littérature. J'ai pour ma part tout simplement privilégié la notation nécessitant le moins de LaTex possible.

Pour alléger les notations, l'équation précédente est généralement écrite sous la forme suivante, puisqu'il est implicite que les variables dépendent du temps :

$$
  \begin{aligned}
    x' &= \mathbf{A}x + \mathbf{B}u \\
    y &= \mathbf{C}x + \mathbf{D}u
  \end{aligned}
$$

Ce système peut s'alléger même davantage, car dans les SSM en apprentissage profond, $$\mathbf{D}u = 0$$ car est vue comme une *skip connexion* facilement calculable.  

$$
  \begin{aligned}
    x' &= \mathbf{A}x + \mathbf{B}u \\
    y &= \mathbf{C}x
  \end{aligned}
$$

Le système que nous avons ici est continu. Il n'est pas possible de le donner tel quel à un ordinateur. Nous devons pour cela le discrétiser.

## <span style="color: #FFBF00"> **Discrétisation** </span>

La discrétisation est l’un, voire le point le plus important dans les SSM. Toute la beauté de cette architecture réside dans cette étape puis qu’elle permet d’aboutir à deux vues différentes d’un modèle en temps continu : la **vue récursive** et la **vue convolutive**.  
Si vous devez retenir une seule chose de cet article, c’est ce point-ci.

| ![image](https://github.com/lbourdois/blog/assets/58078086/cb2dca34-9a3e-481a-8773-2360a1ceaa1c) |
|:--:|
| *Image provenant de l’article [Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers](https://arxiv.org/pdf/2110.13985.pdf), prédécesseur du S4*|

<br>

En effet, la première vue permet d’avoir un mécanisme efficace lors de l’inférence. La seconde permet quant à elle un entraînement rapide car parallélisable.  
Nous verrons aussi dans les prochains articles qu’il existe plusieurs discrétisations possibles. C’est d’ailleurs l’une des différences principales entre les différentes architectures de SSM existantes.  
Pour ce premier article, nous appliquons la discrétisation "originale" appliquée dans le S4 afin d'illustrer ces deux vues possibles pour un SSM.  
Faisons donc un peu de mathématiques.

## <span style="color: #FFBF00"> **Vue récursive d’un SSM** </span>

Pour discrétiser le cas continu, utilisons la méthode des trapèzes où le principe est d'assimiler la région sous la courbe représentative d'une fonction $$f$$ définie sur un segment $$[a , b]$$ à un trapèze et d'en calculer l'aire $$T$$ : $$T=(b-a){\frac  {f(a)+f(b)}{2}}$$.  

Posons $$x’(t) = f(t,x)$$. On a alors :  $$x_{n+1} = x_n + \frac{1}{2}\Delta(f(t_n,y_n) + f(t_{n+1},y_{n+1}))$$ avec $$\Delta = t_{n+1} - t_n$$.  
Si $$x'_n = \mathbf{A}x_n + \mathbf{B} u_n$$ (première ligne de l’équation d’un SSM), correspond à $$f$$, alors :  

$$
\begin{align}
x_{n+1} & = x_n + \frac{\Delta}{2} (\mathbf{A}x_n + \mathbf{B} u_n + \mathbf{A}x_{n+1} + \mathbf{B} u_{n+1}) \\
\Longleftrightarrow  x_{n+1} - \frac{\Delta}{2}\mathbf{A}x_{n+1} & = x_n + \frac{\Delta}{2}\mathbf{A}x_{n} + \frac{\Delta}{2}\mathbf{B}(u_{n+1} + u_n) \\
(*) \Longleftrightarrow  (\mathbf{I}  - \frac{\Delta}{2} \mathbf{A}) x_{n+1} & = (\mathbf{I}  + \frac{\Delta}{2} \mathbf{A}) x_{n} + \Delta \mathbf{B} u_{n+1}\\
\Longleftrightarrow  x_{n+1} & = (\mathbf{I}  - \frac{\Delta}{2} \mathbf{A})^{-1} (\mathbf{I}  + \frac{\Delta}{2} \mathbf{A}) x_n + (\mathbf{I}  - \frac{\Delta}{2} \mathbf{A})^{-1} \Delta \mathbf{B} u_{n+1}
\end{align}
$$

(*) On pose $$u_{n+1} = u_n$$ car le vecteur de contrôle est supposé constant sur un petit $$\Delta$$.

Nous venons d’obtenir notre SSM discretisé !  
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
Dans le système précédent, nous avons d’écrire une version enroulée d’un réseau de neurones récurrents. La version déroulée de la récurrence étant la suivante :  
[IMAGE]  

Vous pouvez observer ici que $$\mathbf{\bar{A}}$$ est nécessaire pour initialiser le réseau.   
Son choix est très important puisqu’un mauvais $$\mathbf{\bar{A}}$$ (techniquement $$\mathbf{A}$$ car $$\mathbf{\bar{A}}$$ est juste une réécriture de la matrice) peut aboutir à un SSM donnant de très mauvais résultats.    
Dans le cadre de cet article, nous donnerons plus bas la formule de la matrice $$\mathbf{A}$$ utilisée dans le S4. Nous verrons dans les articles suivants quels choix de $$\mathbf{\bar{A}}$$ sont possbiles.  

## <span style="color: #FFBF00"> **Vue convolutive d’un SSM** </span>

La récurrence que nous venons d’écrire peut s’écrire sous la forme d’une convolution. Pour cela, il suffit d’itérer les équations du système 

$$
  \begin{aligned}
    x_k &= \mathbf{\bar{A}}x_{k-1} + \mathbf{\bar{B}}u_k \\
    y_k &= \mathbf{\bar{C}}x_k
  \end{aligned}
$$

Commençons par la première ligne du système :  
Etape 0 : $$x_0 = \mathbf{\bar{B}}  u_0$$  
Etape 1 : $$x_1 = \mathbf{\bar{A}}x_{0} + \mathbf{\bar{B}}u_1  = \mathbf{\bar{A}} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{B}}u_1$$  
Etape 2 : $$x_2 = \mathbf{\bar{A}}x_{1} + \mathbf{\bar{B}}u_2  = \mathbf{\bar{A}} (\mathbf{\bar{A}} \mathbf{\bar{B}}  u_0) + \mathbf{\bar{B}}u_2 = \mathbf{\bar{A}}^{2} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{A}} \mathbf{\bar{B}}  u_1 + \mathbf{\bar{B}}u_2$$  
Nous avons $$x_k$$ qui peut s’écrire sous la forme d’une fonction $$f$$ paramétrée par $$u_0, u_1, … u_k$$.  

Passons ensuite à la seconde ligne du système où il est à présent possible d’injecter les valeurs $$x_k$$ calculées à l’instant :  
Etape 0 : $$y_0 = \mathbf{\bar{C}} x_0  = \mathbf{\bar{C}}  \mathbf{\bar{B}}  u_0$$  
Etape 1 : $$y_1 = \mathbf{\bar{C}} x_1  =  \mathbf{\bar{C}} ( \mathbf{\bar{A}} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{B}}u_1) =  \mathbf{\bar{C}} \mathbf{\bar{A}} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{C}} \mathbf{\bar{B}}u_1$$  
Etape 2 : $$y_2 = \mathbf{\bar{C}} x_2 =  \mathbf{\bar{C}}(\mathbf{\bar{A}}^{2} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{A}} \mathbf{\bar{B}}  u_1 + \mathbf{\bar{B}}u_2 ) = \mathbf{\bar{C}}\mathbf{\bar{A}}^{2} \mathbf{\bar{B}}  u_0 + \mathbf{\bar{C}}\mathbf{\bar{A}} \mathbf{\bar{B}}  u_1 + \mathbf{\bar{C}}\mathbf{\bar{B}}u_2$$  
On peut observer le noyau de convolution $$\mathbf{\bar{K}} _k = (\mathbf{\bar{C}}  \mathbf{\bar{B}}, \mathbf{\bar{C}} \mathbf{\bar{A}}  \mathbf{\bar{B}}, …, \mathbf{\bar{C}}  \mathbf{\bar{A}}^{k} \mathbf{\bar{B}})$$ applicable aux $$u_k$$, d’où $$K \ast u$$.  

Comme pour les matrices, nous appliquons une barre sur le $$\mathbf{\bar{K}}$$  pour spécifier qu’il s’agit du noyau de convolution obtenu après discrétisation. Il est généralement appelé noyau de convolution SSM dans la littérature et sa taille est équivalente à l’entièreté de la séquence d’entrée.  
Ce noyau de convolution est calculé par Transformation de Fourier rapide (FFT) mais nous détaillerons cela dans les prochains articles (vous aimez la Flash Attention des *transformers* ? Vous adorerez la Flash FFT Convolution que nous verrons dans le troisième article de blog).

## <span style="color: #FFBF00"> **Apprentissage des matrices A, B et C** </span>

Dans le noyau de convolution développé à l’instant, $$\mathbf{\bar{C}}$$ et $$\mathbf{\bar{B}}$$, sont des scalaires apprenables.   
Concernant $$\mathbf{\bar{A}}$$, nous avons vue que dans notre noyau de convolution, elle s’exprime comme une puissance de $$k$$ au temps $$k$$. Cela peut être très long à calculer c’est pourquoi, on cherche à avoir $$\mathbf{\bar{A}}$$ fixe. Pour cela, la meilleure option est de la rendre diagonale. C'est à dire avoir :

$$
A =  \begin{bmatrix} 
\lambda_{1} & 0 & \cdots & 0 \\ 
0 & \lambda_{2} & \cdots & 0 \\ 
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_{n} 
\end{bmatrix} 

\Rightarrow

A^k =  \begin{bmatrix} 
\lambda_{1}^k & 0 & \cdots & 0 \\ 
0 & \lambda_{2}^k & \cdots & 0 \\ 
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_{n}^k 
\end{bmatrix} 
$$

La façon d’avoir une matrice diagonale dans un SSM n’est pas évidente au premier abord.  
Ainsi, en plus du choix de la discrétisation citée ci-dessus, la manière de définir et initier $$\mathbf{\bar{A}}$$ est l’un des points qui différencient les différentes architectures de SSM développées dans la littérature que nous développerons dans le prochain article de blog. En effet, empiriquement, il apparait qu'un SSM initialisé avec une matrice A aléatoire donne de mauvais résultats alors que si l'initialisation est effectué à partir de la matrice $$HiPPO$$ (pour *High-Order Polynomial Projection Operator*), les résultats sont très bons (MNIST de 60% à 98%).    

La matrice $$HiPPO$$a été introduite par les auteurs du S4 dans un précédent papier [LINK]. Elle est repris dans LSSL [LINK] mais aussi dans l'Appendix du S4.
Sa formule est la suivante :  
$$
    \bm{A} &=
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
    \mathbf{A}_{nk} &=
    \begin{cases}%
      (-1)^{n-k} (2k+1) & n > k \\
      k+1 & n=k \\
      0 & n<k
    \end{cases}
$$

Les auteurs prouvent dans leur papier que cette matrice qui est *Normal Plus Low Rank* (NPLR), est équivalente à une matrice diagonale. Et ainsi que la vue réccurente peut se réécrire sous la forme suivante :
et la vue convolutive sous la forme suivante :  

La preuve est basée sur des mathématiques pas forcément triviales (fonction génératrice, noyaux de Cauchy, identité de Woodbury, etc.) mais très élégentes quand on prend le temps de la refaire. Le point principale est qu'elle s'étend sur plus de 8 pages dans la V1 du S4 (pages 14 à 22 dans l'appendix). La reprendre entièrement rendrait cet article de blog trop long alors qu'il se veut être une introduction aux SSM. Je ne vais donc pas rentrer dans les détails de cette matrice.  
Si je vous invite  à lire la V1 du S4, c'est qu'il faut différencier les différentes versions du S4. En effet, on verra dans l'article suivant, que le S4 a bénéficié de plusieurs versions où le but des auteurs a été justement de simplifier la matrice A, en passant d'une matrice HiPPO NPLR à une matrice diagonale. 
Ainsi, si vous comprenez pas les mathématiques sous-jacentes à la matrice HiPPO, ce n’est pas forcément important car ellee n'est plus du tout utilisée en pratique, au profit de (beaucoup) plus simples. 


# <span style="color: #FF0000"> **Quelques résultats** <span>
AJOUTER UNE SECTION SUR LES RESULTATS POUR ILLUSTRER QUE CA SERT POUR TOUT + COMPLEXITE

# <span style="color: #FF0000"> **Conclusion** <span>
Les SSM sont des modèles possédant trois vues. Une vision continue, et lorsque nous la discrétisons, une vue récurrente ainsi que convolutive.  
La vue convolutive sert à entraîner efficacement le modèle, et la vue récurrente permet de réaliser une inférence potentiellement infinie (pas de limite de taille de contexte ou encore de positionnal encoding comparés aux transformers).  
Ce type de modèle est très versatile puisqu’il est applicable pour les tâches de texte, de vision, d’audio, de séries temporelles ou encore de graphes !  
Nous verrons dans les prochains articles que les principales différences entre les diverses architectures de SSM existantes viennent principalement de la façon de discrétiser l’équation de base des SSM ou encore de définir la matrice A. 

# <span style="color: #FF0000"> **Pour aller plus loin** <span>

# <span style="color: #FF0000"> **Remerciements** </span> 
Je tiens à remercier Boris ALBAR, Pierre BEDU et Nicolas PREVOT d’avoir acceptés de monter un groupe de travail sur le sujet des SSM et de m'avoir ainsi accompagné dans la découverte de ce type de modèle.

# <span style="color: #FF0000"> **Références** </span> 
