---
title: "L'AUGMENTATION DE DONNEES EN NLP"
categories:
  - NLP
tags:
  - Augmentation de données en NLP
  - Data augmentation in NLP French
  - Augmentation de données en NLP français
excerpt : NLP - Un aperçu des techniques disponibles pour réaliser de l'augmentation de données textuelles en traitement du langage naturel
header :
    overlay_image: "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/NLP_radom_blog.png"
    teaser:"https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/WordNet.png"
toc: true
toc_sticky: true
author_profile: false
sidebar:
    nav: sidebar-sample

---


# <span style="color: #FF0000"> **Avant-propos** </span>
Cet article est une traduction de l’article de Amit Chaudhary : [A Visual Survey of Data Augmentation in NLP](https://amitness.com/2020/05/data-augmentation-for-nlp/). Merci à lui de m’avoir autorisé à effectuer cette traduction.
J’ai ajouté des éléments supplémentaires quand j’estimais que cela était pertinent.
<br><br><br>


# <span style="color: #FF0000"> **Introduction** </span>

Contrairement à la vision par ordinateur où l'augmentation de données d'images est une pratique courante, l'augmentation de données textuelles est assez rare en NLP.
Cela s’explique par le fait que cette pratique est moins essentielle qu’en image car en NLP les données sont disponibles en abondance (les modèles de Transformers étant entraînés par exemple sur les millions de pages de Wikipédia). Néanmoins, pour certaines tâches il se peut que vous manquiez de données. 
Voici un exemple simple que j’ai rencontré professionnellement :<br>
à l’INSERM nous travaillons sur un outil de classification afin de déterminer la nature des traumatismes des patients passant par le service des urgences du centre hospitalier universitaire de Bordeaux. Empiriquement, on s’est aperçu que pour avoir des résultats fiables, il faut environ 600 exemples d’entraînement par classes. A cela doit s’ajouter les effectifs nécessaires pour l’échantillon test. Un tel nombre ne pose pas de problème par exemple pour les chutes à domicile (le nombre de personnes âgés admis aux urgences pour une chute est monstrueux), les accidents de la route, le sport, etc… Mais pour d’autres classes, il manque (heureusement) des effectifs comme par exemple pour les noyades, les morsures d’animaux, les tentatives de suicides, etc…  Même en ayant plus de 7 années d'historique de données.
Ainsi pour obtenir des résultats probants, il nous faut augmenter artificiellement les effectifs de certaines classes.<br> 
L’objectif de cet article est de donner un aperçu des approches actuelles utilisées pour augmenter les données textuelles.
<br><br><br>


# <span style="color: #FF0000"> **1.	La substitution lexicale** </span>
Cette approche consiste à substituer des mots présents dans un texte sans pour autant changer le sens de la phrase.
<br>

## <span style="color: #FFBF00"> **1.1 Substitution basée sur un thésaurus** </span>
Dans cette technique, nous prenons un mot aléatoire de la phrase et le remplaçons par son synonyme à l'aide d'un thésaurus. Par exemple, nous pourrions utiliser la base de données [WordNet](https://wordnet.princeton.edu/), pour l'anglais afin de rechercher les synonymes et effectuer ensuite le remplacement. Il s'agit d'une base de données gérée manuellement, avec des relations entre les mots. 
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/WordNet.png">
</figure>
</center>
[Zhang et al.](https://arxiv.org/abs/1509.01626) ont utilisé cette technique dans leur article de 2015 intitulé "Character-level Convolutional Networks for Text Classification". Mueller et al. ont utilisé une stratégie similaire pour générer 10 000 exemples d’entraînement supplémentaires pour leur modèle de similarité des phrases.
<br><br>

Pour le français, quatre bases sont disponibles. Elles consistent toutes en une traduction de WordNet :
-	La partie en français de la base [EuroWordNet](http://projects.illc.uva.nl/EuroWordNet/) qui répertorie plusieurs langues européennes. Elle est cependant limitée, n’est pas accessible librement, et commence à dater (1998)
-	[WOLF](http://pauillac.inria.fr/~sagot/index.html#wolf) de [Sagot Benoît et Fišer Darja](https://www.researchgate.net/publication/228809373_Construction_d'un_wordnet_libre_du_francais_a_partir_de_ressources_multilingues), datant de 2008
-	[JAWS](https://www.researchgate.net/publication/268302040_JAWS_Just_Another_WordNet_Subset) datant de 2010 qui est plus large que WOLF
-	[WoNef](https://wonef.fr/) datant de 2014 qui est peut être vu comme une extension de JAWS
<br>

Il existe aussi une base de données appelée [PPDB](http://paraphrase.org/#/download) contenant des millions de paraphrases (en anglais et multilingues) que vous pouvez télécharger et utiliser. 
<br><br>

## <span style="color: #FFBF00"> **1.2 Substitution basée sur du word embedding** </span>
Dans cette approche, nous prenons des [word embeddings pré-entrainés tels que Word2Vec](https://lbourdois.github.io/blog/nlp/word_embedding/), GloVe, FastText, Sent2Vec, et nous utilisons les mots les plus proches de celui que l’on souhaite remplacer dans l'espace des embedding. [Jiao et al.](https://arxiv.org/abs/1909.10351) ont utilisé cette technique avec GloVe dans leur article "TinyBert" pour améliorer la généralisation de leur modèle linguistique. [Wang et al.](https://www.aclweb.org/anthology/D15-1306.pdf) l'ont utilisée pour augmenter les tweets nécessaires à l'entraînement de leur modèle.

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/NN word2vec.png">
</figure>
</center>

Par exemple, vous pouvez remplacer le mot par les 3 mots les plus similaires et obtenir trois variations du texte. 

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/word2vec synonyme.png">
</figure>
</center>

Pour l’anglais, il est facile d'utiliser des packages comme Gensim pour accéder à des vecteurs de mots pré-entraînés et obtenir les voisins les plus proches. Par exemple, nous trouvons ici les synonymes du mot "awesome" en utilisant des vecteurs de mots entraînés sur des tweets.

```python
# pip install gensim
import gensim.downloader as api

model = api.load('glove-twitter-25')  
model.most_similar('awesome', topn=5)
```
You will get back the 5 most similar words along with the cosine similarities.
```python
[('amazing', 0.9687871932983398),
 ('best', 0.9600659608840942),
 ('fun', 0.9331520795822144),
 ('fantastic', 0.9313924312591553),
 ('perfect', 0.9243415594100952)]
```  

Vous aurez alors en sortis les 5 mots les plus similaires ainsi que les similitudes calculées via le cosinus.

```python
[('amazing', 0.9687871932983398),
 ('best', 0.9600659608840942),
 ('fun', 0.9331520795822144),
 ('fantastic', 0.9313924312591553),
 ('perfect', 0.9243415594100952)]
```
<br>
Pour le français, plusiseurs choix s’offrent à vous :
-	Les différents words embedding mis à disposition par [Jean-Philippe Fauconnier]( https://fauconnier.github.io/#data) (exemple d’implémentation sur sa page)
-	Ceux de [FastText](https://fasttext.cc/docs/en/crawl-vectors.html) (voir le tableau en bas du lien)
<br><br>

## <span style="color: #FFBF00"> **1.3 Substitution basée sur un modèle de langage masqué (Masked Language Model)** </span>

Des modèles de Transformers tels que BERT (voir partie 2.2 de l’[article du blog]( https://lbourdois.github.io/blog/nlp/BERT/#-21-architecture-du-mod%C3%A8le-)), ROBERTA et ALBERT (voir partie 1.1 de l’[article du blog](https://lbourdois.github.io/blog/nlp/ALBERT/) ont été entraînés sur une grande quantité de texte en utilisant une tâche prétexte "Modélisation du langage masqué" où le modèle doit prédire des mots masqués en fonction du contexte.
<br>
Cette tâche peut être utilisée pour compléter certains textes. Par exemple, nous pourrions utiliser un modèle BERT pré-entraîné, masquer certaines parties du texte et demander au modèle BERT de prédire le token masqué.

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/BERT Mask.png">
</figure>
</center>

Ainsi, nous pouvons générer des variations d'un texte en utilisant les prédictions du masque. Par rapport aux approches précédentes, le texte généré est plus cohérent d'un point de vue grammatical, car le modèle tient compte du contexte lors des prédictions.

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/BERT Mask synonyme.png">
</figure>
</center>

Cette approche est facile à mettre en œuvre avec la librairie open source [Transformers d’Hugging Face](https://github.com/huggingface/transformers). Vous pouvez définir le jeton que vous souhaitez remplacer par <mask> et générer des prédictions.

```python
from transformers import pipeline
nlp = pipeline('fill-mask')
nlp('This is <mask> cool')
```

```python
[{'score': 0.515411913394928,
  'sequence': '<s> This is pretty cool</s>',
  'token': 1256},
 {'score': 0.1166248694062233,
  'sequence': '<s> This is really cool</s>',
  'token': 269},
 {'score': 0.07387523353099823,
  'sequence': '<s> This is super cool</s>',
  'token': 2422},
 {'score': 0.04272908344864845,
  'sequence': '<s> This is kinda cool</s>',
  'token': 24282},
 {'score': 0.034715913236141205,
  'sequence': '<s> This is very cool</s>',
  'token': 182}]
```

De même pour le français, vous pouvez utiliser le code suivant : 

```python
camembert_fill_mask = pipeline("fill-mask", model="camembert-base", tokenizer="camembert-base")

results = camembert_fill_mask("Le camembert est <mask> :)")
```
A noter cependant que pour cette méthode le fait de décider quelle partie du texte est à masquer n'est pas triviale. Vous devrez utiliser l'heuristique pour décider du masque, sinon le texte généré pourrait ne pas conserver le sens de la phrase originale.
<br><br>

## <span style="color: #FFBF00"> **1.4 Substitution basée sur le TF-IDF** </span>

Cette méthode d'augmentation a été proposée par [Xie et al.]( https://arxiv.org/abs/1904.12848) dans le document intitulé Unsupervised Data Augmentation. L'idée de base est que les mots qui ont un score TF-IDF faible ne sont pas informatifs et peuvent donc être remplacés sans affecter le label d’une phrase.

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/TF-IDF.png">
</figure>
</center>

Les mots qui remplacent le mot original sont choisis en calculant les scores TF-IDF des mots sur l'ensemble du document et en prenant les plus bas. Pour l’implémentation, vous pouvez vous référer au code fournit avec la publication, disponible [ici](https://github.com/google-research/uda/blob/master/text/augmentation/word_level_augment.py).
<br><br><br>


# <span style="color: #FF0000"> **2. La rétro-traduction** </span>
Dans cette approche, nous utilisons la traduction automatique pour paraphraser un texte tout en en retravaillant le sens. [Xie et al.](https://arxiv.org/abs/1904.12848) ont utilisé cette méthode pour augmenter leur corpus de texte non labellisé et ont entraîné un modèle semi-supervisé sur un jeu de données [IMDB](https://fr.wikipedia.org/wiki/Internet_Movie_Database) avec seulement 20 exemples étiquetés. Leur modèle a surpassé le précédent modèle de pointe entraîné sur 25 000 exemples étiquetés.
Le processus de rétro-traduction est le suivant :
-	Prendre une phrase (par exemple en anglais) et la traduire dans une autre langue, par exemple français
-	Traduire la phrase en français obtenue précédemment en anglais 
-	Vérifier si la nouvelle phrase est différente de la phrase d'origine. Si c'est le cas, utiliser cette nouvelle phrase comme une version augmentée du texte original

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Traduction1.png">
</figure>
</center>


Vous pouvez également effectuer une rétro-traduction en utilisant différentes langues à la fois pour générer plus de variations. Comme illustré ci-dessous, nous traduisons une phrase anglaise vers une langue cible et inversement vers l'anglais pour trois langues cibles : français, mandarin et italien.

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Traduction2.png">
</figure>
</center>

Cette technique a été utilisée par le gagnant du "Toxic Comment Classification Challenge" sur [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557). Il l'a utilisée pour l'augmentation des données d'entraînement ainsi que pendant le test où les probabilités prédites pour la phrase anglaise ainsi que la rétro-traduction en trois langues (français, allemand, espagnol) ont été calculées pour obtenir la prédiction finale.
<br>
Pour la mise en œuvre de la rétro-traduction, vous pouvez utiliser la librairie [TextBlob]( https://textblob.readthedocs.io/en/dev/) et notamment la fonction *Translator*. Vous pouvez également utiliser Google Sheets et suivre les instructions données par [Amit](https://amitness.com/2020/02/back-translation-in-google-sheets/) pour appliquer Google Translate (en anglais). Cette approche est néanmoins manuelle. Pour automatiser la chose, je vous conseille l'API [Googletrans](https://pypi.org/project/googletrans/)
<br><br><br>


# <span style="color: #FF0000"> **3.	Transformation de la surface du texte** </span>
Il s'agit de transformations introduites par Claude Coulombe dans sa [publication](https://arxiv.org/abs/1812.0471) : Text Data Augmentation Made Simple By Leveraging NLP Cloud APIs.
<br>
Dans son article, il donne un exemple de transformation de formes verbales de la contraction à l'expansion et vice versa. Nous pouvons générer des textes augmentés en appliquant cette transformation.

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Contraction1.png">
</figure>
</center>

Comme la transformation ne doit pas changer le sens de la phrase, nous pouvons voir que cela peut échouer en cas d'expansion de formes verbales ambiguës comme : 

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Contraction2.png">
</figure>
</center>

Pour résoudre ce problème, le document propose de permettre des contractions ambiguës mais de sauter les expansions ambiguës.

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Contraction3.png">
</figure>
</center>

Vous pouvez trouver une liste des contractions pour la langue anglaise [ici](https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions). Pour l'expansion, vous pouvez utiliser la librairie [contractions](https://github.com/kootenpv/contractions) en Python.
<br>
Cette technique ne semble pas avoir d’intérêt pour la langue française puisque celle-ci ne contient pas de contraction/expansion comme en anglais.
<br><br><br>

# <span style="color: #FF0000"> **4.	Injection aléatoire de bruit** </span>
L'idée de ces méthodes est d'injecter du bruit dans le texte afin que le modèle entrainé soit robuste aux perturbations.
<br>

## <span style="color: #FFBF00"> **4.1 Injection de fautes d’orthographe** </span>
Dans cette méthode, nous ajoutons des fautes d'orthographe à un mot aléatoire de la phrase. Ces fautes d'orthographe peuvent être ajoutées par programmation ou à l'aide d'un lexique des fautes d'orthographe courantes, comme cette [liste]( https://github.com/makcedward/nlpaug/blob/master/model/spelling_en.txt) pour l'anglais.

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Erreur_orthographe.png">
</figure>
</center>

Pour le français, j’ai essayé de trouver une liste comme celle en anglais citée à l’instant, sans succès.
<br><br>

## <span style="color: #FFBF00"> **4.2 Injection de fautes de frappe** </span>
Cette méthode tente de simuler les erreurs courantes qui se produisent lors de la saisie sur un clavier à disposition QWERTY en raison des touches très proches les unes des autres. Les erreurs sont injectées en fonction de la distance entre les touches du clavier.

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Erreur_QWERTY.png">
</figure>
</center>

Cette approche est bien sur reproductible pour les claviers AZERTY utilisés par les francophones.
<br><br>

## <span style="color: #FFBF00"> **4.3 Bruits d'unigramme** </span>
Cette méthode a été utilisée par [Xie et al.](https://arxiv.org/abs/1703.02573) ou encore par [Qizhe Xie et al.](https://arxiv.org/abs/1904.12848). L'idée est d'effectuer le remplacement en utilisant des mots échantillonnés à partir de la distribution de fréquence des unigrammes. Cette fréquence est essentiellement le nombre de fois que chaque mot apparaît dans le corpus d’entraînement.

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Erreur_unigramme.png">
</figure>
</center>

## <span style="color: #FFBF00"> **4.4 Bruits parasites** </span>
Cette méthode a été proposée par [Xie et al.](https://arxiv.org/abs/1703.02573) dans leur article. L'idée est de remplacer aléatoirement un mot par un token de remplacement que l’on aura choisi préalablement. 
Dans l'article les auteurs utilisent ' ' comme caractère de remplacement. C’est un moyen d'éviter de trop s'adapter à des contextes spécifiques ainsi qu'un mécanisme de lissage du modèle linguistique. Cette technique a permis d'améliorer la perplexité et les scores BLEU.
 
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Erreur_blank.png">
</figure>
</center>
<br>

## <span style="color: #FFBF00"> **4.5 Mélanges de phrases** </span>
Il s'agit d'une technique naïve qui consiste à mélanger des phrases présentes dans un texte de d’entraînement afin de créer une version augmentée.
 
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Shuffle.png">
</figure>
</center>


## <span style="color: #FFBF00"> **4.6 Insertion aléatoire** </span>
Cette technique a été proposée par [Wei et al.](https://arxiv.org/abs/1901.11196) dans leur article "Easy Data Augmentation". Dans cette technique, nous choisissons d'abord un mot aléatoire dans la phrase qui n'est pas un mot d'arrêt. Ensuite, nous trouvons son synonyme et nous l'insérons dans une position aléatoire de la phrase.
 
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Random_insertion.png">
</figure>
</center>


## <span style="color: #FFBF00"> **4.7 Echange aléatoire** </span>
Cette technique a également été proposée par Wei et al. dans leur article "Easy Data Augmentation". L'idée est d'échanger de manière aléatoire deux mots quelconques dans la phrase.
 
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Swap.png">
</figure>
</center>


## <span style="color: #FFBF00"> **4.8 Suppression aléatoire** </span>
Cette technique a également été proposée par Wei et al. dans leur article "Easy Data Augmentation". Dans ce cas, nous retirons aléatoirement chaque mot de la phrase avec une probabilité *p*.
 
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Deletion.png">
</figure>
</center>
<br><br>



# <span style="color: #FF0000"> **5.	Augmentation par Crossover** </span>
Cette technique a été introduite par [Luque](https://arxiv.org/abs/1909.11241) dans son article sur l'analyse des sentiments pour la TASS 2019. Elle s'inspire de l'opération de [croisement des chromosomes qui se produit en génétique](https://fr.wikipedia.org/wiki/Enjambement_(g%C3%A9n%C3%A9tique)).
Dans cette méthode, un tweet est divisé en deux moitiés et deux tweet aléatoires ayant le même label que le tweet divisé voient leurs moitiés échangées (cf. image ci-dessous). L'hypothèse est que, même si le résultat sera peu grammatical et peu solide sur le plan sémantique, le nouveau texte préservera quand même le sentiment. 
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/crossover1.png">
</figure>
</center> 
Les résultats de l’article montrent que cette technique n'a pas eu d'impact sur la précision mais a permis d’améliorer le F1-score, notamment pour les classes minoritaires (Neutral dans l’article).
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/crossover2.png">
</figure>
</center> 
<br><br>


# <span style="color: #FF0000"> **6.	Manipulation de l’arbre syntaxique** </span>
Cette technique a été utilisée dans le papier de [Coulombe](https://arxiv.org/abs/1812.04718). L'idée est d'analyser et de générer l'arbre des dépendances de la phrase originale, de le transformer à l'aide de règles et de générer une phrase paraphrasée.
<br>
Par exemple, une transformation qui ne change pas le sens de la phrase est la transformation de la voix active à la voix passive de la phrase et vice versa.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Active_voice.png">
</figure>
</center> 
<br><br>


# <span style="color: #FF0000"> **7.	Le mélange de texte (MixUp)** </span>
Le MixUp est une technique simple mais efficace d'augmentation d'images introduite par [Zhang et al.](https://arxiv.org/abs/1710.09412) en 2017. L'idée est de combiner deux images aléatoires dans une certaine proportion dans un mini-batch afin de générer des exemples synthétiques pour l'entraînement. Pour les images, cela signifie combiner des pixels d'image de deux classes différentes. Cela agit comme une forme de régularisation pendant l'entraînement.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Mixup1.png">
</figure>
</center> 

[Guo et al.](https://arxiv.org/abs/1905.08941) ont appliqué cette idée au NLP pour travailler avec du texte. Ils proposent deux nouvelles approches pour appliquer Mixup au texte :
<br><br>

## <span style="color: #FFBF00"> **7.1 WordMixUp** </span>
Dans cette méthode, deux phrases d'un mini-batch sont prises aléatoirement et dimensionnées à la même longueur. Ensuite, les mots qui les composent sont combinés dans une certaine proportion. Le word embedding qui en résulte est transmis au flux habituel pour la classification du texte. L'entropie croisée est calculée pour les deux labels du texte original dans la proportion donnée.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Mixup2.png">
</figure>
</center>

## <span style="color: #FFBF00"> **7.2	SentMixup** </span>
Dans cette méthode, on prend deux phrases et on les met à la même longueur. Ensuite, leurs word embedding sont passés dans un encoder LSTM/CNN et nous prenons le dernier état caché comme embedding de la phrase. Ces embedding sont combinés dans une certaine proportion et sont ensuite transmis à la couche de classification finale. La perte d'entropie croisée est calculée sur la base des deux labels des phrases originales dans la proportion donnée. 
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Mixup3.png">
</figure>
</center>
<br><br>


# <span style="color: #FF0000"> **8.	Méthodes génératives** </span>
[Kumar et al](https://arxiv.org/abs/2003.02245) proposent dans leur article d’utiliser des transformers pré-entrainés afin d’augmenter les données d'entraînement. La formulation du problème est la suivante :
-	Ajouter le label de la classe à chaque texte de vos données d'entraînement
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Generative1.png">
</figure>
</center>
-	Fine-tuner un grand modèle de langue préformé (BERT/GPT2/BART) sur ces données de d’entraînement modifiées. Pour le GPT2, la tâche est la génération tandis que pour BERT, l'objectif est la prédiction du jeton masqué.
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Generative2.png">
</figure>
</center>
-	En utilisant le modèle de langage fine-tuné, de nouveaux échantillons peuvent être générés en utilisant le label de la classe et quelques mots initiaux comme « prompt ».  
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/Generative3.png">
</figure>
</center>
<br>
Pour le français, cette méthode est difficilement applicable, c’est pourquoi je ne vous la recommande pas. En effet, il n’existe pas de GPT2 pré-entrainé sur du français avec un vocabulaire en français (c’est ce qu’on essaye de faire à l’INSERM). Le créer nécessite d’importantes ressources de calcul, ainsi que d’importantes masse de données textuelles. Deux points qui sont difficile à réunir. 
<br><br><br>


# <span style="color: #FF0000"> **9. La simplification de textes** </span>
J’ajoute une méthode supplémentaire qu’Amit n’a pas cité dans son article : les modèles permettant la simplification de texte. Ils permettent de conserver le sens de la phrase mais avec une syntaxe différente et souvent plus courte.
Pour la langue anglaise, le jeu de données [ASSET]( https://github.com/facebookresearch/asset) de [Fernando Alva-Manchego, Louis Martin et al.]( https://arxiv.org/pdf/2005.00481.pdf) est disponible depuis mai 2020. Il permet de fine-tuner les modèles de simplification de texte.
 <center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Data_augmentation/ASSET.png">
</figure>
</center>
Pour la langue française, il n’existe pas à ma connaissance un tel jeu de données.
<br><br><br>


# <span style="color: #FF0000"> **Implémentation** </span>
Les librairies Python comme [nlpaug](https://github.com/makcedward/nlpaug) et [textattack](https://github.com/QData/TextAttack) fournissent une API simple afin d’appliquer les méthodes ci-dessus pouvant ainsi être facilement intégrées dans une pipeline.
<br><br><br>


# <span style="color: #FF0000"> **Conclusion** </span>
Il ressort de la revue de la littérature que nombre de ces méthodes d'augmentation sont très spécifiques et que leur impact sur les performances n'a été étudié que pour certains cas d'utilisation particuliers. Il serait intéressant de comparer systématiquement ces méthodes et d'analyser leur impact sur les performances pour de nombreuses tâches.
<br><br><br>


# <span style="color: #FF0000"> **Citation** </span>
Si vous venez à utiliser des éléments de cet article, veillez s'il vous plait en créditer les auteurs en utilisant par exemple comme suit :
“*L'augmentation de données en NLP* par Loïck BOURDOIS (https://lbourdois.github.io/blog/nlp/ALBERT/), d’après Amit CHAUDHARY, *A Visual Survey of Data Augmentation in NLP* (https://amitness.com/2020/05/data-augmentation-for-nlp/)”
Merci :)

