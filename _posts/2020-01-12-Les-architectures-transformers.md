---
title: "LES ARCHITECTURES ISSUES DU TRANSFORMER"
categories:
  - NLP
tags:
  - Etat de l'art Transformers NLP en français
  - Transformers NLP en français
  - CamemBERT NLP français
  - FlauBERT NLP français
excerpt : "NLP - Aperçu des architectures basées sur le Transformer de Vaswani et al."
header :
    overlay_image: "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/NLP_radom_blog.png"
toc: true
toc_sticky: true
author_profile: false
sidebar:
    nav: sidebar-sample
    
---

# <span style="color: #FF0000"> **Avant-propos** </span>
Les architectures que j'ai eu l'occasion d'utiliser professionnellement font l'objet d'articles détaillés, à savoir le [GPT2](https://lbourdois.github.io/blog/nlp/GPT2/) et [BERT](https://lbourdois.github.io/blog/nlp/BERT/).
Néanmoins énormément d'autres architectures issues du Transformer existes. Le nombre de modèles étant conséquent, je ne prévois pas de faire une présentation détaillée de chacun.
L'objectif de cet article est de faire un résumé succint de quelques unes de ces architectures notamment les principales utilisées en langue anglaise et en langue française.<br>


J'espère qu'après avoir lu les articles sur le [Transformer](https://lbourdois.github.io/blog/nlp/Transformer/), le [GPT2](https://lbourdois.github.io/blog/nlp/GPT2/), [BERT](https://lbourdois.github.io/blog/nlp/BERT/), [les Tokenizers](https://lbourdois.github.io/blog/nlp/Les-tokenizers/) et les [Bases de données utilisées dans les articles scientifiques](https://lbourdois.github.io/blog/nlp/Taches-et-jeux-de-donnees-en-NLP/), le lecteur est capable de comprendre de lui même ces autres architectures.<br>
<br><br><br>



# <span style="color: #FF0000"> **Diverses architectures issues du Transformer** </span>

| Modèle  | Description | Nb de paramètres  | Taille du vocabulaire  | Tokenizer utilisé  | 
|:-:|:-:|:-:|:-:|:-:|:-:|
| [GPT-2](https://openai.com/blog/better-language-models/) | Transformer unidirectionnel développé par OpenAI utilisant uniquement la partie decoder du Transformer original. | 4 versions : 124M, 355M, 774M ou 1.5 Mds  | 50K  | BPE  |
| [BERT](https://github.com/google-research/bert) |Transformer bidirectionnel développé par Google utilisant uniquement la partie encoder du Transformer original.   | Tiny : 4.4M, Mini : 11.3M, Small : 29.1M, Medium : 41.7M, Base : 110M, Large : 334M, XLarge : 1270M  | 30K  | WordPiece   |
| [RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta)  | Version optimisée de BERT développée par Facebook.   | Base : 125M, Large : 355M  | 50K | BPE |
| [CamemBERT](https://camembert-model.fr/)   | Modèle basé sur RoBERTa pré-entraîné sur des textes en français et développé par Facebook, l’INRIA et la Sorbonne | 110M   | 32K  | SentencePiece  |
| [FlauBERT](https://github.com/getalp/Flaubert)  | Modèle basé sur BERT pré-entraîné sur des textes en français et développé par l'Univ. Grenoble Alpes, le CNRS et l'Univ. Paris Diderot | small-cased : 54M, base-uncased : 137M, base-cased : 138M et large-cased : 373M   | 50K  | FastBPE  |
| [Distil*](https://github.com/huggingface/transformers/tree/master/examples/distillation)   | Versions distillées (plus petites donc plus rapides) de BERT, du GPT-2 et de RoBERTa développées par Huggingface.   | 66M pour DistillBERT, 82M pour DistillGPT2 et DistilRoBERTa  | Identique au modèle original   | Identique au modèle original  |
| [XLNet](https://github.com/zihangdai/xlnet/)  | Transformer bidirectionnel développé par Google et la Carnegie Mellon University avec pré-entraînement autorégressif.    | Identique à BERT  | 32K  | SentencePiece  |
| [XLM](https://github.com/facebookresearch/XLM/)  | Transformer proposé par Facebook entrainé en en prenant en compte plusieurs langues (de 15 jusqu’à 100). Il utilise uniquement la partie encoder.   | 665M  | entre 30K et 200K (varie en fonction du nombre de langues pris en compte (cf la partie II. du GitHub))  | fastBPE   | 
| [ALBERT](https://github.com/google-research/google-research/tree/master/albert)   | Transformer développé par Google utilisant uniquement la partie encoder du Transformer original. C’est une version avec moins de paramètres de BERT permettant de réduire la consommation de mémoire et augmenter la vitesse d’entraînement.   | Base : 12M, Large : 18M, XLarge : 60M XXLarge : 235M  | 30K | SentencePiece    |
| [BART](https://github.com/pytorch/fairseq/tree/master/examples/bart)   | Transformer développé par Facebook reprenant les parties encoder (bidirectionnel) et decoder (autorégressif) du Transformer original.   | Large : 400M  | 50K  | BPE  | 
| [T5](https://github.com/google-research/text-to-text-transfer-transformer)  | Transformer développé par Google reprenant les parties encoder et decoder du Transformer original.  | Base : 220M, Small : 60M, Large : 770M, Très Large : 2,8Mds et 11Mds.  | 32K  | SentencePiece  | 

<br><br><br>



# <span style="color: #FF0000"> **Conclusion** </span>
Les architectures sont régulièrement améliorées (fine-tuning, augmentation du nombre de paramètres utilisés, ou au contraire diminution dans le cadre de version distillée, etc…). On peut citer par exemple GPT-2 qui est la seconde version du GPT, RoBERTa qui est une version optimisée de BERT, etc…  
De nouvelles architectures font également leur apparition chaque mois (pratiquement semaine).

Tout ceci fait qu’il est actuellement difficile de pouvoir dire quel modèle se démarquera nettement des autres dans le futur et donc nous concentrer exclusivement sur celui-ci. Cette réflexion est d'ailleurs actuellement utopique puisqu'il n'existe pas (**au moment où j'écris ces lignes et à ma connaissance**) d'architecture permettant de réaliser l'ensemble des tâches de NLP existantes. Par exemple, BERT ne permet pas de générer du texte comme le permet le GPT2. A contrario, le GPT2 n'est pas fait de base pour réaliser de la classification, comme le permet BERT.<br>
Pour les modèles actuellement disponibles en français, CamemBERT fait de la reconnaissance d'entités nommées alors que ce n'est pas le cas de FlauBERT. FlauBERT peut traiter le cas du question-answering ce que ne peut pas faire CamemBERT, etc...

Ainsi dans le cadre d’un projet, il est important de connaître la tâche à effectuer pour appliquer le modèle le plus adéquate à votre problème (vous pouvez par exemple consulter le [Github de l’équipe d’Huggingface](https://github.com/huggingface/transformers) qui facilite grandement l’utilisation de nombreuses architectures de Transformers).
Pensez également à vous tenir régulièrement informés sur les dernières nouveautés (vous pouvez pour ça par exemple consulter le classement [Glue Benchmark](https://gluebenchmark.com/leaderboard) ). Procédez également à des tests pour déterminer le modèle le plus efficace pour votre problème.
<br><br><br>



# <span style="color: #FF0000"> **Citation** <span>
Si vous venez à utiliser des éléments de cet article, veillez s'il vous plait en créditer les auteurs en utilisant par exemple comme suit :<br>
“*Les architectures issues du Transformer* par Loïck BOURDOIS (https://lbourdois.github.io/blog/nlp/Les-architectures-transformers/), 2020<br>
Merci :)
