---
title: "LES ARCHITECTURES ISSUES DU TRANSFORMER"
categories:
  - NLP
tags:
  - Etat de l'art Transformers NLP en français
  - Transformer NLP en français
excerpt : "NLP"
header :
    overlay_image: "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/Transformer/Transformers_blog.png"
toc: true
toc_sticky: true
author_profile: false
sidebar:
    nav: sidebar-sample
    
---

# <span style="color: #FF0000"> **Avant-propos** </span>
Les architectures que j'ai eu l'occasion d'utiliser professionnellement font l'objet d'articles détaillés, à savoir le [GPT2]() et [BERT]().
Nénanmoins énormément d'autres architectures issues du Transformer existes. Le nombre de modèles étant conséquent, je ne prévois pas de faire une présentation détaillée de chacun.
L'objectif de cet article est de faire un résumé succint de quelques unes de ces architectures notamment les principales utilisées en langue anglaise et en langue française.<br>


Cet article est également le dernier de la série NLP. J'espère qu'après avoir lu les articles sur le [Transformer](), le [GPT2](), [BERT](), [les Tokenizers]() et les [Bases de données utilisées dans les articles scientifiques](), le lecteur est capable de comprendre de lui même ces autres architectures.<br>

Enfin, alors que les autres articles de la série NLP sont terminés, celui-ci à vocation à être actualisé au fur et à mesure de l'apparition de nouveaux modèles.
<br><br><br>



# <span style="color: #FF0000"> **Diverses architectures issues du Transformer** </span>

| Modèle  | Description | Nb de paramètres  | Taille du vocabulaire  | Tokenizer utilisé  | 
|:-:|:-:|:-:|:-:|:-:|:-:|
| [GPT-2](https://openai.com/blog/better-language-models/) | Transformer unidirectionnel développé par OpenAI utilisant uniquement la partie decoder du Transformer original. | 4 versions : 124M, 355M, 774M ou 1.5 Mds  | 50K  | BPE  |
| [BERT](https://github.com/google-research/bert) |Transformer bidirectionnel développé par Google utilisant uniquement la partie encoder du Transformer original.   | Base : 108M, Large : 334M, XLarge : 1270M  | 30K  | WordPiece   |
| [RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta)  | Version optimisée de BERT développée par Facebook.   | Base : 125M, Large : 355M  | 50K | BPE |
| [Distil*](https://github.com/huggingface/transformers/tree/master/examples/distillation)   | Versions distillées (plus petites donc plus rapides) de BERT, du GPT-2 et de RoBERTa développées par Huggingface.   | 66M pour DistillBERT, 82M pour DistillGPT2 et DistilRoBERTa  | Non communiqué   | Non communiqué   |
| [CamemBERT](https://camembert-model.fr/)   | Modèle basé sur RoBERTa pré-entraîné sur des textes en français et développé par Facebook, l’INRIA et la Sorbonne | 110M   | 32K  | SentencePiece  |
| [XLNet](https://github.com/zihangdai/xlnet/)  | Transformer bidirectionnel développé par Google et la Carnegie Mellon University avec pré-entraînement autorégressif.    | Identique à BERT  | 32K  | SentencePiece  |
| [XLM](https://github.com/facebookresearch/XLM/)  | Transformer proposé par Facebook entrainé en en prenant en compte plusieurs langues (de 15 jusqu’à 100). Il utilise uniquement la partie encoder.   | 665M  | entre 30K et 200K (varie en fonction du nombre de langues pris en compte (cf la partie II. du GitHub))  | fastBPE   | 
| [ALBERT](https://github.com/google-research/google-research/tree/master/albert)   | Transformer développé par Google utilisant uniquement la partie encoder du Transformer original. C’est une version avec moins de paramètres de BERT permettant de réduire la consommation de mémoire et augmenter la vitesse d’entraînement.   | Base : 12M, Large : 18M, XLarge : 60M XXLarge : 235M  | 30K | SentencePiece    |
| [BART](https://github.com/pytorch/fairseq/tree/master/examples/bart)   | Transformer développé par Facebook reprenant les parties encoder (bidirectionnel) et decoder (autorégressif) du Transformer original.   | Large : 400M  | Non communiqué dans le publication ou sur le GitHub   | BPE  | 
| [T5](https://github.com/google-research/text-to-text-transfer-transformer)  | Transformer développé par Google reprenant les parties encoder et decoder du Transformer original.  | Base : 220M, Small : 60M, Large : 770M, Très Large : 2,8Mds et 11Mds.  | 32K  | SentencePiece  | 


