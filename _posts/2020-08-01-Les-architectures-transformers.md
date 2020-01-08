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

| Modèle  | Description | Nb de paramètres  | Taille du vocabulaire  | Tokenizer utilisé  | Nb de tokens max pris en compte   |
|:-:|:-:|:-:|:-:|:-:|:-:|
| [GPT-2](https://openai.com/blog/better-language-models/) | Transformer unidirectionnel développé par OpenAI utilisant uniquement la partie decoder du Transformer original. | 4 versions : 124M, 355M, 774M ou 1.5 Mds  | 50K  | BPE  |  1024 |
| [BERT](https://github.com/google-research/bert) |Transformer bidirectionnel développé par Google utilisant uniquement la partie encoder du Transformer original.   | Base : 108M,<br> Large : 334M,<br> XLarge : 1270M  | 30K  | WordPiece   |   |
| [RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta)  | Version optimisée de BERT développée par Facebook.   | Base : 125M, <br> Large : 355M  | 50K | BPE |   |
| [Distil*](https://github.com/huggingface/transformers/tree/master/examples/distillation)   | Versions distillées (plus petites donc plus rapides) de BERT, du GPT-2 et de RoBERTa développées par Huggingface.   | 66M pour DistillBERT,<br> 82M pour DistillGPT2 et DistilRoBERTa  | Non communiqué   | Non communiqué   |   |
| [CamemBERT](https://camembert-model.fr/)   | Modèle basé sur RoBERTa développé par Facebook, l’INRIA et la Sorbonne. Il est
pré-entraîné sur des textes en français  | 110M   | 32K  | SentencePiece  |  512 |
| [XLNet](https://github.com/zihangdai/xlnet/)  | Transformer bidirectionnel développé par Google et la Carnegie Mellon University avec pré-entraînement autorégressif.    | Identique à BERT  | 32K  | SentencePiece  |   |
| [XLM](https://github.com/facebookresearch/XLM/)  | Transformer proposé par Facebook entrainé en en prenant en compte plusieurs langues (de 15 jusqu’à 100). Il utilise uniquement la partie encoder.   | 665M  | entre 30K et 200K
(varie en fonction du nombre de langues pris en compte (cf la partie II. du GitHub))  | fastBPE   |   |
|   |   |   |   |   |   |
|   |   |   |   |   |   |
|   |   |   |   |   |   |
