---
permalink: /other/
title: "Autre"
classes: wide
---

Sur cette page vous pouvez trouver les contenus sur lesquels j'ai travaillé (à titre personnel ou professionnel) mais qui sont référencés sur d'autres sites que mon blog personnel. Il s'agit principalement de traductions de cours, et des créations de jeux de données et de modèles.

# Traductions
## Cours de Yann Le Cun et Alfredo Canziani de la NYU
Cette traduction a été la plus longue à effectuer s'étalant de 2020 à 2022.  
Le contenu est structuré en 19 unités réparties sur 33 vidéos 🎥 de cours (cours magistraux et travaux dirigés) d’une durée totale d’environ 45H, 74 pages web 🌐 résumant les vidéos via les notes prises par les étudiants pendant le cours, et 16 notebooks Jupyter 📓 (en PyTorch) utilisés lors des TD. Enfin un jeu de données de plus de 3000 données parallèles vérifiées manuellement a été créé pour entraîner un modèle de traduction.  
Vous pouvez retrouver toutes ces ressources sur le site internet dédié qui a été conçu à l'occasion : [https://lbourdois.github.io/cours-dl-nyu/](https://lbourdois.github.io/cours-dl-nyu/).


## Cours de Hugging Face 🤗
### Cours de NLP
En 2022, j'ai traduit le cours de traitement automatique du langage de Hugging Face.  
Le contenu est structuré en 10 chapitres comprenant un total de 76 vidéos 🎥 d'une durée totale d'environ 5H, de 78 pages web 🌐 et 61 notebooks Jupyter 📓  (en PyTorch et Tensorflow).  
Vous pouvez retrouver toutes ces ressources sur le site de [Hugging Face](https://huggingface.co/learn/nlp-course/fr/chapter1/1).

### Cours d'audio
En 2023, j'ai traduit le cours de traitement automatique du langage de Hugging Face.  
Le contenu est structuré en 8 unités réparties sur 46 pages web 🌐.  
Vous pouvez retrouver toutes ces ressources sur le site de [Hugging Face](https://huggingface.co/learn/audio-course/fr/).

### Cours sur les modèles de diffusion
En 2023, j'ai traduit le cours sur les modèles de diffusion de Hugging Face.  
Le contenu est structuré en 4 chapitres portant sur 17 pages web 🌐 et 8 notebooks Jupyter 📓 (en PyTorch).  
Vous pouvez retrouver toutes ces ressources sur le GitHub de [Hugging Face](https://github.com/huggingface/diffusion-models-class/tree/main/units/fr) (le contenu n'ayant pas encore été propagé sur le site officiel).

<br><br>

# Modèles et jeux de données

## FAT5
Le FAT5 est une implémentation du T5 en PyTorch avec un objectif UL2 optimisé pour GPGPU développé avec [Boris ALBAR](https://b-albar.github.io/portfolio/).  
Elle utilise des noyaux CUDA et Triton personnalisés ainsi que des optimisations spécifiques pour augmenter le débit et réduire l'utilisation de la mémoire pour l'entraînement et l'inférence d'un facteur 4 par rapport à l'implémentation originale disponible dans Hugging Face.  
Nous l'avons appliquée en pré-entraînant un modèle en français de 147M paramètres en utilisant uniquement une A100. Nous estimons ainsi pouvoir ramener le prix de pré-entraînement d'un tel modèle à seulement 2200€ (estimation faite sur une instance OVH).   
Le code de pré-entrainement est disponible sur [GitHub](https://github.com/catie-aq/flashT5) sous licence Apache-2.0 et les poids du modèle entraîné sur le compte [Hugging Face du CATIE](https://huggingface.co/CATIE-AQ). Un article de blog détaillant notre méthodologie est disponible [ici](https://huggingface.co/spaces/CATIE-AQ/FAT5-rapport).

## NER
A compléter.

## Question Answering
A compléter.
