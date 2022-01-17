---
title: "Le NLP : Natural Language Processing"
permalink: /nlp/
classes: wide
---


## Qu’est-ce que le NLP ?

Le *Natural Language Processing* (NLP) ou Traitement Automatique du Langage Naturel (TALN) en français, est un domaine à l’intersection de l’informatique, de l’intelligence artificielle et de la linguistique.
L’objectif de cette discipline est de permettre à des ordinateurs de traiter le langage naturel produit par des humains.
<br><br>


## Quelles sont les applications concrètes du NLP ?

Voici une liste non exhaustive d'exemples d’applications :
- la traduction automatique
- la classification de texte 
- la génération de texte
- la synthèse de document
- la génération de réponses à des questions (chatbox)
<br><br>


## Quelles sont les méthodes employées pour mettre en place ces applications ?
J'ai réalisé une série d'articles dédiée aux modèles statistiques utilisés en NLP.

Dans un premier temps, vous trouverez les articles de présentation des modèles. Ils sont classés dans l’ordre chronologique de leur popularité d’utilisation.<br> 
Dans un second temps, vous trouverez les articles concernant les sujets utiles à connaître en NLP : les *tokenizers*, les bases de données, etc.

* Le *bag-of-word* : l’[article de présentation](https://lbourdois.github.io/blog/nlp/Bag-of-word/)
    
* Le *word embedding* : l’[article de présentation](https://lbourdois.github.io/blog/nlp/word_embedding/)

* Les RNNs, les LSTMs et ELMo : l’[article de présentation](https://lbourdois.github.io/blog/nlp/RNN-LSTM-GRU-ELMO/)

* Le Seq2seq et le processus d’attention : l’[article de présentation](https://lbourdois.github.io/blog/nlp/Seq2seq-et-attention/)

* Le *transformer* : l’[article de présentation](https://lbourdois.github.io/blog/nlp/Transformer/)

* Les architectures issues du *transformer*<br>
Il en existe ENORMEMENT (plus de 10 000 modèles dans l'API d'Hugging Face en novembre 2021). Une liste non exhaustive est disponible dans le dernier article de la série consacrée au NLP : lire [l'article en question](https://lbourdois.github.io/blog/nlp/Les-architectures-transformers/)<br>
Je présente plus en détails les architectures que j’ai eu l’occasion d’utiliser professionnellement :<br>
    * BERT : l’[article de présentation](https://lbourdois.github.io/blog/nlp/BERT/)<br> 
    * le GPT-2 : l’[article de présentation](https://lbourdois.github.io/blog/nlp/GPT2/)<br>
    
<br>
* Optimisations du *transformer*<br>
L'architecture *transformer* est très performante mais possède quelques lacunes (nombres importants de paramètres, calculs pas forcément les plus efficients, etc.).<br>
Différents travaux cherchent à résoudre ces problèmes. On peut par exemple évoquer :
    * les versions distillées de modèles,<br>
    * les optimisations sur les matrices d'attention comme :  [ALBERT](https://lbourdois.github.io/blog/nlp/ALBERT/), le [Reformer](https://lbourdois.github.io/blog/nlp/Reformer/), Big Bird, Performer, etc.
    
<br> 
* Enfin des articles concernant des sujets utiles à connaître en NLP :
    * Le prétraitement et les *tokenizers* : l'[article](https://lbourdois.github.io/blog/nlp/Les-tokenizers/)
    * Les tâches et jeux de données test fréquemment utilisés : l'[article](https://lbourdois.github.io/blog/nlp/Taches-et-jeux-de-donnees-en-NLP/)
    * L'augmentation de données en NLP : l'[article](https://lbourdois.github.io/blog/nlp/Data-augmentation-in-NLP/)
