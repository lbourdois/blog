---
title: "ILLUSTRATION DU WORD EMBEDDING ET DU WORD2VEC"
categories:
  - NLP
tags:
  - word embedding en français
---



# Avant-propos

Cet article est une traduction de l’article de Jay Alammar : [The illustrated word2vec](https://jalammar.github.io/illustrated-word2vec/). Ainsi si dans l’article vous lisez des « Je » ou des « Jay », cela provient d’exemple de l’article original que je n’ai pas pu traduire autrement. 


# 1. Exemple d’introduction

Sur une échelle de 0 à 100, dans quelle mesure êtes-vous introverti/extraverti (où 0 est le plus introverti et 100 le plus extraverti) ? Avez-vous déjà passé un test de personnalité comme le MBTI – ou mieux encore, le test des cinq grands traits de personnalité ? Si vous ne l’avez pas fait, ce sont des tests qui vous posent une liste de questions, puis vous notent sur un certain nombre d’axes, l’introversion/extraversion étant l’un d’eux.


<div class="img-div-any-width" markdown="0">
  <image src="/assets/images/word_embeddings/big-five-personality-traits-score.png>
  <br />
   Exemple de résultat d’un test des cinq grands traits de personnalité. Il aurait une capacité prédictive de réussite scolaire, personnelle, et professionnelle. Pour effectuer ce test : https://projects.fivethirtyeight.com/personality-quiz/. 
</div>
