---
title: "Les SSM : State Space Models"
permalink: /ssm/
classes: wide
---

7 octobre 2021, alors que je me demandais si [AK](https://hf.co/akhaliq) était un bot ou un humain, je vis passer l'un de ses [tweets](https://twitter.com/_akhaliq/status/1445931206030282756).
Un lien vers une publication sur [open-review.net](https://openreview.net/forum?id=uYLFoz1vlAC) accompagné de l'image suivante :
![image/png](https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/QMpNVGwdQV2jRw-jYalxa.png)

Intrigué par les résultats annoncés, je suis allé lire en quoi consistait ce modèle S3 qui sera renommé moins d'un mois plus tard en [S4](https://twitter.com/_albertgu/status/1456031299194470407) ([lien](https://github.com/lbourdois/blog/blob/master/assets/efficiently_modeling_long_sequences_s3.pdf) de la version de quand il s'appelait encore S3 pour les intéressés).  
Cet article brillant m’avait impressionné. A l'époque, j'étais persuadé que les State Space Models (SSM) allaient être une révolution et remplacer les transformers dans les prochains mois. Deux ans plus tard, force est de constater que je me suis complètement trompé devant le raz-de-marée de LLM qui font l'actualité en NLP.  

Néanmoins ce lundi 4 décembre 2023, l'annonce de Mamba par [Albert Gu](https://twitter.com/_albertgu/status/1731727672286294400) et [Tri Dao](https://twitter.com/tri_dao/status/1731728602230890895) a ressuscité leur intérêt. Phénomène accentué 4 jours plus tard avec l’annonce de [StripedHyena](https://twitter.com/togethercompute/status/1733213267185762411) par Together AI.  
Une bonne occasion pour moi d'écrire quelques mots sur les développements des SSM au cours de ces deux années.  
Je prévois d'écrire trois articles d'abord où le but est d'illustrer les bases des SSM avec le S4 (le "Attention is all you need" du domaine) avant d'effectuer une revue de littérature de l'évolution des SSM depuis ce premier papier :
- [Introduction aux SSM et au S4](https://lbourdois.github.io/blog/ssm/introduction_ssm/)
- [Historique des SSM en 2022](https://lbourdois.github.io/blog/ssm/ssm_en_2022/)
- [Historique des SSM en 2023](https://lbourdois.github.io/blog/ssm/ssm_en_2023/)
  
J'espère également dans un second temps entrer dans les détails des architectures de certains SSM spécifiques avec des animations ✨
