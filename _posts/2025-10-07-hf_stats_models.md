---
title: "TEST"
tags:
  - Hugging Face
excerpt: "Divers – Explorer les composants principaux des agents individuels et multiples"
header:
    overlay_color: "#1C2A4D"
    teaser: "https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Agents/image_0.png"
author_profile: false
sidebar: false
classes: wide
layout: single
---

<style>
/* Forcer la largeur maximale pour le contenu */
.page__content {
    max-width: 100% !important;
    width: 100% !important;
}

/* S'assurer que le conteneur principal utilise toute la largeur */
.page {
    max-width: 100% !important;
    width: 100% !important;
}

/* Optimiser l'iframe pour la responsivité */
.full-width-iframe {
    width: 100%;
    height: 1200px;
    border: none;
    margin: 0;
    padding: 0;
}

/* Media query pour les écrans plus petits */
@media (max-width: 768px) {
    .full-width-iframe {
        height: 400px;
    }
}

h1 {
    color: red; /* Rouge */
    font-weight: bold;
}

h2 {
    color: #FFBF00; /* Jaune-orangé (eFFBF00 est souvent interprété comme FFBF00, qui est une nuance jaune-orangé) */
    font-weight: bold;
}

h3 {
    color: #51C353; /* Vert clair */
    font-weight: bold;
}

</style>

<div style="width: 100%; margin: 0; padding: 0;">

<h1>Introduction</h1>
<p>Dans cet article de blog, nous nous intéressons à analyser les <strong>modèles open-source les plus impactants en pratique</strong>.
Pour cela nous nous focalisons sur une métrique très pragmatique : « quels sont les modèles les plus téléchargés sur le Hub d'Hugging Face ? ».
L'hypothèse étant que des modèles massivement téléchargés sont ceux qui sont utilisés dans le monde réel.
C'est une approche qui se veut également plus juste vis-à-vis des individus/organisations qui n'ont pas un service de communication ou n'étant pas suivis/likés massivement sur le Hub.</p>

<br><br><br>

<h1>Données collectées</h1>
<p>Les données visibles dans cet article ont été collectées le 1er octobre 2025.<br>
Après avoir établi les 50 entités open-source comptant le plus de téléchargements, nous avons collecté tous les modèles associés à celles-ci. 
Cela représente 72 423 modèles sur les 2 126 833 hébergés sur le Hub soit 3,41% du total.<br>
Ces comptes représentent exactement 36 450 707 797 téléchargements sur les 45 438 836 957 totaux soit 80,22%.<br><br>

Pour chacun des modèles, les tags <code>pipeline</code> et de <code>language</code> ont également été recueillis lorsqu'ils étaient disponibles. 
De même, la taille estimée d'après le fichier <code>.safetensors</code>.<br><br>

Quand des informations étaient manquantes (pas de tags ou de taille de modèle notamment), 
nous avons manuellement corrigé les données en consultant la carte de modèle ou les publications des 1000 modèles open-source les plus téléchargés. 
Ces derniers représentant à eux seuls 77,89% de tous les téléchargements du Hub et 97,10% des 50 entités analysées.<br>

Le tout a finalement été stocké dans un dataframe ressemblant à ceci :</p>

<iframe
  src="https://huggingface.co/datasets/lbourdois/huggingface_downloads_October_2025"
  width="100%"
  height="600"
  frameborder="0"
></iframe>
<center><i>Figure 1: Jeu de données utilisé, disponible  <a href="https://huggingface.co/datasets/lbourdois/huggingface_downloads_October_2025" target="_blank">ici</a></i></center>

<br><br>

<p>C'est à partir de celui-ci que nous générons l'ensemble des graphiques visibles ci-après.<br><br>

Les montants sont arrondis au million le plus proche pour une question de visibilité mais aussi parce que le Hub rencontre quelques soucis d'affichage. 
En effet, par exemple pour le modèle <a href="https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1">sentence-transformers/static-retrieval-mrl-en-v1</a>, aucun téléchargement n'est <a href="https://huggingface.co/docs/hub/models-download-stats">disponible</a>. 
L'objectif ici est surtout de connaître les ordres de grandeurs plutôt que de s'intéresser aux nombres à l'unité près.</p>

<figure>
    <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/bJTNpy1u_HxRw3AfpTepK.png" alt="Download stats issue example" style="max-width:70%;">
    <figcaption>
        Figure 2: Exemple de ce que l'on peut observer pour les modèles où le Hub n'affiche pas correctement le nombre de téléchargement
    </figcaption>
</figure>

<br><br><br>

<h1>Graphiques</h1>

<h2>Aperçu global</h2>
<p>Dans cette première section, nous affichons les téléchargements globaux de chacune des 50 principales entités contribuant à l'open-source, 
ainsi que leur type de catégorie et leur pays d'origine. Nous revenons sur ces deux derniers points dans des sections dédiées.<br><br>

Notons que nous utilisons le terme « entité » et pas « compte » (Hugging Face) car une entité peut être composée de plusieurs comptes. 
Par exemple Google est composée de <a href="https://huggingface.co/google">google</a>, <a href="https://huggingface.co/google-bert">google-bert</a>, <a href="https://huggingface.co/google-t5">google-t5</a> et <a href="https://huggingface.co/albert">albert</a>.<br>
Ainsi, nous proposons un graphique <em>global</em> permettant de comparer entre-elles les entités sont les plus téléchargées et un autre <em>par sous-comptes</em> pour visualiser comment se répartissent les différents comptes en leur sein.</p>

<br>

<b>Vision globale</b>

<br>

<iframe
    class="full-width-iframe"
    src="{{ '/assets/images/hf_models_stats/huggingface_downloads.html' | relative_url }}" 
    allowfullscreen
></iframe>
Figure 3: Top 50 des entités par nombre total de téléchargements

<br>

<b>Vision par sous-comptes</b>

<br>

<iframe
    class="full-width-iframe"
    src="{{ '/assets/images/hf_models_stats/huggingface_downloads_breakdown.html' | relative_url }}" 
    allowfullscreen
></iframe>
        Figure 4: Top 50 des entités par nombre total de téléchargements (détaillé au niveau des sous-comptes)

<br>

<details>
    <summary>Un mot sur chacune des entités</summary>
    <p>1. L'entitée Google est composée de <a href="https://huggingface.co/google">google</a>, <a href="https://huggingface.co/google-bert">google-bert</a>, <a href="https://huggingface.co/google-t5">google-t5</a> et <a href="https://huggingface.co/albert">albert</a>. Plus de 74% de ses téléchargements proviennent de « vieux modèles » à savoir 64% de BERT, 6,8% de T5 et 3,2% d'ALBERT.</p>
    <p>2. L'entitée Meta est composée de <a href="https://huggingface.co/FacebookAI">FacebookAI</a>, <a href="https://huggingface.co/facebook">facebook</a> et <a href="https://huggingface.co/meta-llama">meta-llama</a>. Observations similaires que pour Google mais dans une moindre mesure avec 48,3% des téléchargements provenant de ses RoBERTa contre 9% pour les Llamas.</p>
    <p>3. L'entitée Sentence-transformers (de l'<em>Ubiquitous Knowledge Processing Lab</em> à Darmstadt en Allemagne et plus précisément les travaux de Nils Reimers) complète le trio. Cette entitée est composée de <a href="https://huggingface.co/sentence-transformers">sentence-transformers</a> et <a href="https://huggingface.co/cross-encoder">cross-encoder</a>. Le compte <code>sentence-transformers</code> est d'ailleurs le plus téléchargé de tout Hugging Face.</p>
    <p>4. L'entitée Hugging Face est composée de <a href="https://huggingface.co/timm">timm</a> (52,4% des téléchargements), de <a href="https://huggingface.co/distilbert">distilbert</a> (44,6%), <a href="https://huggingface.co/llava-hf">llava-hf</a>, <a href="https://huggingface.co/HuggingFaceTB">HuggingFaceTB</a> et <a href="https://huggingface.co/HuggingFaceM4">HuggingFaceM4</a>.</p>
    <p>5. L'entitée OpenAI est composée d'<a href="https://huggingface.co/openai">openai</a> (72%) et <a href="https://huggingface.co/openai-community">openai-community</a> (28%). Bien qu'elle publie peu en open-source, l'entreprise est extrêmement impactante lorsqu'elle le fait (ses CLIP et Whisper sont particulièrement téléchargés).</p>
    <p>6. 99% des téléchargements du MIT proviennent de son modèle <a href="https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593">ast-finetuned-audioset-10-10-0.4593</a> qui est le deuxième modèle le plus téléchargé de tout le Hub.</p>
    <p>7. Microsoft est une autre Big Tech populaire. Le graphique de la section sur les modalités montre que c'est l'organisation la plus diversifiée en termes de modalité adressée (là où, à l'exception de quelques autres, la plupart des entités du top 50 sont hyper spécialisées sur une modalité donnée).</p>
    <p>8. Jonatas Grosman est l'individuel le plus téléchargé. Avec à peine 300 followers sur le Hub, il est une illustration du fait que ce ne sont pas forcément les entités les plus suivies / likées qui sont les plus impactantes. Il s'est un temps spécialisé dans le finetuning de wav2vec2 mais ne publie plus de nouveaux modèles depuis 3 ans maintenant.</p>
    <p>9. Pyannote est spécialisée dans des petits modèles de segmentation et diarization d'audio.</p>
    <p>10. 99% des téléchargements de Falcons.ai (350 followers) proviennent de son modèle <a href="https://huggingface.co/Falconsai/nsfw_image_detection">nsfw_image_detection</a> qui est le septième modèle le plus téléchargé de tout le Hub.</p>
    <p>11. BAAI est l'entitée chinoise la plus téléchargée sur Hugging Face notamment via ses modèles <a href="https://huggingface.co/collections/BAAI/bge-66797a74476eb1f085c7446d">bge</a>. C'est aussi l'entitée non-profit la plus téléchargée.</p>
    <p>12. L'entitée Alibaba est composée de <a href="https://hf.co/Qwen">Qwen</a> (81,5%), <a href="https://hf.co/Alibaba-NLP">Alibaba-NLP</a> (9,5%), <a href="https://hf.co/thenlper">thenlper</a> (7,7%), <a href="https://hf.co/Wan-AI">Wan-AI</a>, <a href="https://hf.co/AIDC-AI">AIDC-AI</a>, <a href="https://hf.co/alibaba-pai">alibaba-pai</a>, <a href="https://hf.co/alibaba-damo">alibaba-damo</a>. Elle est ainsi principalement téléchargée pour ses modèles Qwen (surtout le <a href="https://hf.co/Qwen/Qwen2.5-1.5B-Instruct">Qwen2.5-1.5B-Instruct</a> qui représente 20% de ses téléchargments et est le LLM le plus téléchargé du Hub).</p>
    <p>13. L'entitée Amazon est composée d'<a href="https://hf.co/amazon">amazon</a> (80,1%) et d'<a href="https://hf.co/autogluon">autogluon</a> (19,9%). C'est la seule entitée dont les téléchargements portent massivement sur les séries temporelles.</p>
    <p>14. Dima806 est principalement téléchargé pour son modèle <a href="https://huggingface.co/dima806/fairface_age_image_detection">dima806/fairface_age_image_detection</a>. C'est un individuel encore actif en 2025.</p>
    <p>15. Cardiffnlp est téléchargée pour ses nombreux modèles de classification de sentiment.</p>
    <p>16. L'entitée Stability AI est composée de <a href="https://hf.co/stabilityai">stabilityai</a> (80%) et de <a href="https://hf.co/stable-diffusion-v1-5">stable-diffusion-v1-5</a> (20%). Elle est principalement téléchargée pour ses différentes version de Stable diffusion.</p>
    <p>17. L'entitée Maziyar Panahi est composée de <a href="https://hf.co/MaziyarPanahi">MaziyarPanahi</a> (80,7%) qui est son compte individuel où il propose des version GGUF de LLM, et d'<a href="https://hf.co/OpenMed">OpenMed</a> (19,3%) qui est une organisation non-profit qu'il a créé et dédiée aux modèles médicaux. C'est un individuel encore actif en 2025.</p>
    <p>18. Helsinki-NLP est téléchargée pour ses nombreux modèles de traduction automatique.</p>
    <p>19. Laion est principalement téléchargée pour ses nombreux modèles reproduisant <a href="https://huggingface.co/collections/laion/openclip-laion-2b-64fcade42d20ced4e9389b30">CLIP</a>.</p>
    <p>20. Juan Manuel Pérez via l'organsiation <code>pysentimiento</code> est principalement téléchargé pour ses modèles de classification de sentiment en espagnol. Il n'a plus publié de modèles depuis 2023.</p>
    <p>21. Bingsu est principalement téléchargé pour les différents yolo contenus dans <a href="https://huggingface.co/Bingsu/adetailer">Bingsu/adetailer</a>. C'est un individuel encore actif en 2025.</p>
    <p>22. La moitié des téléchargements d'AllenAI provient de son <a href="https://huggingface.co/allenai/longformer-base-4096">longformer</a>.</p>
    <p>23. Tohoku-nlp est un groupe universitaire téléchargé majoritairement pour <a href="https://huggingface.co/tohoku-nlp/bert-base-japanese">sa version japonaise de BERT</a>.</p>
    <p>24. Manuel Romero est massivement téléchargé pour ses finetunings de t5 mais surtout son modèle financier <a href="https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis">distilroberta-finetuned-financial-news-sentiment-analysis</a>. C'est un individuel encore actif en 2025.</p>
    <p>25. Mistral AI est principalement téléchargé pour les versions 7B instruct de ses modèles.</p>
    <p>26. Les téléchargements de Prajjwal1 reposent sur des convertions en Pytorch de bert-tiny (4M de paramètres) et bert-small (29M). Il n'a plus publié de modèles depuis 2023.</p>
    <p>27. Deepset est téléchargé pour son modèle de <a href="https://huggingface.co/deepset/roberta-base-squad2">QA en anglais</a>.</p>
    <p>28. Salesforce connait du succès avec ses différentes versions de son modèle <a href="https://huggingface.co/collections/Salesforce/blip-models-65242f40f1491fbf6a9e9472">BLIP</a>.</p>
    <p>29. Intfloat est considéré comme un individuel chinois puisque Liang Wang a décidé de publier son travail sous son nom sur Hugging Face. En pratique, il s'agit des modèles <a href="https://huggingface.co/collections/intfloat/multilingual-e5-text-embeddings-67b2b8bb9bff40dec9fb3534">e5</a> créés dans le cadre de son travail à Microsoft Asia. C'est un individuel encore actif en 2025.</p>
    <p>30. TheBloke est connu pour proposer des versions quantifiées de modèles. Sa version ayant eu le plus de succès est le <a href="https://huggingface.co/TheBloke/phi-2-GGUF">phi-2-GGUF</a>. Il n'a plus publié de modèles depuis 2024.</p>
    <p>31. CompVis est populaire pour son modèle <a href="https://huggingface.co/collections/CompVis/stable-diffusion-safety-checker">stable-diffusion-safety-checker</a>.</p>
    <p>32. CIDAS est très utilisé pour son modèle de segmentation <a href="https://huggingface.co/collections/CIDAS/clipseg-rd64-refined">clipseg-rd64-refined</a>.</p>
    <p>33. Emily Alsentzer a connu un certain succès avec son <a href="https://huggingface.co/collections/emilyalsentzer/Bio_ClinicalBERT">Bio_ClinicalBERT</a>. Elle n'a plus publié de modèles depuis 2020.</p>
    <p>34. NVIDIA est extrêmement équilibrée (ce n'est pas un unique modèle qui tire tous les téléchargements de l'entité). Son modèle <a href="https://huggingface.co/collections/nvidia/speakerverification_en_titanet_large">speakerverification_en_titanet_large</a> se démarque légèrement des autres.</p>
    <p>35. LM Studio est composée de <a href="https://huggingface.co/lmstudio-community">lmstudio-community</a> (44,1%), <a href="https://huggingface.co/bartowski">bartowski</a> (55,8%) et <a href="https://huggingface.co/lmstudio-ai">lmstudio-ai</a>.</p>
    <p>36. David S. Lim est téléchargé pour ses modèles de NER en anglais. Il n'a plus publié de modèles depuis 2024.</p>
    <p>37. Unsloth est principalement téléchargé pour ses versions quantifiées de LLM.</p>
    <p>38. mradermacher est principalement téléchargé pour ses versions quantifiées de LLM. C'est un groupe d'individuels encore actif en 2025.</p>
    <p>39. Moritz Laurer est principalement téléchargé pour ses modèles de NLI, notamment <a href="https://hf.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli">DeBERTa-v3-base-mnli-fever-anli</a>. Il n'a plus publié de modèles depuis 2024.</p>
    <p>40. Jean-Baptiste Polle est téléchargé pour ses modèles de NER en français. Il n'a plus publié de modèles depuis 2023.</p>
    <p>41. HFL est un laboratoire hybride (université x entreprise) téléchargé majoritairement pour <a href="https://huggingface.co/hfl/chinese-macbert-base">sa version chinoise de BERT</a>.</p>
    <p>42. Deepseek est assez équilibrée avec une petite prévalence des modèles <a href="https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d">R1</a>.</p>
    <p>43. Big Science est un projet collaboratif international (porté par Hugging Face et le CNRS) par ses modèles bloom, bloomz et t0.</p>
    <p>44. Flair est principalement téléchargé pour ses modèles de NER monolingue dans plusieurs langues.</p>
    <p>45. Sam Lowe est principalement téléchargé pour son modèle de classification d'emotions. Il n'a plus publié de modèles depuis 2023.</p>
    <p>46. Patrick John Chia, dont nous n'avons pas pu déterminer la nationalité, est connu pour son modèle <a href="https://huggingface.co/patrickjohncyh/fashion-clip">fashion-clip</a>. Il n'a plus publié de modèles depuis 2023.</p>
    <p>47. Almanach est un groupe universitaire téléchargé majoritairement pour <a href="https://huggingface.co/almanach/camembert-base">sa version française de BERT</a>.</p>
    <p>48. Supabase est principalement téléchargé pour son modèle <a href="https://huggingface.co/Supabase/gte-small">gte-small</a>.</p>
    <p>49. Jina AI est principalement téléchargé pour ses modèles d'embeddings.</p>
    <p>50. colbert-ir est principalement téléchargé pour son modèle <a href="https://huggingface.co/colbert-ir/colbertv2.0">colbertv2.0</a>.</p>
</details>

<br>

<h2>Vision par pays d'origine du compte</h2>
<p>Pour le pays d'origine, nous nous intéressons dans cette version, aux localisations de l'individu et du siège social de son entreprise. Le but ici est d'estimer le nombre de pays disposant d'un environnement permettant de créer les modèles les plus téléchargés.</p>


<h1>Résumé</h1>
<p>L'analyse des 50 entités les plus téléchargées sur le Hub d'Hugging Face (80,22% du total des téléchargements du Hub) montre que :</p>

<ol>
    <li>Sur l'ensemble des modèles open-source où il est possible de connaître la taille du modèle (96,94% du top 50 et 77,76% du Hub), les petits modèles sont ceux qui sont de loin les plus téléchargés :
        <ul>
            <li>92,48% des téléchargements portent sur des modèles de moins d'un milliard de paramètres,</li>
            <li>86,33% sur des modèles de moins de 500 millions de paramètres,</li>
            <li>69,83% sur des modèles de moins de 200 millions de paramètres,</li>
            <li>40,17% sur des modèles de de moins de 100 millions de paramètres.</li>
        </ul>
    </li>
    <li>Les téléchargements de modèles concernent en premier lieu le NLP (58,1%), suivi par la CV à 21,2%, l'audio à 15,1%, les différentes formes de multimodalité 3,3%, les séries temporelles 1,7% et le reste est indéterminé faute de métadonnées correctement annotées.</li>
    <li>Les encodeurs textuels représentent (entre les modèles de base et leur finetuing sur une tâche particulière), plus de 45% des téléchargements totaux (ou 77,5% de la modalité NLP), contre seulement 9,5% pour les décodeurs (16,5% de la modalité) et 3% pour les encodeurs-décodeurs (5% de la modalité).<br>
    Ainsi, contrairement à l'effervescence autour de ces modèles, les LLM ne sont pas massivement téléchargés en open-source. Leur usage dans le monde réel serait-il alors plutôt du côté des API privées ?</li>
    <li>L'anglais représente plus de 79,46% des téléchargements des modèles (monolingues ou multilingues) utilisant une langue (et même 92,85% si on ne considère que les modèles possédant un tag de langue). Cette langue est loin devant les autres. Par exemple, le français qui arrive en deuxième position ne l'est qu'avec 17,48% (20,43% des modèles possédant un tag de langue).</li>
    <li>Les entreprises sont le contributeur majoritaire à l'open-source avec 63,2% des téléchargements (20 entités sur les 50), suivies à 20,7% par les universités (10 entités), puis par des individuels à 12,1% (16 entités), par des structures non-profit à 3,8% (4 entités) et enfin par les laboratoires hybrides à 0,3% (1 entité).</li>
    <li>Les États-Unis sont présents partout, maillant toutes les modalités (NLP, vision, audio, séries temporelles, multimodalités) et toutes les tailles de modèles (de moins de 5M de paramètres aux dizaines/centaines de milliards). Ils sont notamment portés par leurs entreprises open-source (écrasant toute concurrence sur ce segment) mais disposent également d'atouts dans tous les types d'organisation existantes (hors laboratoires hybrides) puisqu'ils sont représentés à 18 reprises dans ce top 50.<br>
    L'Europe (l'Allemagne, la France et le Royaume-Uni notamment) se positionne également dans tous les types d'organisation existantes hors laboratoires hybrides (présente à 19 reprises) mais se démarque par l'impact de ses universités spécialisées sur des modèles de petites tailles (&lt;200M de paramètres). Elle est aussi présente sur toutes les modalités à l'exception des séries temporelles.<br>
    La Chine (représentée par 5 entités) est bien présente sur le segment des grands modèles open-source (31,8% vs 43,1% pour les États-Unis et 24% pour l'Europe sur les modèles de plus de 7,5 milliards de paramètres). Elle est cependant mal placée sur toutes les autres tranches de tailles des modèles (seulement 130M de téléchargements de modèles de moins de 100M de paramètres contre 7,05 milliards pour les États-Unis et 5,3 milliards pour l'Europe). Son non-positionnement sur la vision (à peine 4 millions de téléchargements) et l'audio (0 téléchargements) la pénalise également. Il ne s'agit pas de modalités sur lesquelles elle est connue pour avoir du retard, mais force est de constater qu'elle ne produit pas, à l'heure actuelle, de contenus open-source dans ces modalités sur Hugging Face (plateforme non accessible dans le pays). Elle domine le secteur des organisations non-profit et est seule sur celui des laboratoires hybrides universités/entreprises.<br>
    Enfin, les autres pays ne bénéficient uniquement d'un acteur spécialisé sur une modalité donnée.</li>
</ol>


</div>



