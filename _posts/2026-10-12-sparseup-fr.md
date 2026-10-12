---
title: "SPARSEUP-FR : APPLICATION AU FRANÇAIS DE LA MÉTHODOLOGIE DE LINKUP"
tags:
  - NLP
  - Sparse
excerpt : "Sparseup-fr - Réplication de l'approche de Linkup pour créer un modèle d'embedding sparse SOTA en français"
header :
    overlay_color: "#1C2A4D"
    teaser : "https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Trimming/thumbnail_trimming.png"
author_profile: false
sidebar:
    nav: sidebar-reduction
classes: wide
---
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<center>An English version is available on <a href="https://huggingface.co/blog/alphaedge-ai/french-sparseup">Hugging Face</a>.</center>
<br><br><br>

# <span style="color: #FF0000"> **Avant-Propos** </span> 

Cet article est une republication de celui que j’ai rédigé sur le blog d'[AlphaEdge](https://huggingface.co/alphaedge-ai).  
Je les remercie de m'avoir permis de travailler sur ce sujet.
<br><br><br>

# <span style="color: #FF0000"> **Introduction** </span>

Il y a 3 semaines, [Linkup](https://huggingface.co/Linkup-Platform) a publié son premier modèle open-source, le [linkup-sparseup-embed-v1](https://huggingface.co/Linkup-Platform/linkup-sparseup-embed-v1) [\[1\]](#ref1), accompagné d'un [article de blog](https://www.linkup.so/blog/introducing-sparseup-by-linkup) [\[2\]](#ref2) expliquant leur façon de procéder.  
Il s'agit d'un modèle d'embedding sparse de 150 M de paramètres basé sur les travaux de [LightOn](https://huggingface.co/lightonai) [\[3\]](#ref3) publiés plus tôt dans l'année. Les détails sont dans son article de blog et dans les fichiers `.py` disponibles sur Hugging Face mais, globalement, son approche consiste à partir du [LateOn-unsupervised](https://huggingface.co/lightonai/LateOn-unsupervised), à y greffer la tête MLM du [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-large) [\[4\]](#ref4), puis à finetuner le tout sur les [données](https://huggingface.co/collections/lightonai/denseon-and-lateon) mises à disposition par LightOn.  
Le résultat final est un modèle sparse gagnant 2 points sur BEIR-13 par rapport au précédent SOTA de taille équivalente pour cette famille de modèles.  
Pour se familiariser avec l'approche, nous avons eu envie de répliquer le modèle de Linkup, mais en français plutôt qu'en anglais.
<br><br><br>

# <span style="color: #FF0000"> **Conception du modèle** </span>

Pour pouvoir appliquer la méthodologie de Linkup au français, nous avons procédé à deux changements architecturaux.  
Le premier consiste à changer de *backbone* en utilisant le [mLateOn-unsupervised](https://huggingface.co/lightonai/mLateOn-unsupervised) [\[5\]](#ref5), qui est la version multilingue du LateOn-unsupervised. Celle-ci gère 10 langues, dont le français qui nous intéresse.  
Le second est d'utiliser comme tête MLM le [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) [\[6\]](#ref6) plutôt que le ModernBERT. Nous ne l'avons cependant pas repris tel quel puisque nous l'avons [trimé](https://huggingface.co/blog/lbourdois/introduction-to-trimming) [\[7\]](#ref7) afin de ne garder que 50 240 tokens [minés sur du français](https://huggingface.co/datasets/lbourdois/fineweb-2-trimming) plutôt que les 256 000 originaux.  
L'idée ici est d'obtenir un modèle français de taille équivalente à la version anglaise (cette dernière utilise 50 370 tokens dans son vocabulaire, mais nous n'avons pas souhaité faire correspondre exactement ce nombre afin de conserver un multiple de 64, bénéfique aux GPU).
<br><br><br>

# <span style="color: #FF0000"> **Entraînement** </span>

Nous avons entraîné notre modèle sur une seule RTX 6000 Pro pendant environ 55 h, soit le temps nécessaire pour réaliser une epoch complète sur les [données d'entraînement en français](https://huggingface.co/datasets/lightonai/embeddings-fine-tuning-filtered-fr) mises à disposition par LightOn.  
Concernant les hyperparamètres, nous vous invitons à consulter la [carte de modèle](https://huggingface.co/alphaedge-ai/sparseup-fr-v0). Il s'agit des mêmes que ceux utilisés par Linkup. La seule exception est qu'avec la mémoire disponible sur notre GPU, nous nous sommes permis de doubler la taille de la requête et du document (passant respectivement de 128 à 256 et de 512 à 1024).
<br><br><br>

# <span style="color: #FF0000"> **Résultats** </span>

Pour évaluer les modèles, nous utilisons une version nettoyée de NanoBEIR-fr de LightOn. C'est-à-dire que nous nous sommes assurés qu'il n'y avait pas de fuites de données entre les données d'entraînement de LightOn et ces benchmarks, mais avons également supprimé les lignes suspectes (voir par exemple cette [issue GitHub](https://github.com/embeddings-benchmark/mteb/issues/3708) listant des problèmes rencontrés pour NanoBEIR). Nous ne nous sommes néanmoins pas penchés sur d'éventuels problèmes d'annotation (pointés par exemple par [\[8\]](#ref8) ou [\[9\]](#ref9)).  
Nous mettons la version de ce benchmark à disposition sur [notre profil Hugging Face](https://huggingface.co/datasets/alphaedge-ai/NanoBEIR-clean).

Concernant les modèles, nous nous intéressons à deux comparaisons.  
Premièrement, nous évaluons le modèle obtenu face au mLateOn et au mDenseOn, comme l'ont fait les équipes de Linkup. En effet, comme ils utilisent les *backbones* et les données de LightOn, cela permet de comparer pleinement les différentes grandes approches architecturales en IR (recherche d'information). Pour l'anglais, ils observent que le LateOn (late interaction) est nettement plus performant que le DenseOn (dense), qui lui-même est légèrement au-dessus de leur modèle sparse. Nous souhaitons alors déterminer si nous observons ou non les mêmes tendances en français.  
Deuxièmement, nous comparons le modèle entraîné aux autres modèles sparse en français disponibles sur le Hub d'Hugging Face, afin de se faire une idée de sa position parmi les solutions open source existantes. Plus précisément, il s'agit du [splade-camembert-base-v2](https://huggingface.co/sparse-encoder/splade-camembert-base-v2) [\[10\]](#ref10) d'Arthur BRESNU, qui est monolingue, et de l'[opensearch-neural-sparse-encoding-multilingual-v1](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1) [\[11\]](#ref11) d'Amazon/Linux, qui est multilingue et inclut le français.

À noter  que nous avons lancé les évaluations sur deux configurations différentes : l'une où les documents sont découpés à la taille maximale de ce que chacun des modèles peut prendre en contexte, et l'autre où l'on applique une fenêtre coulissante sur l'ensemble du document, avec une taille de fenêtre égale à la taille de contexte, plus un chevauchement de 64 tokens. Ce dernier cas de figure, qui de surcroît ressemble davantage à un vrai cas de RAG, doit permettre d'évaluer plus justement les différents modèles sparse, qui ont une taille de requête et de document plus faible que les modèles de LightOn.

Finalement, nous précisons que dans les tableaux suivants, les cellules bleues indiquent la performance du meilleur modèle toutes familles confondues, là où les cellules vertes indiquent la performance du meilleur modèle sparse sur le benchmark.

<br><br>

## <span style="color: #FFBF00"> **Configuration avec troncage** </span>

<table>
  <thead>
    <tr>
      <th style="padding:6px 10px;width:12.5%">Jeu de données</th>
      <th style="padding:6px 10px;width:12.5%"><a href="https://huggingface.co/lightonai/mLateOn">mLateOn (307 M)</a></th>
      <th style="padding:6px 10px;width:12.5%"><a href="https://huggingface.co/lightonai/mDenseOn">mDenseOn (307 M)</a></th>
      <th style="padding:6px 10px;width:12.5%"><a href="https://huggingface.co/Linkup-Platform/linkup-sparseup-embed-v1">Sparseup (150 M)</a></th>
      <th style="padding:6px 10px;width:12.5%"><a href="https://huggingface.co/alphaedge-ai/sparseup-fr-v0">Sparseup-fr-v0 (150 M)</a></th>
      <th style="padding:6px 10px;width:12.5%"><a href="https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1">Opensearch (167 M)</a></th>
      <th style="padding:6px 10px;width:12.5%"><a href="https://huggingface.co/sparse-encoder/splade-camembert-base-v2">Splade-camembert (111 M)</a></th>
      <th style="padding:6px 10px;width:12.5%">bm25</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding:6px 10px">arguana</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">68.51</td><td style="padding:6px 10px">59.64</td><td style="padding:6px 10px">47.01</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">49.46</td><td style="padding:6px 10px">41.41</td><td style="padding:6px 10px">46.12</td><td style="padding:6px 10px">41.18</td></tr>
    <tr><td style="padding:6px 10px">climatefever</td><td style="padding:6px 10px">48.19</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">48.98</td><td style="padding:6px 10px">34.58</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">34.66</td><td style="padding:6px 10px">31.85</td><td style="padding:6px 10px">27.84</td><td style="padding:6px 10px">27.44</td></tr>
    <tr><td style="padding:6px 10px">dbpedia</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">69.03</td><td style="padding:6px 10px">65.76</td><td style="padding:6px 10px">55.86</td><td style="padding:6px 10px">61.71</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">62.22</td><td style="padding:6px 10px">58.34</td><td style="padding:6px 10px">51.42</td></tr>
    <tr><td style="padding:6px 10px">fever</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">93.39</td><td style="padding:6px 10px">90.99</td><td style="padding:6px 10px">87.24</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">91.91</td><td style="padding:6px 10px">87.74</td><td style="padding:6px 10px">77.31</td><td style="padding:6px 10px">75.73</td></tr>
    <tr><td style="padding:6px 10px">fiqa2018</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">59.68</td><td style="padding:6px 10px">56.10</td><td style="padding:6px 10px">29.01</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">41.68</td><td style="padding:6px 10px">33.56</td><td style="padding:6px 10px">39.06</td><td style="padding:6px 10px">27.76</td></tr>
    <tr><td style="padding:6px 10px">hotpotqa</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">91.85</td><td style="padding:6px 10px">85.80</td><td style="padding:6px 10px">76.69</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">77.55</td><td style="padding:6px 10px">74.08</td><td style="padding:6px 10px">70.43</td><td style="padding:6px 10px">67.84</td></tr>
    <tr><td style="padding:6px 10px">msmarco</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">60.39</td><td style="padding:6px 10px">59.67</td><td style="padding:6px 10px">51.52</td><td style="padding:6px 10px">50.06</td><td style="padding:6px 10px">49.51</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">57.55</td><td style="padding:6px 10px">44.00</td></tr>
    <tr><td style="padding:6px 10px">nfcorpus</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">35.50</td><td style="padding:6px 10px">34.35</td><td style="padding:6px 10px">28.43</td><td style="padding:6px 10px">28.53</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">34.27</td><td style="padding:6px 10px">30.14</td><td style="padding:6px 10px">30.73</td></tr>
    <tr><td style="padding:6px 10px">nq</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">69.09</td><td style="padding:6px 10px">63.75</td><td style="padding:6px 10px">50.49</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">56.36</td><td style="padding:6px 10px">55.05</td><td style="padding:6px 10px">51.39</td><td style="padding:6px 10px">44.95</td></tr>
    <tr><td style="padding:6px 10px">quoraretrieval</td><td style="padding:6px 10px">87.65</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">89.61</td><td style="padding:6px 10px">77.96</td><td style="padding:6px 10px">85.44</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">85.82</td><td style="padding:6px 10px">82.82</td><td style="padding:6px 10px">70.94</td></tr>
    <tr><td style="padding:6px 10px">scidocs</td><td style="padding:6px 10px">40.99</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">44.17</td><td style="padding:6px 10px">33.51</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">34.97</td><td style="padding:6px 10px">30.96</td><td style="padding:6px 10px">31.54</td><td style="padding:6px 10px">32.31</td></tr>
    <tr><td style="padding:6px 10px">scifact</td><td style="padding:6px 10px">79.42</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">80.91</td><td style="padding:6px 10px">65.05</td><td style="padding:6px 10px">67.47</td><td style="padding:6px 10px">66.74</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">67.99</td><td style="padding:6px 10px">65.57</td></tr>
    <tr><td style="padding:6px 10px">touche2020</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">55.94</td><td style="padding:6px 10px">55.05</td><td style="padding:6px 10px">43.78</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">52.99</td><td style="padding:6px 10px">48.08</td><td style="padding:6px 10px">50.49</td><td style="padding:6px 10px">50.70</td></tr>
    <tr><td colspan="8" style="padding:0;height:3px;line-height:0;background-color:#9aa0a6"></td></tr>
    <tr><td style="padding:6px 10px">Moyenne</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important"><b>66.13</b></td><td style="padding:6px 10px">64.21</td><td style="padding:6px 10px">52.40</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important"><b>56.37</b></td><td style="padding:6px 10px">53.94</td><td style="padding:6px 10px">53.15</td><td style="padding:6px 10px">48.50</td></tr>
  </tbody>
</table>
<center>Tableau 1 : nDCG@10 pour NanoBEIR-fr dans la configuration avec troncage</center>

<br><br>

## <span style="color: #FFBF00"> **Configuration avec fenêtre coulissante** </span>

<table>
  <thead>
    <tr>
      <th style="padding:6px 10px;width:12.5%">Jeu de données</th>
      <th style="padding:6px 10px;width:12.5%"><a href="https://huggingface.co/lightonai/mLateOn">mLateOn (307 M)</a></th>
      <th style="padding:6px 10px;width:12.5%"><a href="https://huggingface.co/lightonai/mDenseOn">mDenseOn (307 M)</a></th>
      <th style="padding:6px 10px;width:12.5%"><a href="https://huggingface.co/Linkup-Platform/linkup-sparseup-embed-v1">Sparseup (150 M)</a></th>
      <th style="padding:6px 10px;width:12.5%"><a href="https://huggingface.co/alphaedge-ai/sparseup-fr-v0">Sparseup-fr-v0 (150 M)</a></th>
      <th style="padding:6px 10px;width:12.5%"><a href="https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1">Opensearch (167 M)</a></th>
      <th style="padding:6px 10px;width:12.5%"><a href="https://huggingface.co/sparse-encoder/splade-camembert-base-v2">Splade-camembert (111 M)</a></th>
      <th style="padding:6px 10px;width:12.5%">bm25</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding:6px 10px">arguana</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">68.51</td><td style="padding:6px 10px">59.64</td><td style="padding:6px 10px">46.28</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">49.46</td><td style="padding:6px 10px">41.67</td><td style="padding:6px 10px">46.12</td><td style="padding:6px 10px">41.18</td></tr>
    <tr><td style="padding:6px 10px">climatefever</td><td style="padding:6px 10px">48.19</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">48.98</td><td style="padding:6px 10px">27.27</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">33.89</td><td style="padding:6px 10px">31.46</td><td style="padding:6px 10px">26.02</td><td style="padding:6px 10px">27.44</td></tr>
    <tr><td style="padding:6px 10px">dbpedia</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">69.03</td><td style="padding:6px 10px">65.76</td><td style="padding:6px 10px">55.99</td><td style="padding:6px 10px">61.92</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">62.39</td><td style="padding:6px 10px">58.21</td><td style="padding:6px 10px">51.42</td></tr>
    <tr><td style="padding:6px 10px">fever</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">93.39</td><td style="padding:6px 10px">90.99</td><td style="padding:6px 10px">86.67</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">91.91</td><td style="padding:6px 10px">88.74</td><td style="padding:6px 10px">78.51</td><td style="padding:6px 10px">75.73</td></tr>
    <tr><td style="padding:6px 10px">fiqa2018</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">59.68</td><td style="padding:6px 10px">56.14</td><td style="padding:6px 10px">27.84</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">41.67</td><td style="padding:6px 10px">33.29</td><td style="padding:6px 10px">38.79</td><td style="padding:6px 10px">27.76</td></tr>
    <tr><td style="padding:6px 10px">hotpotqa</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">91.85</td><td style="padding:6px 10px">85.80</td><td style="padding:6px 10px">76.64</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">77.60</td><td style="padding:6px 10px">74.08</td><td style="padding:6px 10px">70.44</td><td style="padding:6px 10px">67.84</td></tr>
    <tr><td style="padding:6px 10px">msmarco</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">60.39</td><td style="padding:6px 10px">59.67</td><td style="padding:6px 10px">51.52</td><td style="padding:6px 10px">50.06</td><td style="padding:6px 10px">49.13</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">57.55</td><td style="padding:6px 10px">44.00</td></tr>
    <tr><td style="padding:6px 10px">nfcorpus</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">35.50</td><td style="padding:6px 10px">34.35</td><td style="padding:6px 10px">25.69</td><td style="padding:6px 10px">28.62</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">33.67</td><td style="padding:6px 10px">30.11</td><td style="padding:6px 10px">30.73</td></tr>
    <tr><td style="padding:6px 10px">nq</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">69.09</td><td style="padding:6px 10px">63.75</td><td style="padding:6px 10px">50.81</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">56.40</td><td style="padding:6px 10px">54.52</td><td style="padding:6px 10px">51.57</td><td style="padding:6px 10px">44.95</td></tr>
    <tr><td style="padding:6px 10px">quoraretrieval</td><td style="padding:6px 10px">87.65</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">89.61</td><td style="padding:6px 10px">77.96</td><td style="padding:6px 10px">85.44</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">85.82</td><td style="padding:6px 10px">82.85</td><td style="padding:6px 10px">70.94</td></tr>
    <tr><td style="padding:6px 10px">scidocs</td><td style="padding:6px 10px">40.99</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">44.17</td><td style="padding:6px 10px">31.95</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">34.72</td><td style="padding:6px 10px">30.95</td><td style="padding:6px 10px">31.30</td><td style="padding:6px 10px">32.31</td></tr>
    <tr><td style="padding:6px 10px">scifact</td><td style="padding:6px 10px">79.42</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">80.91</td><td style="padding:6px 10px">64.42</td><td style="padding:6px 10px">67.47</td><td style="padding:6px 10px">67.94</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">68.04</td><td style="padding:6px 10px">65.57</td></tr>
    <tr><td style="padding:6px 10px">touche2020</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important">56.03</td><td style="padding:6px 10px">55.05</td><td style="padding:6px 10px">48.53</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important">55.38</td><td style="padding:6px 10px">52.68</td><td style="padding:6px 10px">54.54</td><td style="padding:6px 10px">50.70</td></tr>
    <tr><td colspan="8" style="padding:0;height:3px;line-height:0;background-color:#9aa0a6"></td></tr>
    <tr><td style="padding:6px 10px">Moyenne</td><td style="padding:6px 10px;background:#D9E7FF!important;color:#12233D!important"><b>66.13</b></td><td style="padding:6px 10px">64.22</td><td style="padding:6px 10px">51.66</td><td style="padding:6px 10px;background:#E7FFDE!important;color:#16300F!important"><b>56.50</b></td><td style="padding:6px 10px">54.33</td><td style="padding:6px 10px">53.39</td><td style="padding:6px 10px">48.50</td></tr>
  </tbody>
</table>
<center>Tableau 2 : nDCG@10 pour NanoBEIR-fr dans la configuration avec fenêtre coulissante</center>

<br><br>
## <span style="color: #FFBF00"> **Observations** </span>

Nous pouvons constater, comme les équipes de Linkup, que le modèle à late interaction performe mieux que le dense, et lui-même mieux que les différents modèles sparse. L'usage d'une fenêtre coulissante n'améliore pas les modèles de LightOn, qui bénéficient déjà d'une taille de contexte de requête et de document de 8 192 tokens. Elle apparaît légèrement bénéfique aux modèles sparse français, mais dégrade étrangement ceux de Sparseup. Nous ne rapportons pas les résultats ici car nous faisons un focus sur le français, mais pour vérifier ce comportement nous avons aussi évalué les modèles sur le NanoBEIR-en nettoyé et avons alors observé que la fenêtre coulissante bénéficie bien à Sparseup (permettant également alors de surpasser le mDenseOn).

Concernant les modèles en français, le modèle entraîné surpasse les deux autres disponibles jusqu'à présent sur le Hub de plus de 3 points pour le `splade-camembert-base-v2` (notre modèle étant néanmoins 35 % plus gros) et de 2 points pour l'`opensearch-neural-sparse-encoding-multilingual-v1` (en étant cette fois 11 % plus petit). Il est premier sur 8 jeux de données de NanoBEIR-fr, là où `opensearch` l'est sur 3 et `splade-camembert-base-v2` sur 2.  
Le Sparseup ets en dessous mais c'est attendu car monolingue anglais (fait tout de même mieux qu'un bm25).  

À noter que pour une comparaison à nombre de paramètres équivalent, on pourrait envisager de réentraîner deux modèles (un en ne gardant que 32 768 tokens pour atteindre la taille du `splade-camembert-base-v2` et un autre en gardant 73 556 tokens pour atteindre celle de l'`opensearch-neural-sparse-encoding-multilingual-v1`), mais cela nécessiterait à nouveau plusieurs dizaines d'heures GPU d'entraînement que nous n'avions pas pour cette première expérimentation consistant simplement à vérifier la méthodologie appliquée par Linkup.
<br><br>

## <span style="color: #FFBF00"> **Retours d'expérience** </span>

Nous voulions initialement pouvoir présenter un modèle à la Linkup en français, issu d'une fusion de plusieurs poids sauvegardés. Nous nous sommes rendu compte à la fin de l'entraînement que beaucoup de sauvegardes n'avaient pas été effectuées.  
Il s'avère que la fonction `push_to_hub` refuse de pousser les poids lorsque le README fait plus de 1 Mo. La librairie `sentence_transformers` écrivant les résultats intermédiaires dans les métadonnées YAML du README à chaque évaluation, ce problème peut alors apparaître si vous faites trop de sauvegardes. Nous vous invitons ainsi à faire attention à ce point en rédigeant un petit patch dans votre code d'entraînement pour ne pas sauvegarder les résultats dans le YAML.
<br><br><br>

# <span style="color: #FF0000"> **Conclusion et suite** </span>

Cette simple reproduction de la méthodologie de Sparseup dans une autre langue se révèle concluante puisque nous obtenons un nouveau SOTA en français parmi les modèles sparse disponibles en accès libre.  
Forts de ce premier retour, nous envisageons de pousser plus loin les expérimentations, avec notamment plus de données, une fusion de poids, ou encore une version plus grande du modèle en français. Une autre possibilité serait de proposer une version multilingue.

En l'état, nous mettons à disposition cette v0, dans cette [collection Hugging Face](https://huggingface.co/collections/alphaedge-ai/sparse-models-french-sparseup), le modèle entraîné ainsi que le benchmark nettoyé.

En espérant que cela vous soit utile :)
<br><br><br>

# <span style="color: #FF0000"> **Références** </span>

1. <a id="ref1"></a>*linkup-sparseup-embed-v1* de Thibault FORMAL, Pierre ERBACHER, Cosimo RULLI  (2026). [lien](https://huggingface.co/Linkup-Platform/linkup-sparseup-embed-v1)
2. <a id="ref2"></a>*Introducing SPARSEUP by Linkup: a move towards more competitive open sparse models* de Thibault FORMAL, Pierre ERBACHER, Cosimo RULLI (2026). [lien](https://www.linkup.so/blog/introducing-sparseup-by-linkup)
3. <a id="ref3"></a>*DenseOn with the LateOn: Fully Open Dense and Late-Interaction Models for Multilingual, Long-Context, and Code Search* de Raphaël SOURTY, Antoine CHAFFIN, Paulo Roberto MOURA JUNIOR, Amélie CHATELAIN (2026). [lien](https://arxiv.org/abs/2607.27178)
4. <a id="ref4"></a>*Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference* de Benjamin WARNER, Antoine CHAFFIN, Benjamin CLAVIÉ, et al. (2024). [lien](https://arxiv.org/abs/2412.13663)
5. <a id="ref5"></a>*mDenseOn and mLateOn: more signal, less noise for multilingual agentic search* de Raphaël SOURTY, Antoine CHAFFIN, Paulo Roberto MOURA JUNIOR, Amélie CHATELAIN (2026). [lien](https://huggingface.co/blog/lightonai/mdenseon-mlateon)
6. <a id="ref6"></a>*mmBERT: A Modern Multilingual Encoder with Annealed Language Learning* de Marc MARONE, Orion WELLER, William FLESHMAN, Eugene YANG, Dawn LAWRIE, Benjamin VAN DURME (2025). [lien](https://arxiv.org/abs/2509.06888)
7. <a id="ref7"></a>*Introduction to Trimming ✂* de Loïck BOURDOIS, et al. (2026). [lien](https://huggingface.co/blog/lbourdois/introduction-to-trimming)
8. <a id="ref8"></a>*Systematic Evaluation of Neural Retrieval Models on the Touché 2020 Argument Retrieval Subset of BEIR* de Nandan THAKUR, Luiz BONIFACIO, Maik FRÖBE, et al. (2024). [lien](https://arxiv.org/abs/2407.07790)
9. <a id="ref9"></a>*Hard Negatives, Hard Lessons: Revisiting Training Data Quality for Robust Information Retrieval with LLMs* de Nandan THAKUR, Crystina ZHANG, Xueguang MA, Jimmy LIN (2025). [lien](https://aclanthology.org/2025.findings-emnlp.481/)
10. <a id="ref10"></a>*splade-camembert-base-v2* d'Arthur BRESNU (2025). [lien](https://huggingface.co/sparse-encoder/splade-camembert-base-v2)
11. <a id="ref11"></a>*Towards Competitive Search Relevance For Inference-Free Learned Sparse Retrievers* de Zhichao GENG, Yiwen WANG, Dongyu RU, Yang YANG (2025). [lien](https://arxiv.org/abs/2411.04403)

<br><br><br>
# <span style="color: #FF0000"> **Citation** </span>

```
@misc{hf_blogpost_french_sparseup,
      title={Sparseup-fr: A French replication of Linkup's sparse embedding model}, 
      author={Loïck BOURDOIS},
      year={2026},
      url={https://huggingface.co/blog/alphaedge-ai/french-sparseup}, 
}
```
