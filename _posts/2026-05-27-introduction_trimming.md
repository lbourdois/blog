---
title: "INTRODUCTION AU TRIMMING"
tags:
  - NLP
  - Trimming
excerpt : "Trimming - Une introduction à une technique permettant de réudire le nombre de paramètres d'un modèle tout en maintenant ses performances d'origine"
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

<center>An English version is available on <a href="https://huggingface.co/blog/lbourdois/introduction-to-trimming">Hugging Face</a>.</center>

<br>

# <span style="color: #FF0000"> **Avant-Propos** </span> 

Cet article est une republication de celui que j’ai rédigé sur le blog d'[AlphaEdge](https://huggingface.co/alphaedge-ai).  
Je les remercie de m'avoir permis de travailler sur ce sujet.

Le *trimming* étant particulièrement intéressant pour le multilinguisme, ce travail a été l'occasion d'une collaboration avec différents Hugging Face Fellows pour évaluer cette approche sur d'autres langues que le français ou l'anglais.
À savoir [Tom AARSEN](https://huggingface.co/tomaarsen) (anglais/néerlandais), [Bram VANROY](https://huggingface.co/BramVanroy) (néerlandais), [Christopher AKIKI](https://huggingface.co/christopher) (arabe/allemand), [Woojun JUNG](https://huggingface.co/woojun-jung) (coréen), [Manuel ROMERO](https://huggingface.co/mrm8488) (espagnol) et [Prithiv SAKTHI](https://huggingface.co/prithivMLmods) (Tamil).  

Je tiens finalement à indiquer que l'estimation du temps de lecture est fortement surestimée du fait de nombreux tableaux de résultats, références ou d'exemples de textes pour montrer les sorties obtenues avec les modèles trimmés (ainsi que potentiellement leur traduction en français quand ces exemples portent sur une autre langue)
<br><br><br>

# <span style="color: #FF0000"> **Introduction** </span>

Dans cet article de blog, nous procédons à une introduction à la technique du *trimming*. Cette méthode simple, ne nécessitant pas de réentraînement et s'exécutant sur un simple CPU, permet d'obtenir un modèle plus léger que l'original tout en maintenant ses performances.  
Dans la partie **Pratique 👨‍💻** où nous avons effectué des expérimentations, nous listons les points essentiels dans des encadrés **🧠 À retenir**. Une synthèse de tous les avantages de cette approche est également trouvable dans la conclusion.  
Pour accompagner nos propos, nous dévoilons [5526 modèles](https://huggingface.co/spaces/alphaedge-ai/Trimming_models_search) issus de l'application de cette technique.

<br><br>

## <span style="color: #FFBF00"> **Théorie 👨‍🏫** </span>

Le *trimming* peut être vu comme un sous-ensemble du *pruning*. En effet, comme pour ce dernier, le but de la technique est de modifier/supprimer des poids du modèle pour le rendre in fine plus léger (que ce soit le nombre de paramètres ou sa taille mémoire).   
Néanmoins la spécificité de cette méthode est que nous nous **focalisons uniquement sur les parties de l'architecture en lien avec le vocabulaire**, là où dans le cadre du *pruning* nous modifions généralement plutôt les poids/couches du reste de l'architecture (i.e. le *backbone*).  
Ainsi, **pour le *trimming*, nous supprimons des *tokens* dans le vocabulaire original du modèle** (et par conséquent devons par la même occasion actualiser le *tokenizer*) **mais modifions également la couche d'*embedding* finale gérant la distribution de probabilité du vocabulaire du modèle** (de même pour la couche d'entrée si les *embeddings* sont partagés).

<br>

**À quoi sert le *trimming* ?**  

Considérons un modèle avec une taille de vocabulaire donnée.
Cette taille peut ne pas être pertinente pour deux raisons principales :  
1) Si notre modèle est multilingue ;  
    En effet, toutes les langues ne nous intéressent pas forcément. Il est alors possible de supprimer celles qui sont inutiles pour notre cas d'usage.  
2) Si la taille du vocabulaire n'est pas un multiple de 8 ou de 64.  
    En effet, pour optimiser l'usage de ses GPU, ces multiples sont à privilégier (lire notamment [ici](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc) et [ici](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/)).  
   Cette préconisation simple permet d'accélérer facilement l'entraînement d'un modèle (de 25% d'après les observations de [Karpathy](https://x.com/karpathy/status/1621578354024677377)). Notons que depuis 2023-2024, c'est un usage qui s'est répandu et les modèles utilisent ainsi généralement de base un multiple de 8 ou de 64 pour le vocabulaire. Cependant pour des modèles plus anciens, il peut être intéressant de modifier la taille (nous verrons ci-dessous dans la partie **Pratique 👨‍💻** que 7 des 16 modèles testés ont ce problème).

Cette réduction de la taille du vocabulaire (aux langues d'intérêt et à un multiple de 8 ou 64) permet de réduire la taille du modèle (que ce soit le nombre de paramètres ou in fine sa taille mémoire). Illustrons ce point à travers un exemple simple : le [GPT2-small](https://huggingface.co/openai-community/gpt2) de [RADFORD, WU et al. (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).  
Ce modèle permet de comprendre facilement le cas 2) décrit à l'instant mais est aussi un exemple que le *trimming* fonctionne également sur des modèles monolingues (alors que dans la partie **Pratique 👨‍💻** la plupart des exemples portent sur des modèles multilingues).  

Calculons alors le nombre de paramètres du GPT2-small :

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")

for name, param in model.named_parameters():
    print(f"{name:50s} {str(param.shape):25s} {param.numel():>12,}")
```

<details markdown="1">
<summary><b>Sortie complète :</b></summary>

```python
# Sortie :
# transformer.wte.weight                             torch.Size([50257, 768])    38,597,376
# transformer.wpe.weight                             torch.Size([1024, 768])        786,432
# transformer.h.0.ln_1.weight                        torch.Size([768])                  768
# transformer.h.0.ln_1.bias                          torch.Size([768])                  768
# transformer.h.0.attn.c_attn.weight                 torch.Size([768, 2304])      1,769,472
# transformer.h.0.attn.c_attn.bias                   torch.Size([2304])               2,304
# transformer.h.0.attn.c_proj.weight                 torch.Size([768, 768])         589,824
# transformer.h.0.attn.c_proj.bias                   torch.Size([768])                  768
# transformer.h.0.ln_2.weight                        torch.Size([768])                  768
# transformer.h.0.ln_2.bias                          torch.Size([768])                  768
# transformer.h.0.mlp.c_fc.weight                    torch.Size([768, 3072])      2,359,296
# transformer.h.0.mlp.c_fc.bias                      torch.Size([3072])               3,072
# transformer.h.0.mlp.c_proj.weight                  torch.Size([3072, 768])      2,359,296
# transformer.h.0.mlp.c_proj.bias                    torch.Size([768])                  768
# transformer.h.1.ln_1.weight                        torch.Size([768])                  768
# transformer.h.1.ln_1.bias                          torch.Size([768])                  768
# transformer.h.1.attn.c_attn.weight                 torch.Size([768, 2304])      1,769,472
# transformer.h.1.attn.c_attn.bias                   torch.Size([2304])               2,304
# transformer.h.1.attn.c_proj.weight                 torch.Size([768, 768])         589,824
# transformer.h.1.attn.c_proj.bias                   torch.Size([768])                  768
# transformer.h.1.ln_2.weight                        torch.Size([768])                  768
# transformer.h.1.ln_2.bias                          torch.Size([768])                  768
# transformer.h.1.mlp.c_fc.weight                    torch.Size([768, 3072])      2,359,296
# transformer.h.1.mlp.c_fc.bias                      torch.Size([3072])               3,072
# transformer.h.1.mlp.c_proj.weight                  torch.Size([3072, 768])      2,359,296
# transformer.h.1.mlp.c_proj.bias                    torch.Size([768])                  768
# transformer.h.2.ln_1.weight                        torch.Size([768])                  768
# transformer.h.2.ln_1.bias                          torch.Size([768])                  768
# transformer.h.2.attn.c_attn.weight                 torch.Size([768, 2304])      1,769,472
# transformer.h.2.attn.c_attn.bias                   torch.Size([2304])               2,304
# transformer.h.2.attn.c_proj.weight                 torch.Size([768, 768])         589,824
# transformer.h.2.attn.c_proj.bias                   torch.Size([768])                  768
# transformer.h.2.ln_2.weight                        torch.Size([768])                  768
# transformer.h.2.ln_2.bias                          torch.Size([768])                  768
# transformer.h.2.mlp.c_fc.weight                    torch.Size([768, 3072])      2,359,296
# transformer.h.2.mlp.c_fc.bias                      torch.Size([3072])               3,072
# transformer.h.2.mlp.c_proj.weight                  torch.Size([3072, 768])      2,359,296
# transformer.h.2.mlp.c_proj.bias                    torch.Size([768])                  768
# transformer.h.3.ln_1.weight                        torch.Size([768])                  768
# transformer.h.3.ln_1.bias                          torch.Size([768])                  768
# transformer.h.3.attn.c_attn.weight                 torch.Size([768, 2304])      1,769,472
# transformer.h.3.attn.c_attn.bias                   torch.Size([2304])               2,304
# transformer.h.3.attn.c_proj.weight                 torch.Size([768, 768])         589,824
# transformer.h.3.attn.c_proj.bias                   torch.Size([768])                  768
# transformer.h.3.ln_2.weight                        torch.Size([768])                  768
# transformer.h.3.ln_2.bias                          torch.Size([768])                  768
# transformer.h.3.mlp.c_fc.weight                    torch.Size([768, 3072])      2,359,296
# transformer.h.3.mlp.c_fc.bias                      torch.Size([3072])               3,072
# transformer.h.3.mlp.c_proj.weight                  torch.Size([3072, 768])      2,359,296
# transformer.h.3.mlp.c_proj.bias                    torch.Size([768])                  768
# transformer.h.4.ln_1.weight                        torch.Size([768])                  768
# transformer.h.4.ln_1.bias                          torch.Size([768])                  768
# transformer.h.4.attn.c_attn.weight                 torch.Size([768, 2304])      1,769,472
# transformer.h.4.attn.c_attn.bias                   torch.Size([2304])               2,304
# transformer.h.4.attn.c_proj.weight                 torch.Size([768, 768])         589,824
# transformer.h.4.attn.c_proj.bias                   torch.Size([768])                  768
# transformer.h.4.ln_2.weight                        torch.Size([768])                  768
# transformer.h.4.ln_2.bias                          torch.Size([768])                  768
# transformer.h.4.mlp.c_fc.weight                    torch.Size([768, 3072])      2,359,296
# transformer.h.4.mlp.c_fc.bias                      torch.Size([3072])               3,072
# transformer.h.4.mlp.c_proj.weight                  torch.Size([3072, 768])      2,359,296
# transformer.h.4.mlp.c_proj.bias                    torch.Size([768])                  768
# transformer.h.5.ln_1.weight                        torch.Size([768])                  768
# transformer.h.5.ln_1.bias                          torch.Size([768])                  768
# transformer.h.5.attn.c_attn.weight                 torch.Size([768, 2304])      1,769,472
# transformer.h.5.attn.c_attn.bias                   torch.Size([2304])               2,304
# transformer.h.5.attn.c_proj.weight                 torch.Size([768, 768])         589,824
# transformer.h.5.attn.c_proj.bias                   torch.Size([768])                  768
# transformer.h.5.ln_2.weight                        torch.Size([768])                  768
# transformer.h.5.ln_2.bias                          torch.Size([768])                  768
# transformer.h.5.mlp.c_fc.weight                    torch.Size([768, 3072])      2,359,296
# transformer.h.5.mlp.c_fc.bias                      torch.Size([3072])               3,072
# transformer.h.5.mlp.c_proj.weight                  torch.Size([3072, 768])      2,359,296
# transformer.h.5.mlp.c_proj.bias                    torch.Size([768])                  768
# transformer.h.6.ln_1.weight                        torch.Size([768])                  768
# transformer.h.6.ln_1.bias                          torch.Size([768])                  768
# transformer.h.6.attn.c_attn.weight                 torch.Size([768, 2304])      1,769,472
# transformer.h.6.attn.c_attn.bias                   torch.Size([2304])               2,304
# transformer.h.6.attn.c_proj.weight                 torch.Size([768, 768])         589,824
# transformer.h.6.attn.c_proj.bias                   torch.Size([768])                  768
# transformer.h.6.ln_2.weight                        torch.Size([768])                  768
# transformer.h.6.ln_2.bias                          torch.Size([768])                  768
# transformer.h.6.mlp.c_fc.weight                    torch.Size([768, 3072])      2,359,296
# transformer.h.6.mlp.c_fc.bias                      torch.Size([3072])               3,072
# transformer.h.6.mlp.c_proj.weight                  torch.Size([3072, 768])      2,359,296
# transformer.h.6.mlp.c_proj.bias                    torch.Size([768])                  768
# transformer.h.7.ln_1.weight                        torch.Size([768])                  768
# transformer.h.7.ln_1.bias                          torch.Size([768])                  768
# transformer.h.7.attn.c_attn.weight                 torch.Size([768, 2304])      1,769,472
# transformer.h.7.attn.c_attn.bias                   torch.Size([2304])               2,304
# transformer.h.7.attn.c_proj.weight                 torch.Size([768, 768])         589,824
# transformer.h.7.attn.c_proj.bias                   torch.Size([768])                  768
# transformer.h.7.ln_2.weight                        torch.Size([768])                  768
# transformer.h.7.ln_2.bias                          torch.Size([768])                  768
# transformer.h.7.mlp.c_fc.weight                    torch.Size([768, 3072])      2,359,296
# transformer.h.7.mlp.c_fc.bias                      torch.Size([3072])               3,072
# transformer.h.7.mlp.c_proj.weight                  torch.Size([3072, 768])      2,359,296
# transformer.h.7.mlp.c_proj.bias                    torch.Size([768])                  768
# transformer.h.8.ln_1.weight                        torch.Size([768])                  768
# transformer.h.8.ln_1.bias                          torch.Size([768])                  768
# transformer.h.8.attn.c_attn.weight                 torch.Size([768, 2304])      1,769,472
# transformer.h.8.attn.c_attn.bias                   torch.Size([2304])               2,304
# transformer.h.8.attn.c_proj.weight                 torch.Size([768, 768])         589,824
# transformer.h.8.attn.c_proj.bias                   torch.Size([768])                  768
# transformer.h.8.ln_2.weight                        torch.Size([768])                  768
# transformer.h.8.ln_2.bias                          torch.Size([768])                  768
# transformer.h.8.mlp.c_fc.weight                    torch.Size([768, 3072])      2,359,296
# transformer.h.8.mlp.c_fc.bias                      torch.Size([3072])               3,072
# transformer.h.8.mlp.c_proj.weight                  torch.Size([3072, 768])      2,359,296
# transformer.h.8.mlp.c_proj.bias                    torch.Size([768])                  768
# transformer.h.9.ln_1.weight                        torch.Size([768])                  768
# transformer.h.9.ln_1.bias                          torch.Size([768])                  768
# transformer.h.9.attn.c_attn.weight                 torch.Size([768, 2304])      1,769,472
# transformer.h.9.attn.c_attn.bias                   torch.Size([2304])               2,304
# transformer.h.9.attn.c_proj.weight                 torch.Size([768, 768])         589,824
# transformer.h.9.attn.c_proj.bias                   torch.Size([768])                  768
# transformer.h.9.ln_2.weight                        torch.Size([768])                  768
# transformer.h.9.ln_2.bias                          torch.Size([768])                  768
# transformer.h.9.mlp.c_fc.weight                    torch.Size([768, 3072])      2,359,296
# transformer.h.9.mlp.c_fc.bias                      torch.Size([3072])               3,072
# transformer.h.9.mlp.c_proj.weight                  torch.Size([3072, 768])      2,359,296
# transformer.h.9.mlp.c_proj.bias                    torch.Size([768])                  768
# transformer.h.10.ln_1.weight                       torch.Size([768])                  768
# transformer.h.10.ln_1.bias                         torch.Size([768])                  768
# transformer.h.10.attn.c_attn.weight                torch.Size([768, 2304])      1,769,472
# transformer.h.10.attn.c_attn.bias                  torch.Size([2304])               2,304
# transformer.h.10.attn.c_proj.weight                torch.Size([768, 768])         589,824
# transformer.h.10.attn.c_proj.bias                  torch.Size([768])                  768
# transformer.h.10.ln_2.weight                       torch.Size([768])                  768
# transformer.h.10.ln_2.bias                         torch.Size([768])                  768
# transformer.h.10.mlp.c_fc.weight                   torch.Size([768, 3072])      2,359,296
# transformer.h.10.mlp.c_fc.bias                     torch.Size([3072])               3,072
# transformer.h.10.mlp.c_proj.weight                 torch.Size([3072, 768])      2,359,296
# transformer.h.10.mlp.c_proj.bias                   torch.Size([768])                  768
# transformer.h.11.ln_1.weight                       torch.Size([768])                  768
# transformer.h.11.ln_1.bias                         torch.Size([768])                  768
# transformer.h.11.attn.c_attn.weight                torch.Size([768, 2304])      1,769,472
# transformer.h.11.attn.c_attn.bias                  torch.Size([2304])               2,304
# transformer.h.11.attn.c_proj.weight                torch.Size([768, 768])         589,824
# transformer.h.11.attn.c_proj.bias                  torch.Size([768])                  768
# transformer.h.11.ln_2.weight                       torch.Size([768])                  768
# transformer.h.11.ln_2.bias                         torch.Size([768])                  768
# transformer.h.11.mlp.c_fc.weight                   torch.Size([768, 3072])      2,359,296
# transformer.h.11.mlp.c_fc.bias                     torch.Size([3072])               3,072
# transformer.h.11.mlp.c_proj.weight                 torch.Size([3072, 768])      2,359,296
# transformer.h.11.mlp.c_proj.bias                   torch.Size([768])                  768
# transformer.ln_f.weight                            torch.Size([768])                  768
# transformer.ln_f.bias                              torch.Size([768])                  768
# ─────────────────────────────────────────────────────────────────────────────
# Total unique : 124,439,808
```

</details>

<br>

**Sortie groupée par composant** :
```python
# ── Embeddings ──────────────────────────────
#    wte.weight                         38,597,376
#  Sous-total : 39,383,808

# ── Encodage positionnel ────────────────────
#    wpe.weight                            786,432
#  Sous-total : 786,432

# ── Blocs transformer (×12) ─────────────────
#  Par bloc :
#    ln_1.weight                               768
#    ln_1.bias                                 768
#    attn.c_attn.weight                  1,769,472
#    attn.c_attn.bias                        2,304
#    attn.c_proj.weight                    589,824
#    attn.c_proj.bias                          768
#    ln_2.weight                               768
#    ln_2.bias                                 768
#    mlp.c_fc.weight                     2,359,296
#    mlp.c_fc.bias                           3,072
#    mlp.c_proj.weight                   2,359,296
#    mlp.c_proj.bias                           768
#  Sous-total 1 bloc   :  7,087,872
#  Sous-total 12 blocs : 85,054,464

# ── LayerNorm finale ─────────────────────────
#    ln_f.weight                               768
#    ln_f.bias                                 768
#  Sous-total : 1,536

# ══════════════════════════════════════════════════
# Total unique : 124,439,808
```

<br>

**Vue graphique** :
<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/-qWQSsbtNXsCYTj74pnIr.png" 
       alt="Nombre de paramètres du GPT2 original vs. un GPT2 trimmé" 
       style="max-width:100%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Nombre de paramètres du GPT2 original vs. un GPT2 trimmé</center>
  </figcaption></center>
</figure>

<br>

Les **85 056 000** paramètres (les 12 blocs transformer + la `LayerNorm` finale) ne sont pas modifiés dans le cadre du *trimming* (il faudrait faire du *pruning* pour réduire cette partie), de même que les **786 432** paramètres de l'encodage positionnel (`wpe.weight`).  
Par contre, avec cette technique, nous pouvons modifier les **38 597 376** paramètres de la couche d'*embedding* (`wte.weight`). Dans la sortie détaillée, nous pouvons voir que cette couche est de taille `[50257, 768]` soit un vocabulaire de taille 50 257 où chaque token est représenté sur 768 dimensions.  

<div class="notice--info" style="line-height:1.25;">
  <p style="font-size:0.9rem;"><strong>📝 Note</strong></p>
  <p style="font-size:0.9rem;">
    Cette information est également trouvable sur le Hub d'Hugging Face sans avoir à télécharger le modèle. Il faut consulter le <em>widget "Files infos"</em> qui est disponible pour tous les modèles du Hub dont les poids ont été poussés au format <code>safetensors</code>. Par exemple pour le GPT2, nous pouvons consulter <a href="https://huggingface.co/openai-community/gpt2?show_file_info=model.safetensors">cette page</a>. En passant votre curseur sur la ligne <code>wpe.weight</code> vous verrez même l'information indiquant que cette couche représente 28,17% de la taille totale du modèle.
  </p>
</div>

Nous pouvons constater que la taille du vocabulaire n'est pas un multiple de 64 ! Appliquer du *trimming* sur ce modèle pourrait être pertinent.  
Prenons un exemple théorique. Si nous ramenons la taille de vocabulaire de **50 257** à **32 768** (512 × 64), les 38 597 376 paramètres sont alors réduits à **32 768 × 768 = 25 165 824**. Soit **13 431 552** de paramètres en moins.    
Le nombre de paramètres total passe alors de **124 439 808** à **111 008 256**. C'est une réduction de **10,79 %**.  
Nous verrons dans la section suivante que pour certains modèles, cette réduction peut être beaucoup plus importante.

<br>

Une question que nous pouvons alors nous poser est : *quel est l'impact de cette réduction de paramètres sur les performances du modèle ?*    
Pour y répondre, nous avons analysé 16 modèles portant sur des architectures et des modalités différentes : encodeurs textuels, encodeur-décodeur textuels, décodeurs textuels, modèles d'*embeddings* textuels, modèle d'*embeddings* visuels, encodeur-décodeur textuels/visuels (VLM).  
<br><br>

## <span style="color: #FFBF00"> **Pratique 👨‍💻** </span>

**Comment faire du trimming en pratique ?**    
Plusieurs auteurs se sont intéressés au problème ces dernières années, et parmi ceux qui ont partagé leur code voire proposé une librairie dédiée au sujet, nous pouvons lister : 
- `smaller-transformers` de Geotrend permettant de trimmer un mBERT, la version multilingue du [BERT](https://arxiv.org/abs/1810.04805) de DEVLIN et al. (2018). Une publication est disponible [ici](https://arxiv.org/abs/2010.05609) par ABDAOUI et al. (2021), de même qu'un répertoire [GitHub](https://github.com/Geotrend-research/smaller-transformers) et certains modèles bilingues/*n*-lingues sur leur [compte Hugging Face](https://huggingface.co/Geotrend). Mais leur approche se limite à ce modèle et les vocabulaires sont de tailles différentes en fonction de la langue considérée et jamais un multiple de 64.
- David DALE dans un [article Medium](https://towardsdatascience.com/how-to-adapt-a-multilingual-t5-model-for-a-single-language-b9f94f3d9c90/) (2021) montre comment trimmer un mT5 pour ne garder que l’anglais et le russe. Une approche manuelle pour un contrôle total, mais encore une fois, cela se limite à un modèle particulier.
- Aditya SRIVASTAVA a développé la librairie [`hf-trim`](https://github.com/IamAdiSri/hf-trim) (2022) où il annonce prendre en compte le mT5 et mBART mais il y a plusieurs limitations en pratique. La plus grande est qu'il n'est pas possible de contrôler la taille finale du vocabulaire souhaitée. Il devrait être possible de résoudre cela en modifiant le code source.  
- [`lm-vocab-trimmer`](https://github.com/asahi417/lm-vocab-trimmer) (2023) par [USHIO, ZHOU et CAMACHO-COLLADOS](https://arxiv.org/abs/2305.15020) gère le mT5, mBART et XLM-RoBERTa. Cette librairie est probablement la plus aboutie sur le sujet mais a quelques faiblesses, comme le fait que l’argument `target_vocab_size` ne donne pas le résultat attendu (si vous voulez un multiple de 64). Il n’est pas possible de réduire à *n* langues (un modèle bilingue, par exemple). Finalement tous les modèles gérés reposent sur un *tokenizer* [sentencepiece](https://arxiv.org/abs/1808.06226) de KUDO et RICHARDSON (2018).
- Antoine LOUIS via ce [Space](https://huggingface.co/spaces/antoinelouis/mtem-pruner) (2024), propose de *trimmer* des modèles d’*embeddings* déjà finetunés (en utilisant un tokenizer *sentencepiece*) comme le mE5, le BGE ou le GTE. Il n’est pas possible de sélectionner le nombre souhaité de *tokens* dans le vocabulaire final, et il ne prend en charge que 6 langues. 

L'ensemble de ces outils se focalisent uniquement sur des modèles basés sur un tokenizer *sentencepiece*, des encodeurs ou encodeurs-décodeurs, et permettent très difficilement ou sinon pas du tout de gérer la taille du vocabulaire souhaitée.  
Ainsi, dans le cadre de cette introduction au *trimming*, nous avons voulu traiter des modèles basés sur d'autres tokenizers (notamment le [BPE](https://arxiv.org/abs/1508.07909) de SENNRICH, HADDOW et BIRCH (2015), sur d'autres architectures, sur d'autres modalités, et être capable de choisir la taille du nouveau vocabulaire.  
Dans ce qui suit, nous testons l'ensemble des modèles sur une diversité de langue afin d'évaluer la robustesse de la méthode par des évaluateurs indépendants et dont la langue n'utilise pas forcément l'alphabet latin (coréen, tamoul, arabe entre autres).

<br>


### <span style="color: #51C353"> **Modèles d'embeddings textuels** </span>

Observons l'impact du *trimming* sur les performances de différents types de modèles en commençant avec les modèles d'*embedding* textuels.

#### Modèles et benchmarks sélectionnés 
Dans cette section, nous focalisons notamment sur  :
- Les [granite embedding](https://huggingface.co/collections/ibm-granite/granite-embedding-models) de Granite Team (2025) disponibles en [107M](https://huggingface.co/ibm-granite/granite-embedding-107m-multilingual) et [278M](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual) de paramètres ;
- Les [multilingual-e5](https://huggingface.co/collections/intfloat/multilingual-e5-text-embeddings) de [WANG et al.](https://arxiv.org/abs/2402.05672) (2024) disponibles en taille `small`, `base`, `large` et `instruct` ;
- L'[embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) de SCHECHTER VERA, DUA et al. (2025) ;
- Le [bge-m3](https://huggingface.co/BAAI/bge-m3) de CHEN, XIAO et al. (2024) ;
- Le [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) de ZHANG, LI, LONG, ZHANG et al. (2025).


Concernant les benchmarks, nous évaluons les modèles sur de l'anglais et du néerlandais via : 
- Le [MTEB](https://arxiv.org/abs/2210.07316) de MUENNIGHOFF et al. (2022)
- Le [MTEB-NL](https://huggingface.co/collections/clips/mteb-nl) de [BANAR, LOTFI et al.](https://arxiv.org/abs/2509.12340) (2025). Nous avons choisi cette langue car avec ce benchmark, ces auteurs ont également introduit les modèles [e5-NL](https://huggingface.co/collections/clips/e5-nl) qui ont été construit en utilisant la technique du *trimming*.


#### Réductions de paramètres observées
  
<div id="table-reductions-parametres" style="overflow-x:auto; margin:1.5rem 0; max-width:100%;">
<style>
#table-reductions-parametres table {
  border-collapse: collapse;
  width: 100%;
  table-layout: fixed;
  font-size: 0.72rem !important;
  box-shadow: 0 2px 7px rgba(0,0,0,0.12);
  border-radius: 8px;
  overflow: hidden;
}

#table-reductions-parametres th,
#table-reductions-parametres td {
  padding: 7px 6px !important;
  line-height: 1.25 !important;
  text-align: center !important;
  vertical-align: middle !important;
  white-space: normal !important;
  word-break: normal !important;
  overflow-wrap: normal !important;
}

#table-reductions-parametres th {
  font-weight: 700 !important;
}

#table-reductions-parametres th:first-child,
#table-reductions-parametres td:first-child {
  width: 24%;
  text-align: left !important;
}

#table-reductions-parametres th:not(:first-child),
#table-reductions-parametres td:not(:first-child) {
  width: 12.6%;
}
</style>

<table>
    <thead>
      <tr style="background-color:#2d3748;">
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Modèle</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Taille du vocabulaire (original)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Taille du vocabulaire (trimmé)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">% Réduction vocabulaire</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Paramètres (original)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Paramètres (trimmé)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">% Réduction paramètres</th>
      </tr>
    </thead>
    <tbody>
      <!-- ibm-granite — bleu IBM -->
      <tr style="background-color:#d9eeff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/ibm-granite/granite-embedding-107m-multilingual" target="_blank" style="color:#0043a4;font-weight:500;">granite-embedding-107m-multilingual</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">250 002</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,45%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">107,0M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">17,3M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 83,84%</td>
      </tr>
      <tr style="background-color:#d9eeff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/ibm-granite/granite-embedding-107m-multilingual" target="_blank" style="color:#0043a4;font-weight:500;">granite-embedding-107m-multilingual</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">250 002</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 86,89%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">107,0M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">23,6M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 77,96%</td>
      </tr>
      <tr style="background-color:#d9eeff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual" target="_blank" style="color:#0043a4;font-weight:500;">granite-embedding-278m-multilingual</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">250 002</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,45%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">278,0M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">98,6M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 64,53%</td>
      </tr>
      <tr style="background-color:#d9eeff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual" target="_blank" style="color:#0043a4;font-weight:500;">granite-embedding-278m-multilingual</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">250 002</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 86,89%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">278,0M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">111,2M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 60,00%</td>
      </tr>
      <!-- intfloat — doré/ambre -->
      <tr style="background-color:#fdf3d0;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/intfloat/multilingual-e5-small" target="_blank" style="color:#b07d00;font-weight:500;">multilingual-e5-small</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">250 037</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,45%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">117,7M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">27,9M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 76,26%</td>
      </tr>
      <tr style="background-color:#fdf3d0;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/intfloat/multilingual-e5-small" target="_blank" style="color:#b07d00;font-weight:500;">multilingual-e5-small</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">250 037</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 86,89%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">117,7M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">34,2M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 70,91%</td>
      </tr>
      <tr style="background-color:#fdf3d0;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/intfloat/multilingual-e5-base" target="_blank" style="color:#b07d00;font-weight:500;">multilingual-e5-base</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">250 037</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,45%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">278,0M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">98,6M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 64,53%</td>
      </tr>
      <tr style="background-color:#fdf3d0;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/intfloat/multilingual-e5-base" target="_blank" style="color:#b07d00;font-weight:500;">multilingual-e5-base</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">250 037</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 86,89%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">278,0M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">111,2M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 60,00%</td>
      </tr>
      <tr style="background-color:#fdf3d0;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/intfloat/multilingual-e5-large" target="_blank" style="color:#b07d00;font-weight:500;">multilingual-e5-large</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">250 037</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,45%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">559,9M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">320,7M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 42,7%</td>
      </tr>
      <tr style="background-color:#fdf3d0;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/intfloat/multilingual-e5-large" target="_blank" style="color:#b07d00;font-weight:500;">multilingual-e5-large</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">250 037</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 86,89%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">559,9M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">337,4M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 39,7%</td>
      </tr>
      <!-- Google Gemma — rouge Google -->
      <tr style="background-color:#fde4e1;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/google/embeddinggemma-300m" target="_blank" style="color:#c5221f;font-weight:500;">embeddinggemma-300m</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">262 144</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,75%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">307,6M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">118,8M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 61,36%</td>
      </tr>
      <tr style="background-color:#fde4e1;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/google/embeddinggemma-300m" target="_blank" style="color:#c5221f;font-weight:500;">embeddinggemma-300m</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">262 144</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 87,50%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">307,6M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">131,4M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 57,27%</td>
      </tr>
      <!-- BAAI — bleu marine/indigo -->
      <tr style="background-color:#e2e8f8;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/BAAI/bge-m3" target="_blank" style="color:#1a3a8f;font-weight:500;">bge-m3</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">250 002</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,45%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">567,8M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">328,5M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 42,14%</td>
      </tr>
      <tr style="background-color:#e2e8f8;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/BAAI/bge-m3" target="_blank" style="color:#1a3a8f;font-weight:500;">bge-m3</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">250 002</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 86,89%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">567,8M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">345,3M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 39,18%</td>
      </tr>
      <!-- Qwen — violet -->
      <tr style="background-color:#ece8ff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B" target="_blank" style="color:#4a2db5;font-weight:500;">Qwen3-Embedding-0.6B</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">151 669</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 89,20%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">595,8M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">457,2M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 23,25%</td>
      </tr>
      <tr style="background-color:#ece8ff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B" target="_blank" style="color:#4a2db5;font-weight:500;">Qwen3-Embedding-0.6B</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">151 669</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 78,40%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">595,8M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">474,0M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 20,44%</td>
      </tr>
    </tbody>
  </table>

  
</div>
<div class="notice--info" style="line-height:1.25;">
  <p style="font-size:0.9rem;"><strong>📝 Note</strong></p>
  <p style="font-size:0.9rem;">
    Les granite-embedding, mE5, BGE-M3 et Qwen3-Embedding ayant une taille de vocabulaire n'étant pas un multiple de 64, le <i>trimming</i> semble particulièrement adapté même dans le cas où nous ne souhaitons pas spécifiquement créer un modèle monolingue/<i>n</i>-lingue.
  </p>
</div>

Dans le tableau ci-dessus, et pour l'ensemble de l'article, nous donnons des chiffres pour des vocabulaires de taille 16 384 et 32 768 *tokens*. Nous revenons dans la section de l'article, **Questions ouvertes❓**, sur ce choix et plus généralement sur le défi du nombre de *tokens* à conserver dans le vocabulaire du modèle trimmé.

#### Résultats

##### Anglais


<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/Vl1Mhq-ZOFBsxdx0X1CV1.png" 
       alt="Résultats sur MTEB des modèles d'embedding en anglais" 
       style="max-width:100%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Résultats sur MTEB des modèles d'embedding en anglais</center>
  </figcaption></center>
</figure>

Nous pouvons constater que globalement le *trimming* permet de maintenir les performances du modèle original tout en ayant des modèles plus petits. Dans 5 cas sur 8, nous avons même un modèle de 32 768 *tokens* qui est légèrement meilleur que le modèle original. Notre hypothèse est que la suppression de *tokens* parasites bénéficie au modèle.    
Notons un problème néanmoins, la détérioration des performances de l'embeddinggemma. Nous pensions initialement avoir fait une erreur dans notre code, mais avons exclu cette possibilité lorsque nous nous sommes intéressés au gemma-3 (voir les parties sur les décodeurs et les VLM) qui utilise le même *tokenizer* et où nous n'observons pas de baisse de performances (il s'agit même probablement d'un des modèles les plus robustes au *trimming*).
L'explication que nous avançons, et nous avons donc laissé ce modèle pour illustrer ce phénomène, est que le *trimming* ne fonctionne que si la couche d'*embedding* est la dernière couche du réseau (ou également la première en cas d'*embeddings* partagés). Là où cette couche d'*embedding* est bien la dernière (hors *pooling*) pour tous les autres modèles présentés, pour l'embeddinggemma, il y a [deux couches *denses* supplémentaires](https://huggingface.co/google/embeddinggemma-300m/tree/main). 

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/gghbbAdtyMldL8p-dr22p.png" 
       alt="Embeddinggemma avec 2 couche denses au sommet" 
       style="max-width:100%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Embeddinggemma original et trimmé (32 768) avec 2 couche denses au sommet</center>
  </figcaption></center>
</figure>

Pour cette même raison, nous présentons ici uniquement des modèles bi-encodeurs. En effet, nos expérimentations sur des *rerankers* ont toutes échoué du fait de la couche de classification située au-dessus de celle d*'embedding* dans ces types de modèles.  
Avec l'embeddinggemma, nous souhaitions illustrer la seule limite que nous avons identifiée à la méthode du *trimming*.  

<div class="notice--info" style="line-height:1.25;">
  <p style="font-size:0.9rem;"><strong>📝 Note</strong></p>
  <p style="font-size:0.9rem;">
Nous avons choisi de tout de même proposer des versions monolingues de l'embeddinggemma pour deux raisons :<br>
1) Pour l'anglais, la version 32 768 reste meilleure que le Qwen3-Embedding (original ou trimmé) même avec cette dégradation.<br>
2) La dégradation de 2,5 à 5,5 points (en fonction de la configuration retenue) est probablement la plus extrême du fait que le modèle est principalement entraîné sur de l'anglais. Sur d'autres langues testées, la dégradation est ramenée à 1,7 pour le français (65,13 en 32 768 vs. 66,87 pour l'original) et à 0,6 point pour le néerlandais (cf. paragraphe suivant).
  </p>
</div>

##### Néerlandais

Dans le tableau ci-dessous, les modèles ayant le suffixe `-trm` correspondent aux modèles trimmés par les auteurs des e5-NL. Ils utilisent pour leur part 50 002 *tokens* (qui n'est pas un multiple de 64). Les modèles ayant le suffixe `-trm-nl` sont les modèles trimmés évoqués à l'instant sur lesquels les auteurs ont appliqué un finetuning sur 950 000 données (réelles et synthétiques) supplémentaires.

À noter que les résultats diffèrent légèrement de ceux disponibles sur le [*leaderboard*](https://huggingface.co/spaces/mteb/leaderboard) car nous n'avons pu obtenir des résultats que pour 39 des 40 jeux de données constituant le MTEB-NL. Plus précisément, il n'a pas été possible d'évaluer sur `BelebeleRetrieval` du fait d'une erreur de configuration de la langue dans l'API de MTEB. La colonne `Ret` ci-dessous est donc un peu sous-évaluée par rapport au *leaderboard*. 
Ainsi, nous avons également relancé MTEB-NL sur les modèles originaux pour que l'ensemble des modèles soient comparés sur 39 jeux de données.  

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/KQVCrHtXc6iNlBgwzUaim.png" 
       alt="Résultats sur MTEB-NL des modèles d'embedding en néerlandais" 
       style="max-width:100%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Résultats sur MTEB-NL des modèles d'embedding en néerlandais</center>
  </figcaption></center>
</figure>


Comme pour l'anglais, nous pouvons constater que le *trimming* matche voire permet un très léger gain (généralement inférieur à 0,1 point) de performance lorsque nous gardons 32 768 *tokens*.

Un résultat encore plus intéressant est que si l'on conserve davantage de *tokens* (50 002 par les auteurs du MTEB-NL, mais nous pouvons penser que 50 048 serait un nombre plus pertinent vis-à-vis du matériel), le gain monte entre 0,4 et 0,6 point pour les mE5. Sans aucun réentraînement.  
Si en plus, nous finetunons le modèle trimmé, nous pouvons constater que le gain monte alors de 2,5 à 3,1 points par rapport au modèle original.  

<div class="notice--info" style="line-height:1.25;">
  <p style="font-size:0.9rem;"><strong>📝 Note</strong></p>
  <p style="font-size:0.9rem;">
Nous voyons que garder 50 002 <i>tokens</i> semble plus intéressant que d'en garder 32 768.<br> 
Suite à ces résultats, nous nous sommes questionnés sur le fait de savoir si nous devions ou non proposer des modèles avec un vocabulaire de 50 048 <i>tokens</i>. Nous ne l'avons finalement pas fait. D'une part parce que dans le cadre de cet article de blog, notre objectif est de proposer une simple introduction à la méthode du <i>trimming</i> et non pas les meilleurs modèles possibles pour illustrer nos propos. D'autre part parce que cette question de savoir quel est le nombre de *tokens* optimal pour un <i>tokenizer</i> n'est pas évidente.<br> 
Nous revenons sur ce dernier point dans une partie dédiée dans la dernière section de cet article de blog portant sur les <b>Questions ouvertes❓</b>.
  </p>
</div>

<div class="notice--success" style="line-height:1.25;">
  <p style="font-size:0.9rem;"><strong>🧠 À retenir</strong></p>
  <p style="font-size:0.9rem;">
    Pour les modèles d'<em>embeddings</em> textuels, <strong>le <em>trimming</em> permet d'obtenir un modèle monolingue léger par rapport à un modèle multilingue plus gros</strong>.
    Nous pouvons notamment constater que <strong>les performances sont conservées voire légèrement améliorées</strong>.<br>
    <strong>Si vous observez une légère baisse, il est conseillé d'effectuer un finetuning pour retrouver les performances originales, voire les surpasser</strong>.<br>
    Ce gain de performance via finetuning s'avère même rapide à obtenir car <strong>le modèle trimmé étant plus petit que l'original, le temps d'entraînement est accéléré</strong>. À budget de calcul équivalent, cela ouvre alors la possibilité de montrer davantage de données au modèle trimmé qu'à un modèle original qu'on finetunerait (nous menons cette expérience plus bas dans la partie sur les encodeurs-décodeurs).
    À noter que <strong>les avantages listés ci-dessus ne sont applicables qu'à des modèles où la couche d'<em>embedding</em> est la dernière du réseau. Le <em>trimming</em> ne fonctionnera pas s'il y a des couches positionnées au-dessus de celle d'<em>embedding</em></strong>.
  </p>
</div>

<br>

#### Collections

Nous nous sommes concentrés sur l'anglais et le néerlandais dans le cadre de cet article de blog. Néanmoins, pour chacun des 5 modèles listés ci-dessus, nous avons généré l'ensemble des modèles monolingues possibles à partir des langues gérées par chacun des modèles originaux.  
Pour chaque, nous les fournissons en versions 16 384 *tokens* et 32 768 *tokens*, ainsi que dans l'ensemble des tailles disponibles.  
Certains acteurs ne fournissant pas la liste complète des langues gérées par leur modèle, nous avons généré du mieux que nous pouvions à partir de l'analyse des caractères unicode trouvés dans le tokenizer du modèle original.

<div style="line-height:1.25; background:#2e1f47; color:#f5f3ff; padding:1rem; border-radius:4px;">
  <p style="font-size:0.9rem;"><strong>🎁 Important</strong></p>
  <p style="font-size:0.9rem;">
    La collection rassemblant les <strong>48</strong> modèles (<strong>12</strong> langues différentes) trimmés à partir des granite-embedding est disponible <a href="https://huggingface.co/collections/alphaedge-ai/trimming-granite-embedding" style="color:#c4b5fd;">ici</a>.<br>
    La collection rassemblant les <strong>948</strong> modèles (<strong>119</strong> langues différentes) trimmés à partir des multilingual-e5 est disponible <a href="https://huggingface.co/collections/alphaedge-ai/trimming-multilingual-e5" style="color:#c4b5fd;">ici</a>.<br>
    La collection rassemblant les <strong>243</strong> modèles (<strong>122</strong> langues différentes) trimmés à partir de l'embeddinggemma est disponible <a href="https://huggingface.co/collections/alphaedge-ai/trimming-embeddinggemma" style="color:#c4b5fd;">ici</a>.<br>
    La collection rassemblant les <strong>178</strong> modèles (<strong>89</strong> langues différentes) trimmés à partir du bge-m3 est disponible <a href="https://huggingface.co/collections/alphaedge-ai/trimming-bge-m3" style="color:#c4b5fd;">ici</a>.<br>
    La collection rassemblant les <strong>166</strong> modèles (<strong>83</strong> langues différentes) trimmés à partir du qwen3-embedding est disponible <a href="https://huggingface.co/collections/alphaedge-ai/trimming-multilingual-qwen3-embedding-06b" style="color:#c4b5fd;">ici</a>.
  </p>
</div>

<br>


### Modèles encodeur

Continuons d'analyser l'impact du *trimming* sur les performances de différents types de modèles en passant à présent aux modèles encodeur.

#### Modèles et benchmarks sélectionnés 
Dans cette section, nous nous focalisons sur le [mmBERT](https://huggingface.co/collections/jhu-clsp/mmbert-a-modern-multilingual-encoder) de MARONE, WELLER et al. (2025) et plus particulièrement la version [small](https://huggingface.co/jhu-clsp/mmBERT-small) et la version [base](https://huggingface.co/jhu-clsp/mmBERT-base), mais aussi la [version 32k](https://huggingface.co/llm-semantic-router/mmbert-32k-yarn) de vLLM Semantic Router Team (2026).  

Nous évaluons l'ensemble de ces modèles sur du français, et plus particulièrement sur la tâche de reconnaissance d'entités nommées et le jeu de données [frenchNER_3entities](https://huggingface.co/datasets/CATIE-AQ/frenchNER_3entities) de BOURDOIS (2024) portant sur les entités Personnes, Lieux et Organisations. L'idée ici est de se concentrer sur un seul jeu de données mais de mener plusieurs expériences dessus pour déterminer :
- L'impact de l'ordre du *trimming* (faut-il le faire avant ou après le finetuning ?);
- L'impact de la base de données ayant servi à miner les *tokens*;
- L'impact du nombre de données vues lors du minage des *tokens*;
- Une comparaison avec un modèle en français entraîné de zéro ayant la même architecture (i.e. le [modernBERT](https://arxiv.org/abs/2412.13663) de WARNER, CHAFFIN, CLAVIÉ et al. (2024));
- Une comparaison du *trimming* avec la méthode de la distillation.

Nous évaluons ce dernier point également sur de l'anglais en comparant un mmBERT `small` trimmé sur de l'anglais avec le [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) de [SANH et al.](https://arxiv.org/abs/1910.01108) (2019) sur le benchmark [GLUE](https://arxiv.org/abs/1804.07461) de WANG et al. (2018). L'idée étant ici à l'inverse de se concentrer sur un seul modèle mais l'évaluer sur plusieurs tâches.


#### Réductions de paramètres observées

<div id="table-reductions-parametres" style="overflow-x:auto; margin:1.5rem 0; max-width:100%;">
<style>
#table-reductions-parametres table {
  border-collapse: collapse;
  width: 100%;
  table-layout: fixed;
  font-size: 0.72rem !important;
  box-shadow: 0 2px 7px rgba(0,0,0,0.12);
  border-radius: 8px;
  overflow: hidden;
}

#table-reductions-parametres th,
#table-reductions-parametres td {
  padding: 7px 6px !important;
  line-height: 1.25 !important;
  text-align: center !important;
  vertical-align: middle !important;
  white-space: normal !important;
  word-break: normal !important;
  overflow-wrap: normal !important;
}

#table-reductions-parametres th {
  font-weight: 700 !important;
}

#table-reductions-parametres th:first-child,
#table-reductions-parametres td:first-child {
  width: 24%;
  text-align: left !important;
}

#table-reductions-parametres th:not(:first-child),
#table-reductions-parametres td:not(:first-child) {
  width: 12.6%;
}
</style>

<table>
    <thead>
      <tr style="background-color:#2d3748;">
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Modèle</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Taille du vocabulaire (original)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Taille du vocabulaire (trimmé)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">% Réduction vocabulaire</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Paramètres (original)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Paramètres (trimmé)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">% Réduction paramètres</th>
      </tr>
    </thead>
    <tbody>
      <!-- jhu-clsp — vert -->
      <tr style="background-color:#dff5de;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/jhu-clsp/mmBERT-small" target="_blank" style="color:#1e6b1e;font-weight:500;">mmBERT-small</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">256 000</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,60%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">140,5M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">48,5M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 65,49%</td>
      </tr>
      <tr style="background-color:#dff5de;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/jhu-clsp/mmBERT-small" target="_blank" style="color:#1e6b1e;font-weight:500;">mmBERT-small</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">256 000</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 87,20%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">140,5M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">54,8M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 61,01%</td>
      </tr>
      <tr style="background-color:#dff5de;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/jhu-clsp/mmBERT-base" target="_blank" style="color:#1e6b1e;font-weight:500;">mmBERT-base</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">256 000</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,60%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">306,9M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">122,9M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 59,96%</td>
      </tr>
      <tr style="background-color:#dff5de;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/jhu-clsp/mmBERT-base" target="_blank" style="color:#1e6b1e;font-weight:500;">mmBERT-base</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">256 000</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 87,20%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">306,9M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">135,5M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 55,86%</td>
      </tr>
      <!-- llm-semantic-router — orange -->
      <tr style="background-color:#ffe8cc;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/llm-semantic-router/mmbert-32k-yarn" target="_blank" style="color:#a04000;font-weight:500;">mmbert-32k-yarn</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">256 000</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,60%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">306,9M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">122,9M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 59,96%</td>
      </tr>
      <tr style="background-color:#ffe8cc;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/llm-semantic-router/mmbert-32k-yarn" target="_blank" style="color:#a04000;font-weight:500;">mmbert-32k-yarn</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">256 000</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 87,20%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">306,9M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">135,5M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 55,86%</td>
      </tr>
    </tbody>
  </table>
</div>

#### Résultats

##### Français

**Ordre du trimming** :

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/6IF3xzixCUiQr3yq5OlEL.png" 
       alt="Ordre du trimming" 
       style="max-width:100%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Ordre du trimming</center>
  </figcaption></center>
</figure>

D'après ces expérimentations, dans le cadre du mmBERT, il est impératif de trimmer le modèle original puis finetuner le modèle résultant plutôt que de finetuner le modèle original puis de le trimmer.  
Au-delà d'éviter un effondrement des performances, finetuner un modèle trimmé est aussi plus rapide que de finetuner un modèle original car le modèle est plus petit.

<div class="notice--info" style="line-height:1.25;">
  <p style="font-size:0.9rem;"><strong>📝 Note</strong></p>
  <p style="font-size:0.9rem;">
Nous revenons sur ce point de l'ordre du <i>trimming</i> dans une partie dédiée dans la dernière section de l'article portant sur less <b>Questions ouvertes❓</b>.
  </p>
</div>

Nous pouvons noter que comme pour les modèles d'*embeddings* textuels, garder 32 768 *tokens* donne des résultats semblables au modèle d'origine (petite baisse sur l'entité Organisations à souligner), et qu'en garder 16 384 fait légèrement moins bien.  

Enfin le modèle YaRN apparaît comme moins performant que le modèle original non trimmé, que le modèle original trimmé à 32 768 ou 16 384 *tokens*.

<br>

**Impact de la base de données de textes de minage** :

Ayant découvert le *trimming* il y a quelques années via l'article Medium de David DALE, nous utilisons comme lui le [Leipzig Corpora](https://corpora.uni-leipzig.de) de [GOLDHAHN et al.](http://www.lrec-conf.org/proceedings/lrec2012/pdf/327_Paper.pdf) (2012) pour le minage des *tokens*.  
À l'occasion de cet article de blog, nous nous sommes intéressés à explorer l'impact des données de minage sur les performances des modèles. Dans cette partie, nous comparons donc le Leipzig Corpora à [Fineweb 2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) de [PENEDO et al.](https://arxiv.org/abs/2506.20920) (2025) publié il y a quelques mois. Pour Leipzig Corpora nous avons utilisé les 200 000 premières données du jeu de données `fra_wikipedia_2021` et pour Fineweb 2 les 200 000 premières données du *split* `fra_Latn`.

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/JeCR5a03WChL7HiQ1Mp9e.png" 
       alt="Impact de la base de données de textes de minage" 
       style="max-width:95%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Impact de la base de données de textes de minage</center>
  </figcaption></center>
</figure>

Nous pouvons observer des résultats similaires entre les deux corpus, quel que soit le nombre de *tokens* gardé dans le vocabulaire.  
Pour notre part, et ceci pour l'ensemble des résultats présentés dans cet article ainsi que pour les 5526 modèles générés, nous avons utilisé Fineweb 2.  
Ce choix s'est imposé uniquement d'un point de vue pratique. En effet, Fineweb 2 est disponible directement sur le Hub d'Hugging Face regroupant toutes les langues et fichiers en un unique endroit requêtable de manière automatisée. Là où pour le Leipzig Corpora, il est nécessaire de télécharger les corpus manuellement.  

<br>

**Impact du nombre de textes lors du minage** :

Nous venons de voir que la source utilisée pour miner les *tokens* n'avait pas une grande importance sur les performances. Qu'en est-il du nombre de textes utilisés ?

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/0rWAy3CAsc9nNYmyBfcHt.png" 
       alt="Impact de la base de données de textes de minage" 
       style="max-width:95%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Impact de la base de données de textes de minage</center>
  </figcaption></center>
</figure>

Il s'avère que le nombre de textes n'a pas non plus une grande importance sur les performances.  
Le nombre choisi a cependant un impact sur le temps d'exécution du minage. En effet, obtenir la version trimmée sur 20 000 textes a pris 2 min 04s sur un CPU Intel(R) Core(TM) Ultra 7 255H (2.00 GHz) pour l'ensemble du processus (minage des *tokens*, modification des fichiers de vocabulaire et des couches de l'architecture, pousser le modèle sur le Hub). La version trimmée sur 200 000 textes a pris 8 min 56s. Enfin la version trimmée sur 2 000 000 textes a pris 2h 26min 21s.  

<div style="line-height:1.25; background:#241216; color:#e5e7eb; padding:1rem; border-radius:4px;">
  <p style="font-size:0.9rem;"><strong>Vigilance ⚠️</strong></p>
  <p style="font-size:0.9rem;">
    Nous avons choisi pour notre part de miner l'ensemble des modèles présentés dans cet article de blog sur 200 000 textes au lieu de 20 000 (moins si Fineweb 2 ne propose pas 200 000 textes pour la langue en question).<br>
    En effet, nous voulions sécuriser les choses vis-à-vis de deux points que nous n'avons pas pu tester :<br>
    1) Nous avons miné jusqu'à 32 768 <em>tokens</em> dans le cadre de cet article de blog, mais est-ce que 20 000 textes suffisent si nous souhaitons garder par exemple 65&nbsp;536 <em>tokens</em> ?<br>
    2) Bien que nous ayons voulu tester le <em>trimming</em> sur plusieurs langues portant sur divers alphabets, nous avons évalué exclusivement des langues richement dotées en ressources. C'est-à-dire des langues avec un nombre suffisamment de <em>tokens</em> dans le modèle original qui, in fine, impacte les performances du modèle trimmé, et qui disposent également de jeux de données d'évaluation pour plusieurs tâches.<br>
    Nous n'avons pas pu vérifier l'impact du nombre de textes sur les langues peu dotées en ressources. Une observation qui nous a invité à rédiger cet encart est que pour les mmBERT nous avons pu par exemple générer un modèle en tchétchène en 16 384 <em>tokens</em> mais pas en 32 768 par manque de <em>tokens</em> (unique langue néanmoins où ce phénomène s'est produit).
  </p>
</div>

<div class="notice--info" style="line-height:1.25;">
  <p style="font-size:0.9rem;"><strong>📝 Note</strong></p>
  <p style="font-size:0.9rem;">
  Le jeu de données Fineweb 2 étant lourd (540GB pour les 360M lignes en français par exemple et plus de 8,67 TB au total), et comme nous venons de le voir, le nombre de lignes vues lors du minage n'influençant pas les performances, nous avons créé <strong>un sous-ensemble de Fineweb 2 pour 124 langues disponible <a href="https://huggingface.co/datasets/lbourdois/fineweb-2-trimming" target="_blank">ici</a></strong> (seulement 44GB). Ces 124 langues correspondent à l'intersection des langues proposées par le mBERT et le XLM-RoBERTa.
  </p>
</div>

<div class="notice--info" style="line-height:1.25;">
  <p style="font-size:0.9rem;"><strong>📝 Note</strong></p>
  <p style="font-size:0.9rem;">
Nous nous focalisions ici sur des textes génériques issus de Fineweb 2. Nous n'avons pas eu le temps d'effectuer l'expérience, mais le fait que seuls 20 000 textes soient nécessaires pour le <i>trimming</i> ouvre la porte à un minage sur des données spécialisées sur un domaine donné (médecine, banque, industrie, etc.) qui sont généralement moins nombreuses que les généralistes. Cela permettrait de ne garder que les <i>tokens</i> indispensables au domaine et qu'ils soient alignés avec le jeu de données nécessaire pour réaliser l'adaptation au domaine via un pré-entraînement supplémentaire.
  </p>
</div>

<br>

**Comparaison avec un modèle entraîné de zéro** :

Pour cette comparaison d'un modèle monolingue entraîné de zéro vs. un modèle multilingue trimmé, il nous semblait important de comparer des modèles partageant exactement la même architecture à savoir celle du modernBERT (plutôt qu'un modernBERT trimmé vs. un BERT/RoBERTa/DeBERTa entraîné de zéro par exemple).  
Pour le français, il existe une version entraînée de zéro du modernBERT : le [ModernCamemBERT](https://huggingface.co/almanach/moderncamembert-cv2-base) d'[ANTOUN et al.](https://arxiv.org/abs/2504.08716) (2025).

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/u1QvikbNPrh10ssfY7pLY.png" 
       alt="Ordre du trimming" 
       style="max-width:80%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Comparaison avec un modèle entraîné de zéro</center>
  </figcaption></center>
</figure>

Nous pouvons clairement voir que le modèle monolingue entraîné de zéro surpasse le modèle multilingue original ainsi que sa version trimmée.  
Dès lors, lorsqu'il existe une version monolingue d'un modèle dans votre langue, nous conseillons de l'utiliser.   Dans le cas contraire, une version trimmée peut être une alternative simple en attendant un modèle monolingue entraîné de zéro.  
Accessoirement, vous profiterez également d'une optimisation du nombre de *tokens* par mot (1,42 pour moderncamembert contre 1,58 pour la version trimmée).

<br>

**Comparaison avec la distillation :**

Pour comparer un modèle trimmé à un modèle obtenu par distillation, utilisons le [distilCamemBERT](https://huggingface.co/cmarkea/distilcamembert-base) de [DELESTRE et AMAR](https://hal.archives-ouvertes.fr/hal-03674695/file/cap2022.pdf) (2022) distillé à partir du [CamemBERT base](https://arxiv.org/abs/1911.03894) de MARTIN, MULLER, ORTIZ SUÁREZ, et al. (2019). Pour cela, en plus des options en 16 384 et 32 768 *tokens*, nous avons également généré un modèle de 67 476 *tokens* pour avoir une version trimmée possédant le même nombre de paramètres que distilCamemBERT à savoir 67,5M.

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/_xqO3QbLONtH8irEazj3B.png" 
       alt="Comparaison avec la distillation" 
       style="max-width:80%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Comparaison avec la distillation</center>
  </figcaption></center>
</figure>

Nous pouvons remarquer que le modèle distillé fait presque jeu égal avec le modèle mmBERT original. Cela va dans le sens du point précédent qu'un modèle monolingue entraîné de zéro est supérieur à un modèle multilingue.  
Une autre observation est que le modèle trimmé à 67 476 *tokens*, ayant le même nombre de paramètres que la version distillée, performe de manière équivalente à la distillation.  
La différence fondamentale est le temps nécessaire pour obtenir les deux modèles. Pour la version distillée, les auteurs indiquent que le modèle a tourné 18j sur un GPU Titan RTX.  
De notre côté, nous avons obtenu la version trimmée en 8 min 56s sur un CPU Intel(R) Core(TM) Ultra 7 255H (2.00 GHz) pour l'ensemble du processus.  

Accessoirement le modèle trimmé gère une taille de contexte 16 fois plus longue.

<br>

<div class="notice--success" style="line-height:1.25;">
  <p style="font-size:0.9rem;"><strong>🧠 À retenir</strong></p>
  <p style="font-size:0.9rem;">
    À l'issue de ces différentes expérimentations sur le mmBERT, nous retenons que :
  </p>
  <ul style="font-size:0.9rem;">
    <li><strong>Il faut trimmer un modèle puis le finetuner plutôt que l'inverse</strong>.</li>
    <li><strong>Il faut un nombre suffisant de données pour le minage des <em>tokens</em> mais celui-ci est en réalité relativement faible</strong> (20 000 textes généralistes pouvant suffire pour les langues richement dotées) et <strong>la source des textes n'a pas un grand impact sur les performances</strong>.</li>
    <li><strong>Un modèle monolingue entraîné de zéro fera mieux qu'un modèle multilingue trimmé mais ce dernier peut être une alternative en attendant la conception d'un tel modèle</strong>.</li>
    <li><strong>Le <em>trimming</em> peut être plus intéressant que la distillation. Nous conseillons de le préférer à cette dernière lorsque le nombre de paramètres à réduire souhaités est équivalent</strong>.</li>
  </ul>
</div>

<br>

##### Anglais

**Comparaison avec la distillation :**

Pour l'anglais, nous nous focalisons sur un seul modèle mais faisons une comparaison incluant plusieurs tâches (à savoir 8 tâches du benchmark GLUE).  
Précisons ici que pour le finetuning, l'entraînement est effectué sur l'échantillon `train` et le test sur l'échantillon `dev` et non sur celui de `test`. Nous procédons de la sorte surtout pour pouvoir nous comparer au DistilBERT qui a suivi cette méthodologie.  
Nous reprenons ainsi les chiffres que les auteurs indiquent dans leur publication.  

<div id="table-reductions-parametres" style="overflow-x:auto; margin:1.5rem 0; max-width:100%;">
<style>
#table-reductions-parametres table {
  border-collapse: collapse;
  width: 100%;
  table-layout: fixed;
  font-size: 0.72rem !important;
  box-shadow: 0 2px 7px rgba(0,0,0,0.12);
  border-radius: 8px;
  overflow: hidden;
}

#table-reductions-parametres th,
#table-reductions-parametres td {
  padding: 7px 6px !important;
  line-height: 1.25 !important;
  text-align: center !important;
  vertical-align: middle !important;
  white-space: normal !important;
  word-break: normal !important;
  overflow-wrap: normal !important;
}

#table-reductions-parametres th {
  font-weight: 700 !important;
}

#table-reductions-parametres th:first-child,
#table-reductions-parametres td:first-child {
  width: 24%;
  text-align: left !important;
}

#table-reductions-parametres th:not(:first-child),
#table-reductions-parametres td:not(:first-child) {
  width: 12.6%;
}
</style>

<table>
    <thead>
      <tr style="background-color:#2d3748;">
        <th style="padding:12px;text-align:center;font-weight:600;color:#ffffff;">Modèle</th>
        <th style="padding:12px;text-align:center;font-weight:600;color:#ffffff;">#</th>
        <th style="padding:12px;text-align:center;font-weight:600;color:#ffffff;">MNLI</th>
        <th style="padding:12px;text-align:center;font-weight:600;color:#ffffff;">QQP</th>
        <th style="padding:12px;text-align:center;font-weight:600;color:#ffffff;">QNLI</th>
        <th style="padding:12px;text-align:center;font-weight:600;color:#ffffff;">SST-2</th>
        <th style="padding:12px;text-align:center;font-weight:600;color:#ffffff;">CoLA</th>
        <th style="padding:12px;text-align:center;font-weight:600;color:#ffffff;">STS-B</th>
        <th style="padding:12px;text-align:center;font-weight:600;color:#ffffff;">MRPC</th>
        <th style="padding:12px;text-align:center;font-weight:600;color:#ffffff;">RTE</th>
        <th style="padding:12px;text-align:center;font-weight:600;color:#ffffff;">Moyenne</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background-color:#f8f9fa;">
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/distilbert/distilbert-base-uncased" target="_blank" style="color:#185FA5;font-weight:500;">distilbert-base-uncased</a></td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">67M</td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">82,2</td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">88,5</td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">89,2</td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">91,3</td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">51,3</td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">85,8</td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">87,5</td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">59,9</td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">79,5</td>
      </tr>
      <tr style="background-color:#dff5de;">
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/alphaedge-ai/mmBERT-small-eng-32768" target="_blank" style="color:#1e6b1e;font-weight:500;">mmBERT-small-eng-32768</a></td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">54,8M</td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">84,1</td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">88,0</td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">90,1</td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">91,7</td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">50,3</td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">88,0</td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">89,9</td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">69,7</td>
        <td style="padding:10px;text-align:center;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">81,5</td>
      </tr>
    </tbody>
  </table>
</div>

Nous pouvons noter qu'à l'exécution des tâches QQP et CoLA, les résultats du mmBERT small trimmé sont supérieurs au DistilBERT.  
Cela s'explique sûrement par le fait que le mmBERT small non trimmé score à 84,7 sur ce benchmark.  

<div style="line-height:1.25; background:#2e1f47; color:#f5f3ff; padding:1rem; border-radius:4px;">
  <p style="font-size:0.9rem;"><strong>🎁 Important</strong></p>
  <p style="font-size:0.9rem;">
    Pour l'anglais, le <a href="https://huggingface.co/alphaedge-ai/mmBERT-small-eng-32768" style="color:#c4b5fd;">mmBERT small trimmé</a> de 54,8M de paramètres se présente comme une alternative au DistilBERT tout en étant 18,2% plus petit, capable de gérer 8192 <em>tokens</em> au lieu de 512, et de proposer des résultats plus performants.<br>
  </p>
</div>

À nouveau, comme pour ce que nous avons observé sur le français, un atout important du *trimming* porte sur le temps nécessaire pour obtenir ces versions plus petites des modèles originaux (du mmBERT small donc pour la version trimmé, et du BERT base pour le DistilBERT). Le DistilBERT a nécessité environ 90h de calcul sur 8 GPU V100 16GB contre 9 min 01s sur un CPU Intel(R) Core(TM) Ultra 7 255H (2.00 GHz) pour l'ensemble du processus.


<div class="notice--success" style="line-height:1.25;">
  <p style="font-size:0.9rem;"><strong>🧠 À retenir</strong></p>
  <p style="font-size:0.9rem;">
    À l'issue de ces différentes expérimentations sur le mmBERT, nous retenons que :
  </p>
  <ul style="font-size:0.9rem; margin-top:0.4rem; margin-bottom:0; padding-left:1.2rem;">
    <li><strong>Il faut trimmer un modèle puis le finetuner plutôt que l'inverse</strong>.</li>
    <li><strong>Il faut un nombre suffisant de données pour le minage des <em>tokens</em> mais celui-ci est en réalité relativement faible</strong> (20 000 textes généralistes pouvant suffire pour les langues richement dotées) et <strong>la source des textes n'a pas un grand impact sur les performances</strong>.</li>
    <li><strong>Un modèle monolingue entraîné de zéro fera mieux qu'un modèle multilingue trimmé mais ce dernier peut être une alternative en attendant la conception d'un tel modèle</strong>.</li>
    <li><strong>Le <em>trimming</em> peut être plus intéressant que la distillation. Nous conseillons de le préférer à cette dernière lorsque le nombre de paramètres à réduire souhaités est équivalent</strong>.</li>
  </ul>
</div>


#### Collections

<div style="line-height:1.25; background:#2e1f47; color:#f5f3ff; padding:1rem; border-radius:4px;">
  <p style="font-size:0.9rem;"><strong>🎁 Important</strong></p>
  <p style="font-size:0.9rem;">
    La collection rassemblant les <strong>514</strong> modèles (<strong>124</strong> langues différentes) trimmés à partir des mmBERT est disponible <a href="https://huggingface.co/collections/alphaedge-ai/trimming-mmbert" style="color:#c4b5fd;">ici</a>.
  </p>
</div>

<br>


### Modèles encodeur-décodeur

Intéressons-nous à présent aux modèles encodeur-décodeur. 

#### Modèles et benchmarks sélectionnés 

Pour les modèles encodeur-décodeur, nous nous concentrons sur :
- Le [mT5](https://huggingface.co/collections/google/mt5-release) de XUE, CONSTANT, ROBERTS et al. (2021) et plus particulièrement la version [small](https://huggingface.co/google/mt5-small) et la version [base](https://huggingface.co/google/mt5-base);
- Le [mBART](https://huggingface.co/facebook/mbart-large-50) de LIU, GU, GOYAL et al. (2020).


Nous effectuons les évaluations de ces modèles sur du coréen et plus précisément sur la tâche de génération de question via le benchmark [qg_koquad](https://huggingface.co/datasets/lmqg/qg_koquad) de [USHIO et al.](https://arxiv.org/abs/2210.03992) (2022).

Nous les évaluons également sur de l'arabe et de l'allemand dans le cadre d'une tâche de traduction. En effet, jusqu'ici nous avons montré des usages du *trimming* pour générer des modèles monolingues à partir d'un modèle multilingue. Notre souhait est de montrer que la méthode peut s'appliquer aussi pour générer des modèles *n*-lingues à partir d'un modèle multilingue. Le jeu de données retenu pour la traduction est la partie `ar-de` de [MultiUN](https://huggingface.co/datasets/Helsinki-NLP/multiun) d'[EISELE et CHEN](http://www.lrec-conf.org/proceedings/lrec2010/pdf/686_Paper.pdf) (2010) où nous avons gardé les 100 000 premiers échantillons que nous avons divisé en 90% *train*, 5% éval et 5% *test*.


#### Réductions de paramètres observées

<div id="table-reductions-parametres" style="overflow-x:auto; margin:1.5rem 0; max-width:100%;">
<style>
#table-reductions-parametres table {
  border-collapse: collapse;
  width: 100%;
  table-layout: fixed;
  font-size: 0.72rem !important;
  box-shadow: 0 2px 7px rgba(0,0,0,0.12);
  border-radius: 8px;
  overflow: hidden;
}

#table-reductions-parametres th,
#table-reductions-parametres td {
  padding: 7px 6px !important;
  line-height: 1.25 !important;
  text-align: center !important;
  vertical-align: middle !important;
  white-space: normal !important;
  word-break: normal !important;
  overflow-wrap: normal !important;
}

#table-reductions-parametres th {
  font-weight: 700 !important;
}

#table-reductions-parametres th:first-child,
#table-reductions-parametres td:first-child {
  width: 24%;
  text-align: left !important;
}

#table-reductions-parametres th:not(:first-child),
#table-reductions-parametres td:not(:first-child) {
  width: 12.6%;
}
</style>

<table>
    <thead>
      <tr style="background-color:#2d3748;">
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Modèle</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Taille du vocabulaire (original)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Taille du vocabulaire (trimmé)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">% Réduction vocabulaire</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Paramètres (original)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Paramètres (trimmé)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">% Réduction paramètres</th>
      </tr>
    </thead>
    <tbody>
      <!-- google/mt5 — vert sauge -->
      <tr style="background-color:#fde4e1;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/google/mt5-small" target="_blank" style="color:#c5221f;font-weight:500;">mt5-small</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">250 112</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,45%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">300,2M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">60,8M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 79,73%</td>
      </tr>
      <tr style="background-color:#fde4e1;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/google/mt5-small" target="_blank" style="color:#c5221f;font-weight:500;">mt5-small</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">250 112</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 86,90%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">300,2M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">77,6M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 74,14%</td>
      </tr>
      <tr style="background-color:#fde4e1;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/google/mt5-base" target="_blank" style="color:#c5221f;font-weight:500;">mt5-base</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">250 112</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,45%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">582,4M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">223,4M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 61,64%</td>
      </tr>
      <tr style="background-color:#fde4e1;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/google/mt5-base" target="_blank" style="color:#c5221f;font-weight:500;">mt5-base</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">250 112</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 86,90%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">582,4M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">248,6M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 57,32%</td>
      </tr>
      <!-- facebook/mbart — tan/bois -->
      <tr style="background-color:#e2e8f8;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/facebook/mbart-large-50" target="_blank" style="color:#1a3a8f;font-weight:500;">mbart-large-50</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">250 054</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,45%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">610,9M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">371,6M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 39,17%</td>
      </tr>
      <tr style="background-color:#e2e8f8;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/facebook/mbart-large-50" target="_blank" style="color:#1a3a8f;font-weight:500;">mbart-large-50</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">250 054</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 86,90%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">610,9M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">388,4M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 36,42%</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="notice--info" style="line-height:1.25;">
  <p style="font-size:0.9rem;"><strong>📝 Note</strong></p>
  <p style="font-size:0.9rem;">
Le mBART ayant une taille de vocabulaire n'étant pas un multiple de 64, le <i>trimming</i> semble particulièrement adapté même dans le cas où nous ne souhaitons pas spécifiquement créer un modèle monolingue/<i>n</i>-lingue.
  </p>
</div>

<div class="notice--info" style="line-height:1.25;">
  <p style="font-size:0.9rem;"><strong>📝 Note</strong></p>
  <p style="font-size:0.9rem;">
Nous pensions initialement proposer également la version <code>large</code> du mt5 (réduction du nombre de paramètres de 1,230B à 784,5M soit -36,20%) mais y avons finalement renoncé pour proposer à la place des modèles décodeurs plus récents (voir la section suivante).
  </p>
</div>

#### Résultats

##### Coréen

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/bjK2Hjpc_LuNXXdGQGPAa.png" 
       alt="Résultats de la tâche de génération de question via un encodeur-décodeur" 
       style="max-width:55%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Résultats de la tâche de génération de question via encodeur-décodeur</center>
  </figcaption></center>
</figure>

Pour le mT5 `small`, nous pouvons voir une légère amélioration sur la métrique BLEU et une légère dégradation sur la métrique ROUGE-L.  
Pour le mt5 `base`, la version 32 768 donne les meilleurs résultats tout modèle (originaux ou trimmés) confondu.  

Le mBART se comporte moins bien que le mT5 puisque nous pouvons constater une dégradation des résultats. 

<br>

##### Arabe-Allemand

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/zY_AdP2XROS0rYNHkp6aS.png" 
       alt="Résultats de la tâche de traduction via un encodeur-décodeur" 
       style="max-width:55%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Résultats de la tâche de traduction via un encodeur-décodeur</center>
  </figcaption></center>
</figure>

Pour les mT5, nous pouvons observer une légère dégradation pour la version `small` (surtout en version 16 384 *tokens*) et un match voire une légère amélioration pour la version `base`.  

Pour le mBART, au lieu de générer la version 16 384 *tokens*, nous avons choisi de réaliser une autre expérience. Profitant du fait que le modèle trimmé est plus petit que l'original et que le *finetuning* s'exécute alors plus rapidement, nous avons poursuivi l'entraînement. Cela permet alors de ne pas faire une comparaison à nombre d'*epochs* équivalentes (5 pour chaque modèle pour 58,45 vs 58,77), mais à temps d'exécution équivalent (50 min pour chaque). Nous pouvons alors observer un gain de 1,76 point de BLEU à budget temps similaire.

<div class="notice--success" style="line-height:1.25;">
  <p style="font-size:0.9rem;"><strong>🧠 À retenir</strong></p>
  <p style="font-size:0.9rem;">
    Pour les encodeurs-décodeurs, le <em>trimming</em> permet d'avoir des résultats plus ou moins équivalents au modèle original.
    <strong>Profiter du fait que les modèles trimmés soient plus petits pour les entraîner plus longtemps permet un gain important par rapport au modèle original</strong>.
  </p>
</div>

#### Collections

<div style="line-height:1.25; background:#2e1f47; color:#f5f3ff; padding:1rem; border-radius:4px;">
  <p style="font-size:0.9rem;"><strong>🎁 Important</strong></p>
  <p style="font-size:0.9rem;">
    Une collection rassemblant <strong>392</strong> modèles (<strong>98</strong> langues différentes) trimmés à partir des mt5 est disponible <a href="https://huggingface.co/collections/alphaedge-ai/trimming-mt5" style="color:#c4b5fd;">ici</a>.<br>
    Une collection rassemblant <strong>104</strong> modèles (<strong>52</strong> langues différentes) trimmés à partir du mbart est disponible <a href="https://huggingface.co/collections/alphaedge-ai/trimming-mbart" style="color:#c4b5fd;">ici</a>.
  </p>
</div>

<br>

### Modèles décodeur

Poursuivons nos expérimentations avec les modèles décodeur.

#### Modèles et benchmarks sélectionnés 

Nous nous intéressons ici à quatre familles de décodeurs : 
- Les [Granite 4.0](https://huggingface.co/collections/ibm-granite/granite-40-language-models) d'IBM Research (2025) dans ses tailles `350m` et `1b` (que ce soit en version attention ou hybride) ;
- Les [Qwen 3](https://huggingface.co/collections/Qwen/qwen3) de Qwen Team (2025) dans ses tailles `0.6B` et `1.7B` ;
- Les [Gemma 3](https://huggingface.co/collections/google/gemma-3-release) de Google DeepMind (2025) dans ses tailles `270m` et `1b` ;
- Le [SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) d'Hugging Face (2025).

Nous évaluons ces modèles sur le néerlandais et le coréen.  
Pour le néerlandais, nous utilisons l'échantillon portant sur cette langue du *leaderboard* [EuroEval](https://euroeval.com/leaderboards/Monolingual/dutch/) de [NIELSEN et al.](https://arxiv.org/abs/2406.13469) (2024). Nous faisons notamment un focus sur les tâches génératives proposées par ce *benchmark* et laissons de côté les tâches de type classification sur lesquelles les encodeurs sont plus efficaces.  
Pour le coréen, nous appliquons les modèles sur le benchmark [KoBEST](https://huggingface.co/datasets/skt/kobest_v1) de [KIM et al.](https://arxiv.org/abs/2204.04541) (2022) où nous avons procédé à une configuration 5-*shots*.


#### Réductions de paramètres observées

<div id="table-reductions-parametres" style="overflow-x:auto; margin:1.5rem 0; max-width:100%;">
<style>
#table-reductions-parametres table {
  border-collapse: collapse;
  width: 100%;
  table-layout: fixed;
  font-size: 0.72rem !important;
  box-shadow: 0 2px 7px rgba(0,0,0,0.12);
  border-radius: 8px;
  overflow: hidden;
}

#table-reductions-parametres th,
#table-reductions-parametres td {
  padding: 7px 6px !important;
  line-height: 1.25 !important;
  text-align: center !important;
  vertical-align: middle !important;
  white-space: normal !important;
  word-break: normal !important;
  overflow-wrap: normal !important;
}

#table-reductions-parametres th {
  font-weight: 700 !important;
}

#table-reductions-parametres th:first-child,
#table-reductions-parametres td:first-child {
  width: 24%;
  text-align: left !important;
}

#table-reductions-parametres th:not(:first-child),
#table-reductions-parametres td:not(:first-child) {
  width: 12.6%;
}
</style>

<table>
    <thead>
      <tr style="background-color:#2d3748;">
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Modèle</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Taille du vocabulaire (original)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Taille du vocabulaire (trimmé)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">% Réduction vocabulaire</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Paramètres (original)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Paramètres (trimmé)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">% Réduction paramètres</th>
      </tr>
    </thead>
    <tbody>
      <!-- ibm-granite/granite-4.0 — bleu acier -->
      <tr style="background-color:#d9eeff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/ibm-granite/granite-4.0-350m" target="_blank" style="color:#0043a4;font-weight:500;">granite-4.0-350m</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">100 352</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 83,67%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">352,4M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">266,4M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 24,40%</td>
      </tr>
      <tr style="background-color:#d9eeff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/ibm-granite/granite-4.0-350m" target="_blank" style="color:#0043a4;font-weight:500;">granite-4.0-350m</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">100 352</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 67,35%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">352,4M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">283,2M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 19,64%</td>
      </tr>
      <tr style="background-color:#d9eeff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/ibm-granite/granite-4.0-h-350m" target="_blank" style="color:#0043a4;font-weight:500;">granite-4.0-h-350m</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">100 352</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 83,67%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">340,3M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">275,8M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 18,95%</td>
      </tr>
      <tr style="background-color:#d9eeff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/ibm-granite/granite-4.0-h-350m" target="_blank" style="color:#0043a4;font-weight:500;">granite-4.0-h-350m</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">100 352</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 67,35%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">340,3M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">288,4M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 15,25%</td>
      </tr>
      <tr style="background-color:#d9eeff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/ibm-granite/granite-4.0-1b" target="_blank" style="color:#0043a4;font-weight:500;">granite-4.0-1b</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">100 352</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 83,67%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">1,632B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">1,460B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 10,54%</td>
      </tr>
      <tr style="background-color:#d9eeff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/ibm-granite/granite-4.0-1b" target="_blank" style="color:#0043a4;font-weight:500;">granite-4.0-1b</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">100 352</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 67,35%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">1,632B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">1,493B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 8,48%</td>
      </tr>
      <tr style="background-color:#d9eeff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/ibm-granite/granite-4.0-h-1b" target="_blank" style="color:#0043a4;font-weight:500;">granite-4.0-h-1b</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">100 352</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 83,67%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">1,462B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">1,333B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 8,82%</td>
      </tr>
      <tr style="background-color:#d9eeff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/ibm-granite/granite-4.0-h-1b" target="_blank" style="color:#0043a4;font-weight:500;">granite-4.0-h-1b</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">100 352</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 67,35%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">1,462B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">1,358B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 7,10%</td>
      </tr>
      <!-- Qwen/Qwen3 — menthe -->
      <tr style="background-color:#ece8ff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/Qwen/Qwen3-0.6B" target="_blank" style="color:#4a2db5;font-weight:500;">Qwen3-0.6B</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">151 936</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 89,22%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">751,6M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">507,6M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 36,93%</td>
      </tr>
      <tr style="background-color:#ece8ff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/Qwen/Qwen3-0.6B" target="_blank" style="color:#4a2db5;font-weight:500;">Qwen3-0.6B</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">151 936</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 78,43%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">751,6M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">474,0M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 32,47%</td>
      </tr>
      <tr style="background-color:#ece8ff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/Qwen/Qwen3-1.7B" target="_blank" style="color:#4a2db5;font-weight:500;">Qwen3-1.7B</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">151 936</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 89,22%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">2,032B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">1,543B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 27,33%</td>
      </tr>
      <tr style="background-color:#ece8ff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/Qwen/Qwen3-1.7B" target="_blank" style="color:#4a2db5;font-weight:500;">Qwen3-1.7B</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">151 936</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 78,43%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">2,032B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">1,476B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 24,02%</td>
      </tr>
      <!-- google/gemma-3 (270m, 1b) — terracotta -->
      <tr style="background-color:#fde4e1;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/google/gemma-3-270m-it" target="_blank" style="color:#c5221f;font-weight:500;">gemma-3-270m-it</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">262 144</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,75%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">268,1M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">60,8M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 58,67%</td>
      </tr>
      <tr style="background-color:#fde4e1;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/google/gemma-3-270m-it" target="_blank" style="color:#c5221f;font-weight:500;">gemma-3-270m-it</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">262 144</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 87,50%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">268,1M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">77,6M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 54,76%</td>
      </tr>
      <tr style="background-color:#fde4e1;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/google/gemma-3-1b-it" target="_blank" style="color:#c5221f;font-weight:500;">gemma-3-1b-it</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">262 144</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,75%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">999,9M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">716,7M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 28,31%</td>
      </tr>
      <tr style="background-color:#fde4e1;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/google/gemma-3-1b-it" target="_blank" style="color:#c5221f;font-weight:500;">gemma-3-1b-it</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">262 144</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 87,50%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">999,9M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">735,6M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 26,43%</td>
      </tr>
      <!-- HuggingFaceTB/SmolLM3 — crème chaude -->
      <tr style="background-color:#fdebd0;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/HuggingFaceTB/SmolLM3-3B" target="_blank" style="color:#6b5a2d;font-weight:500;">SmolLM3-3B</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">128 256</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 87,23%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">3,075B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">2,845B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 7,45%</td>
      </tr>
      <tr style="background-color:#fdebd0;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/HuggingFaceTB/SmolLM3-3B" target="_blank" style="color:#6b5a2d;font-weight:500;">SmolLM3-3B</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">128 256</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 74,45%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">3,075B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">2,880B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 6,36%</td>
      </tr>
    </tbody>
  </table>
</div>

#### Résultats

##### Néerlandais

EuroEval proposant des intervalles de confiance, dans le tableau ci-dessous les moyennes en bleu indiquent que les intervalles des modèles originaux/trimmés se chevauchent. S'ils ne se chevauchent pas le meilleur modèle est indiqué en vert et le moins bon en orange.  
Nous donnons également les chiffres des exécutions effectuées par les équipes d'EuroEval quand elles existent.

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/3RXgsxxpAlHgrcpZqN5TP.png" 
       alt="Résultats sur l'échantillon néerlandais d'EuroEval" 
       style="max-width:100%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Résultats sur l'échantillon néerlandais d'EuroEval</center>
  </figcaption></center>
</figure>

Nous pouvons constater que les modèles trimmés font légèrement moins bien que les originaux mais restent dans l'intervalle de confiance à l'exception des Qwen3 où le modèle trimmé fait systématiquement moins bien.  
En analysant de plus près, cela semble dû à la tâche MMLU où nous observons une dégradation, contrairement aux autres où les modèles originaux et trimmés continuent de se chevaucher.

Nous pouvons également observer que nos exécutions semblent nettement meilleures que celles rapportées sur le *leaderboard*. Cela illustre le problème de réplication des évaluations des modèles génératifs.  


Notez que pour vérifier le Qwen3, nous avons effectué un peu de *vibechecking* pour nous faire une idée des capacités qualitatives des modèles. Nous en avons profité pour en faire avec l'ensemble des modèles. Vous pouvez déplier ci-dessous les sorties que nous obtenons avec le *prompt* *Geef me een korte introductie over grote taalmodellen.* (Donne moi une courte introduction aux modèles de langage.) :

<details>
<summary>granite-4.0-350m : Voir la comparaison original / trimmé</summary>
<table>
<tr>
  <th width="50%">Original</th>
  <th width="50%">Trimmé</th>
</tr>
<tr>
  <td width="50%" valign="top">Graag! Hier is een kort introductie over grote taalmodellen:

Grote taalmodellen zijn machine learning modelle met grote hoeveelheden data die ze kunnen leren om taal te begrijpen en te genereren. Ze zijn ontworpen door grote datasets om taalvaardigheid te verbeteren.<br><br>
  <i>Bien sûr ! Voici une brève présentation des grands modèles de langage :
  
Les grands modèles de langage sont des modèles d'apprentissage automatique qui, grâce à d'énormes quantités de données, sont capables d'apprendre à comprendre et à générer du langage. Ils sont conçus à partir de vastes ensembles de données afin d'améliorer leurs compétences linguistiques.</i></td>
  <td width="50%" valign="top">Het is een grote taalmodell, die ontworpen werd door een onderzoeker bij een onderzoeksinstelling. Het is een computerprogramma dat kan begrijpen en schrijven in een taal die door mensen kan worden geïntroduceerd. Het is gebaseerd op de principes van natuurlijke taalverwerking en machine learning. Het is een krachtig hulpmiddel dat kan helpen bij het genereren van textuele inhoud, zoals boeken, artikelen en andere documenten. Het is een belangrijk onderdeel van de toekomst van de technologie en kan een veelzijdigere manier bieden om taal te begrijpen en te genereren.<br><br>
  <i>Il s'agit d'un grand modèle de langage, conçu par un chercheur au sein d'un institut de recherche. C'est un programme informatique capable de comprendre et de produire du texte à partir d'entrées fournies par des humains. Il repose sur les principes du traitement du langage naturel et de l'apprentissage automatique. C'est un outil puissant qui peut aider à générer du contenu textuel, tel que des livres, des articles et d'autres documents. Il constitue un élément important de l'avenir de la technologie et peut offrir une approche plus polyvalente pour comprendre et générer du langage.</i></td>
</tr>
</table>
</details>

<details>
<summary>granite-4.0-h-350m : Voir la comparaison original / trimmé</summary>
<table>
<tr>
  <th width="50%">Original</th>
  <th width="50%">Trimmé</th>
</tr>
<tr>
  <td width="50%" valign="top">Grote taalmodellen, ook bekend als machine learning modellen, zijn een klasse van kunstmatige intelligentie die zich specialiseert in het proces van leren van grote hoeveelheden tekstgegevens. Deze modellen worden ontworpen om patronen, contexten en betekenis in de taal te herkennen en te begrijpen, wat leidt tot verbeterde natuurlijke taalverwerking, zoals tekstgeneratie, tekstanalyse en tekstcompressie. Ze zijn gebruikt in verschillende toepassingen, zoals chatbots, virtuele assistenten, tekstgeneratie, natuurlijke taalverwerking en zelfs in de AI zelf.<br><br>
  <i>Les grands modèles de langage, également appelés modèles d'apprentissage automatique, constituent une catégorie d'intelligence artificielle spécialisée dans l'apprentissage à partir de vastes quantités de données textuelles. Ces modèles sont conçus pour reconnaître et comprendre les schémas, les contextes et le sens du langage, ce qui permet d'améliorer le traitement du langage naturel, notamment la génération, l'analyse et la compression de textes. Ils sont utilisés dans diverses applications, telles que les chatbots, les assistants virtuels, la génération de textes, le traitement du langage naturel et même l'IA elle-même.</i></td>
  <td width="50%" valign="top">Grote taalmodellen zijn een klasse van kunstmatige intelligentie die zich specialiseert in het procesen van menselijke taal. Ze zijn ontworpen om de complexiteiten en nuances van de taal te begrijpen en te genereren, waardoor ze in staat zijn om menselijke taal te verwerken en te interpreteren. Deze modellen worden vaak gebruikt in verschillende toepassingen, zoals chatbots, virtuele assistenten, tekstgeneratie, natuurlijke taalverwerking en meer.<br><br>
  <i>Les grands modèles de langage sont une catégorie d'intelligence artificielle spécialisée dans le traitement du langage humain. Ils sont conçus pour comprendre et reproduire les complexités et les nuances du langage, ce qui leur permet de traiter et d'interpréter le langage humain. Ces modèles sont souvent utilisés dans diverses applications, telles que les chatbots, les assistants virtuels, la génération de texte, le traitement du langage naturel, etc.</i></td>
</tr>
</table>
</details>

<details>
<summary>granite-4.0-1b : Voir la comparaison original / trimmé</summary>
<table>
<tr>
  <th width="50%">Original</th>
  <th width="50%">Trimmé</th>
</tr>
<tr>
  <td width="50%" valign="top">Grote taalmodellen, ook wel bekend als GPT (Generative Pre-trained Transformer) modellen, zijn een type van machine learning model dat is ontworpen om menselijke taal te begrijpen en te genereren. Deze modellen worden getraind op enorme hoeveelheden tekstgegevens uit verschillende bronnen, waardoor ze een breed scala aan taalstijlen, onderwerpen en contexten kunnen begrijpen en reproduceren.
 
De belangrijkste kenmerken van grote taalmodellen zijn:
 
1. **Grootte**: Deze modellen bestaan uit miljoenen van parameters, waardoor ze zeer geavanceerde taalverwerking kunnen uitvoeren. Hun grootte maakt het mogelijk om complexe taalpatronen en contexten te begrijpen en te genereren.

2. **Pre-training**: Vooraf getraind op grote hoeveelheden tekstgegevens, kunnen deze modellen een breed scala aan informatie en taalstijlen leren. Dit maakt ze zeer flexibel en aanpasbaar aan verschillende taakken.
 
3. **Generatie**: Met behulp van hun getrainde kennis, kunnen grote taalmodellen tekst genereren op basis van de invoer die ze ontvangen. Dit maakt ze nuttig voor een breed scala aan toepassingen, zoals tekstgeneratie, vertaling, antwoorden op vragen en meer.
 
4. **Contextbegrip**: Deze modellen zijn uitstekend in het begrijpen van context en het genereren van geschikte en relevante antwoorden of tekst op basis van de gegeven context.
 
Grote taalmodellen hebben een significante invloed gehad op de manier waarop we interageren met technologie en hebben het potentieel om de manier waarop we communiceren en informatie consumeren te revolutioneren. Ze worden gebruikt in verschillende toepassingen, van chatbots en virtuele assistenten tot contentgeneratie en taalvertaling.<br><br>
  <i>Les grands modèles de langage, également connus sous le nom de modèles GPT (Generative Pre-trained Transformer), constituent un type de modèle d'apprentissage automatique conçu pour comprendre et générer le langage humain. Ces modèles sont entraînés sur d'énormes quantités de données textuelles provenant de diverses sources, ce qui leur permet de comprendre et de reproduire un large éventail de styles linguistiques, de sujets et de contextes.
 
Les principales caractéristiques des grands modèles de langage sont les suivantes :

1. **Taille** : Ces modèles comportent des millions de paramètres, ce qui leur permet d'effectuer des traitements linguistiques très sophistiqués. Leur taille leur permet de comprendre et de générer des schémas linguistiques et des contextes complexes.

2. **Pré-entraînement** : Pré-entraînés sur de vastes quantités de données textuelles, ces modèles peuvent assimiler un large éventail d'informations et de styles linguistiques. Cela les rend très flexibles et adaptables à différentes tâches.
 
3. **Génération** : Grâce à leurs connaissances acquises lors de l'entraînement, les grands modèles de langage peuvent générer du texte à partir des données d'entrée qu'ils reçoivent. Cela les rend utiles pour un large éventail d'applications, telles que la génération de texte, la traduction, la réponse à des questions et bien plus encore.
 
4. **Compréhension du contexte** : ces modèles excellent dans la compréhension du contexte et la génération de réponses ou de textes appropriés et pertinents en fonction du contexte donné.
 
Les grands modèles de langage ont eu un impact significatif sur la manière dont nous interagissons avec la technologie et ont le potentiel de révolutionner la façon dont nous communiquons et consommons l'information. Ils sont utilisés dans diverses applications, des chatbots et assistants virtuels à la génération de contenu et à la traduction.</i></td>
  <td width="50%" valign="top">Grote taalmodellen zijn een type kunstmatige intelligentie die is ontworpen om menselijke taal te begrijpen en te genereren. Ze zijn ontwikkeld met behulp van machine learning technieken, met name die gebaseerd op diepe neurale netwerken. Deze modellen zijn getraind op enorme hoeveelheden tekstdata, waardoor ze een breed scala aan woorden, grammatica en context kunnen begrijpen en gebruiken. Ze kunnen vragen beantwoorden, verhalen genereren, tekst corrigeren en zelfs helpen bij het schrijven van artikelen of het begrijpen van complexe concepten.<br><br>
  <i>Les grands modèles de langage sont un type d'intelligence artificielle conçu pour comprendre et générer le langage humain. Ils ont été développés à l'aide de techniques d'apprentissage automatique, notamment celles basées sur les réseaux de neurones profonds. Ces modèles ont été entraînés sur d'énormes quantités de données textuelles, ce qui leur permet de comprendre et d'utiliser un large éventail de mots, de structures grammaticales et de contextes. Ils peuvent répondre à des questions, générer des récits, corriger des textes et même aider à rédiger des articles ou à comprendre des concepts complexes.</i></td>
</tr>
</table>
</details>

<details>
<summary>granite-4.0-h-1b : Voir la comparaison original / trimmé</summary>
<table>
<tr>
  <th width="50%">Original</th>
  <th width="50%">Trimmé</th>
</tr>
<tr>
  <td width="50%" valign="top">Grote taalmodellen, ook wel bekend als transformatormodellen, zijn een soort kunstmatige intelligentie die is ontwikkeld om menselijke taal te begrijpen, genereren en te manipuleren. Deze modellen zijn gebaseerd op een type neurale netwerk dat in staat is om grote hoeveelheden tekstgegevens te verwerken en patronen te herkennen. Ze zijn in staat om complexe taalbehoudingen te begrijpen, zoals context, betekenis en syntaxis, en kunnen daarmee een breed scala aan taakken uitvoeren, zoals tekstgeneratie, vertaling, antwoordgeven op vragen en taalvertaling. Grote taalmodellen zijn steeds meer geavanceerd en worden steeds vaker gebruikt in verschillende toepassingen, zoals chatbots, virtuele assistenten, contentgeneratie en taalonderwijs.<br><br>
  <i>Les grands modèles de langage, également appelés modèles de transformateurs, constituent un type d'intelligence artificielle conçu pour comprendre, générer et manipuler le langage humain. Ces modèles s'appuient sur un type de réseau neuronal capable de traiter de grandes quantités de données textuelles et d'identifier des schémas récurrents. Ils sont capables de comprendre des aspects linguistiques complexes, tels que le contexte, le sens et la syntaxe, et peuvent ainsi effectuer un large éventail de tâches, telles que la génération de texte, la traduction, la réponse à des questions et la traduction linguistique. Les grands modèles de langage sont de plus en plus sophistiqués et sont de plus en plus utilisés dans diverses applications, telles que les chatbots, les assistants virtuels, la génération de contenu et l'enseignement des langues.</i></td>
  <td width="50%" valign="top">Grote taalmodellen zijn geavanceerde kunstmatige intelligentie systemen die zijn ontworpen om menselijke taal te begrijpen en te genereren. Ze zijn gebaseerd op diepgaande leermodellen, zoals die gebruikt worden in transformator-architecturen, die in staat zijn om patronen in grote hoeveelheden tekstgegevens te leren. Deze modellen kunnen een breed scala aan taakken uitvoeren, zoals tekstgeneratie, vertaling, antwoorden op vragen en antwoordgeneratie. Ze zijn in staat om complexe contexten te begrijpen en passende antwoorden te genereren op basis van de gegeven context. Grote taalmodellen hebben een groot aantal parameters, vaak in de miljoenen tot miljarden bereik, wat hen in staat stelt om een zeer grote hoeveelheid informatie te verwerken en te genereren. Ze zijn steeds meer geavanceerder en worden steeds vaker gebruikt in verschillende toepassingen, zoals chatbots, virtuele assistenten en contentgeneratie.<br><br>
  <i>Les grands modèles de langage sont des systèmes d'intelligence artificielle avancés conçus pour comprendre et générer le langage humain. Ils s'appuient sur des modèles d'apprentissage profond, tels que ceux utilisés dans les architectures de transformateurs, capables d'apprendre des schémas récurrents à partir de vastes quantités de données textuelles. Ces modèles peuvent accomplir un large éventail de tâches, telles que la génération de texte, la traduction, la réponse à des questions et la génération de réponses. Ils sont capables de comprendre des contextes complexes et de générer des réponses appropriées en fonction du contexte donné. Les grands modèles de langage ont un grand nombre de paramètres, souvent de l'ordre de millions à milliards, ce qui leur permet de traiter et de générer une très grande quantité d'informations. Ils sont de plus en plus sophistiqués et sont de plus en plus utilisés dans diverses applications, telles que les chatbots, les assistants virtuels et la génération de contenu.</i></td>
</tr>
</table>
</details>

<details>
<summary>Qwen3-0.6B : Voir la comparaison original / trimmé</summary>
<table>
<tr>
  <th width="50%">Original</th>
  <th width="50%">Trimmé</th>
</tr>
<tr>
  <td width="50%" valign="top">Grote taalmodellen zijn grote, complexe modelle die grote hoeveelheden tekst aanpassen, zoals in de Nederlandse taal. Ze zijn ontwikkeld met de behulp van grote datasetten, zoals millions of woorden, en worden geschikt voor verschillende taalgebruiktes, zoals het antwoorden van vragen of het schrijven van teksten. Ze kunnen ook complexe taalfuncties uitvoeren, zoals het onderluiden van stijl of het ontwerpen van stories.<br><br>
  <i>Les grands modèles de langage sont des modèles volumineux et complexes capables de traiter de grandes quantités de texte, notamment en néerlandais. Ils sont développés à partir d'énormes bases de données, comprenant des millions de mots, et sont adaptés à diverses applications linguistiques, telles que la réponse à des questions ou la rédaction de textes. Ils peuvent également effectuer des tâches linguistiques complexes, comme l'adaptation du style ou la création d'histoires.</i></td>
  <td width="50%" valign="top">**Intro over grote taalmodellen** – grote taalmodellen zijn AI-technologieën die kunnen leren te weten te spelen met taal, zowel in tekst als in andere medium. Ze zijn ontwikkeld met grote gegevens en kunnen kwalitatief en kwalificeel onderzoek, samen met het verwerken van tekst, uitvoeren. Deze modellen zijn gebruikt in verschillende sectoren om interactieve en luidere communicatie te versterken, samen met hun capaciteit om complexe taalgebruik te ondersteunen.<br><br>
  <i>**Introduction aux grands modèles de langage** – Les grands modèles de langage sont des technologies d'IA capables d'apprendre à manipuler le langage, tant dans le texte que dans d'autres supports. Développés à partir de vastes ensembles de données, ils permettent d'effectuer des analyses qualitatives et quantitatives, ainsi que le traitement de texte. Ces modèles sont utilisés dans divers secteurs pour améliorer la communication interactive et plus fluide, grâce à leur capacité à prendre en charge un langage complexe.</i></td>
</tr>
</table>
</details>

<details>
<summary>Qwen3-1.7B : Voir la comparaison original / trimmé</summary>
<table>
<tr>
  <th width="50%">Original</th>
  <th width="50%">Trimmé</th>
</tr>
<tr>
  <td width="50%" valign="top">Grote taalmodellen zijn machine learning-roboten die op basis van veel tekstgegevens worden gemaakt. Ze kunnen vaak honderden van duizenden boeken, artikelen of internetinformatie oefenen. Deze modellen gebruiken een specifieke architectuur, zoals de transformer, om lange teksten en context te begrijpen. Ze worden gebruikt voor taalverwijzing, coderen, en inhoudsgeneratie. Hoewel ze veel potentieel bieden, hebben ze ook problemen met bias, ethieken, en het gebruik van rijke resourcebenodigingen.<br><br>
  <i>Les grands modèles de langage sont des robots d'apprentissage automatique créés à partir d'une grande quantité de données textuelles. Ils peuvent souvent être entraînés sur des centaines de milliers de livres, d'articles ou d'informations provenant d'Internet. Ces modèles utilisent une architecture spécifique, telle que le transformer, pour comprendre les textes longs et le contexte. Ils sont utilisés pour la recherche linguistique, le codage et la génération de contenu. Bien qu'ils offrent un grand potentiel, ils posent également des problèmes liés aux biais, à l'éthique et à l'utilisation de ressources importantes.</i></td>
  <td width="50%" valign="top">Grote taalmodellen (zoals GPT, BERT, en andere modellen) zijn AI-modellen met een grote hoeveelheid parameters (parameters zijn de hoeveelheid informatie die het model leert). Ze worden geïntraineerd op miljarden van de tekstopzoeken om te leren zinnetjes te genereren, te begrijpen en te converseren. Deze modellen zijn van toepassing op veel NLP-taken, zoals het genereren van tekst, het analyseren van taal, en het beantigen van vragen. Zij zijn een belangrijk stap voorwaarts in de AI-ontwikkeling, omdat ze de mogelijkheid bieden om complexe taalproblemen op te lossen en gebruikers ervaring te geven die duidelijk en relevant is.<br><br>
  <i>Les grands modèles de langage (tels que GPT, BERT et d'autres modèles) sont des modèles d'IA dotés d'un nombre considérable de paramètres (les paramètres correspondent à la quantité d'informations que le modèle apprend). Ils sont entraînés sur des milliards de requêtes textuelles afin d'apprendre à générer des phrases, à comprendre et à converser. Ces modèles s'appliquent à de nombreuses tâches de traitement du langage naturel (NLP), telles que la génération de texte, l'analyse linguistique et la réponse aux questions. Ils constituent une avancée majeure dans le développement de l'IA, car ils permettent de résoudre des problèmes linguistiques complexes et d'offrir aux utilisateurs une expérience claire et pertinente.</i></td>
</table>
</details>


<details>
<summary>gemma-3-270m-it : Voir la comparaison original / trimmé</summary>
<table>
<tr>
  <th width="50%">Original</th>
  <th width="50%">Trimmé</th>
</tr>
<tr>
  <td width="50%" valign="top">Een grote taalmodel (LLM) is een objectief model dat de taal kan begrijpen en genereren. Het is ontworpen om een brede kennis van de wereld te behouden en te verwerken, wat resulteert in een betere en meer intelligentere menselijke interactie met de taal.<br><br>
  <i>Un grand modèle de langage (LLM) est un modèle capable de comprendre et de générer du langage. Il est conçu pour stocker et traiter une vaste connaissance du monde, ce qui se traduit par une interaction humaine avec le langage à la fois plus efficace et plus intelligente.</i></td>
  <td width="50%" valign="top">Zeker! Hier is een korte introductie over grote taalmodellen:
 
Grote taalmodellen (LLM's) zijn een vorm van machine learning die toegang heeft tot enorme hoeveelheden tekstdata. Ze kunnen een breed scala aan taken uitvoeren, zoals tekstgeneratie, verwerking, vraag-antwoord en zelfs tekst-analyse. Deze modellen worden vaak gebruikt voor verschillende toepassingen, van spraakherkenning tot het genereren van content.<br><br>
  <i>Bien sûr ! Voici une brèveintroduction aux grands modèles de langage :
 
Les grands modèles de langage (LLM) constituent une forme d'apprentissage automatique qui exploite d'énormes quantités de données textuelles. Ils sont capables d'effectuer un large éventail de tâches, telles que la génération de texte, le traitement de texte, les systèmes de questions-réponses et même l'analyse de texte. Ces modèles sont souvent utilisés pour diverses applications, allant de la reconnaissance vocale à la génération de contenu.</i></td>
</tr>
</table>
</details>

<details>
<summary>gemma-3-1b-it : Voir la comparaison original / trimmé</summary>
<table>
<tr>
  <th width="50%">Original</th>
  <th width="50%">Trimmé</th>
</tr>
<tr>
  <td width="50%" valign="top">Oké, hier is een korte introductie over grote taalmodellen (LLM's):
 
*Grote taalmodellen (LLM's) zijn een nieuw soort AI-technologie die zich steeds meer richten op het begrijpen en genereren van menselijke taal.* Het zijn complexe computersystemen die zijn getraind op enorme hoeveelheden tekst en code, waardoor ze in staat zijn om:
 
*  *Tekst te begrijpen:* Ze kunnen complexe vragen beantwoorden, samenvattingen maken en de betekenis van tekst analyseren.
*  *Tekst te genereren:* Ze kunnen nieuwsartikelen, gedichten, code, scripts, muziekstukken, e-mails, brieven, enzovoort, op basis van een prompt.
*  *Tekst te vertalen:* Ze kunnen teksten van de ene taal naar de andere vertalen.
*  *Chatten:* Ze kunnen interactieve gesprekken voeren.
 
*Denk aan ze als enorme 'boeken' die geleerd hebben hoe taal te gebruiken.* Ze zijn niet echt 'denkend' zoals mensen, maar ze simuleren intelligente redenering op basis van de patronen die ze in hun training hebben opgevangen.
 
*Voorbeelden van populaire LLM's zijn:* GPT-3, LaMDA, Bard, en Llama 2.

*Wil je dat ik je verder uitleg, of specifiekere informatie geeft over een bepaald aspect van LLM's?* Bijvoorbeeld, wil je weten hoe ze werken, hoe ze worden gebruikt, of wat hun beperkingen zijn?<br><br>
  <i>Bon, voici une brève introduction aux grands modèles de langage (LLM) :
 
*Les grands modèles de langage (LLM) constituent un nouveau type de technologie d'IA qui se concentre de plus en plus sur la compréhension et la génération du langage humain.* Il s'agit de systèmes informatiques complexes qui ont été entraînés sur d'énormes quantités de texte et de code, ce qui leur permet de :
 
*  *Comprendre du texte :* ils peuvent répondre à des questions complexes, rédiger des résumés et analyser le sens d'un texte.
*  *Générer du texte :* ils peuvent créer des articles d'actualité, des poèmes, du code, des scripts, des morceaux de musique, des e-mails, des lettres, etc., à partir d'une consigne.
*  *Traduire du texte :* Ils peuvent traduire des textes d'une langue à une autre.
*  *Discuter :* Ils peuvent mener des conversations interactives.
 
*Considérez-les comme d'énormes « livres » qui ont appris à utiliser le langage.* Ils ne « pensent » pas vraiment comme les humains, mais ils simulent un raisonnement intelligent en se basant sur les schémas qu'ils ont captés lors de leur entraînement.
 
*Parmi les LLM populaires, on peut citer :* GPT-3, LaMDA, Bard et Llama 2.

*Souhaitez-vous que je vous donne plus d'explications ou des informations plus précises sur un aspect particulier des LLM ?* Par exemple, souhaitez-vous savoir comment ils fonctionnent, comment ils sont utilisés ou quelles sont leurs limites ?</i></td>
  <td width="50%" valign="top">Oké, hier is een korte introductie over grote taalmodellen (LLM's):
 
*Grote Taalmodellen (LLM's) zijn een nieuwe klasse van AI-modellen die op enorme hoeveelheden tekstgegevens zijn getraind. Ze kunnen bijvoorbeeld vragen beantwoorden, teksten genereren, vertalen, samenvattingen maken en zelfs code schrijven!*
 
*Hier zijn de belangrijkste punten om het te begrijpen, zonder te ingewikkeld te worden (maar het is toch belangrijk):*
 
* *Getraind op enorme datasets van tekst en code van het internet, boeken en meer…* Ze eten heel veel tekst!
* *Gebruiken "neurale netwerken" (een soort intelligentie-circuit) om patronen in de data te leren en te voorspellen wat er volgend zal komen in een tekst.*
* *Ze kunnen "attenderen" op specifieke delen van een tekst om te beantwoorden, of te genereren, en te reageren op je input. Dit noemen we "context begrijpen".*
 
*Voorbeelden van bekende LLM's zijn Google's LaMDA, Meta's LLaMA, en de modellen van Microsoft (bijv. Copype).*
 
*Wil je dat ik je iets specifieks over LLM's uitleg? Misschien wil je weten hoe ze werken, of welke toepassingen ze hebben?*.<br><br>
  <i>Ok, voici une brève introduction aux grands modèles de langage (LLM) :
 
*Les grands modèles de langage (LLM) constituent une nouvelle catégorie de modèles d'IA entraînés sur d'énormes quantités de données textuelles. Ils peuvent, par exemple, répondre à des questions, générer des textes, traduire, rédiger des résumés et même écrire du code !*
 
*Voici les points essentiels pour comprendre ce concept, sans entrer dans les détails (mais c'est tout de même important) :*
 
* *Entraînés sur d'énormes ensembles de données textuelles et de code provenant d'Internet, de livres et bien plus encore…* Ils avalent des tonnes de texte !
* *Ils utilisent des « réseaux neuronaux » (une sorte de circuit intelligent) pour apprendre des modèles dans les données et prédire ce qui va suivre dans un texte.*
* *Ils peuvent « se concentrer » sur des parties spécifiques d’un texte pour répondre, générer du contenu ou réagir à votre saisie. C’est ce qu’on appelle « comprendre le contexte ».*
 
*Parmi les LLM connus, on peut citer LaMDA de Google, LLaMA de Meta et les modèles de Microsoft (par exemple Copype).*
 
*Souhaitez-vous que je vous explique quelque chose de précis sur les LLM ? Peut-être souhaitez-vous savoir comment ils fonctionnent ou quelles sont leurs applications ?*</i></td>
</tr>
</table>
</details>

Sur la forme, les textes issus des modèles trimmés ne sont jamais sortis du sujet et sont donc semblables aux originaux.  
Sur le fond, que ce soit les modèles trimmés ou originaux, les sorties générées ont tendance à traduire des termes techniques en lien avec les LLM/l'IA de l'anglais vers le néerlandais mais le font mal. Nous avons aussi observé que les modèles ont du mal avec le pluriel de certains mots. Concrètement, nos évaluateurs indiquent que les textes originaux ou trimmés ressemblent à des rédactions d'enfants pouvant parfois ici ou là utiliser quelques termes sophistiqués.
Sur ce point du fond, notez que les modèles montrés ici servent à illustrer le principe du *trimming*. En pratique, ces différents modèles datent un peu à présent et ne sont sûrement pas les plus utiles. Au moment où nous écrivons ces lignes, le granite 4.1, le gemma 4 et le qwen 3.6 viennent de sortir et seront probablement meilleurs (dans la section suivante nous pouvons déjà voir que le 3.5 est meilleur que le 3).

<div style="line-height:1.25; background:#241216; color:#e5e7eb; padding:1rem; border-radius:4px;">
  <p style="font-size:0.9rem;"><strong>Vigilance ⚠️</strong></p>

  <p style="font-size:0.9rem;">
    Nos expériences de <em>vibechecking</em> ont également porté sur les traces et les <em>prompts</em> systèmes.<br>
    Nous observons que lorsqu'un modèle passe sous la barre des 300M de paramètres une fois trimmé, il a tendance à répondre en anglais par défaut (le Gemma 3 270M non trimmé a ainsi également ce comportement) ou bien pas dans la bonne langue. Il faut alors spécifier dans le <em>prompt</em> système la langue qu'il doit utiliser pour forcer la génération dans celle-ci. Là où pour les modèles plus grands que 300M de paramètres, il n'est pas nécessaire de modifier leur <em>prompt</em> original.<br>
    Concernant les traces, pour les modèles trimmés pour lesquels il est possible d'activer / désactiver le mode <em>thinking</em>, à savoir les Qwen 3 et le SmolLM3, les comportements sont variables. Le SmolLM3 générera systématiquement sa trace en anglais. Pour le Qwen 3, nous avons un comportement variable. Le modèle peut ne plus en générer, en générer en anglais ou bien en générer dans la langue trimmée. Ce dernier point nous semblait intéressant pour pouvoir par exemple générer directement des traces dans une langue d'intérêt (au lieu de devoir en générer en anglais puis les traduire). Cependant ce comportement ne semble pas se produire sur toutes les langues. Nous pouvons l'observer par exemple sur du <a href="https://huggingface.co/alphaedge-ai/Qwen3-1.7B-fra-32768/blob/main/Qwen3_French_trace.ipynb" style="color:#fca5a5;">français</a> qui est proche de l'anglais mais nous n'avons pas réussi à observer ce phénomène sur des alphabets non latins testés comme le coréen, l'arabe ou le tamil. Cela semble aussi dépendre de la difficulté de la tâche.
  </p>
</div>

<br>


##### Coréen

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/IWAQCcevTDilvp7DpaPY8.png" 
       alt="Résultats sur le benchmark KoBEST" 
       style="max-width:100%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Résultats sur le benchmark KoBEST</center>
  </figcaption></center>
</figure>

Pour le coréen, nous observons que nous obtenons des résultats similaires aux modèles originaux à une exception : le granite 4.0-h où nous avons une dégradation de 3 points (ce que nous n'avons pas sur le néerlandais). Nous pouvons voir que c'est dû au jeu de données `boolq` sur lequel nous perdons 12 points. En analysant les résultats, il s'avère que le modèle trimmé s'effondre sur cette tâche sans que nous réussissions à modifier ce comportement (via le *prompt*, un changement des labels à trouver, ou encore de nombre de *shots*).  
De toutes nos expériences (environ 90 modèles trimmés testés), c'est le seul modèle où nous observons une dégradation non attendue. C'est donc un cas isolé mais pouvant illustrer un cas limite du *trimming*.

<div class="notice--success" style="line-height:1.25;">
  <p style="font-size:0.9rem;"><strong>🧠 À retenir</strong></p>
  <p style="font-size:0.9rem;">
    Le <em>trimming</em> fonctionne également sur les décodeurs, mais nous invitons les utilisateurs à être vigilants vis-à-vis de potentiels cas limites rares.
  </p>
</div>


#### Collections

<div style="line-height:1.25; background:#2e1f47; color:#f5f3ff; padding:1rem; border-radius:4px;">
  <p style="font-size:0.9rem;"><strong>🎁 Important</strong></p>
  <p style="font-size:0.9rem;">
    Une collection rassemblant <strong>96</strong> modèles (<strong>12</strong> langues différentes) trimmés à partir des granite 4.0 est disponible <a href="https://huggingface.co/collections/alphaedge-ai/trimming-granite-40" style="color:#c4b5fd;">ici</a>.<br>
    Une collection rassemblant <strong>332</strong> modèles (<strong>83</strong> langues différentes) trimmés à partir des Qwen 3 est disponible <a href="https://huggingface.co/collections/alphaedge-ai/trimming-qwen3" style="color:#c4b5fd;">ici</a>.<br>
    Une collection rassemblant <strong>486</strong> modèles (<strong>122</strong> langues différentes) trimmés à partir des Gemma 3 est disponible <a href="https://huggingface.co/collections/alphaedge-ai/trimming-gemma-3" style="color:#c4b5fd;">ici</a>.<br>
    Une collection rassemblant <strong>16</strong> modèles (<strong>8</strong> langues différentes) trimmés à partir du smolLM3 est disponible <a href="https://huggingface.co/collections/alphaedge-ai/trimming-smollm3" style="color:#c4b5fd;">ici</a>.
  </p>
</div>

<br>

### Modèles multimodaux (VLM)

Observons à présent l'impact du *trimming* sur les performances des VLM.  

#### Modèles et benchmarks sélectionnés 

Nous nous sommes dirigés sur deux VLM pour cette section :
- Les [Qwen 3.5](https://huggingface.co/collections/Qwen/qwen35) de Qwen Team (2026) en taille `0.8B`, `2B` et `4B` ;
- Le [Gemma 3 4B](https://huggingface.co/google/gemma-3-4b-it) de Google DeepMind (2025).

Ne disposant pas de *benchmarks* pour évaluer ces modèles dans d'autres langues que l'anglais, nous nous sommes limités dans cette section à du *vibechecking* en soumettant des images au modèle et en lui demandant de nous les décrire.  
Pour une question de place, nous exposons un exemple pour le français et un pour l'espagnol. 


#### Réductions de paramètres observées

<div id="table-reductions-parametres" style="overflow-x:auto; margin:1.5rem 0; max-width:100%;">
<style>
#table-reductions-parametres table {
  border-collapse: collapse;
  width: 100%;
  table-layout: fixed;
  font-size: 0.72rem !important;
  box-shadow: 0 2px 7px rgba(0,0,0,0.12);
  border-radius: 8px;
  overflow: hidden;
}

#table-reductions-parametres th,
#table-reductions-parametres td {
  padding: 7px 6px !important;
  line-height: 1.25 !important;
  text-align: center !important;
  vertical-align: middle !important;
  white-space: normal !important;
  word-break: normal !important;
  overflow-wrap: normal !important;
}

#table-reductions-parametres th {
  font-weight: 700 !important;
}

#table-reductions-parametres th:first-child,
#table-reductions-parametres td:first-child {
  width: 24%;
  text-align: left !important;
}

#table-reductions-parametres th:not(:first-child),
#table-reductions-parametres td:not(:first-child) {
  width: 12.6%;
}
</style>

<table>
    <thead>
      <tr style="background-color:#2d3748;">
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Modèle</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Taille du vocabulaire (original)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Taille du vocabulaire (trimmé)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">% Réduction vocabulaire</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Paramètres (original)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Paramètres (trimmé)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">% Réduction paramètres</th>
      </tr>
    </thead>
    <tbody>
      <!-- Qwen/Qwen3.5 — bleu azur -->
      <tr style="background-color:#ece8ff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/Qwen/Qwen3.5-0.8B" target="_blank" style="color:#4a2db5;font-weight:500;">Qwen3.5-0.8B</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">248 320</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,40%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">853,0M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">615,5M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 27,84%</td>
      </tr>
      <tr style="background-color:#ece8ff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/Qwen/Qwen3.5-0.8B" target="_blank" style="color:#4a2db5;font-weight:500;">Qwen3.5-0.8B</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">248 320</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 86,80%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">853,0M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">632,3M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 25,88%</td>
      </tr>
      <tr style="background-color:#ece8ff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/Qwen/Qwen3.5-2B" target="_blank" style="color:#4a2db5;font-weight:500;">Qwen3.5-2B</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">248 320</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,40%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">2,213B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">1,738B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 21,46%</td>
      </tr>
      <tr style="background-color:#ece8ff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/Qwen/Qwen3.5-2B" target="_blank" style="color:#4a2db5;font-weight:500;">Qwen3.5-2B</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">248 320</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 86,80%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">2,213B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">1,772B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 19,95%</td>
      </tr>
      <tr style="background-color:#ece8ff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/Qwen/Qwen3.5-4B" target="_blank" style="color:#4a2db5;font-weight:500;">Qwen3.5-4B</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">248 320</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,40%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">4,539B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">3,946B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 13,08%</td>
      </tr>
      <tr style="background-color:#ece8ff;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/Qwen/Qwen3.5-4B" target="_blank" style="color:#4a2db5;font-weight:500;">Qwen3.5-4B</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">248 320</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 86,80%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">4,539B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">3,987B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 12,16%</td>
      </tr>
      <!-- google/gemma-3 — pêche -->
      <tr style="background-color:#fde4e1;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/google/gemma-3-4b-it" target="_blank" style="color:#c5221f;font-weight:500;">gemma-3-4b-it</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">262 208</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 93,75%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">4,300B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">3,671B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 14,63%</td>
      </tr>
      <tr style="background-color:#fde4e1;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/google/gemma-3-4b-it" target="_blank" style="color:#c5221f;font-weight:500;">gemma-3-4b-it</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">262 208</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 87,50%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">4,300B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">3,713B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 13,66%</td>
      </tr>
    </tbody>
  </table>
</div>

#### Résultats

##### Français

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/Uj45gAfddTqTi0hkGgats.png" 
       alt="Photo par Merve Noyan" 
       style="max-width:70%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Photo par Merve NOYAN</center>
  </figcaption></center>
</figure>


Prompt utilisé pour tous les modèles : « Décris cette image en détail. »

<details>
<summary>Qwen3.5 0.8B : Voir la comparaison original / trimmé</summary>
<table>
<tr>
  <th width="50%">Original</th>
  <th width="50%">Trimmé</th>
</tr>
<tr>
  <td width="50%" valign="top">Cette image montre une collection de fleurs de cosmos en rose, avec une bumblebee au centre de la composition. La bumblebee, avec ses taches noires et blanches, est assise sur la fleur principale, qui est l'objet central de l'image. Les autres fleurs de cosmos, de différentes couleurs (rose, rouge, blanche), sont dispersées autour de la fleur principale, certaines étant en fleurs et d'autres en fleurs mortes. Le fond est composé de feuilles vertes et de fleurs de cosmos en fleurs mortes, créant un arrière-plan naturel et coloré.</td>
  <td width="50%" valign="top">L'image montre une scène de nature avec des fleurs de cosmos en violet pâle, une bumblebee au centre de la composition, et d'autres fleurs de cosmos et de gerberas en arrière-plan. La bumblebee, avec ses taches noires et blanches, est assise sur la fleur principale, probablement en quête de nourriture. Les autres fleurs, certaines encore en fleurs et d'autres déjà séchées, ajoutent de la couleur et de la texture à l'arrière-plan. Le fond est rempli de feuilles vertes et de branches, créant un cadre naturel pour cette image de fleurs et d'insectes.</td>
</tr>
</table>
</details>

<details>
<summary>Qwen3.5 2B : Voir la comparaison original / trimmé</summary>
<table>
<tr>
  <th width="50%">Original</th>
  <th width="50%">Trimmé</th>
</tr>
<tr>
  <td width="50%" valign="top">Cette image capture une scène de jardin florale vibrante, centrée sur une grande fleur rose pâle en pleine floraison. Une abeille noire et jaune est visible au centre de la fleur, collectant du pollen. Autour d'elle, d'autres fleurs colorées — rouges, violettes et blanches — s'ouvrent également, tandis que des tiges et des feuilles vertes forment un arrière-plan naturel et luxuriant. L'ensemble de l'image évoque la vie et la beauté du jardin, avec une composition équilibrée et une ambiance apaisante.</td>
  <td width="50%" valign="top">Cette image capture une scène de jardin en pleine vie, centrée sur une fleur de cosmos rose vif qui attire l’attention. Au centre, une abeille noire et jaune est en train de collecter du pollen sur le centre de la fleur, ses ailes translucides légèrement déployées. La fleur principale, avec ses pétales larges et douces, contraste avec d’autres fleurs autour : des roses plus pâles, des fleurs rouges éclatantes en arrière-plan, et des tiges de plantes séchées ou en fleurs blanches, suggérant une saison printanière ou estivale. Le fond est flou, mettant en valeur la profondeur de champ et la netteté du sujet. Des feuilles vertes et des tiges fines complètent le cadre naturel, créant une ambiance apaisante et vivante.</td>
</tr>
</table>
</details>

<details>
<summary>Qwen3.5 4B : Voir la comparaison original / trimmé</summary>
<table>
<tr>
  <th width="50%">Original</th>
  <th width="50%">Trimmé</th>
</tr>
<tr>
  <td width="50%" valign="top">Cette image montre une fleur rose vif, probablement un cosmos, au centre de l’encadrement, avec un bourdon noir et blanc en train de butiner son pollen jaune. Autour d’elle, d’autres fleurs roses sont visibles — certaines encore fraîches, d’autres déjà fanées ou séchées, montrant des tiges brunes et des pétales effilochés. En bas à droite, une petite fleur rouge vive contraste avec les tons roses dominants. Le fond est flou, mais on devine des feuilles vertes et d’autres éléments floraux, créant une ambiance naturelle et paisible. L’ensemble évoque un jardin en pleine saison, où la vie et la décomposition coexistent harmonieusement.</td>
  <td width="50%" valign="top">Cette image présente une fleur rose vif, probablement un cosmos, au centre de l’encadrement, avec un grand bourdon noir et blanc en train de butiner son pollen jaune. Autour d’elle, d’autres fleurs roses sont visibles — certaines encore fraîches, d’autres déjà fanées et séchées, montrant le cycle naturel de la vie végétale. En bas à droite, une petite fleur rouge écarlate ajoute un contraste vibrant, tandis que des feuilles vertes floues forment un fond doux qui met en valeur les sujets principaux. La composition évoque une scène paisible dans un jardin ou un massif floral, où la nature se déploie avec harmonie et simplicité.</td>
</tr>
</table>
</details>

<details>
<summary>Qwen3.5 4B (<i>thinking</i>) : Voir la comparaison original / trimmé</summary>
<table>
<tr>
  <th width="50%">Original</th>
  <th width="50%">Trimmé</th>
</tr>
<tr>
  <td width="50%" valign="top">
&lt;think&gt;<br>
The user wants a detailed description of the image in French.

**Image Analysis:**
- **Main Subject:** A large, light pink cosmos flower (Cosmos bipinnatus) in the center.
- **Action:** A bumblebee is on the center of the flower, collecting pollen.
- **Surroundings:**
    - To the left: Another pink cosmos flower, slightly out of focus.
    - Background: Green foliage, some dried brown flower heads (seed pods).
    - Bottom right: A small red geranium flower.
    - Top right: Some wilting pink flowers.
- **Lighting/Color:** Natural daylight, soft focus background (bokeh), vibrant pinks and greens.

**Drafting the description (internal monologue/rough notes):**
- Une grande fleur rose, un cosmos, est au centre.
- Une abeille bourdonne dessus, elle est noire et blanche.
- Il y a d'autres fleurs roses autour, mais floues.
- En bas à droite, une petite fleur rouge.
- Il y a des tiges sèches et des feuilles vertes en arrière-plan.

**Refining into French:**
1.  Au centre de l'image, une grande fleur cosmos rose pâle s'épanouit avec élégance.
2.  Une abeille bourdonne activement au cœur de cette fleur, recueillant le pollen jaune.
3.  À gauche, une autre fleur rose est visible, bien que légèrement floue, tandis que des tiges séchées et des graines brunes parsèment le champ.
4.  En arrière-plan, on distingue des feuilles vertes et quelques fleurs roses fanées qui ajoutent de la texture à la scène.
5.  En bas à droite, une petite fleur rouge vif contraste avec les tons dominants de rose et de vert.
6.  L'ensemble de la composition capture un instant paisible de la nature, mettant en valeur l'interaction entre l'insecte et la plante.


**Final Polish:**
Au centre de l'image, une grande fleur cosmos rose pâle s'épanouit avec élégance, attirant l'attention par sa couleur douce. Une abeille bourdonne activement au cœur de cette fleur, recueillant le pollen jaune qui contraste avec ses ailes noires et blanches. À gauche, une autre fleur rose est visible, bien que légèrement floue, tandis que des tiges séchées et des graines brunes parsèment le champ. En arrière-plan, on distingue des feuilles vertes et quelques fleurs roses fanées qui ajoutent de la texture à la scène. En bas à droite, une petite fleur rouge vif contraste avec les tons dominants de rose et de vert. L'ensemble de la composition capture un instant paisible de la nature, mettant en valeur l'interaction entre l'insecte et la plante.

Wait, let's make it slightly tighter to ensure it feels natural. Let's go with a 5-sentence version to be safe on the "detailed" part.
1. Au centre de l'image, une grande fleur cosmos rose pâle s'épanouit avec élégance, attirant l'attention par sa couleur douce.
2. Une abeille bourdonne activement au cœur de cette fleur, recueillant le pollen jaune qui contraste avec ses ailes noires et blanches.
3. À gauche, une autre fleur rose est visible, bien que légèrement floue, tandis que des tiges séchées et des graines brunes parsèment le champ.
4. En arrière-plan, on distingue des feuilles vertes et quelques fleurs roses fanées qui ajoutent de la texture à la scène.
5. En bas à droite, une petite fleur rouge vif contraste avec les tons dominants de rose et de vert.

This is good. I will use this.

&lt;/think&gt;<br><br>
    Au centre de l'image, une grande fleur cosmos rose pâle s'épanouit avec élégance, attirant l'attention par sa couleur douce. Une abeille bourdonne activement au cœur de cette fleur, recueillant le pollen jaune qui contraste avec ses ailes noires et blanches. À gauche, une autre fleur rose est visible, bien que légèrement floue, tandis que des tiges séchées et des graines brunes parsèment le champ. En arrière-plan, on distingue des feuilles vertes et quelques fleurs roses fanées qui ajoutent de la texture à la scène. En bas à droite, une petite fleur rouge vif contraste avec les tons dominants de rose et de vert.</td>
  <td width="50%" valign="top">
&lt;think&gt;<br>
The user has provided an image and asked for a description in French.
The image shows a close-up of a garden scene.
Key elements:
- A large, vibrant light-pink cosmos (or similar dahlia-like) in the center.
- A bumblebee is on the center of this main bloom, pollination is visible.
- To the left, another similar light-pink cosmos is visible, but cut off.
- There are several withered, brown, dead florets around the main bloom.
- In the background, there are other plants: a large green lily pad or similar large green shape at the top, and some red geraniums (pelargoniums) at the bottom right.
- There are also some wilting, purplish-pink cosmos in the background.
- The focus is very deep on the central bloom and the bumblebee.

Structure of the description (4-6 sentences):
1.  Start with the main subject: the central light-pink cosmos and the bumblebee.
2.  Mention the action: the bumblebee is pollination.
3.  Mention the context/surrouding: other cosmos, dead florets.
4.  Mention the background elements: red geraniums, greenery.
5.  Convey the general feeling: natural, garden scene.

Let's draft the French sentences:
1.  Cette image présente un plan rapproché d'une fleur rose pâle, probablement un cosmos, au centre de l'encadrement.
2.  Une bourdon, aux couleurs noires et blanches, est perche sur le cœur jaune de la fleur, en train de butiner le pollen.
3.  Autour de cette floraison principale, on observe plusieurs autres tiges portant des fleurs roses, certaines en pleine fleur et d'autres déjà fanées et brunes.
4.  En arrière-plan, on devine des feuilles vertes floues et quelques fleurs rouges, semblables à des géraniums, qui ajoutent de la couleur à la scène.
5.  L'ensemble dépeint une scène de jardin naturelle et vivante, mettant en valeur l'interaction entre l'insecte et la plante.

Refining for flow and detail:
- "Cette image capture une scène de jardin en gros plan, centrée sur une magnifique fleur rose pâle, vraisemblablement un cosmos."
- "Une bourdon, avec son corps noir et blanc caractéristique, est posée au centre de la fleur, occupant le cœur jaune et récoltant le pollen."
- "Autour de cette floraison principale, plusieurs autres tiges portent des fleurs roses, certaines en pleine beauté tandis que d'autres sont déjà fanées et brunes."
- "En arrière-plan, on distingue des feuilles vertes floues et quelques fleurs rouges, probablement des géraniums, qui encadrent la composition."
- "L'ensemble offre une vue détaillée et apaisante de la nature, mettant en évidence l'activité de pollinisation dans un environnement floral varié."

This looks good. It's descriptive, in French, and within the sentence limit.

&lt;/think&gt;<br><br>
Cette image capture une scène de jardin en gros plan, centrée sur une magnifique fleur rose pâle, vraisemblablement un cosmos. Une bourdon, avec son corps noir et blanc caractéristique, est posée au centre de la fleur, occupant le cœur jaune et récoltant le pollen. Autour de cette floraison principale, plusieurs autres tiges portent des fleurs roses, certaines en pleine beauté tandis que d'autres sont déjà fanées et brunes. En arrière-plan, on distingue des feuilles vertes floues et quelques fleurs rouges, probablement des géraniums, qui encadrent la composition. L'ensemble offre une vue détaillée et apaisante de la nature, mettant en évidence l'activité de pollinisation dans un environnement floral varié.</td>
</tr>
</table>
</details>


<details>
<summary>Gemma3 4B : Voir la comparaison original / trimmé</summary>
<table>
<tr>
  <th width="50%">Original</th>
  <th width="50%">Trimmé</th>
</tr>
<tr>
  <td width="50%" valign="top">L'image présente une scène de jardin fleuri, dominée par des cosmos roses vifs. Une grosse abeille noire et jaune est en train de se nourrir sur l'une des fleurs, illustrant l'importance des pollinisateurs dans cet écosystème. Au premier plan, un cosmos particulièrement beau attire l'attention grâce à sa couleur saturée et sa forme élégante. En arrière-plan, on aperçoit d'autres fleurs, dont une rouge, ainsi que des feuilles vertes luxuriantes, créant un tableau coloré et naturel. L'ensemble dégage une impression de calme et de vitalité, typique d'un jardin en pleine floraison.</td>
  <td width="50%" valign="top">L'image montre un champ de cosmos roses en pleine floraison. Une abeille noire et jaune est activement en train de butiner une des fleurs centrales. On observe également quelques fleurs fanées environnantes, de couleur rose et rouge, et des feuilles vertes en arrière-plan. Le tout est entouré d'un fond de végétation dense et luxuriante. La lumière semble douce et diffuse, créant une atmosphère paisible et naturelle.</td>
</tr>
</table>
</details>

<br>

Nous pouvons voir que les réponses des modèles trimmés restent semblables aux modèles originaux.  
Pour le modèle 0.8B, la sortie évoque bien le cosmos dans les deux cas (un rose, l’autre violet pâle), l’insecte *bumblebee* (la vraie réponse attendue étant un bourdon mais nous avons donc une consistance entre les deux modèles) et les fleurs (mortes pour l’un, séchées pour l’autre) en arrière-plan.  
Pour le 2B, nous avons à nouveau le cosmos (rose pâle pour l’un, rose vif pour l’autre), une abeille à la place de *bumblebee*. Le modèle trimmé est un peu plus verbeux.  
Pour le 4B (non *thinking*), nous avons bien le cosmos (rose vif pour tous les deux), le bourdon (le bon animal pour tous les deux) et une description sur le fond variant légèrement entre les deux modèles mais se retrouvant sur le fond.  
Pour le 4B (*thinking*), nous pouvons observer que la trace (en anglais dans les 2 cas) du modèle original est plus long. Ce qui n’indique pas forcément de meilleurs résultats puisque le modèle nous parle d’abeille au lieu de bourdon (le modèle trimmé utilisant cependant « une » bourdon au lieu de « un » bourdon).  
Pour le gemma-3, le modèle original est un peu plus verbeux (ajoute davantage d’adjectifs que le modèle trimmé qui rapporte surtout l’essentiel). Les deux modèles nous parlent d’abeilles au lieu de bourdon.

<br>

##### Espagnol

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/BsZ9lvJyoYvpVav4Xlpu-.png" 
       alt="Photo par Merve Noyan" 
       style="max-width:70%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Photo par Merve NOYAN</center>
  </figcaption></center>
</figure>

Prompt utilisé pour tous les modèles : « Esto es lo que tengo en mi mano. Si como un caramelo de cada color, ¿cuántos caramelos me quedarán al final? » (*Voici ce que j'ai dans ma main. Si je mange un bonbon de chaque couleur, combien me restera t'il de bonbon à la fin ?*)

<details>
<summary>Qwen3.5 0.8B : Voir la comparaison original / trimmé</summary>
<table>
<tr>
  <th width="50%">Original</th>
  <th width="50%">Trimmé</th>
</tr>
<tr>
  <td width="50%" valign="top">Tengo 4 caramelos, 2 de cada color. Si me los da un caramelo de cada color, me quedarán 2 caramelos.<br><br>
  <i>J'ai 4 bonbons, 2 de chaque couleur. S'il m'en donne un de chaque couleur, il m'en restera 2.</i>
  <td width="50%" valign="top">Tengo 4 caramelos de colores diferentes. Si me los come uno por uno, me quedarán 3 caramelos.<br><br>
  <i>J'ai 4 bonbons de couleurs différentes. Si je les mange un par un, il m'en restera 3.</i></td>
</tr>
</table>
</details>

<details>
<summary>Qwen3.5 2B : Voir la comparaison original / trimmé</summary>
<table>
<tr>
  <th width="50%">Original</th>
  <th width="50%">Trimmé</th>
</tr>
<tr>
  <td width="50%" valign="top">Tienes 4 caramelos en total: 2 verdes, 2 naranjas y 1 verde. Si los comes todos, quedarán 0 caramelos.<br><br>
  <i>Tu as 4 bonbons au total : 2 verts, 2 orange et 1 vert. Si tu les manges tous, il ne restera plus aucun bonbon.</i></td>
  <td width="50%" valign="top">Si comes un caramelo de cada color, tendrás 4 caramelos en total. Los caramelos de colores verde, naranja y azul están en tu mano, y el naranja también está visible. Al comer uno de cada color, quedarán 4 caramelos en tu mano.<br><br>
  <i>Si tu manges un bonbon de chaque couleur, tu auras 4 bonbons au total. Les bonbons verts, orange et bleus sont dans ta main, et le bonbon orange est également visible. En mangeant un de chaque couleur, il te restera 4 bonbons dans la main.</i></td>
</tr>
</table>
</details>

<details>
<summary>Qwen3.5 4B : Voir la comparaison original / trimmé</summary>
<table>
<tr>
  <th width="50%">Original</th>
  <th width="50%">Trimmé</th>
</tr>
<tr>
  <td width="50%" valign="top">Tienes 5 caramelos en total: 2 azules, 2 naranjas y 1 verde. Si comes uno de cada color, comerás 3 caramelos. 5 menos 3 es igual a 2. Por lo tanto, te quedarán 2 caramelos al final.<br><br>
<i>Tu as 5 bonbons au total : 2 bleus, 2 orange et 1 vert. Si tu manges un bonbon de chaque couleur, tu en auras mangé 3. 5 moins 3, ça fait 2. Il te restera donc 2 bonbons à la fin.</i>
  </td>
  <td width="50%" valign="top">Tienes 5 caramelos en total: 2 azules, 2 naranjas y 1 verde. Si comes uno de cada color, comerás 3 caramelos (uno azul, uno naranja y uno verde). 5 menos 3 es igual a 2. Al final te quedarán 2 caramelos.<br><br> 
<i>Tu as 5 bonbons au total : 2 bleus, 2 orange et 1 vert. Si tu manges un bonbon de chaque couleur, tu en auras mangé 3 (un bleu, un orange et un vert). 5 moins 3, ça fait 2. Il te restera donc 2 bonbons.</i></td>
</tr>
</table>
</details>

<details>
<summary>Qwen3.5 4B (<i>thinking</i>) : Voir la comparaison original / trimmé</summary>
<table>
<tr>
  <th width="50%">Original</th>
  <th width="50%">Trimmé</th>
</tr>
<tr>
  <td width="50%" valign="top">
&lt;think&gt;<br>
The user wants to know how many candies are left if they eat one of each color.

1.  **Count the candies:** I need to look at the image and count the total number of candies.
    *   There's a teal one at the top.
    *   There's a teal one below it with a leaf symbol.
    *   There's an orange one below that.
    *   There's an orange one with a symbol.
    *   There's a green one at the bottom.
    *   Total count: 5 candies.

2.  **Identify the colors:**
    *   Teal (2 candies)
    *   Orange (2 candies)
    *   Green (1 candy)

3.  **Analyze the condition:** "Si como un caramelo de cada color" (If I eat one candy of each color).
    *   Colors present: Teal, Orange, Green.
    *   Action: Eat one of each.

4.  **Calculate the remaining candies:**
    *   Start with 5.
    *   Eat 1 Teal -> 4 left.
    *   Eat 1 Orange -> 3 left.
    *   Eat 1 Green -> 2 left.
    *   Alternatively: Total (5) - Colors (3) = 2 remaining.

5.  **Formulate the answer:**
    *   State the total number of candies (5).
    *   State the number of colors (3: teal, orange, green).
    *   Explain the subtraction (5 - 3 = 2).
    *   Conclude with the final number.

6.  **Refine the Spanish response:**
    *   "En la imagen hay un total de 5 caramelos."
    *   "Están en tres colores distintos: dos azules turquesa, dos naranjas y uno verde."
    *   "Si comes uno de cada color (azul, naranja y verde), restas 3 caramelos del total."
    *   "Por lo tanto, te quedarán 2 caramelos al final."

7.  **Final Polish:** Ensure natural flow.
    "En la imagen se pueden contar un total de 5 caramelos en la palma de la mano. Estos caramelos están distribuidos en tres colores diferentes: dos de color turquesa, dos de color naranja y uno de color verde. Si decides comer un caramelo de cada uno de los tres colores presentes, estarás comiendo 3 caramelos en total. Al restar esos 3 caramelos del total inicial de 5, te quedarán exactamente 2 dulces."

    Wait, let me double check the count.
    - Top teal
    - Teal with leaf
    - Orange
    - Orange with symbol
    - Green
    Yes, 5 total. 3 colors. 5 - 3 = 2. Correct.

    Let's try to make it slightly more conversational but still formal enough.
    "En la foto se observan un total de 5 caramelos sostenidos en la mano. Estos se dividen en tres colores distintos: dos son de color turquesa, dos son naranjas y uno es verde. Si comes un caramelo de cada color, estarás consumiendo 3 dulces en total. Por lo tanto, al restar esos 3 del total inicial, te quedarán 2 caramelos al final."

    This looks good. 4 sentences.  
&lt;/think&gt;

En la foto se observan un total de 5 caramelos sostenidos en la mano. Estos se dividen en tres colores distintos: dos son de color turquesa, dos son naranjas y uno es verde. Si comes un caramelo de cada color, estarás consumiendo 3 dulces en total. Por lo tanto, al restar esos 3 del total inicial, te quedarán 2 caramelos al final.<br><br>
<i>Sur la photo, on voit au total 5 bonbons tenus dans la main. Ils se répartissent en trois couleurs différentes : deux sont turquoise, deux sont orange et un est vert. Si tu manges un bonbon de chaque couleur, tu en auras consommé 3 au total. Par conséquent, en soustrayant ces 3 bonbons du nombre initial, il t'en restera 2 à la fin.</i>
  </td>
  <td width="50%" valign="top">
&lt;think&gt;  
El usuario quiere saber cuántos caramelos le quedan si come uno de cada color.
Primero, debo contar los caramelos en la imagen.
Veo:
- 2 caramelos azules/verdes (turquesa).
- 2 caramelos naranjas.
- 1 caramelo verde brillante.
Total: 2 + 2 + 1 = 5 caramelos.

Ahora, la pregunta es: "Si me como un caramelo de cada color, ¿cuántos caramelos me quedarán al final?"
Esto es un poco ambiguo. ¿Significa que come un caramelo de *cada* color disponible (azul, naranja, verde)?
Si come un caramelo de cada color:
- Come 1 azul.
- Come 1 naranja.
- Come 1 verde.
Total comido: 3 caramelos.
Caramelos restantes: 5 (totales) - 3 (comidos) = 2 caramelos.

Otra interpretación: ¿Significa que come un caramelo de *cada* color que ve? Sí, eso es lo mismo.
Hay 3 colores: azul, naranja, verde.
Si come uno de cada uno, come 3 caramelos.
Le quedan 2.

Voy a responder directamente.
1. Contar los caramelos: Hay 5 caramelos en total (2 azules, 2 naranjas y 1 verde).
2. Calcular el consumo: Si comes un caramelo de cada color (azul, naranja y verde), comerás 3 caramelos.
3. Calcular el resto: 5 menos 3 son 2.
4. Especificar qué te quedaría: Te quedarían 2 caramelos (probablemente uno azul y uno naranja, o dos naranjas si el azul se come primero, pero la cantidad es 2).

Respuesta final:
Tienes 5 caramelos en total (2 azules, 2 naranjas y 1 verde). Si comes un caramelo de cada color, comerás 3 caramelos en total. Por lo tanto, te quedarán 2 caramelos al final.  
&lt;/think&gt;

Tienes 5 caramelos en total (2 azules, 2 naranjas y 1 verde). Si comes un caramelo de cada color, comerás 3 caramelos en total. Por lo tanto, te quedarán 2 caramelos al final.
<br><br>
<i>Tu as 5 bonbons au total (2 bleus, 2 orange et 1 vert). Si tu manges un bonbon de chaque couleur, tu en auras mangé 3 au total. Il t'en restera donc 2 à la fin.</i></td>
</tr>
</table>
</details>


<details>
<summary>Gemma3 4B : Voir la comparaison original / trimmé</summary>
<table>
<tr>
  <th width="50%">Original</th>
  <th width="50%">Trimmé</th>
</tr>
<tr>
  <td width="50%" valign="top">En tu mano hay tres caramelos: uno naranja, uno azul y uno verde. Si comes uno de cada color, consumirás tres caramelos en total.  Después de comerlos, te quedará cero caramelos en tu mano. ¡Te los has comido todos!<br><br>
  <i>Tu as trois bonbons dans la main : un orange, un bleu et un vert. Si tu manges un bonbon de chaque couleur, tu auras mangé trois bonbons au total.  Une fois que tu les auras mangés, il ne te restera plus aucun bonbon dans la main. Tu les auras tous mangés !</i></td>
  <td width="50%" valign="top">Hay tres caramelos en tu mano: uno naranja, uno turquesa y uno verde. Si comes un caramelo de cada color, consumirás tres caramelos en total. Al final, te quedará cero caramelos. ¡Todo se ha ido!<br><br>
  <i>Tu as trois bonbons dans la main : un orange, un turquoise et un vert. Si tu manges un bonbon de chaque couleur, tu en auras mangé trois au total. À la fin, il ne te restera plus aucun bonbon. Il n'en reste plus un seul !</i></td>
</tr>
</table>
</details>

<br>

Comme pour le français, nous pouvons constater que les réponses des modèles trimmés restent semblables aux modèles originaux.  
Sur le fond du problème à résoudre, les Qwen 0.8B et 2B qu'ils soient trimmés ou pas n'y arrivent pas. De même que le gemma3 4b.  

Le Qwen3.5 4B est capable de donner la bonne réponse que nous activions ou pas le mode *thinking*.  
Une observation intéressante se situe au niveau de la trace. Celle du modèle trimmé est plus courte, mais surtout a été générée en [espagnol](https://huggingface.co/alphaedge-ai/Qwen3.5-4B-spa-32768/blob/main/Qwen3.5_Spanish_trace.ipynb) plutôt qu'en anglais.  
C'est le phénomène que nous décrivions dans la section précédente pour les Qwen3.  
Notons que la trace de l'exemple du Qwen 3.5 4B en français a été générée en anglais et non en français. Ainsi, comme pour les décodeurs, il apparaît comme possible de générer des traces dans la langue du modèle trimmé mais que la consistance de cette génération n'est pas acquise. Nous aurions souhaité pouvoir mener des expériences plus poussées sur le sujet en faisant varier les langues, les types de questions et les tailles de modèles pour savoir si, par exemple, à partir d'une certaine taille, ce phénomène deviendrait possible de manière systématique.

<div style="line-height:1.25; background:#241216; color:#e5e7eb; padding:1rem; border-radius:4px;">
  <p style="font-size:0.9rem;"><strong>Vigilance ⚠️</strong></p>
  <p style="font-size:0.9rem;">
    Nous sommes gênés de ne pas disposer de benchmark pour cette partie et de devoir ainsi nous limiter à du <em>vibechecking</em>. Nous invitons fortement le lecteur à effectuer des tests par lui-même de son côté.<br>
    Notamment être vigilant sur les langues peu dotées ou utilisant un autre alphabet que le latin, sur lesquelles nous n'avons pas pu effectuer de tests.
  </p>
</div>


<div class="notice--success" style="line-height:1.25;">
  <p style="font-size:0.9rem;"><strong>🧠 À retenir</strong></p>
  <p style="font-size:0.9rem;">
    Il semblerait que dans le cadre des VLM, le <em>trimming</em> permet de maintenir des sorties équivalentes aux modèles originaux (à minima pour les langues à alphabet latin et fortement dotée en ressources).<br>
    Comme pour les modèles décodeurs, il est possible de générer des traces dans la langue d'intérêt mais la manière d'obtenir ceci méthodiquement reste encore incertaine.
  </p>
</div>


#### Collections

<div style="line-height:1.25; background:#2e1f47; color:#f5f3ff; padding:1rem; border-radius:4px;">
  <p style="font-size:0.9rem;"><strong>🎁 Important</strong></p>
  <p style="font-size:0.9rem;">
    Une collection rassemblant <strong>498</strong> modèles (<strong>83</strong> langues différentes) trimmés à partir du Qwen 3.5 est disponible <a href="https://huggingface.co/collections/alphaedge-ai/trimming-qwen35" style="color:#c4b5fd;">ici</a>.<br>
    Une collection rassemblant <strong>243</strong> modèles (<strong>122</strong> langues différentes) trimmés à partir du Gemma 3 multimodal est disponible <a href="https://huggingface.co/collections/alphaedge-ai/trimming-gemma3-vlm" style="color:#c4b5fd;">ici</a>.
  </p>
</div>

<br>


### Modèles d'embeddings visuels (CLIP)

Concluons nos expériences en nous intéressant aux modèles d'*embeddings* visuels.

#### Modèles et benchmarks sélectionnés 

Dans cette section de l'article, nous analysons les modèles suivants : 
- Le [clip-ViT-B-32-multilingual-v1](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1) de REIMERS et GUREVYCH (2021) ;
- Le [MetaCLIP2](https://huggingface.co/collections/facebook/meta-clip-2) de Meta (2025) en taille `s16`, `m16`, `b16`, `l14`, `huge` et `giant`.
  
Concernant l'évaluation, les benchmarks non anglais étant rares, nous avons fait le choix de proposer plutôt des *notebooks* d'utilisation des modèles avec la librairie Sentence Transformers puisque la version 5.4 sortie en avril 2026 permet d'utiliser facilement ce type de modèles.  
Nous nous focalisons notamment sur la recherche d'images et la classification d'image en 0-*shot*.


#### Réductions de paramètres observées

<div id="table-reductions-parametres" style="overflow-x:auto; margin:1.5rem 0; max-width:100%;">
<style>
#table-reductions-parametres table {
  border-collapse: collapse;
  width: 100%;
  table-layout: fixed;
  font-size: 0.72rem !important;
  box-shadow: 0 2px 7px rgba(0,0,0,0.12);
  border-radius: 8px;
  overflow: hidden;
}

#table-reductions-parametres th,
#table-reductions-parametres td {
  padding: 7px 6px !important;
  line-height: 1.25 !important;
  text-align: center !important;
  vertical-align: middle !important;
  white-space: normal !important;
  word-break: normal !important;
  overflow-wrap: normal !important;
}

#table-reductions-parametres th {
  font-weight: 700 !important;
}

#table-reductions-parametres th:first-child,
#table-reductions-parametres td:first-child {
  width: 24%;
  text-align: left !important;
}

#table-reductions-parametres th:not(:first-child),
#table-reductions-parametres td:not(:first-child) {
  width: 12.6%;
}
</style>

<table>
    <thead>
      <tr style="background-color:#2d3748;">
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Modèle</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Taille du vocabulaire (original)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Taille du vocabulaire (trimmé)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">% Réduction vocabulaire</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Paramètres (original)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Paramètres (trimmé)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">% Réduction paramètres</th>
      </tr>
    </thead>
    <tbody>
      <!-- sentence-transformers — teal/cyan -->
      <tr style="background-color:#fdebd0;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1" target="_blank" style="color:#6b5a2d;font-weight:500;">clip-ViT-B-32-multilingual-v1</a> (encodeur visuel inclus)</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">119 547</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 86,30%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">223,0M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">143,7M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 35,53%</td>
      </tr>
      <tr style="background-color:#fdebd0;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1" target="_blank" style="color:#6b5a2d;font-weight:500;">clip-ViT-B-32-multilingual-v1</a> (encodeur visuel inclus)</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">119 547</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 72,59%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">223,0M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">156,3M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 29,89%</td>
      </tr>
      <tr style="background-color:#fdebd0;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1" target="_blank" style="color:#6b5a2d;font-weight:500;">clip-ViT-B-32-multilingual-v1</a> (encodeur visuel exclus)</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">119 547</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 86,30%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">135,1M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">55,9M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 58,63%</td>
      </tr>
      <tr style="background-color:#fdebd0;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1" target="_blank" style="color:#6b5a2d;font-weight:500;">clip-ViT-B-32-multilingual-v1</a> (encodeur visuel exclus)</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">119 547</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 72,59%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">135,1M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">68,5M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 49,32%</td>
      </tr>
      <!-- facebook/metaclip-2 — mauve -->
      <tr style="background-color:#e2e8f8;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/facebook/metaclip-2-worldwide-s16-384" target="_blank" style="color:#1a3a8f;font-weight:500;">metaclip-2-worldwide-s16-384</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">901 629</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 98,18%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">389,7M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">49,7M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 87,24%</td>
      </tr>
      <tr style="background-color:#e2e8f8;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/facebook/metaclip-2-worldwide-s16-384" target="_blank" style="color:#1a3a8f;font-weight:500;">metaclip-2-worldwide-s16-384</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">901 629</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 96,37%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">389,7M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">56,0M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 85,62%</td>
      </tr>
      <tr style="background-color:#e2e8f8;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/facebook/metaclip-2-worldwide-m16-384" target="_blank" style="color:#1a3a8f;font-weight:500;">metaclip-2-worldwide-m16-384</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">901 629</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 98,18%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">538,5M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">85,3M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 84,16%</td>
      </tr>
      <tr style="background-color:#e2e8f8;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/facebook/metaclip-2-worldwide-m16-384" target="_blank" style="color:#1a3a8f;font-weight:500;">metaclip-2-worldwide-m16-384</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">901 629</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 96,37%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">538,5M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">93,7M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 82,60%</td>
      </tr>
      <tr style="background-color:#e2e8f8;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/facebook/metaclip-2-worldwide-b16-384" target="_blank" style="color:#1a3a8f;font-weight:500;">metaclip-2-worldwide-b16-384</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">901 629</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 98,18%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">586,2M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">133,0M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 77,31%</td>
      </tr>
      <tr style="background-color:#e2e8f8;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/facebook/metaclip-2-worldwide-b16-384" target="_blank" style="color:#1a3a8f;font-weight:500;">metaclip-2-worldwide-b16-384</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">901 629</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 96,37%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">586,2M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">141,4M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 75,88%</td>
      </tr>
      <tr style="background-color:#e2e8f8;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/facebook/metaclip-2-worldwide-l14" target="_blank" style="color:#1a3a8f;font-weight:500;">metaclip-2-worldwide-l14</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">901 629</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 98,18%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">1,082B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">402,3M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 62,83%</td>
      </tr>
      <tr style="background-color:#e2e8f8;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/facebook/metaclip-2-worldwide-l14" target="_blank" style="color:#1a3a8f;font-weight:500;">metaclip-2-worldwide-l14</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">901 629</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 96,37%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">1,082B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">414,8M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 61,66%</td>
      </tr>
      <tr style="background-color:#e2e8f8;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/facebook/metaclip-2-worldwide-huge-378" target="_blank" style="color:#1a3a8f;font-weight:500;">metaclip-2-worldwide-huge-378</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">901 629</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 98,18%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">1,859B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">952,9M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 48,75%</td>
      </tr>
      <tr style="background-color:#e2e8f8;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/facebook/metaclip-2-worldwide-huge-378" target="_blank" style="color:#1a3a8f;font-weight:500;">metaclip-2-worldwide-huge-378</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">901 629</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 96,37%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">1,859B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">969,7M</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 47,85%</td>
      </tr>
      <tr style="background-color:#e2e8f8;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/facebook/metaclip-2-worldwide-giant-378" target="_blank" style="color:#1a3a8f;font-weight:500;">metaclip-2-worldwide-giant-378</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">901 629</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">16 384</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 98,18%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">3,631B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">2,519B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 31,20%</td>
      </tr>
      <tr style="background-color:#e2e8f8;">
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;"><a href="https://huggingface.co/facebook/metaclip-2-worldwide-giant-378" target="_blank" style="color:#1a3a8f;font-weight:500;">metaclip-2-worldwide-giant-378</a></td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">901 629</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">32 768</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">- 96,37%</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">3,631B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;">2,498B</td>
        <td style="padding:12px 12px;border-bottom:1px solid rgba(0,0,0,0.07);color:#000000;font-weight:700;">- 30,63%</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="notice--info" style="line-height:1.25;">
  <p style="font-size:0.9rem;"><strong>📝 Note</strong></p>
  <p style="font-size:0.9rem;">
Le clip-ViT-B-32-multilingual-v1 et le MetaCLIP-2 ayant une taille de vocabulaire n'étant pas un multiple de 64, le <i>trimming</i> semble particulièrement adapté même dans le cas où nous ne souhaitons pas spécifiquement créer un modèle monolingue/<i>n</i>-lingue.
  </p>
</div>
 
#### Résultats

##### Recherche d'images

Pour la tâche de recherche d'images, vous pouvez consulter les deux *notebooks* suivants qui ont été réalisé pour la [documentation de Sentence Transformers](https://sbert.net/examples/sentence_transformer/applications/image-search/README.html) : [la version monolingue anglais](https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/image-search/Image_Search.ipynb) et la [version multilingue](https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/image-search/image_search_multilingual.ipynb).  
Dans le premier, nous comparons une recherche effectuée via un [CLIP](https://arxiv.org/abs/2103.00020) de RADFORD, KIM et al. (2021) avec un MetaCLIP-2 original et trimmé. Nous pouvons constater aucune perte entre les deux versions (les résultats sont identiques).   
Dans le second, nous appliquons le clip-ViT-B-32-multilingual-v1.

<br>

##### Classification d'images en 0-shot

Pour cette tâche, nous vous invitons à consulter le *notebook* [suivant](https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/image-search/Image_Classification.ipynb).  
Une alternative est d'utiliser ce [Space](https://huggingface.co/spaces/alphaedge-ai/zero-shot-image-classification-trimming) qui permet de comparer, pour toutes les langues utilisées dans cet article de blog, un modèle original à ses versions trimmées en 32 768 et 16 384 *tokens*.  

Le *notebook* portant surtout sur comment effectuer de la classification d'images en 0-*shot* plutôt que sur les modèles trimmés, nous n'avons pas listé dans celui-ci deux observations intéressantes que nous détaillons donc ci-dessous.  


La première est que même si un modèle est trimmé sur une langue donnée, il peut conserver des capacités dans d'autres langues. C'est particulièrement le cas pour les langues proches partageant un même alphabet. 

Par exemple, si nous utilisons le `clip-ViT-B-32-multilingual-v1` avec les classes "Hund" (chien en allemand), "gato" (chat en espagnol), "Paris de jour" et "Paris at night" (Paris de nuit) sur 4 images différentes, nous obtenons :
    
<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/87JAwrEk8gdBfPBWVBzLY.png" 
       alt="Résultats traduction encodeur-décodeur" 
       style="max-width:100%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Classification multilingue d'images en 0-shot avec clip-ViT-B-32-multilingual-v1</center>
  </figcaption></center>
</figure>

Et si nous refaisons le même test mais avec à la place la version trimmée du modèle sur un corpus d'espagnol pour être optimisé dans cette langue, nous constatons que les résultats sont identiques.  

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/87JAwrEk8gdBfPBWVBzLY.png" 
       alt="Résultats traduction encodeur-décodeur" 
       style="max-width:100%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Classification multilingue d'images en 0-shot avec clip-ViT-B-32-multilingual-v1-spa-32768</center>
  </figcaption></center>
</figure>

<br>

Refaisons la même expérience sur des langues utilisant un alphabet différent avec toujours le `clip-ViT-B-32-multilingual-v1` mais avec les classes "Hund" (chien en allemand), "gato" (chat en espagnol), "파리" (Paris en coréen) et "باريس ليلا" (Paris de nuit en arabe) sur les 4 images précédentes, nous obtenons :

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/bmANhB31OTDu68S9fCZe9.png" 
       alt="Classification multilingue (alphabets latins et non latins) d'images en 0-shot avec clip-ViT-B-32-multilingual-v1" 
       style="max-width:100%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Classification multilingue (alphabets latins et non latins) d'images en 0-shot avec clip-ViT-B-32-multilingual-v1</center>
  </figcaption></center>
</figure>

En appliquant, à nouveau le modèle trimmé sur l'espagnol, nous pouvons constater que les performances se dégradent fortement sur les langues n'utilisant pas l'alphabet latin (- 18% pour la première image, et la réponse de la deuxième devient fausse) :

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/jIrr0Q7hywUWHF1FLP0KR.png" 
       alt="Classification multilingue (alphabets latins et non latins) d'images en 0-shot avec clip-ViT-B-32-multilingual-v1-spa-32768" 
       style="max-width:100%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Classification multilingue (alphabets latins et non latins) d'images en 0-shot avec clip-ViT-B-32-multilingual-v1-spa-32768</center>
  </figcaption></center>
</figure>

<br>

La seconde observation porte sur la quantification. En effet, nous avons vu plus haut que le *trimming* peut être une alternative à la distillation. Qu'en est-il de la quantification ?  

Prenons un exemple consistant à faire de la classification 0-*shot* sur l'image suivante :

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/KuiAL1zxpZ4a-bmw7zf2Q.png" 
       alt="Image provenant de COCO (Val 2017) de LIN et al. (2014)" 
       style="max-width:70%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Image provenant de COCO (Val 2017) de LIN et al. (2014)</center>
  </figcaption></center>
</figure>


Faisons un test sur le tamoul avec les classes பூனைகள் (Chats), சிங்கங்கள் (Lions) et புலிகள் (Tigres), et le modèle `metaclip-2-worldwide-s16-384`. Les résultats sont les suivants : 

<div style="overflow-x:auto; margin:1.5rem 0;">
  <table style="border-collapse:collapse;width:100%;font-size:0.95rem;box-shadow:0 2px 7px rgba(0,0,0,0.12);border-radius:8px;overflow:hidden;">
    <thead>
      <tr style="background-color:#2d3748;">
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;width:130px;">Configuration</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">பூனைகள் (Chats)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">சிங்கங்கள் (Lions)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">புலிகள் (Tigres)</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Taille</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background-color:#e2e8f8;">
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">Original FP32</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">&nbsp;0,992750883102417&nbsp;&nbsp;&nbsp;</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">&nbsp;0,005200702231377363&nbsp;&nbsp;&nbsp;</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">&nbsp;0,002048447960987687&nbsp;&nbsp;</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;white-space:nowrap;">&nbsp;1486,42 MB&nbsp;&nbsp;</td>
      </tr>
      <tr style="background-color:#d8e1f3;">
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">Original BF16</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">&nbsp;0,9921875&nbsp;&nbsp;&nbsp;</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">&nbsp;0,005218505859375&nbsp;&nbsp;&nbsp;</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">&nbsp;0,002166748046875&nbsp;&nbsp;</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;white-space:nowrap;">&nbsp;743,21 MB&nbsp;&nbsp;</td>
      </tr>
      <tr style="background-color:#e2e8f8;">
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">Trimming 32768 FP32</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">&nbsp;0,992750883102417&nbsp;&nbsp;</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">&nbsp;0,005200702231377363&nbsp;&nbsp;</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">&nbsp;0,002048447960987687&nbsp;&nbsp;</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;white-space:nowrap;">&nbsp;213,68 MB&nbsp;&nbsp;</td>
      </tr>
      <tr style="background-color:#d8e1f3;">
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">Trimming 32768 BF16</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">&nbsp;0,9921875&nbsp;&nbsp;&nbsp;</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">&nbsp;0,005218505859375&nbsp;&nbsp;&nbsp;</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">&nbsp;0,002166748046875&nbsp;&nbsp;</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;white-space:nowrap;">&nbsp;106,84 MB&nbsp;&nbsp;</td>
      </tr>
      <tr style="background-color:#e2e8f8;">
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">Trimming 16384 FP32</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">&nbsp;0,992750883102417&nbsp;&nbsp;&nbsp;</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">&nbsp;0,005200702231377363&nbsp;&nbsp;</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">&nbsp;0,002048447960987687&nbsp;&nbsp;</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;white-space:nowrap;">&nbsp;189,68 MB&nbsp;&nbsp;&nbsp;</td>
      </tr>
      <tr style="background-color:#d8e1f3;">
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">Trimming 16384 BF16</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">&nbsp;0,9921875&nbsp;&nbsp;&nbsp;</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">&nbsp;0,005218505859375&nbsp;&nbsp;&nbsp;</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">&nbsp;0,002166748046875&nbsp;&nbsp;</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;white-space:nowrap;">&nbsp;&nbsp;94,84 MB&nbsp;&nbsp;</td>
      </tr>
    </tbody>
  </table>
</div>

Nous avons laissé délibérément tous les chiffres pour illustrer que le *trimming* est extrêmement robuste sur les *embeddings* visuels.
Nous pouvons voir qu'il n'y a aucune dégradation entre modèle original (FP32) et ses variantes trimmées en 32 768 et 16 384 *tokens*. Le passage en BF16 dégrade très légèrement les résultats mais pas de quoi remettre en cause les conclusions de la classification.  
Si l'on compare le *trimming* à la quantification, il apparaît que pour ces types de modèles, le *trimming* est plus intéressant. En effet, nous passons de 1486,42 MB à 189,68 MB pour le *trimming* soit 87,24% de réduction de la taille mémoire, contre une réduction de 50% pour le BF16 (passage de 1486,42 MB à 743.21 MB).  
Mais en combinant les deux approches, *trimming* et quantification permettent de se ramener à seulement 94,84 MB soit 93,62% de réduction !


Cet exemple laisse supposer que le *trimming* sur les CLIP donne des résultats identiques au modèle original. C'est un exemple extrême avec le tamoul qui nécessite peu de *tokens* minés pour cette tâche (voir la partie **Nombre de *tokens* à conserver** ci-après). En recommençant la même chose mais cette fois avec le français, nous obtenons :

<div style="overflow-x:auto; margin:1.5rem 0;">
  <table style="border-collapse:collapse;width:100%;font-size:0.95rem;box-shadow:0 2px 7px rgba(0,0,0,0.12);border-radius:8px;overflow:hidden;">
    <thead>
      <tr style="background-color:#2d3748;">
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;width:130px;">Configuration</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Chats</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Lions</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Tigres</th>
        <th style="padding:12px 12px;text-align:center;font-weight:600;color:#ffffff;">Taille</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background-color:#e2e8f8;">
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">Original FP32</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">0,9781160354614258</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">0,0006950251990929246</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">0,021188927814364433</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;white-space:nowrap;">1486,42 MB</td>
      </tr>
      <tr style="background-color:#d8e1f3;">
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">Original BF16</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">0,98046875</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">0,000789642333984375</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">0,020263671875</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;white-space:nowrap;">743,21 MB</td>
      </tr>
      <tr style="background-color:#e2e8f8;">
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">Trimming 32768 FP32</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">0,9983513355255127</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">0,0007094039465300739</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">0,0009392466163262725</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;white-space:nowrap;">213,68 MB</td>
      </tr>
      <tr style="background-color:#d8e1f3;">
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">Trimming 32768 BF16</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">0,98046875</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">0,000789642333984375</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">0,020263671875</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;white-space:nowrap;">106,84 MB</td>
      </tr>
      <tr style="background-color:#e2e8f8;">
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">Trimming 16384 FP32</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">0,9984958171844482</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">0,0013454135041683912</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">0,00015876817633397877</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;white-space:nowrap;">189,68 MB</td>
      </tr>
      <tr style="background-color:#d8e1f3;">
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">Trimming 16384 BF16</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">0,998419761657715</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">0,001411437988281</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;">0,000168800354004</td>
        <td style="border:1px solid #b0bcd8; padding:10px 14px; color:#000000;white-space:nowrap;">94,84 MB</td>
      </tr>
    </tbody>
  </table>
</div>

Nous avons cette fois des modèles trimmés faisant mieux que l'original, et la version BF16 faisant mieux que la FP32 pour le modèle original. À nouveau, la combinaison du *trimming* et de la quantification apparaît comme très pertinente.

<div class="notice--info" style="line-height:1.25;">
  <p style="font-size:0.9rem;"><strong>📝 Note</strong></p>
  <p style="font-size:0.9rem;">
  Deux autres <em>notebooks</em> sont disponibles. L'un sur de la
  <a href="https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/image-search/Image_Duplicates.ipynb" target="_blank">déduplication d'images exactes ou similaires</a> et l'autre sur du <a href="https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/image-search/Image_Clustering.ipynb" target="_blank">clustering d'images</a>.
  </p>
</div>

<div class="notice--success" style="line-height:1.25;">
  <p style="font-size:0.9rem;"><strong>🧠 À retenir</strong></p>
  <p style="font-size:0.9rem;">
    <strong>Sur des modèles d'<em>embeddings</em> visuels de type CLIP, le <em>trimming</em> permet d'obtenir des performances identiques au modèle original plus de 15 chiffres après la virgule</strong>.
    Les modèles trimmés obtenus sont très légers (c'est sur eux que nous observons les plus grandes réductions du nombre de paramètres de tous les modèles présentés dans cet article) et restent <strong>plus performants que des modèles obtenus par quantification BF16</strong>. <strong>L'association de ces deux méthodes apparaît comme très compétitive</strong>.<br>
    Nous pouvons aussi constater que même si nous transformons un modèle multilingue en un modèle monolingue, <strong>il est possible de garder des capacités multilingues, notamment sur des langues partageant un même alphabet</strong>.
  </p>
</div>

#### Collections

<div style="line-height:1.25; background:#2e1f47; color:#f5f3ff; padding:1rem; border-radius:4px;">
  <p style="font-size:0.9rem;"><strong>🎁 Important</strong></p>
  <p style="font-size:0.9rem;">
    Une collection rassemblant <strong>98</strong> modèles (<strong>49</strong> langues différentes) trimmés à partir du clip-ViT-B-32-multilingual-v1 est disponible <a href="https://huggingface.co/collections/alphaedge-ai/trimming-clip-vit-b-32-multilingual-v1" style="color:#c4b5fd;">ici</a>.<br>
    Une collection rassemblant <strong>1&nbsp;152</strong> modèles (<strong>97</strong> langues différentes) trimmés à partir des MetaCLIP2 est disponible <a href="https://huggingface.co/collections/alphaedge-ai/trimming-metaclip2" style="color:#c4b5fd;">ici</a>.
  </p>
</div>

<br><br>


## <span style="color: #FFBF00"> **Questions ouvertes❓** </span>

### <span style="color: #51C353"> **Nombre de *tokens* à conserver** </span>

La question s’est posée de savoir quelle est la meilleure taille de modèle pour le vocabulaire du modèle trimmé. Il s’avère que plus nous avons creusé la question, plus nous remettions en cause notre compréhension des *tokenizers*. En effet, il semble que la taille à conserver dépend de beaucoup de facteurs :

1) Le modèle original  

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/J9IvfTZ-xSv3_3FeDcXoO.png" 
       alt="Nombre de tokens pour couvrir un certain pourcent du jeu de données de minage pour le modèle mmBERT-small sur l'anglais" 
       style="max-width:100%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Nombre de tokens pour couvrir un certain pourcent du jeu de données de minage pour le modèle mmBERT-small sur l'anglais</center>
  </figcaption></center>
</figure>


Sur ce graphique, nous pouvons constater que sur les 200 000 textes issus de Fineweb-2 ayant servi au minage (mais également à l’entraînement du mmBERT), 25 908 *tokens* permettent de couvrir 95% des *tokens* vus dans la base de données de minage. Ce sont 66 002 *tokens* pour 99% et 154 270 *tokens* pour 100%. 
Sur les 256 000 *tokens* d’origine, nous pouvons passer de **101 730** *tokens* qui n’apparaissent jamais dans des textes anglais soit une réduction de 39M de paramètres (-27,8%) du nombre de paramètres assurés. Mais devons-nous réellement garder la longue queue de **88 268** *tokens* représentant le dernier pourcent soit une réduction potentielle de 33,9M de paramètres supplémentaires (- 24,13 %). Et pourquoi s’arrêter à 99% et non pas 98 ou 97 ?

Les choses se corsent quand nous analysons les 14 autres architectures :

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/q_hLLHhofSu2aDQETIYH0.png" 
       alt="Nombre de tokens pour couvrir un certain pourcent du jeu de données de minage en fonction de l'architecture considérée pour l'anglais" 
       style="max-width:100%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Nombre de tokens pour couvrir un certain pourcent du jeu de données de minage pour le modèle mmBERT-small sur l'anglais</center>
  </figcaption></center>
</figure>

Nous pouvons voir que pour une langue donnée nous n’avons donc pas un nombre de *tokens* universel à conserver qui ferait l’affaire pour toutes les architectures. Mais vous pouvez constater que le 32 768 est à chaque fois compris entre 95% et 99%. 
À noter que pour les autres langues utilisant l’alphabet latin que nous avons évaluées dans cet article, nous constatons qu’à l’exception des gemmas, 32 768 *tokens* suffisent pour être situé entre 99% et 100%. Notre hypothèse étant que l’anglais étant le cas limite puisqu’est la langue la plus dotée en ressources. 

2) La langue
   
En s’intéressant aux langues n’utilisant pas l’alphabet latin, nous pouvons constater des tailles de vocabulaires nettement plus basses. Par exemple si nous reprenons le mmBERT mais cette fois-ci en faisant un focus sur le coréen plutôt que l’anglais :

<figure>
  <center>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/613b0a62a14099d5afed7830/9DpDFfHfmyWdcwODFtXqc.png" 
       alt="Nombre de tokens pour couvrir un certain pourcent du jeu de données de minage pour le modèle mmBERT-small sur le coréen" 
       style="max-width:100%; height:auto;">
  <figcaption>
    <center>
    Figure 1 : Nombre de tokens pour couvrir un certain pourcent du jeu de données de minage pour le modèle mmBERT-small sur le coréen</center>
  </figcaption></center>
</figure>

Le nombre de *tokens* pour couvrir 99% de la base de données est de 5859 (3 956 pour l’arabe et à peine 471 pour le tamil). Cependant la queue du dernier pourcent est très importante avec 143 161 *tokens* supplémentaires nécessaires pour atteindre les 100% (108 697 pour l’arabe et 113 391 pour le tamil).

3) La tâche
   
Au-delà du nombre de *tokens* dans le vocabulaire nécessaire à couvrir la base de minage, c’est la tâche en aval finale qui doit être la plus déterminante. Dans la section de 3.1.4 du papier [SAM3-LiteText](https://arxiv.org/abs/2602.12173) de ZENG et al. (2026), les auteurs observent qu'uniquement 17 300 *tokens* du vocabulaire de l'encodeur textuel sont en réalité pertinents, puisque les autres n'apparaissent jamais dans les jeux de données communs de segmentation d’objets. C’est cette observation qui nous a poussé à proposer également des versions des modèles en 16 384 *tokens*.  
Cette combinaison langue / modèle / tâche en aval apparaît envisageable pour un cas d’usage précis, auquel cas nous pouvons avoir un modèle avec un vocabulaire personnalisé. Si l’on souhaite un modèle généraliste, cela semble plus compliqué car nous ne connaissons pas forcément toutes les tâches en aval.
Dans le cas de modèles *n*-lingues et non pas monolingues, l’interaction entre les différentes langues peut aussi être délicat. Il faut s’assurer d’avoir un équilibre entre les langues pour qu’une ne prenne pas le pas sur les autres, anticiper que des *tokens* seront en commun entre *n* langues, etc.  

Enfin pour conclure ce point, une piste que nous aimerions explorer sont les tokenizers des modèles d’ASR. En effet, le [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo) de RADFORD, KIM et al. (2022)  gère 99 langues pour un tokenizer de 51 866 *tokens* soit en moyenne 524 *tokens* par langue. Les auteurs de `vocabtrimmer` observent de leur côté (et nous avons pu le répliquer) qu’en dessous de 10 000 *tokens* les performances des modèles sur les tâches en aval s’écroulent. 
Comment se fait-il que les modèles d’ASR fonctionnent avec si peu de *tokens* et pas les modèles de textes alors que le texte est simplement une forme écrite de la parole ?

<br>

### <span style="color: #51C353"> **Modèles incluant du code** </span>

Dans le cadre de cet article nous nous sommes focalisés exclusivement sur des modèles trimmés portant sur des langues.  
Un cas d'usage fréquent des décodeurs / VLM étant la génération de code, il faudrait mener des vérifications que le *trimming* fonctionne bien avec les langages informatiques pour pouvoir construire des modèles langue de l'utilisateur / langage(s) informatique(s) spécifique à un utilisateur travaillant dans l'informatique dans un pays donné.

<br>

### <span style="color: #51C353"> **Ordre du *trimming*** </span>

Dans la section sur les encodeurs nous trouvons qu'il vaut mieux faire du VT (*vocabulary trimming*) puis du FT (*finetuning*) pour éviter un effondrement des performances (i.e. des modèles ne marchant plus du tout). Il s'avère que nous avons observé les mêmes résultats pour des encodeurs-décodeurs, des décodeurs et des VLM (les performances ne s'effondrant pas forcément au point que le modèle ne soit plus utilisable mais avec de grosses pertes de performances de l'ordre de 10 à 20 points en fonction de la tâche).  
Le gain de performances des auteurs du MTEB-NL sur les modèles d'*embeddings* trimmés puis finetunés que nous avons également reproduits, semble aussi aller dans le sens d'adopter cet ordre opératoire. 
Enfin, en considérant le fait qu'un modèle trimmé se finetune plus rapidement que le modèle original car plus petit, **nous conseillons, et ceci n'engage que nous, de toujours trimmer un modèle puis de le finetuner**.

<div style="line-height:1.25; background:#241216; color:#e5e7eb; padding:1rem; border-radius:4px;">
  <p style="font-size:0.9rem;"><strong>Vigilance ⚠️</strong></p>
  <p style="font-size:0.9rem;">
    Néanmoins, nous devons signaler que les auteurs de <code>vocabtrimmer</code> (dans les tableaux 1 et 2 de leur <a href="https://arxiv.org/abs/2305.15020" style="color:#fca5a5;">papier</a>) trouvent des résultats contradictoires en fonction de la tâche ou de la langue considérée, mais aussi du modèle (tests sur un mT5, mBART et XLM-RoBERTa).<br>
    Notons cependant que toutes ces comparaisons ne sont pas effectuées avec le même nombre de <em>tokens</em> par langue et par modèle.
  </p>
</div>

<br>

### <span style="color: #51C353"> **Impact du trimming sur les biais** </span>

Dans leur papier, les auteurs de `vocabtrimmer` (voir la section 6 notamment) constatent qu'un modèle anglais issu d'un modèle multilingue trimmé semble présenter moins de préjugés sociaux (notamment raciaux) qu'un modèle anglais entraîné de zéro. Ce résultat va dans le sens de travaux antérieurs ([AHN et OH (2021)](https://aclanthology.org/2021.emnlp-main.42/). L'intuition des auteurs est qu'un modèle entraîné avec davantage de langues est alors exposé à une plus grande diversité culturelle et ainsi que des points de vue divergents se compensent mutuellement.  

À noter que cette observation n'a pas été vérifiée pour d'autres langues que l'anglais, et ne porte que sur un seul type de modèle, à savoir un encodeur (un XLM-RoBERTa trimmé vs. un RoBERTa).  
Il faudrait alors tester cette hypothèse sur d'autres langues (la question de l'existence de benchmarks se pose alors) et inclure plus de modèles.

<br>

### <span style="color: #51C353"> **Le code pour trimmer** </span>

Nous avons présenté dans cet article de blog plus de 16 modèles différents. Nous proposons les versions monolingues de chacun d'eux via 5526 modèles disponibles [ici](https://huggingface.co/spaces/alphaedge-ai/Trimming_models_search).   
Néanmoins, comment faire si de votre côté vous souhaitez construire votre propre modèle trimmé portant sur *l* langues de votre choix conservant *t* *tokens* de votre choix ?  
Nous réfléchissons actuellement à la conception d'une librairie permettant de faire ceci facilement. Un point préalable à résoudre est la gestion de ces modèles sur Hugging Face et via la librairie `transformers`. En effet, actuellement nous proposons 5526 modèles différents représentant au total environ 10TB de stockage. Cette approche n'est absolument pas efficiente. Elle duplique inutilement les poids du *backbone* alors qu'entre le modèle original et le modèle trimmé, seules les couches d'*embeddings* de sortie (et éventuellement d'entrée) sont modifiées, ainsi que les fichiers de vocabulaire.  
Pour limiter le stockage, il suffirait alors de charger d'un côté le *backbone* du modèle original et de l'autre les fichiers de config permettant d'appliquer le *trimming* souhaité.  
Il serait aussi probablement nécessaire de revoir un peu notre code actuel afin de proposer, en plus de l'option du nombre de *tokens* fixés par l'utilisateur, l'option de trouver automatiquement la puissance de 64 la plus proche d'un seuil de recouvrement fixé par l'utilisateur. Comme évoqué plus haut dans cette section, le seuil de recouvrement (par exemple 95% ou 99%) étant le nombre de *tokens* dans le vocabulaire pour recouvrir *x* % des *tokens* de la base de minage. Cela permettrait ainsi d'avoir des tailles de vocabulaire plus spécifiques à chacune des langues.   
En attendant de converger vers une solution, vous pouvez effectuer des demandes particulières [ici](https://huggingface.co/spaces/alphaedge-ai/Trimming_models_search/discussions) où, à l'instar de ce qui se fait pour la quantification (avec [mradermacher](https://huggingface.co/mradermacher/model_requests) par exemple), nous pourrons répondre aux requêtes populaires.

<br><br><br>

# <span style="color: #FF0000"> **Conclusion** </span>

Nous avons effectué un tour d'horizon de la méthode du *trimming* permettant de réduire très simplement la taille d'un modèle que ce soit en termes de nombre de paramètres ou de tailles mémoires.  
Cette méthode, fonctionnant uniquement si la couche d'*embedding* est la dernière du réseau, a de nombreux avantages :  
- Elle ne nécessite pas de réentraînement donc pas de GPU. L'ensemble des modèles trimmés que nous avons obtenus l'ont été sur CPU.
- Nous observons des performances équivalentes voire une amélioration possiblement dû à la suppression de *tokens* parasites.
- En cas de légère dégradation des résultats, il suffit de finetuner le modèle trimmé.
- À budget de calculs équivalents, il vaut d'ailleurs même mieux trimmer un modèle puis le finetuner afin de bénéficier de l'accélération de l'entraînement que l'on obtient du fait de la réduction du modèle.
- La méthode est compétitive face à la distillation au point que nous conseillons de préférer le *trimming* à cette dernière lorsque le nombre de paramètres à réduire souhaités est équivalent.
- De la même manière, le *trimming* peut être plus pertinent que la quantification et les deux méthodes se complètent bien.
- Un comportement intéressant que nous observons mais que nous n'arrivons pas actuellement à répliquer de manière automatique, est qu'il apparaît qu'un modèle trimmé dans une langue donnée peut générer des traces de raisonnement dans cette langue ciblée, évitant ainsi d'avoir à les générer en anglais puis à les traduire.
- Enfin, bien que nous trimmons un modèle sur une langue donnée, en fonction de la tâche en aval finale, le modèle peut toujours fonctionner dans d'autres langues.

Des questions restent ouvertes comme le nombre de *tokens* à garder pour une langue donnée / modalité donnée / tâche donnée, l'ordre du *trimming* ou encore l'impact sur les biais.  

Nous travaillons à faciliter l'usage de cette méthode via un outil. En attendant, nous vous invitons à explorer les 5526 modèles disponibles via ce [Space](https://huggingface.co/spaces/alphaedge-ai/Trimming_models_search).


<br><br><br>

# <span style="color: #FF0000"> **Références** </span>

- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) de Alec RADFORD, Jeffrey WU, Rewon CHILD, David LUAN, Dario AMODEI et Ilya SUTSKEVER (2019)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) de Jacob DEVLIN, Ming-Wei CHANG, Kenton LEE et Kristina TOUTANOVA (2018)
- [Load What You Need: Smaller Versions of Multilingual BERT](https://arxiv.org/abs/2010.05609) d'Amine ABDAOUI, Camille PRADEL et Grégoire SIGEL (2020)
- [How to adapt a multilingual T5 model for a single language](https://towardsdatascience.com/how-to-adapt-a-multilingual-t5-model-for-a-single-language-b9f94f3d9c90/) de David DALE (2021)
- [hf-trim](https://github.com/IamAdiSri/hf-trim) d'Aditya SRIVASTAVA (2022)
- [An Efficient Multilingual Language Model Compression through Vocabulary Trimmin](https://arxiv.org/abs/2305.15020) de Asahi USHIO, Yi ZHOU et Jose CAMACHO-COLLADOS (2023)
- [Multilingual Text Embedding Model Pruner](https://huggingface.co/spaces/antoinelouis/mtem-pruner) d'Antoine LOUIS (2024)
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226) de Taku KUDO et John RICHARDSON (2018)
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) de Rico SENNRICH, Barry HADDOW et Alexandra BIRCH (2015)
- [Granite Embedding Models](https://arxiv.org/abs/2502.20204) de Granite Team (2025)
- [Multilingual E5 Text Embeddings: A Technical Report](https://arxiv.org/abs/2402.05672) de Liang WANG, Nan YANG, Xiaolong HUANG, Linjun YANG, Rangan MAJUMDER et Furu WEI (2024)
- [EmbeddingGemma: Powerful and Lightweight Text Representations](https://arxiv.org/abs/2509.20354) de SCHECHTER VERA, DUA et al. (2025)
- [M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation](https://arxiv.org/abs/2402.03216) de Jianlv CHEN, Shitao XIAO, Peitian ZHANG, Kun LUO, Defu LIAN et Zheng LIU (2024)
- [Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](https://arxiv.org/abs/2506.05176) de Yanzhao ZHANG, Mingxin LI, Dingkun LONG, Xin ZHANG, Huan LIN, Baosong YANG, Pengjun XIE, An YANG, Dayiheng LIU, Junyang LIN, Fei HUANG et Jingren ZHOU (2025)
- [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) de Niklas MUENNIGHOFF, Nouamane TAZI, Loïc MAGNE et Nils REIMERS (2022)
- [MTEB-NL and E5-NL: Embedding Benchmark and Models for Dutch](https://arxiv.org/abs/2509.12340) de Nikolay BANAR, Ehsan LOTFI, Jens VAN NOOTEN, Cristina ARHILIUC, Marija KLIOCAITE et Walter DAELEMANS (2025)
- [mmBERT: A Modern Multilingual Encoder with Annealed Language Learning](https://arxiv.org/abs/2509.06888) de Marc MARONE, Orion WELLER, William FLESHMAN, Eugene YANG, Dawn LAWRIE et Benjamin VAN DURME (2025)
- [mmBERT-32K-YaRN: Extended Context Modern Multilingual BERT](https://huggingface.co/llm-semantic-router/mmbert-32k-yarn) de vLLM Semantic Router Team (2026)
- [frenchNER_3entities ](https://huggingface.co/datasets/CATIE-AQ/frenchNER_3entities) de Loïck BOURDOIS (2024)
- [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](https://arxiv.org/abs/2412.13663) de Benjamin WARNER, Antoine CHAFFIN, Benjamin CLAVIÉ, Orion WELLER, Oskar HALLSTRÖM, Said TAGHADOUINI, Alexis GALLAGHER, Raja BISWAS, Faisal LADHAK, Tom AARSEN, Nathan COOPER, Griffin ADAMS, Jeremy HOWARD et Iacopo POLI (2024)
- [Building Large Monolingual Dictionaries at the Leipzig Corpora Collection: From 100 to 200 Languages](http://www.lrec-conf.org/proceedings/lrec2012/pdf/327_Paper.pdf) de Dirk GOLDHAHN, Thomas ECKART et Uwe QUASTHOFF (2012)
- [FineWeb2: One Pipeline to Scale Them All -- Adapting Pre-Training Data Processing to Every Language](https://arxiv.org/abs/2506.20920) de Guilherme PENEDO, Hynek KYDLÍČEK, Vinko SABOLČEC, Bettina MESSMER, Negar FOROUTAN, Amir Hossein KARGARAN, Colin RAFFEL, Martin JAGGI, Leandro VON WERRA et Thomas WOLF (2025)
- [ModernBERT or DeBERTaV3? Examining Architecture and Data Influence on Transformer Encoder Models Performance](https://arxiv.org/abs/2504.08716) de Wissam ANTOUN, Benoît SAGOT et Djamé SEDDAH (2025)
- [DistilCamemBERT : une distillation du modèle français CamemBERT](https://hal.archives-ouvertes.fr/hal-03674695/file/cap2022.pdf) de Cyrile DELESTRE et Abibatou AMAR (2022)
- [CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894) de Louis MARTIN, Benjamin MULLER, Pedro Javier ORTIZ SUÁREZ, Yoann DUPONT, Laurent ROMARY, Éric VILLEMONTE DE LA CLERGERIE, Djamé SEDDAH, Benoît SAGOT (2019)
- [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) de Victor SANH, Lysandre DEBUT, Julien CHAUMOND et Thomas WOLF (2019)
- [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/abs/1804.07461) de Alex WANG, Amanpreet SINGH, Julian MICHAEL, Felix HILL, Omer LEVY et Samuel R. BOWMAN (2018)
- [mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer](https://arxiv.org/abs/2010.11934) de Linting XUE, Noah CONSTANT, Adam ROBERTS, Mihir KALE, Rami AL-RFOU, Aditya SIDDHANT, Aditya BARUA et Colin RAFFEL (2021)
- [Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210) de Yinhan LIU, Jiatao GU, Naman GOYAL, Xian LI, Sergey EDUNOV, Marjan GHAZVININEJAD, Mike LEWIS et Luke ZETTLEMOYER (2020)
- [Generative Language Models for Paragraph-Level Question Generation](https://arxiv.org/abs/2210.03992) de Asahi USHIO, Fernando ALVA-MANCHEGO et Jose CALACHO-COLLADOS (2022)
- [MultiUN: A Multilingual Corpus from United Nation Documents](http://www.lrec-conf.org/proceedings/lrec2010/pdf/686_Paper.pdf) d'Andreas EISELE et Yu CHEN (2010)
- [Encoder vs Decoder: Comparative Analysis of Encoder and Decoder Language Models on Multilingual NLU Tasks](https://arxiv.org/abs/2406.13469) de Dan Saattrup NIELSEN, Kenneth ENEVOLDSEN et Peter SCHNEIDER-KAMP (2024)
- [KOBEST: Korean Balanced Evaluation of Significant Tasks](https://arxiv.org/abs/2204.04541) de Dohyeong KIM, Myeongjun JANG, Deuk Sin KWON et Eric DAVIS (2022)
- [Granite 4.0 Language Models](https://github.com/ibm-granite/granite-4.0-language-models) d'IBM Research (2025)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) de Qwen Team (2025)
- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) de Gemma Team, Google DeepMind (2025)
- [SmolLM3: smol, multilingual, long-context reasoner](https://huggingface.co/blog/smollm3) de Elie BAKOUCH, Loubna BEN ALLAL, Anton LOZHKOV, Nouamane TAZI, Lewis TUNSTALL, Carlos Miguel PATIÑO, Edward BEECHING, Aymeric ROUCHER, Aksel Joonas REEDI, Quentin GALLOUÉDEC, Kashif RASUL, Nathan HABIB, Clémentine FOURRIER, Hynek KYDLÍČEK, Guilherme PENEDO, Hugo LARCHER, Mathieu MORLON, Vaibhav SRIVASTAV, Joshua LOCHNER, Xuan-Son NGUYEN, Colin RAFFEL, Leandro VON WERRA et Thomas WOLF (2025)
- [Qwen3.5: Accelerating Productivity with Native Multimodal Agents](https://qwen.ai/blog?id=qwen3.5) de Qwen Team (2026)
- [clip-ViT-B-32-multilingual-v1](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1) de REIMERS et GUREVYCH (2021) 
- [Meta CLIP 2: A Worldwide Scaling Recipe](https://arxiv.org/abs/2507.22062) de Yung-Sung CHUANG, Yang LI, Dong WANG, Ching-Feng YEH, Kehan LYU, Ramya RAGHAVENDRA, James GLASS, Lifei HUANG, Jason WESTON, Luke ZETTLEMOYER, Xinlei CHEN, Zhuang LIU, Saining XIE, Wen-tau YIH, Shang-Wen LI et Hu XU (2025)
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) d'Alec RADFORD, Jong Wook KIM, Chris HALLACY, Aditya RAMESH, Gabriel GOH, Sandhini AGARWAL, Girish SASTRY, Amanda ASKELL, Pamela MISHKIN, Jack CLARK, Gretchen KRUEGER et Ilya SUTSKEVER (2021)
- [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312) de Tsung-Yi LIN, Michael MAIRE, Serge BELONGIE, Lubomir BOURDEV, Ross GIRSHICK, James HAYS, Pietro PERONA, Deva RAMANAN, C. Lawrence ZITNICK et Piotr DOLLÁR (2014)
- [SAM3-LiteText: An Anatomical Study of the SAM3 Text Encoder for Efficient Vision-Language Segmentation](https://arxiv.org/abs/2602.12173) de Chengxi ZENG, Yuxuan JIANG, Ge GAO, Shuai WANG, Duolikun DANIER, Bin ZHU, Stevan RUDINAC, David BULL et Fan ZHANG (2026)
- [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) d'Alec RADFORD, Jong Wook KIM, Tao XU, Greg BROCKMAN, Christine MCLEAVEY, Ilya SUTSKEVER (2022)
- [Mitigating Language-Dependent Ethnic Bias in BERT](https://aclanthology.org/2021.emnlp-main.42/) de Jaimeen AHN et Alice OH (2021)
  
<br><br>

<br><br><br>

# <span style="color: #FF0000"> **Citation** </span>

```
@misc{hf_blogpost_trimming,
      title={Introduction to Trimming}, 
      author={Loïck BOURDOIS and Tom AARSEN and Bram VANROY and Christopher AKIKI and Woojun JUNG and Manuel ROMERO and Prithiv SAKTHI},
      year={2026},
      url={https://huggingface.co/blog/lbourdois/introduction-to-trimming}, 
}
```
