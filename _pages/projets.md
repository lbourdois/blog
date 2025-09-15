---
permalink: /projets/
title: "Projets"
classes: wide
---

Sur cette page vous pouvez trouver les contenus sur lesquels j'ai travaillé (à titre personnel ou professionnel) mais qui sont référencés sur d'autres sites que mon blog personnel. Il s'agit principalement de traductions de cours, et des créations de jeux de données et de modèles.

<br><br>

<!--
# Vue d'ensemble

<iframe
	src="https://lbourdois-roadmap.static.hf.space"
	frameborder="0"
	width="950"
	height="1500"
></iframe>

<br><br><br>
-->

# Modèles et jeux de données en français

## FAT5
Le FAT5 est une implémentation du [T5](https://arxiv.org/abs/1910.10683) en PyTorch avec un objectif [UL2](https://arxiv.org/abs/2205.05131) optimisé pour GPGPU développé avec [Boris ALBAR](https://b-albar.github.io/portfolio/).  
Elle utilise des noyaux CUDA et Triton personnalisés ainsi que des optimisations spécifiques pour augmenter le débit et réduire l'utilisation de la mémoire pour l'entraînement et l'inférence d'un facteur 2 par rapport à l'implémentation originale disponible dans Hugging Face.  
Nous l'avons appliquée en pré-entraînant un modèle en français de 147M paramètres en utilisant uniquement une A100. Nous estimons ainsi pouvoir ramener le prix de pré-entraînement d'un tel modèle à seulement 1200€ (estimation faite sur une instance Sesterce).   
Le code de pré-entrainement est disponible sur [GitHub](https://github.com/catie-aq/flashT5) sous licence Apache-2.0 et les poids du modèle entraîné sur le compte [Hugging Face du CATIE]([https://huggingface.co/CATIE-AQ](https://huggingface.co/collections/CATIE-AQ/catie-french-fat5-ul2-677697a35feea336389d6403)). Un article de blog détaillant notre méthodologie est disponible [ici](https://huggingface.co/spaces/CATIE-AQ/FAT5-rapport).

## NER
Les NERmemBERT constituent une famille de modèles de Reconnaissance d’Entités Nommées en français capable d’étiqueter jusqu’à 4 entités (Personnalités, Lieux, Organisations, Divers tel que des noms d’œuvre, de maladies, etc.). Ils sont disponibles en taille base (110M ou 136M de paramètres) et large (336M), gérant des contextes allant de 512 à 8192 tokens. Les poids sont disponibles gratuitement en open-source, tout comme les jeux de données ayant servis à l’entraînement. Le tout est disponible sur le compte [Hugging Face du CATIE](https://huggingface.co/collections/CATIE-AQ/catie-french-ner-pack-658aefafe3f7a2dcf0e4dbb4). Un article de blog détaillant la méthodologie adoptée est disponible [ici](https://lbourdois.github.io/blog/NER/).<br>
Ils ont été téléchargés plus de 180 000 fois depuis leur mise en ligne.

## Question Answering
Les QAmemBERT constituent une famille de réponse aux questions en français capable d’indiquer si la réponse à une question est présente ou pas dans un texte de contexte associé. Ils sont disponibles en taille base (110M ou 136M de paramètres) et large (335M), gérant des contextes allant de 512 à 8192 tokens. Les poids sont disponibles gratuitement en open-source, tout comme le jeu de données ayant servis à l’entraînement. Le tout est disponible sur le compte [Hugging Face du CATIE](https://huggingface.co/collections/CATIE-AQ/catie-french-qa-pack-650821750f44c341cdb8ec91). Un article de blog détaillant la méthodologie adoptée est disponible [ici](https://lbourdois.github.io/blog/QA/).<br>
Ils ont été téléchargés plus de 155 000 fois depuis leur mise en ligne.

## DFP
*Dataset of French Prompts* (DFP) contient 113 129 978 lignes portant sur 30 tâches de NLP différentes.  
724 prompts ont été écrits sous forme impérative, de tutoiement et de vouvoiement afin de couvrir autant que possible les données de pré-entraînement utilisées par le modèle qui utilisera ces données et qui nous sont inconnues.

Les colonnes *inputs* et *targets* suivent le même format que l'ensemble de données [xP3](https://huggingface.co/datasets/bigscience/xP3) de Muennighoff et al.  
L'ensemble des détails est disponible sur [Hugging Face](https://huggingface.co/datasets/CATIE-AQ/DFP).<br>
Il a été téléchargé plus de 65 000 fois depuis sa mise en ligne.

## La marmite
Projet toujours en cours.  
Il s'agit de proposer un équivalent en français du jeu de données [*the cauldron*](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) afin de pouvoir entraîner un VLM en français.  
Ce jeu de données prendra en compte des données d'OCR (voir celles déjà disponibles en ligne [ici](https://huggingface.co/collections/lbourdois/french-ocr-datasets-67c8d3152330f11227e0d108)), de captionning (voir celles déjà disponibles en ligne [ici](https://huggingface.co/collections/lbourdois/french-caption-datasets-67c8d2227284a5daa00c50b9)), de VQA (voir celles déjà disponibles en ligne [ici](https://huggingface.co/collections/lbourdois/french-vqa-datasets-67c8d1a162a23ef0e9a2bc89)) et de raisonnement.  
Les sous-jeux de données déjà en ligne ont été téléchargé plus de 15 000 fois depuis leur mise en ligne.

<br><br><br>

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
En 2023, j'ai traduit le cours d'audio de Hugging Face.  
Le contenu est structuré en 8 unités réparties sur 46 pages web 🌐.  
Vous pouvez retrouver toutes ces ressources sur le site de [Hugging Face](https://huggingface.co/learn/audio-course/fr/).

### Cours sur les modèles de diffusion
En 2023, j'ai traduit le cours sur les modèles de diffusion de Hugging Face.  
Le contenu est structuré en 4 chapitres portant sur 17 pages web 🌐 et 8 notebooks Jupyter 📓 (en PyTorch).  
Vous pouvez retrouver toutes ces ressources sur le GitHub de [Hugging Face](https://github.com/huggingface/diffusion-models-class/tree/main/units/fr) (le contenu n'ayant pas encore été propagé sur le site officiel).

### Cours sur les agents IA
En 2025, j'ai participé avec [Kim NOEL](https://github.com/knoel99) à la traduction du cours sur les agents IA de Hugging Face.  
Le contenu est structuré en 4 unités (+ 3 bonus) réparties sur 74 pages web 🌐 et 16 notebooks Jupyter 📓.  
Vous pouvez retrouver toutes ces ressources sur le site de [Hugging Face](https://huggingface.co/learn/agents-course/fr).



<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portail des cours d'apprentissage profond</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #e2e8f0;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            text-align: center;
            margin-bottom: 60px;
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 40px 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .header-content {
            max-width: 800px;
            margin: 0 auto;
        }

        .logo {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #8b5cf6, #06b6d4, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 15px;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        .subtitle {
            font-size: 1.3rem;
            color: #94a3b8;
            margin-bottom: 20px;
            font-weight: 500;
        }

        .description {
            font-size: 1.1rem;
            color: #cbd5e1;
            line-height: 1.7;
        }

        .courses-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }

        .course-card {
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .course-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(135deg, #8b5cf6, #06b6d4, #10b981);
            transform: scaleX(0);
            transition: transform 0.4s ease;
        }

        .course-card:hover::before {
            transform: scaleX(1);
        }

        .course-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.4);
            border-color: rgba(139, 92, 246, 0.3);
        }

        .course-icon {
            font-size: 3rem;
            margin-bottom: 20px;
            display: block;
        }

        .course-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 15px;
            color: #f1f5f9;
            line-height: 1.3;
        }

        .course-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 20px;
            font-size: 0.9rem;
            color: #94a3b8;
        }

        .meta-item {
            display: flex;
            align-items: center;
            gap: 5px;
            background: rgba(139, 92, 246, 0.2);
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: 500;
            border: 1px solid rgba(139, 92, 246, 0.3);
        }

        .course-description {
            color: #cbd5e1;
            line-height: 1.6;
            margin-bottom: 25px;
            font-size: 1rem;
        }

        .course-features {
            list-style: none;
            margin-bottom: 25px;
        }

        .course-features li {
            padding: 8px 0;
            color: #94a3b8;
            font-size: 0.95rem;
            position: relative;
            padding-left: 25px;
        }

        .course-features li::before {
            content: '✓';
            position: absolute;
            left: 0;
            color: #10b981;
            font-weight: bold;
            font-size: 1.1rem;
        }

        .course-actions {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s ease;
            cursor: pointer;
            font-size: 0.95rem;
        }

        .btn-primary {
            background: linear-gradient(135deg, #8b5cf6, #06b6d4);
            color: white;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5);
        }

        .btn-secondary {
            background: rgba(139, 92, 246, 0.2);
            color: #a78bfa;
            border: 1px solid rgba(139, 92, 246, 0.3);
        }

        .btn-secondary:hover {
            background: rgba(139, 92, 246, 0.3);
            transform: translateY(-1px);
        }

        .stats-section {
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 40px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 30px;
        }

        .stat-item {
            text-align: center;
            padding: 20px;
            background: rgba(139, 92, 246, 0.1);
            border-radius: 15px;
            border: 1px solid rgba(139, 92, 246, 0.2);
        }

        .stat-number {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #8b5cf6, #06b6d4, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            display: block;
            margin-bottom: 10px;
        }

        .stat-label {
            color: #94a3b8;
            font-weight: 600;
            font-size: 1rem;
        }

        footer {
            text-align: center;
            padding: 40px 20px;
            color: rgba(255, 255, 255, 0.9);
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            margin-top: 40px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .footer-content {
            max-width: 800px;
            margin: 0 auto;
        }

        .footer-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 15px;
            color: white;
        }

        .footer-text {
            font-size: 1.1rem;
            line-height: 1.7;
            margin-bottom: 25px;
        }

        .social-links {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
        }

        .social-link {
            color: white;
            text-decoration: none;
            padding: 12px 24px;
            background: rgba(139, 92, 246, 0.3);
            border-radius: 12px;
            transition: all 0.3s ease;
            font-weight: 600;
            border: 1px solid rgba(139, 92, 246, 0.4);
        }

        .social-link:hover {
            background: rgba(139, 92, 246, 0.5);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(139, 92, 246, 0.3);
        }

        /* Language Switcher */
        .language-switcher {
            position: fixed;
            top: 30px;
            right: 30px;
            z-index: 1000;
            display: flex;
            align-items: center;
            background: rgba(15, 23, 42, 0.9);
            backdrop-filter: blur(20px);
            border-radius: 50px;
            padding: 8px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .lang-btn {
            padding: 10px 16px;
            border: none;
            background: transparent;
            border-radius: 50px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 6px;
            min-width: 80px;
            justify-content: center;
        }

        .lang-btn.active {
            background: linear-gradient(135deg, #8b5cf6, #06b6d4);
            color: white;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
        }

        .lang-btn:not(.active) {
            color: #94a3b8;
        }

        .lang-btn:not(.active):hover {
            background: rgba(139, 92, 246, 0.2);
            color: #a78bfa;
        }

        .lang-flag {
            font-size: 1.1rem;
        }

        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .courses-grid {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            
            .logo {
                font-size: 2.5rem;
            }
            
            .course-card {
                padding: 25px;
            }
            
            .course-actions {
                flex-direction: column;
            }
            
            .btn {
                text-align: center;
                justify-content: center;
            }
            
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
            }
        }

        @media (max-width: 480px) {
            .stats-grid {
                grid-template-columns: 1fr;
            }
            
            .meta-item {
                font-size: 0.8rem;
                padding: 4px 10px;
            }
        }

        /* Animation d'entrée */
        .course-card {
            animation: fadeInUp 0.6s ease-out forwards;
        }

        .course-card:nth-child(1) { animation-delay: 0.1s; }
        .course-card:nth-child(2) { animation-delay: 0.2s; }
        .course-card:nth-child(3) { animation-delay: 0.3s; }
        .course-card:nth-child(4) { animation-delay: 0.4s; }
        .course-card:nth-child(5) { animation-delay: 0.5s; }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Hover effects améliorés */
        .course-card:hover .course-icon {
            transform: scale(1.1) rotate(5deg);
            transition: transform 0.3s ease;
        }

        .course-icon {
            transition: transform 0.3s ease;
        }
    </style>
</head>
<body>
    <!-- Language Switcher -->
    <div class="language-switcher">
        <button class="lang-btn active" data-lang="fr" onclick="switchLanguage('fr')">
            <span class="lang-flag">🇫🇷</span>
            <span>FR</span>
        </button>
        <button class="lang-btn" data-lang="en" onclick="switchLanguage('en')">
            <span class="lang-flag">🇺🇸</span>
            <span>EN</span>
        </button>
    </div>

    <div class="container">
        <header>
            <div class="header-content">
                <h1 class="logo" data-fr="🎓 Portail éducatif" data-en="🎓 Learning Portal">🎓 Portail éducatif</h1>
                <p class="subtitle" data-fr="Cours en français d'apprentissage profond" data-en="Deep Learning Courses in French">Cours en français d'apprentissage profond</p>
                <p class="description" data-fr="Découvrez une collection de cours d'apprentissage profond traduits, de qualité universitaire accessibles à tous gratuitement, alliant théorie et pratique." data-en="Discover a collection of translated, university-quality deep learning courses that are accessible to everyone for free, combining theory and practice.">
                    Découvrez une collection de cours d'apprentissage profond traduits, de qualité universitaire accessibles à tous gratuitement, alliant théorie et pratique.
                </p>
            </div>
        </header>

        <div class="stats-section">
            <div class="stats-grid">
                <div class="stat-item">
                    <span class="stat-number">60+</span>
                    <span class="stat-label" data-fr="Heures de Contenu" data-en="Hours of Content">Heures de Contenu</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">280+</span>
                    <span class="stat-label" data-fr="Pages Web" data-en="Web Pages">Pages Web</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">100+</span>
                    <span class="stat-label" data-fr="Notebooks Jupyter" data-en="Jupyter Notebooks">Notebooks Jupyter</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">45+</span>
                    <span class="stat-label" data-fr="Chapitres" data-en="Chapters">Unités</span>
                </div>
            </div>
        </div>

        <div class="courses-grid">
            <!-- Cours NYU -->
            <div class="course-card">
                <span class="course-icon">🎓</span>
                <h2 class="course-title" data-fr="Cours d'Apprentissage Profond de la NYU" data-en="NYU Deep Learning Course">Cours d'Apprentissage Profond de la NYU</h2>
                <div class="course-meta">
                    <span class="meta-item">📚 19 unités</span>
                    <span class="meta-item" data-fr="🎥 33 vidéos" data-en="🎥 33 videos">🎥 33 vidéos</span>
                    <span class="meta-item" data-fr="🌐 74 pages" data-en="🌐 74 pages">🌐 74 pages</span>
                    <span class="meta-item">📓 16 notebooks</span>
                    <span class="meta-item">⏱️ 45H</span>
                </div>
                <p class="course-description" data-fr="Cours complet de Yann Le Cun et Alfredo Canziani traduit de 2020 à 2022. Couvre 19 unités avec cours magistraux et travaux dirigés, des ConvNets aux modèles à base d'énergie en passant par les Transformers." data-en="Complete course by Yann LeCun and Alfredo Canziani translated from 2020 to 2022. Covers 19 units with lectures and tutorials, from ConvNets to energy-based models including Transformers.">
                    Cours complet de Yann Le Cun et Alfredo Canziani traduit de 2020 à 2022. Couvre 19 unités avec cours magistraux et travaux dirigés, des ConvNets aux modèles à base d'énergie en passant par les Transformers.
                </p>
                <ul class="course-features" data-course="nyu">
                    <li data-fr="Vision probabiliste de l'apprentissage profond" data-en="Probabilistic view of deep learning">Vision probabiliste de l'apprentissage profond</li>
                    <li data-fr="Modèles à base d'énergie (EBMs)" data-en="Energy-based models (EBMs)">Modèles à base d'énergie (EBMs)</li>
                    <li data-fr="Architectures modernes (Transformers, GANs, GNNs)" data-en="Modern architectures (Transformers, GANs, GNNs)">Architectures modernes (Transformers, GANs, GNNs)</li>
                    <li data-fr="Applications pratiques avec PyTorch" data-en="Practical applications with PyTorch">Applications pratiques avec PyTorch</li>
                    <li data-fr="Plus de 3000 données parallèles vérifiées" data-en="Over 3000 verified parallel data points">Plus de 3000 données parallèles vérifiées</li>
                </ul>
                <div class="course-actions">
                    <a href="https://lbourdois.github.io/cours-dl-nyu/" class="btn btn-primary" target="_blank">
                        <span data-fr="🚀 Accéder au cours" data-en="🚀 Start Course">🚀 Accéder au cours</span>
                    </a>
                    <a href="https://www.youtube.com/playlist?list=PL5v98r88P9ZanM5WxvQcLJK-GKr4hQms5" class="btn btn-secondary" target="_blank">
                        <span data-fr="📺 Playlist YouTube" data-en="📺 YouTube Playlist">📺 Playlist YouTube</span>
                    </a>
                </div>
            </div>

            <!-- Cours NLP HF -->
            <div class="course-card">
                <span class="course-icon">💬</span>
                <h2 class="course-title" data-fr="Cours de NLP d'Hugging Face" data-en="Hugging Face NLP Course">Cours de NLP d'Hugging Face</h2>
                <div class="course-meta">
                    <span class="meta-item">📚 10 unités</span>
                    <span class="meta-item">🎥 76 vidéos</span>
                    <span class="meta-item">🌐 78 pages</span>
                    <span class="meta-item">📓 61 notebooks</span>
                    <span class="meta-item">⏱️ 5H</span>
                </div>
                <p class="course-description" data-fr="Traduction du cours de traitement automatique du langage de Hugging Face. 10 chapitres couvrant les techniques modernes de NLP avec PyTorch et TensorFlow." data-en="Translation of Hugging Face's natural language processing course. 10 chapters covering modern NLP techniques with PyTorch and TensorFlow.">
                    Traduction du cours de traitement automatique du langage de Hugging Face. 10 chapitres couvrant les techniques modernes de NLP avec PyTorch et TensorFlow.
                </p>
                <ul class="course-features" data-course="nlp">
                    <li data-fr="Transformers et attention" data-en="Transformers and attention">Transformers et attention</li>
                    <li data-fr="Fine-tuning de modèles pré-entraînés" data-en="Fine-tuning pre-trained models">Fine-tuning de modèles pré-entraînés</li>
                    <li data-fr="Tokenisation et preprocessing" data-en="Tokenization and preprocessing">Tokenisation et preprocessing</li>
                    <li data-fr="Évaluation et métriques" data-en="Evaluation and metrics">Évaluation et métriques</li>
                    <li data-fr="Applications pratiques en production" data-en="Practical production applications">Applications pratiques en production</li>
                </ul>
                <div class="course-actions">
                    <a href="https://huggingface.co/learn/nlp-course/fr" class="btn btn-primary" target="_blank">
                        <span data-fr="🚀 Accéder au cours" data-en="🚀 Start Course">🚀 Accéder au cours</span>
                    </a>
                </div>
            </div>

            <!-- Cours Audio HF -->
            <div class="course-card">
                <span class="course-icon">🔊</span>
                <h2 class="course-title" data-fr="Cours d'Audio d'Hugging Face" data-en="Hugging Face Audio Course">Cours d'Audio d'Hugging Face</h2>
                <div class="course-meta">
                    <span class="meta-item">📚 8 unités</span>
                    <span class="meta-item">🌐 46 pages</span>
                    <span class="meta-item">⏱️ ?H</span>
                </div>
                <p class="course-description" data-fr="Traduction du cours d'audio d'Hugging Face. 8 unités structurées pour maîtriser l'audio avec les transformers." data-en="Translation of Hugging Face's audio course. 8 structured units for mastering audio with transformers.">
                    Traduction du cours d'audio d'Hugging Face. 8 unités structurées pour maîtriser l'audio avec les transformers.
                </p>
                <ul class="course-features" data-course="audio">
                    <li data-fr="Traitement de signaux audio" data-en="Audio signal processing">Traitement de signaux audio</li>
                    <li data-fr="Reconnaissance vocale automatique" data-en="Automatic speech recognition">Reconnaissance vocale automatique</li>
                    <li data-fr="Synthèse vocale" data-en="Speech synthesis">Synthèse vocale</li>
                    <li data-fr="Classification audio" data-en="Audio classification">Classification audio</li>
                    <li data-fr="Modèles audio pré-entraînés" data-en="Pre-trained audio models">Modèles audio pré-entraînés</li>
                </ul>
                <div class="course-actions">
                    <a href="https://huggingface.co/learn/audio-course/fr" class="btn btn-primary" target="_blank">
                        <span data-fr="🚀 Accéder au cours" data-en="🚀 Start Course">🚀 Accéder au cours</span>
                    </a>
                </div>
            </div>

            <!-- Cours Diffusion HF -->
            <div class="course-card">
                <span class="course-icon">🎨</span>
                <h2 class="course-title" data-fr="Cours sur les modèles de diffusion d'Hugging Face" data-en="Hugging Face Diffusion Models Course">Cours sur les modèles de diffusion d'Hugging Face</h2>
                <div class="course-meta">
                    <span class="meta-item">📚 4 chapitres</span>
                    <span class="meta-item">🌐 17 pages</span>
                    <span class="meta-item">📓 8 notebooks</span>
                    <span class="meta-item">⏱️ ?H</span>
                </div>
                <p class="course-description" data-fr="Traduction du cours sur les modèles de diffusion de Hugging Face en 2023. Apprenez à créer et entraîner des modèles génératifs d'images avec PyTorch." data-en="Translation of Hugging Face's course on diffusion models in 2023. Learn how to create and train generative image models with PyTorch.">
                    Traduction du cours sur les modèles de diffusion de Hugging Face en 2023. Apprenez à créer et entraîner des modèles génératifs d'images avec PyTorch.
                </p>
                <ul class="course-features" data-course="diffusion">
                    <li data-fr="Théorie des modèles de diffusion" data-en="Diffusion models theory">Théorie des modèles de diffusion</li>
                    <li data-fr="DDPM et DDIM" data-en="DDPM and DDIM">DDPM et DDIM</li>
                    <li data-fr="Stable Diffusion" data-en="Stable Diffusion">Stable Diffusion</li>
                    <li data-fr="Fine-tuning personnalisé" data-en="Custom fine-tuning">Fine-tuning personnalisé</li>
                    <li data-fr="Applications créatives" data-en="Creative applications">Applications créatives</li>
                </ul>
                <div class="course-actions">
                    <a href="https://github.com/huggingface/diffusion-models-class/tree/b267aba397673e9b1a6da13fc0822c4ec85f2277/units/fr" class="btn btn-primary" target="_blank">
                        <span data-fr="🚀 Accéder au cours" data-en="🚀 Start Course">🚀 Accéder au cours</span>
                    </a>
                </div>
            </div>

            <!-- Cours Agents IA HF -->
            <div class="course-card">
                <span class="course-icon">🤖</span>
                <h2 class="course-title" data-fr="Cours sur les Agents IA d'Hugging Face" data-en="Hugging Face AI Agents Course">Cours sur les Agents IA d'Hugging Face</h2>
                <div class="course-meta">
                    <span class="meta-item">📚 7 unités</span>
                    <span class="meta-item">🌐 74 pages</span>
                    <span class="meta-item">📓 16 notebooks</span>
                    <span class="meta-item">⏱️ ?H</span>
                </div>
                <p class="course-description" data-fr="Traduction du cours sur les agents IA d'Hugging Face en 2025 comprenant 7 unités (4 unités principales + 3 bonus) pour maîtriser les agents intelligents." data-en="Translation of Hugging Face's course on AI agents in 2025, comprising 7 units (4 main units + 3 bonus units) to master intelligent agents.">
                    Traduction du cours sur les agents IA d'Hugging Face en 2025 comprenant 7 unités (4 unités principales + 3 bonus) pour maîtriser les agents intelligents.
                </p>
                <ul class="course-features" data-course="agents">
                    <li data-fr="Architectures d'agents autonomes" data-en="Autonomous agent architectures">Architectures d'agents autonomes</li>
                    <li data-fr="Planification et raisonnement" data-en="Planning and reasoning">Planification et raisonnement</li>
                    <li data-fr="Multi-modal et multi-agent" data-en="Multi-modal and multi-agent">Multi-modal et multi-agent</li>
                    <li data-fr="Outils et environnements" data-en="Tools and environments">Outils et environnements</li>
                    <li data-fr="Déploiement en production" data-en="Production deployment">Déploiement en production</li>
                </ul>
                <div class="course-actions">
                    <a href="https://huggingface.co/learn/agents-course/fr" class="btn btn-primary" target="_blank">
                        <span data-fr="🚀 Accéder au cours" data-en="🚀 Start Course">🚀 Accéder au cours</span>
                    </a>
                </div>
            </div>
        </div>

        <footer>
            <div class="footer-content">
                <h3 class="footer-title" data-fr="🌟 Tous les liens" data-en="🌟 All the links">🌟 Tous les liens</h3>
                <p class="footer-text" data-fr="Tous ces cours sont disponibles gratuitement sous licence Creative Commons." data-en="All these courses are available for free under Creative Commons license.">
                    Tous ces cours sont disponibles gratuitement sous licence Creative Commons.
                </p>
                <div class="social-links">
                    <a href="https://github.com/lbourdois" class="social-link" target="_blank">
                        💻 GitHub
                    </a>
                    <a href="https://huggingface.co/lbourdois" class="social-link" target="_blank">
                        🤗 Hugging Face
                    </a>
                    <!-- <a href="https://discord.gg/CthuqsX8Pb" class="social-link" target="_blank">
                        💬 Discord
                    </a>
                    <a href="mailto:loick.bourdois@gmail.com" class="social-link">
                        📧 Contact
                    </a> -->
                </div>
            </div>
        </footer>
    </div>

    <script>
        let currentLanguage = 'fr';

        function switchLanguage(lang) {
            if (lang === currentLanguage) return;
            
            currentLanguage = lang;
            
            // Update language buttons
            document.querySelectorAll('.lang-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            document.querySelector(`[data-lang="${lang}"]`).classList.add('active');
            
            // Update all translatable elements
            document.querySelectorAll('[data-fr], [data-en]').forEach(element => {
                const key = element.getAttribute(`data-${lang}`);
                if (key) {
                    element.textContent = key;
                }
            });
            
            // Update document language attribute
            document.documentElement.lang = lang;
            
            // Add smooth transition effect
            document.body.style.transition = 'opacity 0.3s ease';
            document.body.style.opacity = '0.8';
            setTimeout(() => {
                document.body.style.opacity = '1';
            }, 150);
        }

        // Initialize language system
        document.addEventListener('DOMContentLoaded', function() {
            // Set initial language based on browser preference or default to French
            const browserLang = navigator.language.startsWith('en') ? 'en' : 'fr';
            if (browserLang !== 'fr') {
                switchLanguage(browserLang);
            }
        });
    </script>
</body>
</html>
