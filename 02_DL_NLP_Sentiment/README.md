# 🧠 Analyse de Sentiment — Deep Learning

## Problème

Classer les critiques de films IMDB en **positives** ou **négatives** à l’aide du Deep Learning.

## Objectifs du projet (portfolio)

- Entraîner un modèle Deep Learning (Embedding + BiLSTM / GRU).  
- Sauvegarder le modèle entraîné ainsi que les artefacts de prétraitement.  
- Déployer une **application Streamlit** permettant l’inférence en direct (avec un score de confiance).

## Structure du dépôt

- `data/` : fichiers de données locaux optionnels (non nécessaires pour le dataset IMDB de TensorFlow)  
- `notebooks/` : notebooks d’expérimentation et d’entraînement  
- `src/` : code réutilisable (prétraitement + modèle)  
- `models/` : artefacts du modèle sauvegardé (non commités si trop volumineux)  
- `results/` : métriques et figures  
- `app/` : application Streamlit

## Comment exécuter

### 1) Créer un environnement Python compatible

TensorFlow **ne supporte pas** Python 3.14.

Recommandé :

- Python **3.10** ou **3.11**

### 2) Installer les dépendances

Depuis la racine du dépôt :

```bash
pip install -r 02_DL_NLP_Sentiment/requirements.txt
```

### 3) Entraîner le modèle

Ouvrir et exécuter le(s) notebook(s) dans `notebooks/`.

### 4) Lancer l’application Streamlit

```bash
streamlit run app/app.py
```

## Déploiement (Streamlit Cloud)

### Pré-requis

- Le projet force **Python 3.11** via `runtime.txt`.
- Les artefacts `models/` et `*.keras` ne sont pas commités (trop volumineux).

### 1) Publier les artefacts en GitHub Release

Créer une Release GitHub (ex: tag `v1.0`) et uploader ces 2 fichiers :

- `models/sentiment_model.keras`
- `models/text_vectorizer.keras`

Tu obtiens ensuite 2 URLs de téléchargement direct (format `.../releases/download/...`) :

- `MODEL_URL` : `https://github.com/soboure69/data-science-portfolio/releases/download/v1.0/sentiment_model.keras`
- `VECTORIZER_URL` : `https://github.com/soboure69/data-science-portfolio/releases/download/v1.0/text_vectorizer.keras`

### 2) Créer l’app sur Streamlit Cloud

Dans Streamlit Cloud :

- **Repository** : `soboure69/data-science-portfolio`
- **Branch** : `master`
- **Main file path** : `02_DL_NLP_Sentiment/app/app.py`

L’app fonctionne en 2 modes :

- **Local** : utilise `models/` si présent (utile en local)
- **GitHub Release** : télécharge les artefacts (mode recommandé pour Streamlit Cloud)

Optionnel (recommandé) : définir les variables d’environnement Streamlit Cloud :

- `MODEL_URL`
- `VECTORIZER_URL`

## Screenshot

Après déploiement, ajouter une capture d’écran (ex: `results/streamlit_app.png`) puis l’intégrer ici :

![Streamlit App](results/streamlit_app.png)

## Livrables

- Modèle sauvegardé dans `models/`  
- Code de l’application dans `app/`  
- Figures et métriques dans `results/`  
- `learn.md` pour les concepts essentiels

---
