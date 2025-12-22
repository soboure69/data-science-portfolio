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

## Livrables

- Modèle sauvegardé dans `models/`  
- Code de l’application dans `app/`  
- Figures et métriques dans `results/`  
- `learn.md` pour les concepts essentiels

---
