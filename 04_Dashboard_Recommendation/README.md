# Projet #4 — Recommender + Dashboard (Dash)

## Objectif portfolio

Dashboard **Dash** interactif montrant un système de recommandation **content-based** (cosine similarity).

- **Reco** : TF-IDF sur texte (name/category/description/tags) + features numériques (price/rating/num_reviews)
- **UI** : sélection produit + recommandations + charts
- **Mini-exercice UX** : filtre **catégorie** (impact sur les recommandations)

## Comment lancer

Pré-requis : **Python 3.11 ou 3.12**. Python 3.14 peut casser l’exécution (ex : `pkgutil.find_loader` supprimé) et/ou l’installation de dépendances (wheels manquants).

Depuis ce dossier :

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python dashboard/app.py
```

Puis ouvrir : [http://127.0.0.1:8050](http://127.0.0.1:8050)

## Dataset

- Fichier : `data/products.csv`
- Colonnes : `product_id,name,category,price,rating,num_reviews,description,tags`

## Déploiement

### Option recommandée : Render (Web Service)

Prérequis :

- Python `3.12.x` (voir `runtime.txt`)
- Commande de démarrage : voir `Procfile`

Paramètres Render typiques :

- **Build command** : `pip install -r requirements.txt`
- **Start command** : `gunicorn dashboard.app:server`

Notes :

- `dashboard/app.py` expose `server = app.server` pour Gunicorn.

### Alternative : Heroku

Le projet inclut aussi :

- `Procfile`
- `runtime.txt`

Le start command est identique : `gunicorn dashboard.app:server`.

## Structure

- `src/recommendation_engine.py` : moteur de reco (features + cosine similarity)
- `dashboard/app.py` : app Dash (layout + callbacks)

## Business case

- Document : [`Business_Case.md`](Business_Case.md) (exportable en PDF)

## Concepts à savoir expliquer

- Cosine similarity (intuition + usage)
- Feature scaling (pour mixer texte + numériques)
- Cold start (produit inconnu / pas d’historique)
- Dash callbacks (Input/State/Output)