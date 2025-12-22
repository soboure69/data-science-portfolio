# Projet #2 — NLP Deep Learning + Streamlit (Learn Notes)

## Objectif portfolio

Avoir :

- un modèle de **sentiment analysis** entraîné et sauvegardé
- une app **Streamlit** qui exécute l’inférence (prédiction) en conditions “production”

## Plan (rappel)

- J1 : tokenisation + padding + dataset IMDB
- J2 : modèle (Embedding + BiLSTM) + entraînement
- J3 : évaluation + courbes + sauvegarde
- J4 : app Streamlit (UI + prédiction + confiance)
- J5 : packaging + déploiement Streamlit Cloud + README + screenshot

---

## [NLP] Représenter du texte pour un réseau de neurones

### 1) Vocabulaire et indexation

Un modèle de Deep Learning ne comprend pas le texte brut. On transforme chaque mot en un **id** (entier) :

- on construit un vocabulaire (ex: top 10 000 mots)
- chaque mot reçoit un index
- une phrase devient une suite d’indices

Conséquences :

- plus le vocabulaire est grand, plus on couvre de mots mais plus le modèle est lourd
- les mots rares finissent souvent en token “OOV” (out-of-vocabulary)

### 2) Tokenisation

La tokenisation est la règle qui définit comment découper le texte en unités :

- mots (word-level)
- sous-mots (BPE, WordPiece)
- caractères

Dans ce projet : tokenisation type “mots” via `TextVectorization`.

### 3) Padding (séquences de même longueur)

Les réseaux attendent des tenseurs de taille fixe.
On choisit une longueur `MAX_LEN` (ex: 200) :

- si la phrase est plus courte : on **pad** avec des zéros
- si elle est plus longue : on **truncate** (coupe)

Pourquoi c’est important :

- `MAX_LEN` trop petit : on perd de l’information
- `MAX_LEN` trop grand : coût mémoire/temps et risque d’overfitting

### 4) Embeddings

Au lieu d’utiliser des ids directement, on apprend une représentation dense :

- `Embedding(vocab_size, embedding_dim)`
- chaque mot (id) est mappé vers un vecteur de dimension `embedding_dim` (ex: 128)

Intuition :

- le modèle apprend que certains mots ont des usages “proches”
- ces vecteurs se spécialisent pour la tâche (sentiment)

### 5) TextVectorization (pipeline NLP intégré)

`TextVectorization` combine :

- standardisation (lowercase, retirer ponctuation…)
- split en tokens
- mapping token → id
- padding/truncation

En “prod”, il faut absolument réutiliser **le même vectorizer** qu’au training.

---

## [DL] Modèle Embedding + BiLSTM

### 1) Pourquoi un RNN (LSTM / GRU) ?

Le sentiment dépend de l’ordre des mots et du contexte :

- “not good” ≠ “good”

Un LSTM/GRU lit la séquence et garde un état interne qui résume le contexte.

### 2) BiLSTM (bidirectionnel)

Un BiLSTM lit la séquence :

- de gauche à droite
- et de droite à gauche

Intérêt : capturer du contexte “futur” dans la phrase (utile en classification).

Coût : plus lent et plus lourd qu’un unidirectionnel.

### 3) Dropout (régularisation)

Le dropout “éteint” aléatoirement des neurones pendant l’entraînement.

Effet :

- empêche le modèle de trop dépendre d’un sous-ensemble de features
- réduit l’overfitting

Attention :

- en inference, dropout est désactivé
- trop de dropout peut empêcher le modèle d’apprendre

### 4) Early Stopping

On surveille une métrique de validation (ex: `val_loss`) :

- si elle n’améliore plus pendant `patience` epochs → on arrête

Pourquoi :

- évite de sur-entraîner
- réduit le temps de training

### 5) Overfitting vs Generalization

Signes d’overfitting :

- `train_accuracy` monte, `val_accuracy` stagne ou baisse
- `val_loss` remonte alors que `loss` descend

Le vrai objectif : bonne performance sur des données non vues.

---

## [Prod] Training vs Inference

### 1) Training

Pendant l’entraînement :

- calcul du gradient
- mise à jour des poids (optimizer)
- consommation CPU/GPU élevée

Artefacts importants :

- architecture du modèle
- poids entraînés
- preprocessing (vectorizer)

### 2) Inference

En inference (prod) :

- pas de gradient
- pas d’optimizer
- objectif : prédire vite et de manière stable

Bonnes pratiques :

- charger le modèle avec `compile=False` si on ne re-train pas
- réutiliser strictement le même `TextVectorization`

### 3) Latence

La latence est le temps de réponse d’une prédiction.

Facteurs :

- taille du modèle
- longueur des séquences (`MAX_LEN`)
- hardware (CPU vs GPU)
- overhead appli (Streamlit)

### 4) Cache Streamlit

`@st.cache_resource` est utile pour :

- charger le modèle une seule fois
- éviter de recharger à chaque interaction UI

Risques :

- si on change les artefacts (nouvelle release), le cache peut conserver l’ancien
- il faut alors vider le cache / forcer le re-download

### 5) Artefacts et déploiement

En déploiement, un point critique est :

- le fichier `model.keras` doit contenir les **poids**

Symptôme classique quand ce n’est pas le cas :

- `Layer 'embedding' expected 1 variables, but received 0 variables`

---

## [Ablation] LSTM vs GRU (mini-exercice)

### Objectif

Comparer rapidement (2 epochs) :

- Embedding + BiLSTM
- Embedding + BiGRU

### Ce qu’on mesure

- vitesse (temps/epoch)
- `val_accuracy` / `val_loss`

### Ce qu’on attend (intuition)

- GRU a souvent des performances proches du LSTM
- GRU est parfois un peu plus rapide / plus simple

### Conclusion attendue (portfolio)

- montrer que l’architecture a un impact
- justifier le choix final (même si c’est un compromis)

---

## Checklist “je sais expliquer”

- NLP
  - expliquer vocabulaire, OOV, padding
  - expliquer embeddings et pourquoi on les apprend
- DL
  - expliquer LSTM/GRU et Bi-directionnel
  - expliquer dropout + early stopping
- Prod
  - expliquer training vs inference
  - expliquer pourquoi il faut sauvegarder vectorizer + modèle
  - expliquer le rôle du cache Streamlit
