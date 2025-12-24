# Notes / Learn — Projet #4 (Recommender + Dashboard Dash)

## 1) Problématique

Objectif : proposer des recommandations de produits **sans historique utilisateur** (pas de clics/achats) en s’appuyant uniquement sur les **caractéristiques des produits**.

On est dans un cas typique de **cold start** (au sens “pas d’interactions”), donc on choisit une approche :

- **Content-based filtering** : recommander des items similaires au produit consulté.

---

## 2) Données (dataset)

Fichier : `data/products.csv`

Colonnes utilisées :

- `product_id` : identifiant
- `name` : nom du produit
- `category` : catégorie
- `description` : description texte
- `tags` : tags (mots-clés)
- `price`, `rating`, `num_reviews` : variables numériques

### Hypothèses de qualité

- Les champs texte peuvent être vides, mais on veut éviter les `NaN` dans les transformations.
- Les champs numériques doivent être convertibles en `float`/`int`.

---

## 3) Approche de recommandation : content-based

### 3.1) Pourquoi TF-IDF ?

Pour représenter des textes (nom, description, tags), on a besoin d’un vecteur numérique.

- **TF (Term Frequency)** : fréquence d’un mot dans un document.
- **IDF (Inverse Document Frequency)** : pondération qui diminue l’importance des mots très fréquents dans l’ensemble du corpus.

Intuition :

- Un mot rare (ex : “wireless”) discrimine mieux qu’un mot présent partout (ex : “product”).

### 3.2) Pourquoi cosine similarity ?

Une fois les produits vectorisés (TF-IDF + numérique), on compare le produit “requête” aux autres.

- La **cosine similarity** mesure l’angle entre deux vecteurs.
- Avantage : robuste à l’échelle globale des vecteurs (on compare une orientation plutôt qu’une magnitude).

Formule :

- `cosine(u, v) = (u·v) / (||u|| * ||v||)`

Interprétation :

- `1` : identiques
- `0` : orthogonaux (pas de similarité)
- `-1` : opposés (rare dans ce contexte si les features sont non négatives)

---

## 4) Features numériques : standardisation et fusion

On combine :

- Features **textuelles** (TF-IDF)
- Features **numériques** (`price`, `rating`, `num_reviews`)

### 4.1) Pourquoi standardiser ?

Sans scaling :

- `price` peut dominer car l’échelle est plus grande.

On applique une **standardisation** (z-score) :

- `x_scaled = (x - mean) / std`

Objectif : rendre les variables numériques comparables entre elles et mieux “mixables” avec la partie texte.

### 4.2) Pourquoi mixer texte + numérique ?

- Deux produits peuvent être très proches en description mais très éloignés en prix.
- Ou inversement : proches en prix/ratings mais descriptions différentes.

Le mix permet un compromis “pertinence sémantique + cohérence business”.

---

## 5) Filtrage : catégorie et contraintes

Le dashboard propose un filtre **catégorie**.

### 5.1) Effet du filtre (mini-exercice UX)

Si on impose `category == X`, on restreint le pool de produits candidats.

Conséquences :

- On peut augmenter la **cohérence** des recommandations (plus “pertinentes” pour l’utilisateur).
- Mais on peut aussi diminuer la **diversité**.
- Et parfois obtenir **0 recommandations** si le dataset est petit ou si les contraintes sont trop strictes.

Dans l’app, on gère explicitement ce cas via un message UI pour éviter un écran “vide” incompréhensible.

---

## 6) Architecture du projet

- `src/recommendation_engine.py`
  - Charge les données
  - Construit les features
  - Calcule la similarité
  - Retourne les recommandations (avec filtres)

- `dashboard/app.py`
  - UI (layout)
  - Callbacks Dash (interaction)
  - Affichage des recommandations et graphiques

---

## 7) Dash : concepts clés à expliquer

### 7.1) Layout vs callbacks

- **Layout** : déclaration des composants (dropdown, slider, graphiques…)
- **Callbacks** : “réactions” quand une valeur change

### 7.2) Input / State / Output

- `Input` : déclenche l’exécution du callback (ex : dropdown produit)
- `State` : valeur lue sans déclencher à elle seule (ex : un paramètre)
- `Output` : composant(s) mis à jour (ex : table des recommandations)

### 7.3) Gestion des cas limites (UX)

- Aucun résultat : afficher un message clair
- Réduire les surprises : afficher un résumé des filtres appliqués

---

## 8) Déploiement (Render)

### 8.1) Pourquoi Gunicorn ?

En production, on ne lance pas un serveur de dev.

- Gunicorn sert l’app via WSGI.
- Dans Dash, `app.server` expose l’app Flask sous-jacente.

### 8.2) Point clé : Python version

Les wheels (pandas/scikit-learn/numpy) dépendent fortement de la version Python.

- On fixe **Python 3.12.x** pour éviter les builds natifs (souvent lents/fragiles).

### 8.3) Commandes

- Build : `pip install -r requirements.txt`
- Start : `gunicorn dashboard.app:server`

---

## 9) Limites et améliorations possibles

- **Dataset démo** petit : la diversité des recommandations est limitée.
- TF-IDF ne capture pas toujours la sémantique fine (synonymes, contexte).

Améliorations possibles :

- Embeddings (Sentence Transformers) + ANN index (FAISS)
- “Diversification” : pénaliser les items trop similaires entre eux
- Explicabilité : afficher les termes TF-IDF les plus contributeurs
- Feature weighting : pondérer texte vs numérique selon le contexte
