# Learn (minimum) — Prédiction du Churn (ML)

Ce document est une courte checklist des “concepts minimum” à comprendre pour maîtriser ce que j'ai construit dans ce projet.

## [ML] Surapprentissage, biais/variance, validation croisée

### Surapprentissage (overfitting)

Un modèle surapprend lorsqu’il performe très bien sur les données d’entraînement mais mal sur des données jamais vues (test / production).

- Cause typique : modèle trop flexible par rapport à la quantité ou au bruit des données.  
- Symptômes :  
  - Score entraînement très élevé, score test nettement plus faible.  
  - Importances de variables / splits qui semblent “aléatoires” ou instables.  
- Solutions typiques :  
  - Réduire la complexité du modèle (arbres plus petits, régularisation, moins de features).  
  - Utiliser la validation croisée pour ajuster les hyperparamètres.  
  - Obtenir plus de données ou améliorer la qualité du signal.

### Biais vs variance

Un modèle mental utile :

- **Biais élevé** = sous-apprentissage (underfitting).  
  - Modèle trop simple.  
  - Scores train et test tous deux faibles.

- **Variance élevée** = surapprentissage (overfitting).  
  - Modèle trop sensible au jeu d’entraînement.  
  - Score train élevé, score test plus faible.

RandomForest réduit généralement la variance grâce au bagging.  
XGBoost peut être très performant mais nécessite régularisation / tuning pour éviter l’overfitting.

### Validation croisée (Cross-validation)

La validation croisée estime la capacité de généralisation de manière plus fiable qu’un simple split train/test.

- Choix typique : **Stratified K-Fold** pour la classification.  
- Pourquoi : en churn, la classe positive est souvent minoritaire ; la stratification garantit des proportions similaires dans chaque fold.  
- Utilisation : tuning d’hyperparamètres (GridSearchCV / RandomizedSearchCV). La CV réduit le risque que ton modèle soit “chanceux” sur un seul split.

---

## [Metrics] Accuracy vs précision/rappel/F1 vs ROC-AUC

### Accuracy

Proportion de prédictions correctes.

- Pertinent lorsque les classes sont équilibrées et que les coûts d’erreur sont similaires.  
- Trompeur en cas de déséquilibre (ex : prédire “No churn” pour tout le monde peut sembler “bon”).

### Précision / Rappel

Pour la classe positive (churn = 1) :

- **Précision** : parmi les clients prédits comme churners, combien churnent réellement ?  
  - Haute précision = moins de faux positifs.

- **Rappel** : parmi les churners réels, combien sont détectés ?  
  - Haut rappel = moins de faux négatifs.

### F1-score

Moyenne harmonique de la précision et du rappel.

- Utile pour obtenir un seul indicateur équilibrant les deux.  
- Reste une simplification : le “meilleur” compromis dépend des coûts métier.

### ROC-AUC

Mesure la qualité du classement sur tous les seuils possibles.

- Un ROC-AUC élevé signifie que les churners reçoivent en général des probabilités plus élevées que les non-churners.  
- Il **ne choisit pas** le seuil de décision ; il évalue la capacité de séparation du modèle.

---

## [Data] Fuite de données, stratification, pipeline de prétraitement

### Data leakage (fuite de données)

La fuite survient lorsque le modèle accède (directement ou indirectement) à des informations indisponibles au moment de la prédiction.

Exemples :

- Utiliser des informations futures (ex : features dérivées du churn).  
- Appliquer le scaling/encoding sur l’ensemble du dataset avant le split.

Impact :

- Performances test artificiellement gonflées.  
- Mauvaise performance réelle en production.

### Stratification

Lors du split train/test, il faut **stratifier** sur la variable cible.

- Garantit un taux de churn similaire dans train et test.  
- Réduit le risque d’un split “malchanceux”.

### Pipeline de prétraitement (même simple)

Un schéma robuste :

- D’abord splitter les données (train/test).  
- Fit le prétraitement uniquement sur train (encodeurs/scalers), puis transformer test.  
- Conserver les artefacts (scaler, encodeurs, noms de features) pour reproduire les mêmes transformations plus tard.

En production ML, cela se fait généralement via `sklearn.pipeline.Pipeline` / `ColumnTransformer`.

---