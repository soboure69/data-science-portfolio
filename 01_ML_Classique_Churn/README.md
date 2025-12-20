# Prédiction du Churn — Machine Learning

## Objectif

Prédire le churn client et comparer Random Forest à XGBoost.

## Points clés

- Comparer deux solides modèles de référence pour données tabulaires : **RandomForest (bagging)** vs **XGBoost (boosting)**.  
- Évaluer avec plusieurs métriques (pas seulement l’accuracy).  
- Montrer que **le seuil de classification est une décision métier**.

## Résultats (exécution actuelle)

Les métriques sont calculées sur le jeu de test mis de côté.

| Modèle | Accuracy | Précision | Rappel | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.7991 | 0.6553 | 0.5134 | 0.5757 | 0.8414 |
| XGBoost | 0.7977 | 0.6478 | 0.5214 | 0.5778 | 0.8422 |

## Seuil de décision (perspective métier)

Le seuil par défaut est **0.5**.

D’après l’analyse des seuils (0.3 / 0.5 / 0.7), **0.3** offre un meilleur compromis global (F1 plus élevé) si l’on accepte de cibler un groupe de clients plus large.

## Figures

### Matrices de confusion

![confusion_matrix.png](results/confusion_matrix.png)

### Importance des variables (Top 15)

![feature_importance.png](results/feature_importance.png)

## Artéfacts

- `results/model_metrics.json`  
- `results/confusion_matrix.png`  
- `results/feature_importance.png`

## Structure

- `data/raw/` : dataset  
- `notebooks/` : notebook d’analyse  
- `src/` : code réutilisable  
- `results/` : métriques et figures

## Installation

```bash
pip install -r ../requirements.txt
```

## Exécution (VS Code)

- Ouvrir `notebooks/01_churn_prediction.ipynb`  
- Sélectionner l’interpréteur/kernel `.venv`  
- Exécuter toutes les cellules

## Dataset

Placer le fichier CSV Telco Customer Churn ici :

`01_ML_Classique_Churn/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

---