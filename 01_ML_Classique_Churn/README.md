# Churn Prediction - Machine Learning

## Goal

Predict customer churn and compare Random Forest vs XGBoost.

## Results (current run)

### RandomForest

- Accuracy: 0.7991
- Precision: 0.6553
- Recall: 0.5134
- F1: 0.5757
- ROC-AUC: 0.8414

### XGBoost

- Accuracy: 0.7977
- Precision: 0.6478
- Recall: 0.5214
- F1: 0.5778
- ROC-AUC: 0.8422

## Artifacts

- `results/model_metrics.json`
- `results/confusion_matrix.png`
- `results/feature_importance.png`

## Structure

- `data/raw/`: dataset
- `notebooks/`: analysis notebook
- `src/`: reusable code
- `results/`: metrics and figures

## Setup

```bash
pip install -r ../requirements.txt
```

## Run (VS Code)

- Open `notebooks/01_churn_prediction.ipynb`
- Select the `.venv` interpreter/kernel
- Run all cells

## Dataset

Place the Telco Customer Churn CSV file at:

`01_ML_Classique_Churn/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
