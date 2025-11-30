# Poisonous Mushroom Classification

Predict whether a mushroom is edible or poisonous using large-scale categorical morphological data.

## Dataset

This project uses:
* Extended dataset (3.1M rows) for real experiments
* Stratified 100k sample for model development, tuning, and cross-validation

Place files in `./data/`:
train.csv
test.csv
sample_submission.csv

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dependencies
pandas
numpy
scikit-learn
xgboost
matplotlib
jupyter

## Pipeline Overview

* Drop extremely high-missing columns (>80%)
* Impute (most-frequent for categorical, median for numeric)
* Encode categoricals (One-Hot for 100k sample; category codes for larger runs)
* Models:
   * Logistic Regression
   * Decision Tree
   * Random Forest (baseline + tuned)
   * XGBoost (baseline + tuned)
   * SVM (RBF)
   * Ensemble: RF + XGBoost + threshold optimization

## Key Results (100k Validation Sample)

| Model | MCC |
|-------|-----|
| Logistic Regression | 0.606 |
| Decision Tree | 0.605 |
| Random Forest (baseline) | 0.9799 |
| Random Forest (tuned) | 0.9806 |
| XGBoost (baseline) | 0.9768 |
| XGBoost (tuned) | 0.9800 |
| SVM (RBF) | 0.9694 |
| Ensemble (RF + XGB, thr=0.53) | 0.9811 |

**Best model: Ensemble of tuned RF + tuned XGB.**

## Feature Importance

Top predictors (from tuned XGBoost / RF):

* `does-bruise-or-bleed`
* `ring-type`
* `cap-surface`
* `cap-shape`
* `stem-surface`

## Notebooks

Located in `./notebooks/`:

* `01_exploration.ipynb`
* `02_baseline_model.ipynb`
* `03_model_improvement.ipynb`

## Future Work

* SHAP-based interpretability
* Full-scale training
* LightGBM experiments
* Optional Streamlit web app
