#  Task 2: End-to-End ML Pipeline for Customer Churn Prediction

---

## Objective

Build a reusable, production-grade machine learning pipeline using scikit-learn's Pipeline API to predict customer churn from the Telco dataset.

---

## Methodology / Approach

1. **Dataset**: Telco Customer Churn (IBM/Kaggle) — 7,043 customers, 20 features, binary churn target
2. **Preprocessing (inside Pipeline)**:
   - Numerical: Median imputation → StandardScaler
   - Categorical: Mode imputation → OneHotEncoder
   - Combined using `ColumnTransformer`
3. **Models Trained**:
   - Logistic Regression (baseline)
   - Random Forest (strong baseline)
   - Random Forest + GridSearchCV (best model)
4. **Hyperparameter Tuning**: 5-fold cross-validated GridSearchCV over `n_estimators`, `max_depth`, `min_samples_split`
5. **Evaluation**: Accuracy, F1-score, ROC-AUC, Confusion Matrix, Feature Importances
6. **Export**: Full pipeline saved via `joblib` — ready for production deployment

---

## Key Results / Observations

- **Month-to-month contracts** and **short tenure** are the strongest churn predictors
- Pipeline bundles preprocessing + model — no manual preprocessing needed at inference time
- `joblib` export allows a 1-line `load()` and `predict()` in any production environment

---

## Tech Stack

- `scikit-learn` — Pipeline, ColumnTransformer, GridSearchCV, models
- `pandas` / `numpy` — data handling
- `matplotlib` / `seaborn` — visualizations
- `joblib` — pipeline serialization

---

## How to Run

1. Open `Task2_ML_Pipeline_Churn.ipynb` in Google Colab
2. CPU runtime is sufficient (no GPU needed)
3. Run all cells — dataset downloads automatically

---

## Files

```
Task2_ML_Pipeline_Churn.ipynb   ← Main notebook
README.md                        ← This file
churn_pipeline.joblib            ← Exported pipeline (generated when run)
```
