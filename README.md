# 📊 End-to-End Telecom Customer Churn Prediction Engine

A production-ready, explainable machine learning system designed to predict telecom customer churn using multiple business-driven modeling strategies. The system features hyperparameter optimization via `GridSearchCV`, model explainability using `SHAP`, and an interactive decision dashboard powered by `Streamlit`.

---

## 🎯 Project Overview

Rather than relying on a single default model, this system evaluates and deploys **three distinct modeling strategies** tailored for different enterprise goals:

1. **High Accuracy Strategy (LightGBM):**
   * **Focus:** Overall classification correctness.
   * **Technique:** Cost-sensitive weighting (`class_weight='balanced'`) to handle class imbalance naturally without synthetic data noise.
2. **Business Value Strategy (Random Forest):**
   * **Focus:** Catching maximum potential churners (High Recall).
   * **Technique:** Oversampling via `SMOTE` combined with custom decision threshold optimization maximizing F1/Recall trade-offs.
3. **Explainability & Accuracy Strategy (XGBoost):**
   * **Focus:** High ROC-AUC ranking power with seamless feature breakdown.
   * **Technique:** `SMOTE` oversampling paired with `TreeExplainer` for local SHAP attribution.

---

## 🏗️ Project Architecture

```text
telco_churn_project/
│
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Kaggle Telco Churn Dataset
│
├── models/
│   └── xgboost_model.pkl                          # Trained strategy artifacts & metadata
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py                      # Data cleaning & feature engineering pipeline
│   ├── explainability.py                          # SHAP tree explainer & horizontal bar plot utilities
│   └── train.py                                   # Multi-strategy training pipeline with GridSearchCV
│
├── app.py                                         # Streamlit live assessment & evaluation dashboard
├── requirements.txt                               # Project dependencies
└── README.md                                      # Documentation