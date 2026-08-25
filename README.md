# Telecom Customer Churn Prediction

This project predicts whether a telecom customer is likely to leave the service.

The project uses machine learning models such as LightGBM, Random Forest, and XGBoost. It also includes hyperparameter tuning, SMOTE for handling imbalanced data, SHAP for model explanation, and a Streamlit dashboard.

## Project Features

- Customer churn prediction
- Data preprocessing and feature engineering
- Handling class imbalance using SMOTE
- Hyperparameter tuning using GridSearchCV
- LightGBM, Random Forest, and XGBoost models
- Model evaluation using Accuracy, Recall, F1-score and ROC-AUC
- SHAP-based model explanation
- Streamlit dashboard for predictions

## Models Used

### LightGBM
Used mainly for improving overall classification performance.

Class weights are used to handle the imbalance between churn and non-churn customers.

### Random Forest
Used when recall is important because missing a customer who may churn can be costly.

SMOTE is used to balance the training data, and the prediction threshold can be adjusted.

### XGBoost
Used for strong classification performance and feature-level explanations.

SHAP TreeExplainer is used to understand which features influence the prediction.

## Project Structure

```text
telco_churn_project/
│
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── xgboost_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── explainability.py
│   └── train.py
│
├── app.py
├── requirements.txt
└── README.md
