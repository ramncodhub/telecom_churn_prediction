import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Path fix so running directly via `python src/train.py` works seamlessly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import preprocess_pipeline


def find_optimal_threshold(y_true, probs):
    """Finds decision threshold that maximizes F1-Score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluates a fitted estimator using a specific classification threshold."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    return {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'f1_score': f1_score(y_test, preds),
        'roc_auc': roc_auc_score(y_test, probs),
        'confusion_matrix': confusion_matrix(y_test, preds),
        'roc_curve': roc_curve(y_test, probs),
        'threshold': threshold
    }


def train_model():
    data_path = os.path.join('data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset missing at path: {data_path}. Download from Kaggle first.")
        
    print("Preprocessing dataset...")
    X, y = preprocess_pipeline(data_path)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trained_strategies = {}

    # -------------------------------------------------------------
    # Strategy 1: High Accuracy Strategy (LightGBM Cost-Sensitive)
    # -------------------------------------------------------------
    print("\n[Strategy 1] Running GridSearchCV for High Accuracy (LightGBM)...")
    lgb_model = LGBMClassifier(
        class_weight='balanced', 
        n_jobs=1,                # <--- Set to 1 to prevent OpenMP thread deadlock in CV
        random_state=42, 
        verbose=-1
    )
    lgb_param_grid = {
        'n_estimators': [100, 150],
        'learning_rate': [0.03, 0.05],
        'num_leaves': [20, 31]
    }
    
    lgb_grid = GridSearchCV(lgb_model, lgb_param_grid, cv=3, scoring='accuracy', n_jobs=2)
    lgb_grid.fit(X_train, y_train)
    best_lgb = lgb_grid.best_estimator_
    
    trained_strategies['High Accuracy (LightGBM)'] = {
        'model': best_lgb,
        'metrics': evaluate_model(best_lgb, X_test, y_test),
        'best_params': lgb_grid.best_params_,
        'description': 'Optimized for highest overall correct classification without synthetic resampling noise.'
    }

    # -------------------------------------------------------------
    # Strategy 2: Business Value Strategy (Random Forest + Tuned Threshold)
    # -------------------------------------------------------------
    print("[Strategy 2] Running GridSearchCV for Business Value (Random Forest + SMOTE)...")
    rf_pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=1))
    ])
    rf_param_grid = {
        'rf__n_estimators': [100, 150],
        'rf__max_depth': [6, 8]
    }
    
    rf_grid = GridSearchCV(rf_pipe, rf_param_grid, cv=3, scoring='recall', n_jobs=2)
    rf_grid.fit(X_train, y_train)
    best_rf_pipe = rf_grid.best_estimator_
    
    # Calculate optimal threshold for recall/business value
    rf_train_probs = best_rf_pipe.predict_proba(X_train)[:, 1]
    best_thresh = find_optimal_threshold(y_train, rf_train_probs)
    
    trained_strategies['Business Value (Random Forest)'] = {
        'model': best_rf_pipe,
        'metrics': evaluate_model(best_rf_pipe, X_test, y_test, threshold=best_thresh),
        'best_params': rf_grid.best_params_,
        'description': 'Optimized for maximum Recall (catching the maximum number of potential churners).'
    }

    # -------------------------------------------------------------
    # Strategy 3: Explainability + Accuracy Strategy (XGBoost + SMOTE)
    # -------------------------------------------------------------
    print("[Strategy 3] Running GridSearchCV for Explainability + Accuracy (XGBoost)...")
    xgb_pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(eval_metric='logloss', n_jobs=1, random_state=42))
    ])
    xgb_param_grid = {
        'xgb__n_estimators': [100, 150],
        'xgb__max_depth': [3, 4],
        'xgb__learning_rate': [0.03, 0.05]
    }
    
    xgb_grid = GridSearchCV(xgb_pipe, xgb_param_grid, cv=3, scoring='roc_auc', n_jobs=2)
    xgb_grid.fit(X_train, y_train)
    best_xgb_pipe = xgb_grid.best_estimator_
    
    # Extract underlying XGBoost model for direct SHAP Explainer compatibility
    xgb_underlying = best_xgb_pipe.named_steps['xgb']
    
    trained_strategies['Explainability + Accuracy (XGBoost)'] = {
        'model': best_xgb_pipe,
        'shap_model': xgb_underlying,
        'metrics': evaluate_model(best_xgb_pipe, X_test, y_test),
        'best_params': xgb_grid.best_params_,
        'description': 'Optimized for high ROC-AUC and seamless feature-level SHAP explanation breakdown.'
    }

    # Save artifacts
    os.makedirs('models', exist_ok=True)
    artifacts = {
        'strategies': trained_strategies,
        'feature_names': list(X.columns)
    }
    
    save_path = os.path.join('models', 'xgboost_model.pkl')
    joblib.dump(artifacts, save_path)
    print(f"\nAll strategies optimized via GridSearchCV and saved to {save_path}")


if __name__ == "__main__":
    train_model()