import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_shap_explanation(model, input_df, feature_names):
    """
    Calculates SHAP values for a given single input row.
    Unwraps ImbPipeline or Pipeline automatically.
    """
    # Extract underlying estimator if model is wrapped in a Pipeline
    if hasattr(model, 'named_steps'):
        if 'rf' in model.named_steps:
            estimator = model.named_steps['rf']
        elif 'xgb' in model.named_steps:
            estimator = model.named_steps['xgb']
        elif 'lgb' in model.named_steps:
            estimator = model.named_steps['lgb']
        else:
            # Fallback to last step in pipeline
            estimator = model.steps[-1][1]
    else:
        estimator = model

    # Generate explainer and compute SHAP values
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer(input_df)
    
    # Ensure feature names are explicitly attached
    shap_values.feature_names = list(feature_names)
    
    return explainer, shap_values


def plot_shap_bar(shap_values, max_display=8):
    """
    Renders a horizontal bar plot showing top features driving churn vs retention.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Handle multi-output (binary classification) SHAP shapes from TreeExplainer
    values = shap_values.values[0]
    if len(values.shape) > 1 and values.shape[1] == 2:
        # Extract class 1 (churn positive class)
        values = values[:, 1]
    
    # Retrieve feature names safely
    feat_names = getattr(shap_values, 'feature_names', [f"Feature {i}" for i in range(len(values))])
    
    df_shap = pd.DataFrame({
        'Feature': feat_names,
        'SHAP_Value': values,
        'Abs_Value': np.abs(values)
    }).sort_values(by='Abs_Value', ascending=False).head(max_display)
    
    df_shap = df_shap.sort_values(by='Abs_Value', ascending=True)
    
    # Red for Churn Risk (+), Green for Retention (-)
    colors = ['#ff4b4b' if val > 0 else '#1cba63' for val in df_shap['SHAP_Value']]
    
    ax.barh(df_shap['Feature'], df_shap['SHAP_Value'], color=colors)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Impact on Churn Risk (SHAP Value)")
    ax.set_title(f"Top {max_display} Drivers for This Prediction")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    return fig