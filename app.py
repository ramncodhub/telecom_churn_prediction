import sys
import os

# Ensure local imports work cleanly regardless of execution directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.explainability import get_shap_explanation, plot_shap_bar
from src.data_preprocessing import clean_data, feature_engineering

st.set_page_config(
    page_title="Multi-Strategy Customer Churn Engine",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_model.pkl')
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

artifacts = load_artifacts()

st.title("📊 Telecom Customer Churn Engine & Model Evaluation")

if artifacts is None:
    st.error("⚠️ Artifacts missing! Please run `python src/train.py` first.")
    st.stop()

strategies = artifacts['strategies']
feature_names = artifacts['feature_names']

# --- Strategy Selector Header ---
st.markdown("### 🎯 Choose Prediction Goal / Model Strategy")
strategy_choice = st.selectbox(
    "Select Model Strategy:",
    options=list(strategies.keys()),
    index=0
)

selected_strategy = strategies[strategy_choice]
active_model = selected_strategy['model']
active_metrics = selected_strategy['metrics']

st.info(f"**Strategy Focus:** {selected_strategy['description']}")

# --- Sidebar Inputs ---
st.sidebar.header("Customer Profile Input")

tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 2)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 95.0)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])

# Construct raw input payload matching data contract
raw_input = pd.DataFrame([{
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': tenure,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': str(tenure * monthly_charges)
}])

# Pipeline preprocessing
cleaned_df = clean_data(raw_input)
featured_df = feature_engineering(cleaned_df)
encoded_df = pd.get_dummies(featured_df)
input_df = encoded_df.reindex(columns=feature_names, fill_value=0)

# --- App Layout Tabs ---
tab1, tab2 = st.tabs(["🔮 Live Customer Assessment", "📈 Strategy Metrics & Comparison"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Assessment Result")
        probs = active_model.predict_proba(input_df)[0][1]
        custom_thresh = active_metrics.get('threshold', 0.5)
        is_churn = probs >= custom_thresh
        
        st.metric(
            label="Churn Probability Score",
            value=f"{probs * 100:.1f}%",
            delta="High Risk" if is_churn else "Low Risk",
            delta_color="inverse"
        )
        
        st.caption(f"Decision Threshold for this Strategy: `{custom_thresh:.2f}`")
        
        if is_churn:
            st.error("⚠️ High Churn Risk: Trigger Retention Offer")
        else:
            st.success("✅ Low Churn Risk Customer")

    with col2:
        st.subheader("Model Decision Breakdown (SHAP Feature Impact)")
        
        # Unwrap pipeline if needed to pass tree estimator to SHAP
        shap_target_model = selected_strategy.get('shap_model', None)
        if shap_target_model is None:
            if hasattr(active_model, 'named_steps'):
                shap_target_model = active_model.named_steps.get('rf', active_model)
            else:
                shap_target_model = active_model
                
        with st.spinner("Generating Feature Importance Breakdown..."):
            try:
                explainer, shap_values = get_shap_explanation(shap_target_model, input_df, feature_names)
                fig = plot_shap_bar(shap_values, max_display=8)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"SHAP explanation visualization unavailable for this estimator: {e}")

with tab2:
    st.subheader(f"Metrics Scorecard: {strategy_choice}")
    
    # 1. Main Scorecard Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{active_metrics['accuracy']*100:.2f}%")
    m2.metric("Precision", f"{active_metrics['precision']*100:.2f}%")
    m3.metric("Recall", f"{active_metrics['recall']*100:.2f}%")
    m4.metric("F1-Score", f"{active_metrics['f1_score']*100:.2f}%")
    m5.metric("ROC-AUC", f"{active_metrics['roc_auc']:.4f}")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### Best Hyperparameters (GridSearchCV)")
        st.json(selected_strategy.get('best_params', {}))
        
        st.markdown("### Confusion Matrix")
        cm = active_metrics['confusion_matrix']
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)

    with col_b:
        st.markdown("### Full Metric Comparison across All Strategies")
        
        comp_data = []
        for strat_name, strat_data in strategies.items():
            m = strat_data['metrics']
            comp_data.append({
                'Strategy': strat_name,
                'Accuracy (%)': round(m['accuracy'] * 100, 2),
                'Recall (%)': round(m['recall'] * 100, 2),
                'Precision (%)': round(m['precision'] * 100, 2),
                'F1 Score (%)': round(m['f1_score'] * 100, 2),
                'ROC-AUC': round(m['roc_auc'], 4)
            })
            
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp, use_container_width=True)