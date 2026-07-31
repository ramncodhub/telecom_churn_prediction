import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw dataset and convert column types."""
    df = df.copy()
    
    # Drop CustomerID as it carries no signal
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # TotalCharges contains blank spaces ' ' for new customers with tenure = 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
    
    # Target conversion
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Extract domain-specific business features."""
    df = df.copy()
    
    # 1. Average Monthly Spend Ratio over tenure
    df['Average_Monthly_Ratio'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    # 2. Total active extra services subscribed
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    df['Total_Services'] = 0
    for service in services:
        df['Total_Services'] += (df[service] == 'Yes').astype(int)
        
    # 3. High Value Customer Indicator
    df['Is_High_Value'] = ((df['MonthlyCharges'] > df['MonthlyCharges'].median()) & 
                          (df['tenure'] > 12)).astype(int)
                          
    return df

def encode_features(df: pd.DataFrame, target_col='Churn'):
    """One-hot encode categorical variables."""
    X = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()
    y = df[target_col] if target_col in df.columns else None
    
    # One-hot encoding
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    return X_encoded, y

def preprocess_pipeline(raw_csv_path: str):
    """Full preprocessing flow for dataset preparation."""
    df = pd.read_csv(raw_csv_path)
    df_cleaned = clean_data(df)
    df_featured = feature_engineering(df_cleaned)
    X, y = encode_features(df_featured, target_col='Churn')
    return X, y