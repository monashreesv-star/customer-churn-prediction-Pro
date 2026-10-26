"""
data_preprocessing.py
Data Ingestion, Cleaning, and Preprocessing Module for Customer Churn Prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional


NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset:
    - Replace blank spaces in 'TotalCharges' with NaN and coerce to float.
    - Impute NaN in 'TotalCharges' with 0.0 (corresponding to 0 tenure).
    """
    df_clean = df.copy()
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(
            df_clean['TotalCharges'].replace(' ', np.nan),
            errors='coerce'
        ).fillna(0.0)
    return df_clean


def prepare_train_test(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler, List[str]]:
    """
    Perform feature selection, one-hot encoding, train-test splitting, and scaling.
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, fitted_scaler, feature_names
    """
    df_clean = clean_data(df)
    
    # Target encoding: 'Yes' -> 1, 'No' -> 0
    if 'Churn' not in df_clean.columns:
        raise ValueError("Dataset does not contain 'Churn' target column.")
        
    y = (df_clean['Churn'].str.strip().str.lower() == 'yes').astype(int)
    
    # Drop irrelevant columns
    drop_cols = ['Churn']
    if 'customerID' in df_clean.columns:
        drop_cols.append('customerID')
        
    X_raw = df_clean.drop(columns=drop_cols)
    
    # One-hot encoding
    X_encoded = pd.get_dummies(X_raw, drop_first=True, dtype=int)
    feature_names = X_encoded.columns.tolist()
    
    # Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Numerical scaling
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[NUMERICAL_COLS] = scaler.fit_transform(X_train[NUMERICAL_COLS])
    X_test_scaled[NUMERICAL_COLS] = scaler.transform(X_test[NUMERICAL_COLS])
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def transform_unseen_data(
    df: pd.DataFrame,
    feature_names: List[str],
    scaler: StandardScaler
) -> pd.DataFrame:
    """
    Preprocess unseen customer data using the trained feature columns and scaler.
    """
    df_clean = clean_data(df)
    
    drop_cols = [c for c in ['customerID', 'Churn'] if c in df_clean.columns]
    X_raw = df_clean.drop(columns=drop_cols)
    
    X_encoded = pd.get_dummies(X_raw, drop_first=True, dtype=int)
    
    # Align columns with training feature columns (fill missing columns with 0)
    X_aligned = X_encoded.reindex(columns=feature_names, fill_value=0)
    
    # Scale numerical features
    X_aligned[NUMERICAL_COLS] = scaler.transform(X_aligned[NUMERICAL_COLS])
    
    return X_aligned
