"""
train_models.py
Model training, evaluation, comparison, and serialization pipeline for Customer Churn Prediction.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_raw_data, prepare_train_test


def evaluate_model(model, X_test, y_test):
    """Compute key classification metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
    
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc
    }


def main():
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    dataset_path = os.path.join("data", "Telco-Customer-Churn.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please ingest data first.")
        
    print(f"[*] Ingesting dataset from: {dataset_path}")
    raw_df = load_raw_data(dataset_path)
    print(f"[*] Raw shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    
    print("[*] Preparing train/test split (80/20 stratified) and scaling numerical features...")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_train_test(raw_df)
    print(f"[*] Training features shape: {X_train.shape}")
    print(f"[*] Test features shape: {X_test.shape}")
    print(f"[*] Total engineered features: {len(feature_names)}")
    
    # Define candidate models
    models = {
        'Logistic Regression (Standard)': LogisticRegression(
            max_iter=1000, random_state=42
        ),
        'Logistic Regression (Balanced)': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42
        ),
        'Random Forest (Balanced)': RandomForestClassifier(
            n_estimators=100, max_depth=8, class_weight='balanced', random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
    }
    
    comparison_records = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n-> Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics['Model'] = name
        comparison_records.append(metrics)
        trained_models[name] = model
        
    # Build comparison DataFrame
    comparison_df = pd.DataFrame(comparison_records)
    cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    comparison_df = comparison_df[cols]
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 60)
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Save comparison table to reports
    os.makedirs(os.path.join("reports"), exist_ok=True)
    comparison_df.to_csv(os.path.join("reports", "model_comparison.csv"), index=False)
    print("\n[+] Model comparison saved to reports/model_comparison.csv")
    
    # Model Selection:
    # Logistic Regression (Balanced) delivers top Recall (~78.3%) for early retention intervention,
    # while Gradient Boosting delivers top ROC-AUC (0.842) and accuracy (~80%).
    # We serialize Gradient Boosting and Logistic Regression (Balanced) for deployment.
    os.makedirs("models", exist_ok=True)
    
    best_model_name = 'Gradient Boosting'
    best_model = trained_models[best_model_name]
    best_metrics = [m for m in comparison_records if m['Model'] == best_model_name][0]
    
    bundle = {
        'model_name': best_model_name,
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': best_metrics,
        'high_risk_threshold': 0.70,
        'moderate_risk_threshold': 0.40
    }
    
    bundle_path = os.path.join("models", "best_churn_model.pkl")
    joblib.dump(bundle, bundle_path)
    print(f"[+] Primary model artifact ({best_model_name}) exported to: {bundle_path}")
    
    # Also save the Balanced Recall-focused model for business scenarios prioritizing max recall
    balanced_name = 'Logistic Regression (Balanced)'
    balanced_model = trained_models[balanced_name]
    balanced_bundle = {
        'model_name': balanced_name,
        'model': balanced_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': [m for m in comparison_records if m['Model'] == balanced_name][0],
        'high_risk_threshold': 0.70,
        'moderate_risk_threshold': 0.40
    }
    balanced_path = os.path.join("models", "recall_optimized_churn_model.pkl")
    joblib.dump(balanced_bundle, balanced_path)
    print(f"[+] Recall-prioritized model artifact ({balanced_name}) exported to: {balanced_path}")
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
