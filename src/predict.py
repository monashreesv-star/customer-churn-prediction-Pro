"""
predict.py
Inference & Business Logic Engine for Customer Churn Prediction.
Flags high-risk customers (>70% churn probability) and prescribes retention actions.
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import transform_unseen_data, clean_data


def assign_retention_action(risk_tier: str, contract_type: str, monthly_charges: float) -> str:
    """Recommend business action based on risk tier and customer profile."""
    if risk_tier == "High Risk (>70%)":
        if contract_type == "Month-to-month":
            return "Urgent: Assign Customer Retention Specialist; offer discounted 1-year contract bundle."
        elif monthly_charges > 70.0:
            return "Urgent: Review fiber/premium tier pricing; offer customized loyalty package or bundle discount."
        else:
            return "Urgent: Conduct proactive outreach; offer free service checkup and 20% billing discount for 3 months."
    elif risk_tier == "Moderate Risk (40-70%)":
        return "Priority: Send targeted satisfaction survey; trigger automated email campaign highlighting unused features."
    else:
        return "Normal: Standard billing cycle and ongoing service communications."


def score_customers(
    input_df: pd.DataFrame,
    model_bundle_path: str = "models/best_churn_model.pkl",
    high_risk_threshold: float = 0.70,
    moderate_risk_threshold: float = 0.40
) -> pd.DataFrame:
    """
    Score customers, compute churn probabilities, categorize risk tiers,
    and attach retention actions.
    """
    if not os.path.exists(model_bundle_path):
        raise FileNotFoundError(f"Model artifact not found at {model_bundle_path}. Please run train_models.py first.")
        
    bundle = joblib.load(model_bundle_path)
    model = bundle['model']
    scaler = bundle['scaler']
    feature_names = bundle['feature_names']
    
    # Preserve IDs and relevant metadata if present
    output_df = pd.DataFrame()
    if 'customerID' in input_df.columns:
        output_df['customerID'] = input_df['customerID']
    else:
        output_df['customerID'] = [f"CUST_{i:05d}" for i in range(len(input_df))]
        
    contract_col = input_df['Contract'] if 'Contract' in input_df.columns else pd.Series(["Unknown"] * len(input_df))
    charges_col = input_df['MonthlyCharges'] if 'MonthlyCharges' in input_df.columns else pd.Series([0.0] * len(input_df))
    
    # Preprocess features
    X_processed = transform_unseen_data(input_df, feature_names, scaler)
    
    # Predict probabilities
    probabilities = model.predict_proba(X_processed)[:, 1]
    
    # Attach predictions
    output_df['Contract'] = contract_col.values
    output_df['MonthlyCharges'] = charges_col.values
    output_df['Churn_Probability'] = np.round(probabilities, 4)
    output_df['Predicted_Churn'] = (probabilities >= 0.50).astype(int)
    
    # Risk categorization based on business thresholds
    risk_tiers = []
    actions = []
    for prob, contract, charges in zip(probabilities, contract_col, charges_col):
        if prob >= high_risk_threshold:
            tier = "High Risk (>70%)"
        elif prob >= moderate_risk_threshold:
            tier = "Moderate Risk (40-70%)"
        else:
            tier = "Low Risk (<40%)"
            
        action = assign_retention_action(tier, str(contract), float(charges) if pd.notnull(charges) else 0.0)
        risk_tiers.append(tier)
        actions.append(action)
        
    output_df['Risk_Tier'] = risk_tiers
    output_df['Retention_Action'] = actions
    
    return output_df


def main():
    parser = argparse.ArgumentParser(description="Predict Customer Churn & Flag High-Risk Customers")
    parser.add_argument("--input", type=str, default="data/Telco-Customer-Churn.csv",
                        help="Path to input customer data CSV")
    parser.add_argument("--model", type=str, default="models/best_churn_model.pkl",
                        help="Path to trained model artifact .pkl")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="High-risk probability threshold (default: 0.70)")
    parser.add_argument("--output", type=str, default="reports/customer_churn_predictions.csv",
                        help="Path to save predictions CSV")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of high-risk customers to display in console")
    args = parser.parse_args()
    
    print("=" * 65)
    print("CUSTOMER CHURN RISK SCORING & RETENTION DISPATCHER")
    print("=" * 65)
    print(f"[*] Input Data: {args.input}")
    print(f"[*] Model Artifact: {args.model}")
    print(f"[*] High-Risk Cutoff: {args.threshold * 100:.1f}%")
    
    input_df = pd.read_csv(args.input)
    results_df = score_customers(input_df, args.model, high_risk_threshold=args.threshold)
    
    # Summary metrics
    total_customers = len(results_df)
    high_risk_count = (results_df['Risk_Tier'] == "High Risk (>70%)").sum()
    moderate_risk_count = (results_df['Risk_Tier'] == "Moderate Risk (40-70%)").sum()
    low_risk_count = (results_df['Risk_Tier'] == "Low Risk (<40%)").sum()
    
    print("\n--- RISK TIER SUMMARY ---")
    print(f"Total Evaluated: {total_customers:,} customers")
    print(f"  - High Risk (>70%):     {high_risk_count:5,d} ({100 * high_risk_count / total_customers:.1f}%) -> Immediate Intervention")
    print(f"  - Moderate Risk (40-70%):{moderate_risk_count:5,d} ({100 * moderate_risk_count / total_customers:.1f}%) -> Automated Nurture")
    print(f"  - Low Risk (<40%):      {low_risk_count:5,d} ({100 * low_risk_count / total_customers:.1f}%) -> Retained")
    
    # Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"\n[+] Full prediction records exported to: {args.output}")
    
    # Display top high-risk customers
    print(f"\n--- TOP {args.top_n} HIGHEST RISK CUSTOMERS ---")
    high_risk_sample = results_df.sort_values(by='Churn_Probability', ascending=False).head(args.top_n)
    display_cols = ['customerID', 'Contract', 'MonthlyCharges', 'Churn_Probability', 'Risk_Tier']
    print(high_risk_sample[display_cols].to_string(index=False))


if __name__ == '__main__':
    main()
