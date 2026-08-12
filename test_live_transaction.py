"""
BNY Credit Card Fraud Detection — Interactive Live Predictor Tool
Simulates real-time transaction scoring against trained XGBoost model & SHAP engine.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

OUTPUT_DIR = os.path.join(os.getcwd(), "output_artifacts")
DATA_DIR = r"C:\Users\Mayoori\OneDrive\Desktop\Credit Card Fraud Detection"

def predict_custom_transaction(amount, avg_30d, hour, velocity_1h, distance_km, pos_mode="CNP"):
    model_path = os.path.join(OUTPUT_DIR, "best_model.joblib")
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
    ohe_path = os.path.join(OUTPUT_DIR, "ohe.joblib")
    feats_path = os.path.join(OUTPUT_DIR, "feature_names.json")
    summary_path = os.path.join(OUTPUT_DIR, "pipeline_summary.json")

    if not os.path.exists(model_path):
        print("Error: Trained model not found in output_artifacts/!")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    ohe = joblib.load(ohe_path)
    with open(feats_path, "r") as f:
        feature_names = json.load(f)
    with open(summary_path, "r") as f:
        summary = json.load(f)

    tau_star = summary.get("optimal_threshold", 0.2620)

    # Construct feature dataframe
    sample_df = pd.DataFrame([{
        "transaction_hour": hour,
        "transaction_day_of_week": 3,
        "transaction_amount_inr": amount,
        "is_international": 0,
        "velocity_last_1h": velocity_1h,
        "velocity_last_24h": velocity_1h * 3,
        "avg_txn_amount_30d": avg_30d,
        "std_txn_amount_30d": avg_30d * 0.3,
        "distance_from_home_km": distance_km,
        "card_age_days": 500,
        "credit_limit_inr": 150000,
        "amount_to_limit_ratio": amount / 150000.0,
        "amount_vs_avg_ratio": amount / (avg_30d + 1e-5),
        "is_night_transaction": 1 if hour in [22, 23, 0, 1, 2, 3] else 0,
        "amount_zscore_30d": (amount - avg_30d) / (avg_30d * 0.3 + 1e-5),
        "velocity_surge_ratio": velocity_1h / (velocity_1h * 3 / 24.0 + 1e-5),
        "distance_risk_score": distance_km / 100.0,
        "merchant_category": "online_retail",
        "pos_entry_mode": pos_mode
    }])

    num_cols_eng = [
        'transaction_hour', 'transaction_day_of_week', 'transaction_amount_inr',
        'is_international', 'velocity_last_1h', 'velocity_last_24h',
        'avg_txn_amount_30d', 'std_txn_amount_30d', 'distance_from_home_km',
        'card_age_days', 'credit_limit_inr', 'amount_to_limit_ratio',
        'amount_vs_avg_ratio', 'is_night_transaction', 'amount_zscore_30d',
        'velocity_surge_ratio', 'distance_risk_score'
    ]
    cat_cols = ['merchant_category', 'pos_entry_mode']

    cat_encoded = pd.DataFrame(ohe.transform(sample_df[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
    X_raw = pd.concat([sample_df[num_cols_eng], cat_encoded], axis=1)
    X_scaled = scaler.transform(X_raw)

    prob = model.predict_proba(X_scaled)[0, 1]
    is_fraud = prob >= tau_star

    print("\n" + "=" * 70)
    print(f"  REAL-TIME TRANSACTION FRAUD SCORING RESULTS")
    print("=" * 70)
    print(f"  Transaction Amount       : Rs. {amount:,.2f}")
    print(f"  30-Day Average Amount    : Rs. {avg_30d:,.2f}  (Ratio: {amount/avg_30d:.1f}x)")
    print(f"  Execution Window         : {hour}:00 hrs ({'Night Window' if hour in [22,23,0,1,2,3] else 'Day Window'})")
    print(f"  1-Hour Velocity          : {velocity_1h} transaction(s)")
    print(f"  Distance from Registered : {distance_km} km")
    print(f"  POS Entry Mode           : {pos_mode}")
    print("-" * 70)
    print(f"  Model Fraud Probability  : {prob*100:.2f}%")
    print(f"  Optimal Risk Threshold   : {tau_star*100:.2f}%")
    print(f"  FINAL VERDICT            : {'[!] HIGH RISK FRAUD -- BLOCKED' if is_fraud else '[OK] APPROVED'}")
    print("=" * 70)

if __name__ == "__main__":
    print("Test 1: Suspicious Night Transaction")
    predict_custom_transaction(amount=95000, avg_30d=2500, hour=2, velocity_1h=5, distance_km=520, pos_mode="CNP")

    print("\nTest 2: Normal Day-to-Day Transaction")
    predict_custom_transaction(amount=450, avg_30d=500, hour=14, velocity_1h=1, distance_km=8, pos_mode="CHIP")
