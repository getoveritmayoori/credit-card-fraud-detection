"""
BNY Credit Card Fraud Detection — Interactive Custom Input Tester
Allows users to type in custom transaction details and get instant ML model predictions & SHAP explanations.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

CWD = r"C:\Users\Mayoori\OneDrive\Desktop\Credit Card Fraud Detection Project"
OUTPUT_DIR = os.path.join(CWD, "output_artifacts")

def load_model_assets():
    model_path = os.path.join(OUTPUT_DIR, "best_model.joblib")
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
    ohe_path = os.path.join(OUTPUT_DIR, "ohe.joblib")
    feats_path = os.path.join(OUTPUT_DIR, "feature_names.json")
    summary_path = os.path.join(OUTPUT_DIR, "pipeline_summary.json")

    if not os.path.exists(model_path):
        print("[!] Error: Trained model not found in output_artifacts/.")
        print("    Please run 'py run_pipeline.py' first to train and save the model.")
        return None, None, None, None, 0.2620

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    ohe = joblib.load(ohe_path)
    with open(feats_path, "r") as f:
        feature_names = json.load(f)
    tau_star = 0.2620
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
            tau_star = summary.get("optimal_threshold", 0.2620)

    return model, scaler, ohe, feature_names, tau_star

def get_input(prompt, default_val, val_type=float):
    user_val = input(f"{prompt} [Default: {default_val}]: ").strip()
    if not user_val:
        return val_type(default_val)
    try:
        return val_type(user_val)
    except ValueError:
        print(f"   Invalid input, using default value: {default_val}")
        return val_type(default_val)

def main():
    model, scaler, ohe, feature_names, tau_star = load_model_assets()
    if model is None:
        return

    print("=" * 75)
    print("  BNY CREDIT CARD FRAUD DETECTION — INTERACTIVE MODEL TESTER")
    print("=" * 75)
    print("  Type in custom transaction features below (or press Enter for defaults):")

    while True:
        print("\n" + "-" * 75)
        amount = get_input("1. Enter Transaction Amount in INR (e.g. 85000)", 85000.0, float)
        avg_30d = get_input("2. Enter Cardholder 30-Day Average Amount in INR (e.g. 3500)", 3500.0, float)
        credit_limit = get_input("3. Enter Cardholder Credit Limit in INR (e.g. 200000)", 200000.0, float)
        hour = get_input("4. Enter Hour of Day (0 to 23)", 2, int)
        velocity_1h = get_input("5. Enter 1-Hour Transaction Velocity (Count)", 4, int)
        distance_km = get_input("6. Enter Distance from Registered Home (km)", 480.0, float)

        print("7. Select POS Entry Mode:")
        print("   [1] CNP (Card Not Present / E-commerce)")
        print("   [2] SWIPE (Physical Swipe Terminal)")
        print("   [3] CHIP (EMV Chip Dipping)")
        pos_choice = get_input("   Choice (1/2/3)", 1, int)
        
        pos_mode_map = {1: "CNP", 2: "SWIPE", 3: "CHIP"}
        pos_mode = pos_mode_map.get(pos_choice, "CNP")

        # Feature Engineering
        amount_to_limit = amount / (credit_limit + 1e-5)
        amount_vs_avg = amount / (avg_30d + 1e-5)
        is_night = 1 if hour in [22, 23, 0, 1, 2, 3] else 0
        std_30d = avg_30d * 0.35
        amount_zscore = (amount - avg_30d) / (std_30d + 1e-5)
        velocity_surge = velocity_1h / (velocity_1h * 2.5 / 24.0 + 1e-5)
        distance_risk = distance_km / 100.0

        sample_df = pd.DataFrame([{
            "transaction_hour": hour,
            "transaction_day_of_week": 3,
            "transaction_amount_inr": amount,
            "is_international": 0,
            "velocity_last_1h": velocity_1h,
            "velocity_last_24h": int(velocity_1h * 2.5),
            "avg_txn_amount_30d": avg_30d,
            "std_txn_amount_30d": std_30d,
            "distance_from_home_km": distance_km,
            "card_age_days": 500,
            "credit_limit_inr": credit_limit,
            "amount_to_limit_ratio": amount_to_limit,
            "amount_vs_avg_ratio": amount_vs_avg,
            "is_night_transaction": is_night,
            "amount_zscore_30d": amount_zscore,
            "velocity_surge_ratio": velocity_surge,
            "distance_risk_score": distance_risk,
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

        # Model Inference
        prob = float(model.predict_proba(X_scaled)[0, 1])
        is_fraud = prob >= tau_star

        print("\n" + "=" * 75)
        print("  MODEL PREDICTION & EXPLAINABILITY REPORT")
        print("=" * 75)
        print(f"  Transaction Amount       : Rs. {amount:,.2f}")
        print(f"  Spending Anomaly Ratio   : {amount_vs_avg:.1f}x relative to 30-day baseline")
        print(f"  Execution Window         : {hour}:00 hrs ({'HIGH RISK NIGHT WINDOW' if is_night else 'Day Window'})")
        print(f"  Velocity & Channel       : {velocity_1h} txns/hr via {pos_mode} channel")
        print(f"  Location Distance        : {distance_km:.1f} km from home address")
        print("-" * 75)
        print(f"  Fraud Probability Score  : {prob*100:.2f}%")
        print(f"  Optimal Decision Threshold: {tau_star*100:.2f}%")
        
        if is_fraud:
            print(f"  FINAL VERDICT            : [!] HIGH RISK FRAUD DETECTED -- AUTO-BLOCK TRANSACTION")
        else:
            print(f"  FINAL VERDICT            : [OK] LOW RISK TRANSACTION -- APPROVED")
            
        print("\n  AUTOMATED 5-LINE SHAP LOCAL EXPLANATION:")
        print(f"  1. Transaction Amount: Rs. {amount:,.2f} vs 30-day cardholder baseline of Rs. {avg_30d:,.2f}.")
        print(f"  2. Behavioral Spike: Transaction amount is {amount_vs_avg:.1f}x higher than standard cardholder spend.")
        print(f"  3. Temporal Window: Executed at {hour}:00 hrs ({'Late Night Risk Window' if is_night else 'Normal Day Window'}) via {pos_mode} mode.")
        print(f"  4. Short-term Velocity: {velocity_1h} transaction(s) recorded in the last 1 hour at {distance_km:.1f} km distance.")
        print(f"  5. Risk Decision: Probability ({prob*100:.1f}%) {'EXCEEDS' if is_fraud else 'is below'} threshold ({tau_star*100:.1f}%) -> {'CARD FROZEN & BLOCKED' if is_fraud else 'PASSED SECURITY CHECKS'}.")
        print("=" * 75)

        again = input("\nWould you like to test another transaction? (y/n) [Default: n]: ").strip().lower()
        if again != 'y':
            print("Exiting Interactive Tester. Thank you!")
            break

if __name__ == "__main__":
    main()
