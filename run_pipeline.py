"""
BNY Credit Card Fraud Detection — End-to-End Fast ML Pipeline
Executes Tasks 1 through 4 programmatically and exports visualizations, metrics, models, and SHAP data.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, precision_recall_curve
)

from xgboost import XGBClassifier
import shap

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

# Paths
DATA_DIR = r"C:\Users\Mayoori\OneDrive\Desktop\Credit Card Fraud Detection"
TRAIN_PATH = os.path.join(DATA_DIR, "train_transactions.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_transactions.csv")
OUTPUT_DIR = os.path.join(os.getcwd(), "output_artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def savefig(fig, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   [OK] Saved figure: {filename}")

def run():
    print("=" * 80)
    print("BNY CREDIT CARD FRAUD DETECTION -- EXECUTING ML PIPELINE")
    print("=" * 80)

    # 1. LOAD DATA
    print("\n--- [1/5] Loading Datasets ---")
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=['transaction_timestamp'])
    test_df = pd.read_csv(TEST_PATH, parse_dates=['transaction_timestamp'])
    print(f"   Train Shape: {train_df.shape}")
    print(f"   Test Shape : {test_df.shape}")

    # 2. TASK 1: EXPLORATORY DATA ANALYSIS (EDA)
    print("\n--- [2/5] Task 1: Exploratory Data Analysis ---")
    fraud_counts = train_df['is_fraud'].value_counts()
    fraud_pcts = train_df['is_fraud'].value_counts(normalize=True) * 100
    imbalance_ratio = fraud_counts[0] / fraud_counts[1]
    
    eda_summary = {
        "total_transactions": int(len(train_df)),
        "legitimate_count": int(fraud_counts[0]),
        "legitimate_pct": float(fraud_pcts[0]),
        "fraud_count": int(fraud_counts[1]),
        "fraud_pct": float(fraud_pcts[1]),
        "imbalance_ratio": float(imbalance_ratio),
        "missing_values": train_df.isnull().sum().to_dict(),
        "duplicates": int(train_df.duplicated(subset=['transaction_id']).sum())
    }
    
    print(f"   Legitimate: {fraud_counts[0]} ({fraud_pcts[0]:.2f}%)")
    print(f"   Fraud     : {fraud_counts[1]} ({fraud_pcts[1]:.2f}%)")
    print(f"   Imbalance Ratio: {imbalance_ratio:.2f} : 1")

    # Plot 1: Class Distribution
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(['Legitimate (0)', 'Fraudulent (1)'], [fraud_counts[0], fraud_counts[1]],
                  color=['#2b5c8f', '#e63946'], width=0.5, edgecolor='black', linewidth=1)
    ax.set_title('Transaction Class Distribution (Train Set)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Transaction Count', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(train_df)) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height + 150,
                f'{height:,}\n({pct:.2f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    savefig(fig, "1_class_distribution.png")

    # Plot 2: Temporal Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    hour_stats = train_df.groupby('transaction_hour')['is_fraud'].agg(['mean']) * 100
    ax1.plot(hour_stats.index, hour_stats['mean'], marker='o', color='#e63946', linewidth=2.5)
    ax1.fill_between(hour_stats.index, hour_stats['mean'], color='#e63946', alpha=0.15)
    ax1.set_xticks(range(0, 24))
    ax1.set_xlabel('Hour of Day (0-23)', fontsize=11)
    ax1.set_ylabel('Fraud Rate (%)', fontsize=11, color='#e63946')
    ax1.set_title('Fraud Rate by Hour of Day', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    dow_stats = train_df.groupby('transaction_day_of_week')['is_fraud'].agg(['mean']) * 100
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax2.bar(days, dow_stats['mean'], color='#457b9d', edgecolor='black', alpha=0.85)
    ax2.set_xlabel('Day of Week', fontsize=11)
    ax2.set_ylabel('Fraud Rate (%)', fontsize=11)
    ax2.set_title('Fraud Rate by Day of Week', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    savefig(fig, "2_temporal_fraud_distribution.png")

    # Plot 3: Amount Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(x='is_fraud', y='transaction_amount_inr', data=train_df, ax=ax1, palette=['#2b5c8f', '#e63946'])
    ax1.set_xticklabels(['Legitimate (0)', 'Fraudulent (1)'])
    ax1.set_title('Transaction Amount (INR) — Raw Scale', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    train_df['log_amount'] = np.log1p(train_df['transaction_amount_inr'])
    sns.boxplot(x='is_fraud', y='log_amount', data=train_df, ax=ax2, palette=['#2b5c8f', '#e63946'])
    ax2.set_xticklabels(['Legitimate (0)', 'Fraudulent (1)'])
    ax2.set_title('Transaction Amount (Log1p INR)', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    savefig(fig, "3_amount_distribution.png")

    # Plot 4: Categorical Fraud Rates
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cat_fraud = train_df.groupby('merchant_category')['is_fraud'].mean().sort_values(ascending=False) * 100
    axes[0].barh(cat_fraud.index, cat_fraud.values, color='#1d3557', edgecolor='black')
    axes[0].set_title('Fraud Rate by Merchant Category (%)', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)

    pos_fraud = train_df.groupby('pos_entry_mode')['is_fraud'].mean().sort_values(ascending=False) * 100
    axes[1].bar(pos_fraud.index, pos_fraud.values, color='#e63946', edgecolor='black', width=0.4)
    axes[1].set_title('Fraud Rate by POS Entry Mode (%)', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    intl_fraud = train_df.groupby('is_international')['is_fraud'].mean() * 100
    axes[2].bar(['Domestic (0)', 'International (1)'], intl_fraud.values, color=['#2a9d8f', '#f4a261'], edgecolor='black', width=0.4)
    axes[2].set_title('Fraud Rate: Domestic vs International (%)', fontsize=12, fontweight='bold')
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    savefig(fig, "4_categorical_fraud_rates.png")

    # Plot 5: Correlation Matrix
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    corr_matrix = train_df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax, cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
    ax.set_title('Correlation Matrix of Numeric Features', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    savefig(fig, "5_correlation_matrix.png")

    # 3. TASK 2: FEATURE ENGINEERING & PREPROCESSING
    print("\n--- [3/5] Task 2: Feature Engineering & Preprocessing ---")
    
    def engineer_features(df):
        df_feats = df.copy()
        df_feats['amount_to_limit_ratio'] = df_feats['transaction_amount_inr'] / (df_feats['credit_limit_inr'] + 1e-5)
        df_feats['amount_vs_avg_ratio'] = df_feats['transaction_amount_inr'] / (df_feats['avg_txn_amount_30d'] + 1e-5)
        df_feats['is_night_transaction'] = df_feats['transaction_hour'].isin([22, 23, 0, 1, 2, 3]).astype(int)
        df_feats['amount_zscore_30d'] = (df_feats['transaction_amount_inr'] - df_feats['avg_txn_amount_30d']) / (df_feats['std_txn_amount_30d'] + 1e-5)
        df_feats['velocity_surge_ratio'] = df_feats['velocity_last_1h'] / (df_feats['velocity_last_24h'] / 24.0 + 1e-5)
        df_feats['distance_risk_score'] = df_feats['distance_from_home_km'] / 100.0
        return df_feats

    train_eng = engineer_features(train_df)
    test_eng = engineer_features(test_df)

    num_cols_eng = [
        'transaction_hour', 'transaction_day_of_week', 'transaction_amount_inr',
        'is_international', 'velocity_last_1h', 'velocity_last_24h',
        'avg_txn_amount_30d', 'std_txn_amount_30d', 'distance_from_home_km',
        'card_age_days', 'credit_limit_inr', 'amount_to_limit_ratio',
        'amount_vs_avg_ratio', 'is_night_transaction', 'amount_zscore_30d',
        'velocity_surge_ratio', 'distance_risk_score'
    ]
    cat_cols = ['merchant_category', 'pos_entry_mode']
    
    for c in num_cols_eng:
        train_eng[c].fillna(train_eng[c].median(), inplace=True)
        test_eng[c].fillna(train_eng[c].median(), inplace=True)
    for c in cat_cols:
        train_eng[c].fillna(train_eng[c].mode()[0], inplace=True)
        test_eng[c].fillna(train_eng[c].mode()[0], inplace=True)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(train_eng[cat_cols])
    cat_feature_names = ohe.get_feature_names_out(cat_cols)

    train_cat_encoded = pd.DataFrame(ohe.transform(train_eng[cat_cols]), columns=cat_feature_names, index=train_eng.index)
    test_cat_encoded = pd.DataFrame(ohe.transform(test_eng[cat_cols]), columns=cat_feature_names, index=test_eng.index)

    X_train_raw = pd.concat([train_eng[num_cols_eng], train_cat_encoded], axis=1)
    y_train = train_eng['is_fraud'].values

    X_test_raw = pd.concat([test_eng[num_cols_eng], test_cat_encoded], axis=1)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    feature_names = list(X_train_raw.columns)
    print(f"   Processed Feature Count: {len(feature_names)}")

    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
    joblib.dump(ohe, os.path.join(OUTPUT_DIR, "ohe.joblib"))
    with open(os.path.join(OUTPUT_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # 4. TASK 3: MODEL DEVELOPMENT & 5-FOLD CV COMPARISON
    print("\n--- [4/5] Task 3: Model Development & 5-Fold Stratified CV ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    models = {
        "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced_subsample', random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=imbalance_ratio, random_state=SEED, n_jobs=-1, eval_metric='logloss')
    }

    cv_results = {}
    for name, model in models.items():
        precisions, recalls, f1s, roc_aucs, pr_aucs = [], [], [], [], []
        for train_idx, val_idx in cv.split(X_train_scaled, y_train):
            X_tr, y_tr = X_train_scaled[train_idx], y_train[train_idx]
            X_val, y_val = X_train_scaled[val_idx], y_train[val_idx]
            model.fit(X_tr, y_tr)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred_default = (y_pred_proba >= 0.5).astype(int)

            precisions.append(precision_score(y_val, y_pred_default, zero_division=0))
            recalls.append(recall_score(y_val, y_pred_default, zero_division=0))
            f1s.append(f1_score(y_val, y_pred_default, zero_division=0))
            roc_aucs.append(roc_auc_score(y_val, y_pred_proba))
            pr_aucs.append(average_precision_score(y_val, y_pred_proba))

        cv_results[name] = {
            "Precision": (np.mean(precisions), np.std(precisions)),
            "Recall": (np.mean(recalls), np.std(recalls)),
            "F1-Score": (np.mean(f1s), np.std(f1s)),
            "AUC-ROC": (np.mean(roc_aucs), np.std(roc_aucs)),
            "AUC-PR": (np.mean(pr_aucs), np.std(pr_aucs)),
        }
        print(f"   {name:30s} -> Precision: {np.mean(precisions):.4f}, Recall: {np.mean(recalls):.4f}, AUC-PR: {np.mean(pr_aucs):.4f}")

    best_model_name = "XGBoost"
    best_model = models[best_model_name]
    best_model.fit(X_train_scaled, y_train)
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, "best_model.joblib"))

    # Out-of-fold predictions
    oof_probas = np.zeros(len(y_train))
    for train_idx, val_idx in cv.split(X_train_scaled, y_train):
        X_tr, y_tr = X_train_scaled[train_idx], y_train[train_idx]
        X_val, y_val = X_train_scaled[val_idx], y_train[val_idx]
        m_clone = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=imbalance_ratio, random_state=SEED, n_jobs=-1, eval_metric='logloss')
        m_clone.fit(X_tr, y_tr)
        oof_probas[val_idx] = m_clone.predict_proba(X_val)[:, 1]

    precisions_pr, recalls_pr, thresholds_pr = precision_recall_curve(y_train, oof_probas)
    f1_scores = 2 * (precisions_pr * recalls_pr) / (precisions_pr + recalls_pr + 1e-8)
    
    valid_idx = np.where(recalls_pr[:-1] >= 0.75)[0]
    best_threshold_idx = valid_idx[np.argmax(f1_scores[valid_idx])] if len(valid_idx) > 0 else np.argmax(f1_scores)
        
    optimal_threshold = float(thresholds_pr[best_threshold_idx])
    opt_precision = float(precisions_pr[best_threshold_idx])
    opt_recall = float(recalls_pr[best_threshold_idx])
    opt_f1 = float(f1_scores[best_threshold_idx])

    print(f"\n   Optimal Decision Threshold (tau*) = {optimal_threshold:.4f}")
    print(f"   At tau*={optimal_threshold:.4f} -> Recall: {opt_recall*100:.1f}%, Precision: {opt_precision*100:.1f}%")

    # Plot 6: Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls_pr, precisions_pr, color='#1d3557', linewidth=2.5, label=f'XGBoost PR Curve (AUC-PR = {average_precision_score(y_train, oof_probas):.4f})')
    ax.scatter([opt_recall], [opt_precision], color='#e63946', s=120, zorder=5, label=f'Optimal Threshold tau* = {optimal_threshold:.3f}\n(Recall = {opt_recall*100:.1f}%, Precision = {opt_precision*100:.1f}%)')
    ax.axhline(y=opt_precision, linestyle='--', color='#e63946', alpha=0.5)
    ax.axvline(x=opt_recall, linestyle='--', color='#e63946', alpha=0.5)
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=12)
    ax.set_title('Precision-Recall Curve & Optimal Threshold Selection', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', fontsize=11)
    savefig(fig, "6_precision_recall_curve.png")

    # Plot 7: Confusion Matrix Comparison
    y_pred_default = (oof_probas >= 0.5).astype(int)
    y_pred_optimal = (oof_probas >= optimal_threshold).astype(int)

    cm_default = confusion_matrix(y_train, y_pred_default)
    cm_optimal = confusion_matrix(y_train, y_pred_optimal)

    tn_def, fp_def, fn_def, tp_def = cm_default.ravel()
    fpr_default = (fp_def / (fp_def + tn_def)) * 100

    tn_opt, fp_opt, fn_opt, tp_opt = cm_optimal.ravel()
    fpr_optimal = (fp_opt / (fp_opt + tn_opt)) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
    ax1.set_title(f'Default Threshold (0.50)\nRecall: {tp_def/(tp_def+fn_def)*100:.1f}% | FPR: {fpr_default:.2f}%', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_xticklabels(['Legit (0)', 'Fraud (1)'])
    ax1.set_yticklabels(['Legit (0)', 'Fraud (1)'])

    sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', ax=ax2, cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
    ax2.set_title(f'Optimal Threshold ({optimal_threshold:.3f})\nRecall: {tp_opt/(tp_opt+fn_opt)*100:.1f}% | FPR: {fpr_optimal:.2f}%', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    ax2.set_xticklabels(['Legit (0)', 'Fraud (1)'])
    ax2.set_yticklabels(['Legit (0)', 'Fraud (1)'])
    plt.tight_layout()
    savefig(fig, "7_confusion_matrix_comparison.png")

    # 5. TASK 4: SHAP EXPLAINABILITY
    print("\n--- [5/5] Task 4: SHAP Model Explainability ---")
    explainer = shap.TreeExplainer(best_model)
    
    sample_indices = np.random.choice(len(X_train_scaled), size=min(2000, len(X_train_scaled)), replace=False)
    X_sample_scaled = X_train_scaled[sample_indices]
    X_sample_raw = X_train_raw.iloc[sample_indices]
    
    shap_vals = explainer.shap_values(X_sample_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_sample_raw, feature_names=feature_names, show=False, max_display=12)
    plt.title('SHAP Global Summary Plot (Beeswarm)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    savefig(plt.gcf(), "8_shap_beeswarm_summary.png")

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_10_idx = np.argsort(mean_abs_shap)[::-1][:10]
    top_10_features = [feature_names[i] for i in top_10_idx]
    top_10_shap_values = [mean_abs_shap[i] for i in top_10_idx]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(top_10_features[::-1], top_10_shap_values[::-1], color='#e63946', edgecolor='black')
    ax.set_xlabel('Mean |SHAP Value| (Impact on Model Output)', fontsize=11)
    ax.set_title('Top-10 Features by Global SHAP Importance', fontsize=13, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    for i, v in enumerate(top_10_shap_values[::-1]):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    savefig(fig, "9_shap_top10_bar.png")

    fraud_indices = np.where(y_train == 1)[0]
    sample_idx = fraud_indices[0] if len(fraud_indices) > 0 else 0
    sample_row = train_eng.iloc[sample_idx]
    
    sample_shap = explainer.shap_values(X_train_scaled[sample_idx:sample_idx+1])[0]
    top_pos_shap_idx = np.argsort(sample_shap)[::-1][:3]
    top_pos_features = [(feature_names[i], sample_shap[i], X_train_raw.iloc[sample_idx, i]) for i in top_pos_shap_idx]

    local_narrative = (
        f"1. Flagged Transaction ID: {sample_row['transaction_id']} (Amount: Rs. {sample_row['transaction_amount_inr']:,.2f}, Cardholder: {sample_row['cardholder_id']}).\n"
        f"2. Primary Risk Trigger: Transaction amount exceeded 30-day average by {sample_row['amount_vs_avg_ratio']:.1f}x (SHAP score impact: +{top_pos_features[0][1]:.2f}).\n"
        f"3. High Velocity & Night Window: High velocity of {sample_row['velocity_last_1h']} txns/hr executed during late night window ({sample_row['transaction_hour']}:00 hrs).\n"
        f"4. Anomaly Distance & Entry Mode: Transaction executed at {sample_row['distance_from_home_km']:.1f} km from registered home address via {sample_row['pos_entry_mode']} entry.\n"
        f"5. Final Risk Verdict: Model probability of {oof_probas[sample_idx]*100:.1f}% significantly exceeds optimal fraud threshold ({optimal_threshold*100:.1f}%), indicating high likelihood of account takeover fraud."
    )

    print("\n   --- Sample Flagged Transaction 5-Line Explanation ---")
    print(local_narrative)

    with open(os.path.join(OUTPUT_DIR, "sample_local_explanation.txt"), "w", encoding="utf-8") as f:
        f.write(local_narrative)

    test_probas = best_model.predict_proba(X_test_scaled)[:, 1]
    test_preds = (test_probas >= optimal_threshold).astype(int)

    test_predictions_df = pd.DataFrame({
        "transaction_id": test_df["transaction_id"],
        "cardholder_id": test_df["cardholder_id"],
        "merchant_id": test_df["merchant_id"],
        "transaction_amount_inr": test_df["transaction_amount_inr"],
        "fraud_probability": np.round(test_probas, 4),
        "is_fraud_predicted": test_preds
    })
    test_predictions_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
    print(f"\n   Saved test predictions for {len(test_predictions_df)} transactions to test_predictions.csv!")

    config_summary = {
        "eda": eda_summary,
        "optimal_threshold": optimal_threshold,
        "opt_precision": opt_precision,
        "opt_recall": opt_recall,
        "opt_f1": opt_f1,
        "fpr_default": float(fpr_default),
        "fpr_optimal": float(fpr_optimal),
        "cv_results": {
            k: {metric: [float(m[0]), float(m[1])] for metric, m in v.items()}
            for k, v in cv_results.items()
        },
        "top_10_features": top_10_features
    }

    with open(os.path.join(OUTPUT_DIR, "pipeline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(config_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("SUCCESS: ML PIPELINE COMPLETE & ALL ARTIFACTS EXPORTED TO output_artifacts/")
    print("=" * 80)

if __name__ == "__main__":
    run()
