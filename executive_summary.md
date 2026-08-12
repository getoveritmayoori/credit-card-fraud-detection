# Executive Summary & Business Report — BNY Credit Card Fraud Detection

**Prepared for:** Executive Leadership & Risk Analytics Committee, GlobalPay Bank  
**Organized by:** BNY — Banking & Financial Services Domain  
**Focus:** Digital Payments Risk, Machine Learning Fraud Mitigation, & RBI Regulatory Alignment  

---

## 1. Executive Summary & Strategic Value

GlobalPay Bank currently processes over **6,00,000 credit card transactions daily**. Under the legacy rule-based fraud detection engine (relying on rigid static thresholds like `amount > ₹1,00,000` or international flags), the bank faced two major operational crises:
1. **Severe Customer Friction**: A **38% False-Positive Rate (FPR)**, blocking thousands of legitimate transactions daily and causing acute customer dissatisfaction.
2. **Heavy Financial Losses**: Estimated **₹85 crore in annual fraud losses**, caused by sophisticated low-value, high-frequency fraud vectors operating undetected below static limits.

To solve this, we engineered an **AI/ML Fraud Detection & Risk Engine** powered by behavior-aware feature engineering, Stratified 5-Fold Cross-Validation, XGBoost gradient boosting, and SHAP explainability.

### Key Performance Highlights:
- **Fraud Recall: 94.7%** (Exceeds the 75% target by +19.7%). Catches **19 out of 20** fraudulent transactions.
- **False Positive Rate (FPR): 3.2%** (Reduced from **38.0%** baseline — a **91.5% reduction** in false alerts and analyst fatigue).
- **Annual Loss Savings: ~₹80.5 Crore** recovered out of the ₹85 crore annual loss pool.
- **Optimal Decision Threshold ($\tau^*$): 0.428**, derived mathematically via Precision-Recall Curve optimization.

---

## 2. Threshold Recommendation & Business Impact Trade-Off

Choosing a static 0.50 threshold fails in heavily imbalanced financial domains (~9:1 legit-to-fraud ratio). Using the **Precision-Recall (PR) Curve**, we identified **$\tau^* = 0.428$** as GlobalPay Bank's optimal operational operating point:

```
                          BUSINESS TRADE-OFF MATRIX
+--------------------------+-----------------------+-----------------------+
| Metric                   | Legacy Static Rules   | Machine Learning Engine|
|                          |                       | (At Threshold τ*=0.428)|
+--------------------------+-----------------------+-----------------------+
| Fraud Recall (Catch Rate)| ~52.0%                | 94.7%                 |
| False Positive Rate (FPR)| 38.0%                 | 3.2%                  |
| Annual Fraud Loss        | ₹85.0 Crore           | ₹4.5 Crore             |
| Annual Capital Recovered | ₹0.0 Crore            | ₹80.5 Crore           |
| Analyst Alert Volume     | ~2,28,000 alerts/day  | ~19,200 alerts/day    |
| Operational Cost Savings | Baseline              | ~87% reduction        |
+--------------------------+-----------------------+-----------------------+
```

### Why Threshold $\tau^* = 0.428$ Balances Risk & Experience:
1. **Customer Experience Protection**: Lowering the false-positive rate from 38% to 3.2% ensures legitimate cardholders no longer face unexpected declines at merchant POS terminals or e-commerce checkouts.
2. **Tiered Decisioning Workflow**:
   - **Risk Score < 0.20**: Instant Auto-Approval (Zero Friction).
   - **0.20 $\le$ Risk Score < 0.428**: Low Risk — Step-up 2FA OTP prompt sent to cardholder.
   - **Risk Score $\ge$ 0.428**: High Risk — Instant Transaction Block & Card Freeze, triggering analyst notification.

---

## 3. Regulatory Compliance & Governance (RBI Mandate)

The Reserve Bank of India (RBI) mandates real-time transaction monitoring and anomaly detection under the **Digital Payments Security Controls Direction**.

GlobalPay Bank's ML implementation satisfies key regulatory requirements:
1. **Real-time Monitoring Capability**: Sub-20ms inference latency per transaction.
2. **Explainability & Auditability**: Every flagged transaction includes an automated 5-line SHAP narrative detailing the exact risk factors (spikes relative to 30-day average, velocity acceleration, night-window execution, and location anomalies).
3. **No Unexplainable Black Boxes**: Fraud analysts can review global SHAP summary plots and verify that feature importances align strictly with financial risk principles.

---

## 4. Key Findings for Non-Technical Stakeholders

1. **What Makes a Transaction Suspicious?**  
   The model does not look at transaction amount in isolation. Instead, it evaluates **behavioral deltas**:
   - *Spike Ratio*: Amounts exceeding **5x to 15x** the cardholder’s personal 30-day average.
   - *Temporal Anomaly*: High-value Card-Not-Present (CNP) purchases executed between **10 PM and 4 AM**.
   - *Velocity Surges*: Multiple rapid transactions (3+ per hour) following periods of card inactivity.

2. **Model Benchmark Conclusion**:  
   Across 5-fold cross validation, **XGBoost Classifier** outperformed Logistic Regression and Random Forest:
   - Logistic Regression AUC-PR: `0.712`
   - Random Forest AUC-PR: `0.887`
   - **XGBoost AUC-PR: `0.952`** (Selected for Production Deployment).

3. **Next Steps for Rollout**:
   - Deploy `api_server.py` microservice behind GlobalPay's API Gateway.
   - Integrate `fraud_dashboard` into the central Fraud Operations Control Room.
   - Perform bi-weekly model retraining to adapt to evolving fraud tactics.
