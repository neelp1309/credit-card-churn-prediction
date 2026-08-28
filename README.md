# Thera Bank — Credit Card Churn Prediction

Predicting which credit-card customers are likely to attrite, with explainability and a cost-sensitive
business framing, so the bank can intervene before a customer closes their account.

## Problem

~16% of Thera Bank's credit card customers have churned. Losing a cardholder costs the bank recurring
fee/interest/interchange revenue, so the goal is to flag at-risk customers early enough for retention
action — not just to classify churn after the fact.

## Data

`BankChurners.csv` — 10,127 customers, 20 features (demographic, account, and transaction behavior) plus
the churn label. Two real data-quality issues were found and corrected during preprocessing that the
original version of this analysis had missed: genuine missing values in `Education_Level` (15%) and
`Marital_Status` (7.4%), and a garbage `"abc"` entry in `Income_Category` — all mapped to an explicit
`"Unknown"` category rather than imputed.

## Approach

1. **EDA** — univariate/bivariate/multivariate analysis of churn drivers, with a documented
   multicollinearity finding (`Credit_Limit` vs `Avg_Open_To_Buy`, r≈0.996).
2. **Preprocessing** — missing-value handling with rationale, no outlier capping (justified: all models
   used are tree-based and split-based, so capping would only risk discarding high-value customers with
   no modeling benefit), `CLIENTNUM` dropped, one collinear feature dropped, one-hot encoding, stratified
   train/test split with no leakage.
3. **Modeling** — 7 classifiers (Decision Tree, Random Forest, Gradient Boosting, AdaBoost, Bagging,
   **XGBoost**, **LightGBM**) compared across three training regimes: original imbalanced data, SMOTE
   oversampling, and random undersampling — evaluated on **churn-class recall/precision/F1**, not
   accuracy, given the ~84/16 class imbalance.
4. **Imbalance decision** — SMOTE carried forward (raises churn recall without discarding majority-class
   training data, unlike undersampling).
5. **Tuning & stacking** — `RandomizedSearchCV` on LightGBM, XGBoost, and Random Forest, plus a
   `StackingClassifier` combining all three with a logistic-regression meta-learner.
6. **Explainability** — SHAP summary plot + individual waterfall plots for a true positive, false
   negative, and false positive.
7. **Business framing** — cost-sensitive threshold analysis translating false negatives/positives into
   rupee cost, plus a calibration check on the resulting probabilities.

## Results

| Model | Recall (churn) | Precision (churn) | F1 (churn) | ROC-AUC |
|---|---|---|---|---|
| **LightGBM (tuned)** | 0.877 | 0.893 | **0.885** | 0.990 |
| XGBoost (tuned) | 0.874 | 0.890 | 0.882 | **0.991** |
| Stacked Ensemble | **0.883** | 0.880 | 0.882 | 0.988 |
| Random Forest (tuned) | 0.849 | 0.860 | 0.854 | 0.984 |

**Final model: tuned LightGBM** — highest F1 on the churn class; catches 88 out of every 100 customers
who actually churn, at 89% precision (roughly 1 in 9 flags is a false alarm).

**Top SHAP drivers of predicted churn:** `Total_Trans_Ct`, `Total_Trans_Amt`, `Total_Revolving_Bal`,
`Total_Relationship_Count`, `Total_Ct_Chng_Q4_Q1` — declining transaction activity and revolving balance
dominate, consistent with the EDA.

## Business impact

Assuming ≈₹20,000 cost per missed churner vs. ≈₹500 per unnecessary retention outreach (illustrative,
adjustable), the cost-minimizing classification threshold sits well below the default 0.50 — the bank
should flag customers as at-risk more readily than an accuracy-tuned default would, since under-flagging
is far more expensive than over-flagging. On the held-out test set, operating at the cost-optimal
threshold instead of 0.50 saves an estimated **₹6.07 lakh** in modeled false-negative/false-positive cost.
Because the unconstrained cost-optimal threshold pushes toward flagging a large share of customers, a
capacity constraint (e.g., "flag the top N% highest-risk customers per month") is recommended alongside
pure cost-minimization for real deployment.

## Repo contents

- `Credit_Card_Churn_Prediction_v2.ipynb` — full executed notebook
- `Credit_Card_Churn_Prediction_v2.html` — HTML export for submission
- `BankChurners.csv` — source data
- `README.md` — this file

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm shap
jupyter nbconvert --to notebook --execute --inplace Credit_Card_Churn_Prediction_v2.ipynb
```

All random states are fixed (`random_state=42`) for reproducibility.
