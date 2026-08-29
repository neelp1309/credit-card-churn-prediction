# Thera Bank — Credit Card Churn Prediction

Production-oriented credit-card churn project demonstrating model development, leakage-aware validation, explainability, cost-sensitive decisioning, API serving, automated tests, CI, experiment tracking, and monitoring hooks.

> **Repository status:** P0 is fully implemented and the leakage-safe training path has now been executed on the supplied `BankChurners.csv`. The refreshed model artifact, drift reference profile, and training report are included. P1 engineering hooks (CI, MLflow integration, Prometheus metrics, drift checks, and cloud deployment config) are implemented; a public cloud deployment still requires an external account.

## Why this revision matters

The original analysis had a strong untouched test split, but its hyperparameter search applied SMOTE before `RandomizedSearchCV`. The revised training path fixes that by putting SMOTE **inside each CV fold**. It also separates model tuning, threshold selection, and final testing:

```text
training split
   ↓
CV model/imbalance search
   ↓
validation split → choose business threshold
   ↓
refit selected model on train + validation
   ↓
untouched test → final report once
```

The production training module compares three imbalance approaches using training-CV recall:

1. fold-safe SMOTE
2. LightGBM `class_weight="balanced"`
3. LightGBM `scale_pos_weight`

This is more defensible than choosing SMOTE by default.

## Repository

```text
.
├── api/
│   ├── app.py                    # FastAPI + SHAP + Prometheus metrics
│   ├── preprocessing.py
│   ├── requirements.txt
│   └── Dockerfile
├── artifacts/
│   ├── model_artifacts.joblib    # refreshed leakage-safe model
│   ├── reference_profile.json
│   └── training_report.json
├── data/
│   ├── BankChurners.csv          # supplied training dataset
│   └── README.md
├── docs/
│   ├── architecture_diagram.png
│   ├── P0_P1_STATUS.md
│   ├── TEMPORAL_VALIDATION.md
│   ├── TRAINING_RESULTS.md
│   └── README_V1.md              # archived original README
├── monitoring/
│   ├── drift.py
│   └── README.md
├── notebooks/
│   ├── Credit_Card_Churn_Prediction_P0_FIXED.ipynb
│   ├── analysis_v1_original.ipynb
│   └── analysis_v1_original.html
├── src/
│   └── train.py                  # leakage-safe training + MLflow + artifact/profile generation
├── tests/
├── .github/workflows/ci.yml
├── render.yaml
├── requirements.txt
├── requirements-dev.txt
└── Makefile
```


## Leakage-safe retraining results

The corrected pipeline was executed on **10,127 customers** (1,627 attrited / 8,500 existing). Strategy selection used 5-fold training CV recall only.

| Strategy | CV recall | Validation recall | Validation precision | Validation F1 | Validation ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Fold-safe SMOTE | 0.9135 | 0.9004 | 0.8801 | 0.8902 | 0.9922 |
| `class_weight="balanced"` | 0.9347 | **0.9387** | 0.8507 | **0.8925** | 0.9920 |
| `scale_pos_weight` | **0.9376** | 0.9349 | 0.8531 | 0.8921 | **0.9923** |

**Selected strategy:** `scale_pos_weight`.

On the untouched test set at threshold **0.50**: **Recall 0.9200, Precision 0.8470, F1 0.8820, ROC-AUC 0.9914**.

The validation-only business-cost search selected threshold **0.09**. At that operating point the untouched test set gives **Recall 0.9908, Precision 0.5620, F1 0.7171**. The low threshold is intentional because the configured false-negative cost (20,000) is 40x the false-positive cost (500).

See `docs/TRAINING_RESULTS.md` and `artifacts/training_report.json` for the full reproducible results.

## P0 improvements implemented

- SMOTE runs inside CV folds during tuning.
- Imbalance strategy comparison includes class weighting and `scale_pos_weight`.
- Business threshold is selected on validation data rather than optimized on the final test set.
- API threshold is constrained to `[0, 1]`.
- Strong Pydantic field validation and rejection of unknown fields.
- Preprocessing is extracted and tested against the model's exact training-column order.
- Dependencies and repository paths are explicit and reproducible.
- Automated unit/integration tests are included.

## P1 improvements implemented

- **GitHub Actions:** lint, tests, Docker build on every push/PR.
- **MLflow:** logs dataset hash, strategy, hyperparameters, CV/validation/test metrics, model artifact, training report, and reference drift profile.
- **Monitoring:** `/metrics` exposes Prometheus metrics for request volume, latency, prediction counts, and probability distribution.
- **Drift:** structured prediction logs can be compared with the training reference profile using PSI for numeric features and total-variation distance for categorical features.
- **Cloud-ready:** `render.yaml` + Docker deployment configuration are included.

A true temporal validation is **not claimed** because the dataset has no timestamp suitable for an out-of-time feature/label split. See `docs/TEMPORAL_VALIDATION.md`.

## Set up

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
```

The supplied source dataset is included at `data/BankChurners.csv`.

## Train with leakage-safe CV + MLflow

```bash
python src/train.py --data data/BankChurners.csv --output-dir artifacts
```

MLflow uses the default local tracking store. View experiments with:

```bash
mlflow ui
```

Training generates:

```text
artifacts/model_artifacts.joblib
artifacts/reference_profile.json
artifacts/training_report.json
```

The training report contains the chosen imbalance strategy, CV recall, validation threshold/cost, and untouched-test metrics.

## Run tests

```bash
pytest -q
```

After leakage-safe retraining, the refreshed model artifact passed **10/10 automated tests** in the training environment. The model was generated with scikit-learn **1.8.0**, and the runtime dependency files are pinned to the same version for model-persistence compatibility.

## Run the API

```bash
uvicorn api.app:app --reload --port 8000
```

Useful endpoints:

```text
GET  /health
POST /predict
GET  /docs
GET  /metrics
```

Example:

```bash
curl -X POST "http://localhost:8000/predict?threshold=0.25" \
  -H "Content-Type: application/json" \
  -d '{
    "Customer_Age": 45,
    "Gender": "M",
    "Dependent_count": 3,
    "Education_Level": "Graduate",
    "Marital_Status": "Married",
    "Income_Category": "$60K - $80K",
    "Card_Category": "Blue",
    "Months_on_book": 36,
    "Total_Relationship_Count": 3,
    "Months_Inactive_12_mon": 2,
    "Contacts_Count_12_mon": 3,
    "Credit_Limit": 8500.0,
    "Total_Revolving_Bal": 1200.0,
    "Total_Amt_Chng_Q4_Q1": 0.75,
    "Total_Trans_Amt": 4200.0,
    "Total_Trans_Ct": 55,
    "Total_Ct_Chng_Q4_Q1": 0.65,
    "Avg_Utilization_Ratio": 0.14
  }'
```

Invalid thresholds such as `threshold=2` return HTTP **422** rather than silently producing nonsensical decisions.

## Docker

```bash
docker build -f api/Dockerfile -t churn-api .
docker run --rm -p 8000:8000 churn-api
```

The CI workflow repeats the Docker build on GitHub. Docker itself was not available in the execution environment used to prepare this revision, so the local image build could not be executed here.

## Monitoring and drift

Start the API with an optional JSONL prediction log:

```bash
PREDICTION_LOG_PATH=prediction_logs/predictions.jsonl \
uvicorn api.app:app --port 8000
```

After enough traffic:

```bash
python monitoring/drift.py \
  --reference artifacts/reference_profile.json \
  --predictions prediction_logs/predictions.jsonl
```

For a real deployment, ship logs and Prometheus metrics to managed infrastructure rather than depending on the container filesystem.

## Deployment

`render.yaml` is included for an easy public portfolio deployment. Connect the GitHub repo to Render and deploy the Docker service; `/health` is configured as the health check. The same container can be moved to Azure Container Apps, AWS ECS/App Runner, or GCP Cloud Run later.

## Interview story

The strongest engineering story is no longer simply "AUC ≈ 0.99." It is:

- I identified that recent activity variables make the task closer to detecting ongoing disengagement than long-horizon forecasting.
- I fixed fold-level resampling leakage in hyperparameter tuning.
- I benchmarked imbalance-handling alternatives instead of assuming SMOTE was best.
- I separated CV/model selection, business-threshold selection, and untouched final testing.
- I made the model reproducible and testable behind a validated API.
- I added CI, MLflow tracking, operational metrics, and drift monitoring.
- I explicitly refused to fabricate temporal validation when the source data does not support it.
