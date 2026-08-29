# Exact next steps after successful retraining

The leakage-safe P0 training run is complete and the refreshed model artifacts are already in `artifacts/`.

## 1. Create a clean local environment

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -r requirements.txt
```

## 2. Reproduce the training run locally

```powershell
python src/train.py --data data/BankChurners.csv --output-dir artifacts
```

Compare the new `artifacts/training_report.json` with `docs/TRAINING_RESULTS.md`. With the fixed random state and compatible library versions, results should be close/reproducible.

## 3. Inspect MLflow

Training code already logs the dataset hash, imbalance strategy, hyperparameters, CV/validation/test metrics, model artifact, training report, and reference profile when MLflow is installed.

```powershell
mlflow ui
```

Confirm nested runs for `smote`, `class_weight`, and `scale_pos_weight`.

## 4. Run tests

```powershell
python -m pytest -q
```

Expected baseline: **10 tests passing**.

## 5. Run the API

```powershell
$env:PREDICTION_LOG_PATH="prediction_logs/predictions.jsonl"
uvicorn api.app:app --reload --port 8000
```

Verify `/health`, `/predict`, `/docs`, and `/metrics`.

## 6. Build Docker locally

```powershell
docker build -f api/Dockerfile -t churn-api .
docker run --rm -p 8000:8000 churn-api
```

## 7. Push to GitHub

GitHub Actions will run Ruff, the automated tests, and the Docker build.

## 8. Deploy publicly

For the fastest portfolio deployment, connect the repository to Render. `render.yaml` is included. Verify the public `/health`, `/docs`, and `/metrics` endpoints, then add the public `/docs` URL to the README/resume.

## 9. Generate real drift evidence

After collecting real prediction logs:

```powershell
python monitoring/drift.py --reference artifacts/reference_profile.json --predictions prediction_logs/predictions.jsonl
```

Do not manufacture production drift data just for the portfolio; use actual inference traffic or clearly labelled replay data.

## 10. Temporal validation

The current dataset has no event/prediction timestamp suitable for honest out-of-time validation, so this remains a documented data limitation rather than a fabricated split. See `docs/TEMPORAL_VALIDATION.md`.
