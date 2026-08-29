# P0 / P1 completion status

## P0 — must fix

- [x] **Fold-safe SMOTE during tuning** — `src/train.py` puts SMOTE inside `imblearn.Pipeline` used by `RandomizedSearchCV`; the patched notebook does the same.
- [x] **Imbalance alternatives** — production training compares fold-safe SMOTE, LightGBM `class_weight="balanced"`, and `scale_pos_weight` using training-CV recall.
- [x] **Repository reproducibility** — root/API/dev requirements, deterministic random state, explicit dataset location, and a dataset hash in the training report/MLflow run.
- [x] **README/repo cleanup** — README now describes files that actually exist.
- [x] **API validation** — threshold constrained to `[0,1]`, unknown fields rejected, field ranges validated, revolving balance checked against credit limit.
- [x] **Automated tests** — API, validation, preprocessing alignment, metrics, and drift math.

## P1 — strong interview differentiators

- [x] **GitHub Actions CI** — lint + pytest + Docker build on push/PR.
- [x] **MLflow experiment tracking** — nested runs for imbalance strategies, params/metrics/dataset hash, model/reference-profile/report artifacts.
- [x] **Production monitoring hooks** — Prometheus `/metrics`, structured prediction logs, numeric PSI + categorical total-variation drift checks.
- [x] **Cloud deployment config** — Render Docker deployment manifest included.
- [ ] **Public cloud deployment** — requires your Render/AWS/Azure/GCP account; config is ready but no external account was used here.
- [ ] **True temporal validation** — blocked by the dataset because it has no event/prediction timestamp. See `docs/TEMPORAL_VALIDATION.md`; no fake row-order split is used.

## Executed training status

- [x] Exact `BankChurners.csv` supplied and placed at `data/BankChurners.csv`.
- [x] Leakage-safe training executed end-to-end.
- [x] `scale_pos_weight` selected by highest 5-fold training-CV recall (**0.9376**).
- [x] Fresh `model_artifacts.joblib` generated.
- [x] Fresh `reference_profile.json` generated.
- [x] Fresh `training_report.json` generated.
- [x] New artifact validated by **10/10 automated tests**.
- [x] Standard-threshold untouched-test metrics: recall **0.9200**, precision **0.8470**, F1 **0.8820**, ROC-AUC **0.9914**.
- [x] Validation-selected business threshold: **0.09**; untouched-test recall at that operating point: **0.9908**.

See `docs/TRAINING_RESULTS.md` for the comparison and interpretation.
