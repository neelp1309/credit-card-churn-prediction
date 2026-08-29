# Monitoring

The API exposes Prometheus-format metrics at `/metrics`:

- request count
- request latency
- prediction class count
- predicted-probability histogram

For feature drift, set `PREDICTION_LOG_PATH=prediction_logs/predictions.jsonl` when starting the API. After `src/train.py` has generated `artifacts/reference_profile.json`, run:

```bash
python monitoring/drift.py \
  --reference artifacts/reference_profile.json \
  --predictions prediction_logs/predictions.jsonl
```

The lightweight drift check reports numeric PSI and categorical total-variation distance. In a real deployment, send structured logs/metrics to a managed observability backend instead of relying on local disk.
