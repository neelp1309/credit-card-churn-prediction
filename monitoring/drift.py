"""Lightweight feature/prediction drift check for JSONL prediction logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-6


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected.astype(float), EPS, None)
    actual = np.clip(actual.astype(float), EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def total_variation(expected: dict[str, float], actual: pd.Series) -> float:
    keys = set(expected) | set(actual.index.astype(str))
    return 0.5 * sum(abs(float(expected.get(k, 0.0)) - float(actual.get(k, 0.0))) for k in keys)


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Prediction log is empty")
    return pd.DataFrame(rows)


def evaluate_drift(profile: dict, current: pd.DataFrame) -> dict:
    report = {"numeric": {}, "categorical": {}, "prediction": {}}

    for col, spec in profile.get("numeric", {}).items():
        if col not in current:
            continue
        values = pd.to_numeric(current[col], errors="coerce").dropna().to_numpy()
        if len(values) == 0:
            continue
        edges = np.array([-np.inf, *spec["cuts"], np.inf], dtype=float)
        counts, _ = np.histogram(values, bins=edges)
        actual = counts / max(counts.sum(), 1)
        value = psi(np.asarray(spec["proportions"]), actual)
        report["numeric"][col] = {"psi": value, "status": "alert" if value >= 0.20 else "ok"}

    for col, expected in profile.get("categorical", {}).items():
        if col not in current:
            continue
        actual = current[col].fillna("Unknown").astype(str).value_counts(normalize=True)
        value = total_variation(expected, actual)
        report["categorical"][col] = {
            "total_variation": value,
            "status": "alert" if value >= 0.20 else "ok",
        }

    if "churn_probability" in current:
        report["prediction"] = {
            "count": int(len(current)),
            "mean_probability": float(current["churn_probability"].mean()),
            "p90_probability": float(current["churn_probability"].quantile(0.90)),
            "flag_rate": float(current.get("churn_flag", pd.Series(dtype=float)).astype(float).mean())
            if "churn_flag" in current
            else None,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path("artifacts/reference_profile.json"))
    parser.add_argument("--predictions", type=Path, default=Path("prediction_logs/predictions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("monitoring/drift_report.json"))
    args = parser.parse_args()

    profile = json.loads(args.reference.read_text(encoding="utf-8"))
    current = load_jsonl(args.predictions)
    report = evaluate_drift(profile, current)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
