from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.preprocessing import preprocess_record

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_PATH = Path(os.getenv("MODEL_ARTIFACTS_PATH", BASE_DIR / "artifacts" / "model_artifacts.joblib"))
PREDICTION_LOG_PATH = os.getenv("PREDICTION_LOG_PATH")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("churn_api")

app = FastAPI(
    title="Thera Bank Churn Prediction API",
    description="Predicts credit-card churn probability and returns SHAP-based reasons.",
    version="2.0.0",
)

_artifacts = None
_explainer = None
_load_lock = Lock()

REQUEST_COUNT = Counter("churn_api_requests_total", "API requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("churn_api_request_latency_seconds", "API request latency", ["path"])
PREDICTION_COUNT = Counter("churn_predictions_total", "Predictions by class", ["churn_flag"])
PREDICTION_PROB = Histogram(
    "churn_probability",
    "Predicted churn probability",
    buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0),
)


def get_artifacts():
    global _artifacts, _explainer
    if _artifacts is None:
        with _load_lock:
            if _artifacts is None:
                _artifacts = joblib.load(ARTIFACTS_PATH)
                _explainer = shap.TreeExplainer(_artifacts["model"])
    return _artifacts, _explainer


class CustomerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Customer_Age: int = Field(..., ge=18, le=100)
    Gender: Literal["M", "F"]
    Dependent_count: int = Field(..., ge=0, le=10)
    Education_Level: Literal[
        "High School", "Graduate", "Uneducated", "College", "Post-Graduate", "Doctorate", "Unknown"
    ]
    Marital_Status: Literal["Married", "Single", "Divorced", "Unknown"]
    Income_Category: Literal[
        "Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"
    ]
    Card_Category: Literal["Blue", "Silver", "Gold", "Platinum"]
    Months_on_book: int = Field(..., ge=0, le=120)
    Total_Relationship_Count: int = Field(..., ge=1, le=10)
    Months_Inactive_12_mon: int = Field(..., ge=0, le=12)
    Contacts_Count_12_mon: int = Field(..., ge=0, le=20)
    Credit_Limit: float = Field(..., gt=0, le=1_000_000)
    Total_Revolving_Bal: float = Field(..., ge=0, le=1_000_000)
    Total_Amt_Chng_Q4_Q1: float = Field(..., ge=0, le=20)
    Total_Trans_Amt: float = Field(..., ge=0, le=10_000_000)
    Total_Trans_Ct: int = Field(..., ge=0, le=10_000)
    Total_Ct_Chng_Q4_Q1: float = Field(..., ge=0, le=20)
    Avg_Utilization_Ratio: float = Field(..., ge=0, le=1)

    @model_validator(mode="after")
    def validate_financial_relationships(self):
        if self.Total_Revolving_Bal > self.Credit_Limit:
            raise ValueError("Total_Revolving_Bal cannot exceed Credit_Limit")
        return self


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_flag: bool
    threshold_used: float
    model_name: str
    top_reasons: list[dict]


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    with REQUEST_LATENCY.labels(path=request.url.path).time():
        try:
            response = await call_next(request)
        except Exception:
            REQUEST_COUNT.labels(method=request.method, path=request.url.path, status="500").inc()
            raise
    REQUEST_COUNT.labels(method=request.method, path=request.url.path, status=str(response.status_code)).inc()
    return response


def _write_prediction_log(record: dict, probability: float, churn_flag: bool, threshold: float) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **record,
        "churn_probability": probability,
        "churn_flag": churn_flag,
        "threshold": threshold,
    }
    logger.info("prediction=%s", json.dumps(payload, default=str))
    if PREDICTION_LOG_PATH:
        path = Path(PREDICTION_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")


@app.get("/")
def root():
    return {"service": "Thera Bank Churn Prediction API", "version": "2.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    try:
        artifacts, _ = get_artifacts()
        return {
            "status": "ok",
            "model": type(artifacts["model"]).__name__,
            "artifact_path": str(ARTIFACTS_PATH.name),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {exc}") from exc


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(
    record: CustomerRecord,
    threshold: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
):
    artifacts, explainer = get_artifacts()
    model = artifacts["model"]
    used_threshold = float(threshold if threshold is not None else artifacts["cost_optimal_threshold"])

    record_dict = record.model_dump()
    X = preprocess_record(record_dict, artifacts["training_columns"], artifacts["cat_cols"])

    probability = float(model.predict_proba(X)[:, 1][0])
    churn_flag = probability >= used_threshold

    shap_values = explainer.shap_values(X)
    arr = np.asarray(shap_values)
    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[1])[0]
    elif arr.ndim == 3:
        sv = arr[0, :, 1]
    else:
        sv = arr[0]

    contributions = pd.Series(sv, index=X.columns).sort_values(key=np.abs, ascending=False)
    top_reasons = [
        {
            "feature": feature,
            "shap_value": round(float(value), 4),
            "pushes_toward": "churn" if value > 0 else "retention",
        }
        for feature, value in contributions.head(3).items()
    ]

    PREDICTION_COUNT.labels(churn_flag=str(bool(churn_flag)).lower()).inc()
    PREDICTION_PROB.observe(probability)
    _write_prediction_log(record_dict, probability, bool(churn_flag), used_threshold)

    return PredictionResponse(
        churn_probability=round(probability, 4),
        churn_flag=bool(churn_flag),
        threshold_used=used_threshold,
        model_name=type(model).__name__,
        top_reasons=top_reasons,
    )
