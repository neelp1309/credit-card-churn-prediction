"""
Thera Bank Churn Prediction API
--------------------------------
Serves the tuned LightGBM churn model built in Credit_Card_Churn_Prediction_v4.ipynb.

Run locally:
    uvicorn app:app --reload --port 8000

Then POST a customer record to /predict (see README for an example payload),
or open http://localhost:8000/docs for interactive Swagger docs.
"""

from typing import Literal
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Thera Bank Churn Prediction API",
    description="Predicts credit-card customer churn probability with SHAP-based reasons.",
    version="1.0.0",
)

ARTIFACTS_PATH = "model_artifacts.joblib"
_artifacts = None
_explainer = None


def get_artifacts():
    global _artifacts, _explainer
    if _artifacts is None:
        _artifacts = joblib.load(ARTIFACTS_PATH)
        _explainer = shap.TreeExplainer(_artifacts["model"])
    return _artifacts, _explainer


class CustomerRecord(BaseModel):
    Customer_Age: int = Field(..., ge=18, le=100, example=45)
    Gender: Literal["M", "F"] = Field(..., example="M")
    Dependent_count: int = Field(..., ge=0, le=10, example=3)
    Education_Level: Literal[
        "High School", "Graduate", "Uneducated", "College",
        "Post-Graduate", "Doctorate", "Unknown"
    ] = Field(..., example="Graduate")
    Marital_Status: Literal["Married", "Single", "Divorced", "Unknown"] = Field(..., example="Married")
    Income_Category: Literal[
        "Less than $40K", "$40K - $60K", "$60K - $80K",
        "$80K - $120K", "$120K +", "Unknown"
    ] = Field(..., example="$60K - $80K")
    Card_Category: Literal["Blue", "Silver", "Gold", "Platinum"] = Field(..., example="Blue")
    Months_on_book: int = Field(..., ge=0, le=120, example=36)
    Total_Relationship_Count: int = Field(..., ge=1, le=10, example=3)
    Months_Inactive_12_mon: int = Field(..., ge=0, le=12, example=2)
    Contacts_Count_12_mon: int = Field(..., ge=0, le=20, example=3)
    Credit_Limit: float = Field(..., gt=0, example=8500.0)
    Total_Revolving_Bal: float = Field(..., ge=0, example=1200.0)
    Total_Amt_Chng_Q4_Q1: float = Field(..., ge=0, example=0.75)
    Total_Trans_Amt: float = Field(..., ge=0, example=4200.0)
    Total_Trans_Ct: int = Field(..., ge=0, example=55)
    Total_Ct_Chng_Q4_Q1: float = Field(..., ge=0, example=0.65)
    Avg_Utilization_Ratio: float = Field(..., ge=0, le=1, example=0.14)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_flag: bool
    threshold_used: float
    top_reasons: list[dict]


def preprocess(record: CustomerRecord, training_columns: list, cat_cols: list) -> pd.DataFrame:
    row = record.dict()
    df = pd.DataFrame([row])

    # Avg_Open_To_Buy is dropped in training (collinear with Credit_Limit) — never required as input.
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Align to the exact column set/order the model was trained on;
    # any dummy column not present for this record (e.g. a category not hit) is filled with 0.
    df_aligned = df_encoded.reindex(columns=training_columns, fill_value=0)
    return df_aligned


@app.get("/")
def root():
    return {
        "service": "Thera Bank Churn Prediction API",
        "docs": "/docs",
        "predict_endpoint": "/predict",
    }


@app.get("/health")
def health():
    try:
        get_artifacts()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")


@app.post("/predict", response_model=PredictionResponse)
def predict(record: CustomerRecord, threshold: float | None = None):
    artifacts, explainer = get_artifacts()
    model = artifacts["model"]
    training_columns = artifacts["training_columns"]
    cat_cols = artifacts["cat_cols"]
    used_threshold = threshold if threshold is not None else artifacts["cost_optimal_threshold"]

    X = preprocess(record, training_columns, cat_cols)

    proba = float(model.predict_proba(X)[:, 1][0])
    churn_flag = proba >= used_threshold

    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    elif np.asarray(shap_values).ndim == 3:
        sv = np.asarray(shap_values)[0, :, 1]
    else:
        sv = np.asarray(shap_values)[0]

    contributions = pd.Series(sv, index=X.columns).sort_values(key=np.abs, ascending=False)
    top_reasons = [
        {"feature": feat, "shap_value": round(float(val), 4), "pushes_toward": "churn" if val > 0 else "retention"}
        for feat, val in contributions.head(3).items()
    ]

    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_flag=bool(churn_flag),
        threshold_used=used_threshold,
        top_reasons=top_reasons,
    )
