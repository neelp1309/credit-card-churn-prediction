import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MODEL_ARTIFACTS_PATH", str(ROOT / "artifacts" / "model_artifacts.joblib"))

from api.app import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_payload():
    return {
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
        "Avg_Utilization_Ratio": 0.14,
    }
