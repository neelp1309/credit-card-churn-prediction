import joblib
from pathlib import Path

from api.preprocessing import preprocess_record


def test_preprocessing_matches_training_columns(valid_payload):
    root = Path(__file__).resolve().parents[1]
    artifacts = joblib.load(root / "artifacts" / "model_artifacts.joblib")
    X = preprocess_record(valid_payload, artifacts["training_columns"], artifacts["cat_cols"])
    assert X.columns.tolist() == artifacts["training_columns"]
    assert X.shape == (1, len(artifacts["training_columns"]))
    assert not X.isna().any().any()
