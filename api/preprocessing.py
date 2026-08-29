from __future__ import annotations

import pandas as pd


def preprocess_record(record: dict, training_columns: list[str], cat_cols: list[str]) -> pd.DataFrame:
    """Apply the same one-hot encoding/alignment used by the training notebook.

    SMOTE is deliberately absent here: resampling is a training-time operation only.
    """
    df = pd.DataFrame([record])
    encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return encoded.reindex(columns=training_columns, fill_value=0)
