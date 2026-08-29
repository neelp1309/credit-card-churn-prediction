# Dataset

The original project uses `BankChurners.csv` (10,127 rows, 21 columns after excluding the two Naive-Bayes helper columns found in some public copies).

The uploaded ZIP did **not** contain the CSV, so this P0/P1 cleanup does not invent or substitute training data. Place the exact dataset used by the notebook at:

```text
data/BankChurners.csv
```

The training script also accepts the common source-data representation where `Unknown` is already present in the three categorical fields; the original notebook's cleaned representation maps Education/Marital missing values and the `Income_Category="abc"` placeholder back to `Unknown`.
