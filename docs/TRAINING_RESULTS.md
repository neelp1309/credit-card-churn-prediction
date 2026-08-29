# Leakage-safe retraining results

The revised P0/P1 training path was executed on the supplied `BankChurners.csv` dataset.

## Dataset

- Rows: **10,127**
- Columns in source CSV: **21**
- Existing customers: **8,500**
- Attrited customers: **1,627**
- Positive-class prevalence: **16.07%**
- Random state: **42**

## Imbalance-strategy comparison

Model/strategy selection was based only on **5-fold training cross-validation recall**. SMOTE is applied inside the CV pipeline so synthetic samples are generated only from each fold's training partition.

| Strategy | CV recall | Validation recall | Validation precision | Validation F1 | Validation ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Fold-safe SMOTE | 0.9135 | 0.9004 | 0.8801 | 0.8902 | 0.9922 |
| `class_weight="balanced"` | 0.9347 | 0.9387 | 0.8507 | 0.8925 | 0.9920 |
| `scale_pos_weight` | **0.9376** | 0.9349 | 0.8531 | 0.8921 | **0.9923** |

**Selected strategy:** `scale_pos_weight`

Selected LightGBM parameters:

```text
learning_rate = 0.046995735689060486
max_depth     = 5
n_estimators  = 187
num_leaves    = 50
```

## Untouched test-set result

At the standard `0.50` probability threshold:

| Metric | Test result |
|---|---:|
| Recall | **0.9200** |
| Precision | **0.8470** |
| F1 | **0.8820** |
| ROC-AUC | **0.9914** |

The test split was not used for model/strategy selection or threshold optimization.

## Business operating threshold

The validation-only cost function used:

```text
False-negative cost = 20,000
False-positive cost =    500
```

The minimum validation cost was obtained at a threshold of **0.09**, with validation cost **102,500**.

Using that operating threshold on the untouched test set:

| Metric | Test result |
|---|---:|
| Recall | **0.9908** |
| Precision | **0.5620** |
| F1 | **0.7171** |
| ROC-AUC | **0.9914** |

This threshold intentionally favors catching nearly all churners because a false negative is assumed to cost 40x a false positive. It is an operating-policy choice, not a claim that 0.09 is universally optimal.

## Important conclusion

The corrected experiment shows that **SMOTE is not required for the best result**. Under the same leakage-safe CV framework, LightGBM with `scale_pos_weight` produced the highest CV recall. This is a stronger production choice because it avoids generating synthetic customer records while still addressing class imbalance.

Generated artifacts:

```text
artifacts/model_artifacts.joblib
artifacts/reference_profile.json
artifacts/training_report.json
```
