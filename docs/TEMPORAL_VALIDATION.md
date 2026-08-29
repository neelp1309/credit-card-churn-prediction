# Temporal validation status

A genuine out-of-time validation cannot be completed with the current dataset because there is no event/prediction timestamp defining what was known at prediction time and what happened later.

Do **not** create a fake time split from row order.

With production data, the validation contract should be:

1. Pick a prediction cutoff `T`.
2. Build all features using events with `event_time <= T` only.
3. Define the label as churn during `(T, T + horizon]`, e.g. the next 90 days.
4. Train on earlier cutoffs and validate/test on strictly later cutoffs.
5. Keep a gap if the label window overlaps the feature aggregation window.
6. Re-run threshold calibration on the out-of-time validation cohort.

This is intentionally documented as a limitation rather than presented as completed work.
