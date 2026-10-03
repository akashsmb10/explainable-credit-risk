# Decision-oriented chart guide

- `delinquency_event_rate.png`: whether recent reported delay distinguishes review groups, with training-only Wilson intervals. It does not establish why payment events occur.
- `eda_LIMIT_BAL.png`: whether limit quantiles show different observed event rates; limits reflect lender selection as well as borrower characteristics, and are not EAD.
- `eda_utilization_latest.png`: whether statement-to-limit bands provide useful review segmentation. Negative/over-limit values are retained; utilization is not affordability.
- `eda_payment_to_limit.png`: whether historical payment intensity adds behavioural differentiation. The measure is not income or contractual installment coverage.
- `roc.png`: rank discrimination compared with the naive benchmark. Ranking alone cannot set a cost-optimal policy.
- `precision_recall.png`: how event capture trades against review precision under imbalance; prevalence is the reference.
- `calibration.png`: whether scored groups' outcomes track mean predicted event probabilities; finite-bin uncertainty and geographic transfer remain concerns.
- `risk_bands.png`: whether validation-defined bands produce differentiated holdout outcomes, useful for simulated review allocation.
- `shap_summary.png` and local waterfalls: which encoded predictors the raw tree associates with score differences. Attribution is not causal and does not explain calibrated probabilities additively.
