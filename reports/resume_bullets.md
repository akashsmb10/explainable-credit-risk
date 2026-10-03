# Retail Credit Risk & Explainable Loan Decision Engine

Technology: Python, pandas, SciPy, statsmodels, scikit-learn, XGBoost, SHAP, DuckDB, SQL, Streamlit, pytest.

- Built a reproducible credit-risk analysis pipeline for 30,000 public cardholder records, with duplicate-profile holdouts and train-only preprocessing.
- Compared baseline, logistic and XGBoost models with independent calibration; selected xgboost:sigmoid, achieving held-out ROC-AUC 0.7808, average precision 0.5591 and Brier 0.1325.
- Developed SQL-backed segmentation, SHAP explanations and a review simulation capturing 57.31% of holdout events at a 24.08% review rate, with clearly assumed loss scenarios.

Traceability: `artifacts/tables/run_summary.json` for source/model metrics; `artifacts/tables/threshold_tradeoffs.csv` at the summary's review threshold for simulation rates. These are portfolio experiments, not bank deployment or profit impact.
