# Verified completion report

1. **Dataset and selection:** UCI Default of Credit Card Clients, Taiwan, historical existing-cardholder repayment data. Credible official public download, compact and credential-free compared with Home Credit. Target is next-month default payment, not an Indian bank-grade PD.
2. **Architecture:** repository root contains config, docs, reusable src, raw/processed data, eight executed notebooks, executable DuckDB SQL, artifacts, tests, reports and seven-page Streamlit app. No published GitHub repository or bank deployment is claimed.
3. **Verified size:** 30,000 source rows, 25 columns including ID/target, 6,636 events, event rate 22.1200%, missing cells 0. SHA256 `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`. Full-profile duplicates 56; model-input duplicates are grouped across all partitions.
4. **Models:** base-rate benchmark, regularized logistic, regularized shallow XGBoost; selected `xgboost:sigmoid` by validation Brier then log loss, before test scoring.
5. **Verified final metrics:**

| roc_auc | average_precision | precision | recall | f1 | ks | brier | log_loss | tn | fp | fn | tp | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7808 | 0.5591 | 0.5204 | 0.5731 | 0.5455 | 0.4302 | 0.1325 | 0.4242 | 4002 | 694 | 561 | 753 | 0.2840 |

6. **Calibration result:** selected-model mapping comparisons on final test below. This is descriptive holdout evidence; final model selection used validation only.

| candidate | roc_auc | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- |
| xgboost:raw | 0.7808 | 0.5591 | 0.1325 | 0.4242 |
| xgboost:sigmoid | 0.7808 | 0.5591 | 0.1325 | 0.4242 |
| xgboost:isotonic | 0.7799 | 0.5365 | 0.1328 | 0.4260 |

7. **Main statistical findings:** training-only associations, not causes. Positive-delay segmentation:

| recent_delay | size | sum | mean | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| no positive delay | 9234 | 1250 | 0.1354 | 0.1285 | 0.1425 |
| reported delay | 2692 | 1391 | 0.5167 | 0.4978 | 0.5356 |

| feature | test | statistic | p_value | p_holm | effect_size |
| --- | --- | --- | --- | --- | --- |
| LIMIT_BAL | Mann-Whitney U | 9219051.5000 | 9.341e-85 | 5.604e-84 | -0.2481 |
| utilization_latest | Mann-Whitney U | 13583993.5000 | 2.326e-17 | 1.163e-16 | 0.1079 |
| months_delayed | Mann-Whitney U | 17970970.5000 | 0.0000 | 0.0000 | 0.4657 |
| payment_to_limit | Mann-Whitney U | 11839888.5000 | 0.0070 | 0.0140 | -0.0343 |
| SEX | Chi-square | 17.5305 | 2.827e-05 | 8.482e-05 | 0.0383 |
| EDUCATION | Chi-square | 74.4168 | 4.832e-16 | 1.933e-15 | 0.0790 |
| MARRIAGE | Chi-square | 8.4242 | 0.0148 | 0.0148 | 0.0266 |

Statistical significance differs from usefulness; effect size and holdout trade-offs determine practical value. Reduced-logit inference and VIF are in statistical_findings.md.

8. **Raw-tree global drivers:** mean absolute exact TreeSHAP contributions in log-odds; correlated encodings can split importance. This ranking is from the declared held-out explanation sample.

| feature | mean_abs_shap_log_odds |
| --- | --- |
| months_delayed | 0.3319 |
| maximum_delay | 0.2637 |
| PAY_0_2 | 0.1784 |
| BILL_AMT1 | 0.1365 |
| LIMIT_BAL | 0.1155 |
| PAY_AMT3 | 0.1048 |
| utilization_latest | 0.1001 |
| bill_change_to_limit | 0.0838 |

9. **Decision engine:** frozen review threshold 0.283990; simulated test acceptance 75.92%, review 24.08%, review event capture 57.31%, accepted event rate 12.29%. Boundaries `[0.14707765807900644, 0.2839901475140817, 0.5722191552185503]` are validation-score quantiles under a review-capacity assumption. No actual lending or profitability result.
10. **Loss assumptions:** displayed default LGD 0.45 and EAD 100000 NT dollars; sensitivity varies configured LGD and exposure assumptions. Neither LGD nor actual EAD is observed. Dataset event score × assumptions is a scenario loss proxy, not real bank EL/provisions.
11. **Responsible modeling:** demographics excluded from scoring, proxy risks remain. Test subgroup results:

| attribute | group | n | events | event_rate | roc_auc | brier | calibration_gap | false_positive_rate | false_negative_rate | selection_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SEX | 1 | 2346 | 546 | 0.2327 | 0.7767 | 0.1392 | -0.0037 | 0.1583 | 0.4249 | 0.7447 |
| SEX | 2 | 3664 | 768 | 0.2096 | 0.7831 | 0.1282 | 0.0035 | 0.1412 | 0.4284 | 0.7686 |
| age_band | up to 30 | 2200 | 464 | 0.2109 | 0.7822 | 0.1291 | 0.0205 | 0.1682 | 0.3966 | 0.7400 |
| age_band | 31 to 45 | 2907 | 620 | 0.2133 | 0.7760 | 0.1317 | -0.0091 | 0.1277 | 0.4710 | 0.7867 |
| age_band | above 45 | 903 | 230 | 0.2547 | 0.7947 | 0.1435 | -0.0162 | 0.1634 | 0.3696 | 0.7176 |

No fairness certification follows; subgroup differences and sample uncertainty require bank-specific investigation.
12. **Tests:** 28 tests passed, including all Streamlit pages and slider endpoint.
13. **Dashboard:** all seven pages passed programmatic rendering. HTTP health and HTML endpoints returned success during a temporary local launch; the audit server was then stopped. Run the command below for an interactive local Streamlit dashboard. Public frozen-results demo: https://akashsmb10.github.io/explainable-credit-risk/ (publication verified). This is not a hosted Streamlit process.
14. **Reproduction commands (PowerShell, repository root):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m src.pipeline --clean
.\.venv\Scripts\python.exe -m src.notebooks
.\.venv\Scripts\python.exe -m src.final_audit
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

The final audit runs pytest; standalone command: `.\.venv\Scripts\python.exe -m pytest -q`. `--clean` removes only generated models/tables/figures and processed data and retains the downloaded official source. Installation/download may require network access. Direct pins are in requirements.txt; the executed environment is recorded in requirements-lock.txt.

15. **Resume-ready output:**

# Retail Credit Risk & Explainable Loan Decision Engine

Technology: Python, pandas, SciPy, statsmodels, scikit-learn, XGBoost, SHAP, DuckDB, SQL, Streamlit, pytest.

- Built a reproducible credit-risk analysis pipeline for 30,000 public cardholder records, with duplicate-profile holdouts and train-only preprocessing.
- Compared baseline, logistic and XGBoost models with independent calibration; selected xgboost:sigmoid, achieving held-out ROC-AUC 0.7808, average precision 0.5591 and Brier 0.1325.
- Developed SQL-backed segmentation, SHAP explanations and a review simulation capturing 57.31% of holdout events at a 24.08% review rate, with clearly assumed loss scenarios.

Traceability: `artifacts/tables/run_summary.json` for source/model metrics; `artifacts/tables/threshold_tradeoffs.csv` at the summary's review threshold for simulation rates. These are portfolio experiments, not bank deployment or profit impact.


16. **Remaining limitations:** old single-geography observational cardholder sample; unknown precise contractual event definition; no income/affordability, rejected applicants, losses/recovery or actual default exposure; no out-of-time or production validation. Intervals condition on a fitted model/split. Model selection over several candidates has uncertainty. Statistical inference is exploratory. The dashboard is a local artifact explorer, not authenticated bank infrastructure.
17. **Do not claim in an ICICI interview:** ICICI deployment, India-specific bank-grade PD, improved bank profits, causal SHAP drivers, validated underwriting acceptance, real production drift, true expected loss/provisioning or certified fairness. Explain the business assumptions and show the saved numerical artifacts instead.
