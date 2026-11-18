# Explainable Credit Risk

### Retail Credit Risk & Explainable Loan Decision Engine

**A reproducible banking analytics portfolio that turns repayment histories into explainable risk estimates and transparent review simulations.**

[**Open live interactive demo ↗**](https://akashsmb10.github.io/explainable-credit-risk/) · [Model card](reports/model_card.md) · [Executive summary](reports/executive_summary.md) · [Technical methodology](docs/technical_methodology.md)

Python · scikit-learn · XGBoost · SHAP · DuckDB / SQL · Streamlit · pytest

![Interactive dashboard preview](site/assets/dashboard-preview.png)

> Educational decision support using historical public cardholder data. This project does not recommend any bank's lending policy. Its scores estimate the dataset's payment-event label, not bank-grade Indian loan PD.

## What this project demonstrates

- **Statistical judgment:** confidence intervals, distribution comparisons, effect sizes, multiplicity correction and exploratory logistic inference.
- **Honest model validation:** grouped partitions for training, calibration, selection and final testing; preprocessing learns only from training.
- **Probability quality:** separate ranking and calibration evaluation; raw, sigmoid and isotonic mappings are compared without test leakage.
- **Business reasoning:** review-capacity trade-offs, risk bands, explicitly assumed losses and responsible subgroup/monitoring analysis.
- **Usable evidence:** executed SQL, global/local SHAP, a Streamlit dashboard and a public results explorer.

## Verified results

| Evidence | Executed result |
| --- | --- |
| Source clients | 30,000 |
| Positive events | 6,636 (22.1%) |
| Final test clients | 6,010 |
| Selected model | xgboost:sigmoid |
| Test ROC-AUC / average precision | 0.781 / 0.559 |
| Test KS / Brier score | 0.430 / 0.133 |
| Review rate / event capture | 24.1% / 57.3% |
| Event rate in simulated acceptance group | 12.3% |

Figures are from the saved run; re-running on another OS/XGBoost build can shift them by about ±0.001 (logistic results reproduce exactly).

**Calibration nuance:** sigmoid calibration won on validation Brier. Raw XGBoost had a slightly better Brier on the final test holdout; this is disclosed rather than changing the selected model after seeing test results.

| Model / probability mapping | ROC-AUC | Average precision | Brier score |
| --- | --- | --- | --- |
| base_rate:raw | 0.500 | 0.219 | 0.171 |
| logistic:raw | 0.763 | 0.537 | 0.135 |
| xgboost:sigmoid | 0.781 | 0.559 | 0.133 |

All empirical figures come from executed code. [Run summary](artifacts/tables/run_summary.json), [full model comparisons](artifacts/tables/test_models.csv), [threshold trade-offs](artifacts/tables/threshold_tradeoffs.csv) and [confidence intervals](artifacts/tables/test_confidence_intervals.csv) provide traceability. Review figures are retrospective simulations, not actual approvals or profit improvements.

## Champion–challenger: logistic vs XGBoost

| Model | ROC-AUC | Gini | KS | Brier score | Decile inversions |
| --- | --- | --- | --- | --- | --- |
| logistic | 0.763 | 0.527 | 0.419 | 0.135 | 0 |
| xgboost | 0.781 | 0.562 | 0.430 | 0.132 | 1 |

XGBoost ranks risk better by ΔAUC 0.017 [0.010, 0.025] (95% paired-bootstrap interval). Logistic regression rank-orders the deciles more smoothly and is directly interpretable. In a regulated scoring setting, logistic would be the natural champion and XGBoost a challenger.

This compares frozen raw probabilities on the test partition for reporting only; the validation-selected model remains `xgboost:sigmoid`. Deciles contain equal counts by score rank, from low to high risk; tied scores use source row order. An inversion is an adjacent decline in observed event rate. [Model and decile evidence](artifacts/tables/champion_challenger.csv) and [paired-bootstrap results](artifacts/tables/champion_challenger_bootstrap.json) include the mean ΔAUC and share of replicates at or below zero. Intervals condition on these fitted models and this split.

## Explore the evidence

![Observed event rates by validation-derived risk band](artifacts/figures/risk_bands.png)

<details>
<summary><strong>View probability calibration and SHAP explanations</strong></summary>

![Holdout probability calibration](artifacts/figures/calibration.png)

![Raw XGBoost TreeSHAP summary](artifacts/figures/shap_summary.png)

SHAP explains the raw tree's log-odds, not calibrated probabilities or causal effects. The saved app includes local contributions for every test cardholder.

</details>

## Dashboard

The **public demo** is a browser-based explorer of frozen verified results: portfolio overview, risk distribution, cardholder scores, model performance, SHAP explanations, review-threshold simulation and assumed-loss scenarios. It performs no new model inference or retraining. Its predictions are exported by `python -m src.web_demo` from the same artifacts used by Streamlit.

The **full Streamlit application** is included in [app/app.py](app/app.py). Run it locally after generating the artifacts. It also loads saved results instead of retraining at startup. [Dashboard and hosting guide](docs/dashboard.md).

## Reproduce the project

Clone the repository and run commands from its root. Python **3.12** is the verified runtime.

```bash
git clone https://github.com/akashsmb10/explainable-credit-risk.git
cd explainable-credit-risk
```

<details open>
<summary><strong>Windows PowerShell</strong></summary>

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m src.pipeline --clean
.\.venv\Scripts\python.exe -m src.notebooks
.\.venv\Scripts\python.exe -m src.web_demo
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.final_audit
.\.venv\Scripts\python.exe scripts/release_audit.py
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

</details>

<details>
<summary><strong>Linux / macOS</strong></summary>

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m src.pipeline --clean
.venv/bin/python -m src.notebooks
.venv/bin/python -m src.web_demo
.venv/bin/python -m pytest -q
.venv/bin/python -m src.final_audit
.venv/bin/python scripts/release_audit.py
.venv/bin/python -m streamlit run app/app.py
```

Cross-platform execution is checked by GitHub Actions when available; the local verified environment is Windows. The first pipeline run downloads the official UCI archive. `--clean` regenerates only downstream output directories and retains downloaded source data. Model training is required before using Streamlit on a fresh clone. Raw data, serialized models, full prediction dumps and the virtual environment are deliberately excluded from Git.

</details>

The final audit runs pytest, replays saved predictions, reruns every SQL query, checks notebook execution and starts a temporary Streamlit HTTP server. [Final execution audit](reports/final_audit.md).

## Repository map

```text
app/                     Streamlit dashboard
config/                  Reproducible settings and scenario assumptions
docs/                    Data dictionary, scope, methodology and dashboard guide
notebooks/               Executed audit-to-decision narrative
src/                     Reusable Python pipeline and artifact exporters
sql/                     Executed DuckDB analytical queries
tests/                   Feature, leakage, decision and dashboard tests
artifacts/figures/       Generated analytical figures
artifacts/tables/        Shareable evidence and execution summaries
reports/                 Model card, findings and audit
site/                    Static interactive public demo and saved result snapshot
scripts/                 Repository publication and presentation checks
```

## Dataset, assumptions and limitations

[UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), Yeh (2009), [DOI: 10.24432/C55S3H](https://doi.org/10.24432/C55S3H). Historical Taiwanese existing cardholders; next-month default-payment target. Income, actual LGD/EAD, recoveries, rejected applicants and separate temporal cohorts are unavailable.

Risk-band thresholds express a validation-derived review-capacity scenario. LGD and EAD are explicitly assumed in loss sensitivity analysis; credit limit is not measured EAD. Demographics are excluded from scoring, but subgroup differences and proxy risks remain. No actual production or temporal drift evidence is claimed. Statistical associations and SHAP are non-causal. See [model scope](docs/model_scope_and_limitations.md), [statistical findings](reports/statistical_findings.md) and [model card](reports/model_card.md).

## Documentation worth reading

| Document | Purpose |
| --- | --- |
| [Executive summary](reports/executive_summary.md) | Business problem, measured results and boundaries |
| [Technical methodology](docs/technical_methodology.md) | Complete modeling, calibration and statistical rationale |
| [Data dictionary](docs/data_dictionary.md) | Source schema, code issues and audit findings |
| [Model card](reports/model_card.md) | Intended use, validation, fairness and monitoring limitations |
| [Final audit](reports/final_audit.md) | Reproduction and verification results |
| [Publication checks](reports/publication_audit.md) | Exported-score consistency, links and upload hygiene |

## Attribution and license

Project code is [MIT licensed](LICENSE). The UCI dataset is **CC BY 4.0**; its source and attribution are preserved in [NOTICE](NOTICE). The live snapshot contains public benchmark observations and model-derived results, not private bank customer data.
