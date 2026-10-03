"""A concise GitHub front page, rendered from executed artifacts."""
import json
import pandas as pd
from .data import ROOT
from .documentation import table

REPOSITORY='https://github.com/akashsmb10/explainable-credit-risk'

def readme(summary):
    tables=ROOT/'artifacts/tables'; m=summary['test_metrics']
    trade=pd.read_csv(tables/'threshold_tradeoffs.csv')
    policy=trade.iloc[(trade.threshold-summary['review_threshold']).abs().argmin()]
    publication_path=ROOT/'config/publication.json'
    publication=json.loads(publication_path.read_text()) if publication_path.exists() else {}
    if publication.get('verified_live'):
        demo=f"[**Open live interactive demo ↗**]({publication['demo_url']})"
    else:
        demo='**Interactive demo:** prepared for GitHub Pages; publication is not yet verified.'
    model_comparison=pd.read_csv(tables/'test_models.csv')
    chosen=model_comparison[model_comparison.candidate.isin(['base_rate:raw','logistic:raw',summary['selected']])]
    compact=chosen[['candidate','roc_auc','average_precision','brier']].rename(columns={
        'candidate':'Model / probability mapping','roc_auc':'ROC-AUC','average_precision':'Average precision','brier':'Brier score'})
    return f"""# Explainable Credit Risk

### Retail Credit Risk & Explainable Loan Decision Engine

**A reproducible banking analytics portfolio that turns repayment histories into explainable risk estimates and transparent review simulations.**

{demo} · [Model card](reports/model_card.md) · [Executive summary](reports/executive_summary.md) · [Technical methodology](docs/technical_methodology.md)

Python · scikit-learn · XGBoost · SHAP · DuckDB / SQL · Streamlit · pytest

![Interactive dashboard preview](site/assets/dashboard-preview.png)

> Educational decision support using historical public cardholder data. This project does not recommend ICICI lending policy. Its scores estimate the dataset's payment-event label, not bank-grade Indian loan PD.

## What this project demonstrates

- **Statistical judgment:** confidence intervals, distribution comparisons, effect sizes, multiplicity correction and exploratory logistic inference.
- **Honest model validation:** grouped partitions for training, calibration, selection and final testing; preprocessing learns only from training.
- **Probability quality:** separate ranking and calibration evaluation; raw, sigmoid and isotonic mappings are compared without test leakage.
- **Business reasoning:** review-capacity trade-offs, risk bands, explicitly assumed losses and responsible subgroup/monitoring analysis.
- **Usable evidence:** executed SQL, global/local SHAP, a Streamlit dashboard and a public results explorer.

## Verified results

| Evidence | Executed result |
| --- | --- |
| Source clients | {summary['rows']:,} |
| Positive events | {summary['events']:,} ({summary['event_rate']:.2%}) |
| Final test clients | {summary['partitions']['test']['n']:,} |
| Selected model | Sigmoid-calibrated XGBoost |
| Test ROC-AUC / average precision | {m['roc_auc']:.4f} / {m['average_precision']:.4f} |
| Test KS / Brier score | {m['ks']:.4f} / {m['brier']:.4f} |
| Review rate / event capture | {policy.review_rate:.2%} / {policy.event_capture:.2%} |
| Event rate in simulated acceptance group | {policy.accepted_event_rate:.2%} |

**Calibration nuance:** sigmoid calibration won on validation Brier. Raw XGBoost had a slightly better Brier on the final test holdout; this is disclosed rather than changing the selected model after seeing test results.

{table(compact)}

All empirical figures come from executed code. [Run summary](artifacts/tables/run_summary.json), [full model comparisons](artifacts/tables/test_models.csv), [threshold trade-offs](artifacts/tables/threshold_tradeoffs.csv) and [confidence intervals](artifacts/tables/test_confidence_intervals.csv) provide traceability. Review figures are retrospective simulations, not actual approvals or profit improvements.

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
git clone {REPOSITORY}.git
cd explainable-credit-risk
```

<details open>
<summary><strong>Windows PowerShell</strong></summary>

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-lock.txt
.\\.venv\\Scripts\\python.exe -m src.pipeline --clean
.\\.venv\\Scripts\\python.exe -m src.notebooks
.\\.venv\\Scripts\\python.exe -m src.final_audit
.\\.venv\\Scripts\\python.exe -m src.web_demo
.\\.venv\\Scripts\\python.exe -m streamlit run app/app.py
```

</details>

<details>
<summary><strong>Linux / macOS</strong></summary>

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m src.pipeline --clean
.venv/bin/python -m src.notebooks
.venv/bin/python -m src.final_audit
.venv/bin/python -m src.web_demo
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
reports/                 Model card, findings, interview preparation and audit
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
| [Interview guide](reports/interview_guide.md) | Questions, concise answers and deeper technical follow-ups |
| [Verified resume bullets](reports/resume_bullets.md) | Three concise claims linked to executed evidence |
| [Final audit](reports/final_audit.md) | Reproduction and verification results |
| [Publication checks](reports/publication_audit.md) | Exported-score consistency, links and upload hygiene |

## Attribution and license

Project code is [MIT licensed](LICENSE). The UCI dataset is **CC BY 4.0**; its source and attribution are preserved in [NOTICE](NOTICE). The live snapshot contains public benchmark observations and model-derived results, not private bank customer data.
"""
