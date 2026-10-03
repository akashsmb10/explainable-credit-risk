"""Produce the final user-facing result ledger from verified artifacts."""
import json
import pandas as pd
from .data import ROOT
from .documentation import table

def write_completion():
    t=ROOT/'artifacts/tables'
    s=json.loads((t/'run_summary.json').read_text())
    audit=json.loads((t/'final_audit.json').read_text())
    assert all(x['status']=='PASS' for x in audit)
    def read(path): return pd.read_csv(t/path)
    test=read('test_models.csv'); stat=read('statistical_tests.csv'); driver=read('shap_global.csv')
    sub=read('subgroup_metrics.csv'); rates=read('delay_event_rates.csv'); trade=read('threshold_tradeoffs.csv')
    policy=trade.iloc[(trade.threshold-s['review_threshold']).abs().argmin()]
    model=s['selected'].split(':')[0]
    publication_path=ROOT/'config/publication.json'
    publication=json.loads(publication_path.read_text()) if publication_path.exists() else {}
    live_status=(f"Public frozen-results demo: {publication['demo_url']} (publication verified). This is not a hosted Streamlit process."
                 if publication.get('verified_live') else 'No cloud publication is claimed.')
    comparison=test[test.candidate.str.startswith(model+':')][['candidate','roc_auc','average_precision','brier','log_loss']]
    report=f"""# Verified completion report

1. **Dataset and selection:** UCI Default of Credit Card Clients, Taiwan, historical existing-cardholder repayment data. Credible official public download, compact and credential-free compared with Home Credit. Target is next-month default payment, not an Indian bank-grade PD.
2. **Architecture:** repository root contains config, docs, reusable src, raw/processed data, eight executed notebooks, executable DuckDB SQL, artifacts, tests, reports and seven-page Streamlit app. No published GitHub repository or bank deployment is claimed.
3. **Verified size:** {s['rows']:,} source rows, {s['columns']} columns including ID/target, {s['events']:,} events, event rate {s['event_rate']:.4%}, missing cells {s['missing_cells']}. SHA256 `{s['source_sha256']}`. Full-profile duplicates {s['duplicate_profiles']}; model-input duplicates are grouped across all partitions.
4. **Models:** base-rate benchmark, regularized logistic, regularized shallow XGBoost; selected `{s['selected']}` by validation Brier then log loss, before test scoring.
5. **Verified final metrics:**

{table(pd.DataFrame([s['test_metrics']]))}

6. **Calibration result:** selected-model mapping comparisons on final test below. This is descriptive holdout evidence; final model selection used validation only.

{table(comparison)}

7. **Main statistical findings:** training-only associations, not causes. Positive-delay segmentation:

{table(rates)}

{table(stat[['feature','test','statistic','p_value','p_holm','effect_size']])}

Statistical significance differs from usefulness; effect size and holdout trade-offs determine practical value. Reduced-logit inference and VIF are in statistical_findings.md.

8. **Raw-tree global drivers:** mean absolute exact TreeSHAP contributions in log-odds; correlated encodings can split importance. This ranking is from the declared held-out explanation sample.

{table(driver.head(8))}

9. **Decision engine:** frozen review threshold {s['review_threshold']:.6f}; simulated test acceptance {policy.simulated_acceptance_rate:.2%}, review {policy.review_rate:.2%}, review event capture {policy.event_capture:.2%}, accepted event rate {policy.accepted_event_rate:.2%}. Boundaries `{s['band_boundaries']}` are validation-score quantiles under a review-capacity assumption. No actual lending or profitability result.
10. **Loss assumptions:** displayed default LGD 0.45 and EAD 100000 NT dollars; sensitivity varies configured LGD and exposure assumptions. Neither LGD nor actual EAD is observed. Dataset event score × assumptions is a scenario loss proxy, not real bank EL/provisions.
11. **Responsible modeling:** demographics excluded from scoring, proxy risks remain. Test subgroup results:

{table(sub[['attribute','group','n','events','event_rate','roc_auc','brier','calibration_gap','false_positive_rate','false_negative_rate','selection_rate']])}

No fairness certification follows; subgroup differences and sample uncertainty require bank-specific investigation.
12. **Tests:** {next(x['evidence'] for x in audit if x['component'].startswith('Complete pytest'))}.
13. **Dashboard:** all seven pages passed programmatic rendering. HTTP health and HTML endpoints returned success during a temporary local launch; the audit server was then stopped. Run the command below for an interactive local Streamlit dashboard. {live_status}
14. **Reproduction commands (PowerShell, repository root):**

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-lock.txt
.\\.venv\\Scripts\\python.exe -m src.pipeline --clean
.\\.venv\\Scripts\\python.exe -m src.notebooks
.\\.venv\\Scripts\\python.exe -m src.final_audit
.\\.venv\\Scripts\\python.exe -m streamlit run app/app.py
```

The final audit runs pytest; standalone command: `.\\.venv\\Scripts\\python.exe -m pytest -q`. `--clean` removes only generated models/tables/figures and processed data and retains the downloaded official source. Installation/download may require network access. Direct pins are in requirements.txt; the executed environment is recorded in requirements-lock.txt.

15. **Resume-ready output:**

{(ROOT/'reports/resume_bullets.md').read_text()}

16. **Remaining limitations:** old single-geography observational cardholder sample; unknown precise contractual event definition; no income/affordability, rejected applicants, losses/recovery or actual default exposure; no out-of-time or production validation. Intervals condition on a fitted model/split. Model selection over several candidates has uncertainty. Statistical inference is exploratory. The dashboard is a local artifact explorer, not authenticated bank infrastructure.
17. **Do not claim in an ICICI interview:** ICICI deployment, India-specific bank-grade PD, improved bank profits, causal SHAP drivers, validated underwriting acceptance, real production drift, true expected loss/provisioning or certified fairness. Explain the business assumptions and show the saved numerical artifacts instead.
"""
    (ROOT/'reports/completion_report.md').write_text(report,encoding='utf-8')

if __name__=='__main__': write_completion()
