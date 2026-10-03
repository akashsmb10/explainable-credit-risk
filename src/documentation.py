"""Render reported empirical numbers exclusively from executed artifacts."""
import json
import pandas as pd
from .data import ROOT

def table(frame):
    # Avoid a runtime dependency on tabulate.
    head='| '+' | '.join(map(str,frame.columns))+' |\n'
    line='| '+' | '.join(['---']*len(frame.columns))+' |\n'
    rows=[]
    for row in frame.itertuples(index=False,name=None):
        rows.append('| '+' | '.join((f'{x:.3e}' if 0<abs(x)<1e-4 else f'{x:.4f}') if isinstance(x,float) else str(x) for x in row)+' |')
    return head+line+'\n'.join(rows)

def write_documentation(s):
    tables=ROOT/'artifacts/tables'; reports=ROOT/'reports'
    def read(name): return pd.read_csv(tables/name)
    def write(path,body): (ROOT/path).write_text(body,encoding='utf-8')
    metrics=s['test_metrics']; selected=s['selected']
    val=read('validation_models.csv'); test=read('test_models.csv'); stats=read('statistical_tests.csv')
    sub=read('subgroup_metrics.csv'); drivers=read('shap_global.csv').head(10)
    decision=read('threshold_tradeoffs.csv')
    policy=decision.iloc[(decision.threshold-s['review_threshold']).abs().argmin()]
    raw_brier=float(test.loc[test.candidate==selected.split(':')[0]+':raw','brier'].iloc[0])
    reasons={
        'LIMIT_BAL':'Granted-limit association may reflect lender selection and borrower differences; it is not affordability or a causal limit effect.',
        'utilization_latest':'Statement-to-limit differences can guide behavioural review segments; they cannot establish income burden.',
        'months_delayed':'Historical delay frequency is a directly interpretable behavioural review signal; its out-of-sample value is tested by the modeling layer.',
        'payment_to_limit':'The small rank association must not be oversold merely because the p-value is below the conventional significance cutoff.',
        'SEX':'Audit-only association supports investigation of population differences, not demographic scoring or fairness certification.',
        'EDUCATION':'Pooled audit categories describe sampled differences; neither undocumented category meaning nor causal education effects are inferred.',
        'MARRIAGE':'A small audit association is not a basis for demographic underwriting; evaluate proxies and subgroup uncertainty.'}
    statistical_details='\n\n'.join(
        f"### {r.feature}\n\nH0: {r.H0}. H1: {r.H1}. Assumptions: {r.assumptions}. "
        f"Test statistic {r.statistic:.6g}; raw p-value {r.p_value:.6g}; Holm p-value {r.p_holm:.6g}; "
        f"effect size {r.effect_size:.6g} ({r.effect_definition}). "
        f"Interpretation: {'evidence against H0 at the declared conventional significance level' if r.p_holm<.05 else 'insufficient evidence against H0 at the declared conventional significance level'}; this does not measure causal effects. "
        f"Business relevance: {reasons[r.feature]}"
        for r in stats.itertuples())
    empirical=f"""Executed source audit: {s['rows']:,} clients, {s['events']:,} positive events ({s['event_rate']:.2%}), {s['missing_cells']} missing cells, {s['duplicate_profiles']} duplicate predictor profiles, {s['negative_bill_cells']} negative bill cells. Source file SHA256: `{s['source_sha256']}`.

The predeclared validation-Brier selection rule selected **{selected}**. Test ROC-AUC **{metrics['roc_auc']:.4f}**, average precision **{metrics['average_precision']:.4f}**, KS **{metrics['ks']:.4f}**, Brier **{metrics['brier']:.4f}**. At the validation-derived review threshold **{s['review_threshold']:.6f}**, precision is **{metrics['precision']:.4f}**, recall **{metrics['recall']:.4f}**, F1 **{metrics['f1']:.4f}**.

The simulated acceptance fraction on the untouched test is **{policy.simulated_acceptance_rate:.2%}**; accepted clients' observed event rate is **{policy.accepted_event_rate:.2%}**. Review captures **{policy.event_capture:.2%}** of observed events. These are retrospective same-sample simulations, not realized approval or profit impacts.

Calibration is not claimed to improve every holdout metric: the selected model's raw test Brier is **{raw_brier:.6f}**, versus selected-mapping Brier **{metrics['brier']:.6f}**. Selection remains based on the independent validation result; test differences are reported without switching winners.
"""
    partitions=pd.DataFrame([{'partition':k,**v} for k,v in s['partitions'].items()])
    README=f"""# Retail Credit Risk & Explainable Loan Decision Engine

An educational banking analytics portfolio: historical repayment data → statistical understanding → risk ranking → probability calibration → explainability → review-capacity simulation → assumed-loss scenarios → monitoring demonstrations.

This project asks how a risk analyst can prioritize existing cardholders for review while distinguishing predictive evidence from business assumptions. It does not recommend ICICI lending policy.

## Verified results

{empirical}

![Risk bands](artifacts/figures/risk_bands.png)
![Calibration](artifacts/figures/calibration.png)
![SHAP](artifacts/figures/shap_summary.png)

## Reproduce (Windows PowerShell, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-lock.txt
.\\.venv\\Scripts\\python.exe -m src.pipeline --clean
.\\.venv\\Scripts\\python.exe -m src.notebooks
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe -m src.final_audit
.\\.venv\\Scripts\\python.exe -m streamlit run app/app.py
```

Run commands from the repository root. First pipeline run downloads the official UCI Excel archive; later runs reuse the source and regenerate all downstream models/tables/figures without loading old fitted artifacts. Unix equivalents use `.venv/bin/python`. Network access is necessary for dependencies and first data acquisition. The final audit checks the app with Streamlit AppTest and launches a temporary HTTP server for a health check; the interactive command above starts your own dashboard.

## Dataset and target

[UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), Yeh (2009), [DOI](https://doi.org/10.24432/C55S3H), CC BY 4.0. Source documentation describes existing Taiwan cardholders, April–September 2005 histories, and a next-month default-payment label. The exact contractual definition and horizon beyond this label are unavailable. Scores estimate this dataset event, not bank-grade PD. Home Credit's access requirements and large relational download make it less reproducible for this self-contained project.

## Validation design

{table(partitions)}

StratifiedGroupKFold assigns identical predictor profiles to one fold; one fold is calibration, one validation, one test, and the remaining folds training. Preprocessing, imputation, scaling and category vocabularies are fitted only on training. Calibration maps are learned on the dedicated calibration partition. Candidate selection and score quantiles use validation. Test is scored after these are frozen. No time-based generalization can be tested because cohort dates are absent. Group stratification is approximate. Bootstrap intervals are ordinary client resamples, conditional on the fitted model and split, and can understate dependence/selection uncertainty.

## Model hierarchy and calibration

Base-rate benchmark, regularized logistic regression and shallow regularized XGBoost. Fixed model hyperparameters avoid an extensive search on a small validation set. All nontrivial models compare raw, sigmoid and isotonic probabilities. Selection minimizes validation Brier, with log loss as tie-breaker. Classification metrics in the comparison table use the conventional threshold; the policy results use the validation-derived threshold. Average precision is reported explicitly and should not be confused with trapezoidal PR-AUC.

Validation results:

{table(val[['candidate','roc_auc','average_precision','brier','log_loss']])}

Test results (descriptive, never used for selection):

{table(test[['candidate','roc_auc','average_precision','brier','log_loss']])}

Confidence intervals:

{table(read('test_confidence_intervals.csv'))}

Good calibration means that a group scored at a given event probability has roughly that event frequency under the same population and target horizon. It does not guarantee an individual outcome. Rank performance and probability quality can differ; isotonic's ties may alter ranking. Test reliability bins include Wilson event-rate intervals and deciles compare scores with outcomes.

## Decision assumptions

Risk boundaries are the configured validation median, upper quartile and upper-tail quantile: `{s['band_boundaries']}`. Labels describe relative model risk. Manual review starts at the upper-quartile boundary, reflecting a scenario with approximately one quarter of validation cases prioritized. This is an assumed review-capacity rule, not an optimal underwriting cutoff. Ties can change the realized review fraction. No lending benefit, rejection cost, affordability, legal rule or profit function is observed; a bank would require all of these before determining acceptance. Threshold tables retain false positives, false negatives, event capture, acceptance fraction and accepted event rate. Low score does not ensure eligibility or affordability.

## Expected-loss scenarios

Banking EL = PD × LGD × EAD. Here the probability is a model-estimated dataset event score; LGD and EAD are explicitly assumed. LGD is a fraction of exposure lost conditional on default; EAD is outstanding exposure at the default time, not the observed card limit. Scenario currency is NT dollars. The default display assumes LGD 0.45 and EAD 100000; the configuration and scenario table vary both. These scenario values are not empirical findings. Doubling assumed exposure at the same score and LGD doubles scenario loss. No real provision, IFRS 9 ECL or capital estimate is made.

## Statistics, features and explanations

Training-only Mann–Whitney comparisons, point-biserial associations, Spearman correlations, demographic chi-square/Cramer's V, Holm correction, Wilson intervals, VIF and a reduced logit with robust covariance are executed. Rank-biserial effect sizes assess practical strength separately from p-values. Inference is exploratory, conditional on specification and approximate independent clients; it is not causal or a separate confirmatory experiment. The reduced inferential model is distinct from the penalized predictive logistic model. See [statistical findings](reports/statistical_findings.md).

Behavioural features quantify utilization, payment-to-limit, bill change and delay frequency. No income or employment ratio is manufactured. Latest/mean bills divided by limit can be negative or exceed unity; values are retained and documented. Payment-to-positive-bill is a history proxy, not a matched statement repayment or affordability measure.

Exact TreeSHAP in XGBoost explains raw log-odds. SHAP plots use the SHAP plotting package; contributions are checked against raw margins. Global importance is computed on a reproducible held-out sample including representative clients. Local contributions are saved for every test client; waterfalls cover low, medium, high and boundary clients. Calibration changes the output scale; TreeSHAP does not add to the calibrated score. If a different model wins, tree explanations remain benchmark-specific. Correlated features can share attribution; model association does not imply causality.

## Responsible use and monitoring

Sex, age, education and marriage are excluded from scoring, retained for audits. Exclusion does not remove proxy discrimination. Test subgroup tables report counts, event rates, AUC, Brier, calibration-in-the-large, errors and simulated selection rates. These are limited descriptive checks, not fairness certification. AUC needs both classes and uncertainty matters in small groups. No current Indian regulatory compliance claim is made.

PSI compares train-derived bins with random holdouts; a synthetic credit-limit shift checks sensitivity. There is no actual production or temporal drift evidence. Outcome-rate and Brier/missingness comparisons demonstrate future delayed-label monitoring. Train-score comparisons can include overfit effects. PSI cutoffs require context and bin sensitivity. Real deployment needs independent temporal validation, applicant selection-bias assessment, governance and human review.

## Architecture and artifacts

`src/`: reusable audit, feature, modeling, evaluation, calibration, explainability, decision, statistics and orchestration modules. `notebooks/`: executed narrative notebooks covering the analysis. `sql/`: executable DuckDB portfolio, segmentation, monitoring, decile and exposure queries. `app/`: seven-page Streamlit dashboard loading saved artifacts. `tests/`: core logic, leakage and dashboard tests. `docs/`: dataset and scope. `reports/`: executive summary, model card, findings, interview guide, verified resume and execution audit. `config/`: explicit assumptions. `artifacts/tables/run_summary.json`: numerical source of truth. Models, raw/processed data, database and virtual environment are ignored; small tables, reports, figures and notebook outputs are intentionally shareable. Review data attribution before redistributing.

Useful reading: [model card](reports/model_card.md), [interview guide](reports/interview_guide.md), [resume bullets](reports/resume_bullets.md), [final audit](reports/final_audit.md). All empirical headline metrics are rendered by code from the executed artifacts; the final audit verifies consistency and saved prediction replay.
"""
    write('docs/technical_methodology.md',README.replace('](reports/','](../reports/').replace('](artifacts/','](../artifacts/'))
    from .presentation import readme
    write('README.md',readme(s))
    write('reports/executive_summary.md',f"# Executive summary\n\n{empirical}\n\nPriority is reliable review triage, not automatic rejection. Utilization and repayment-history associations support segmentation; they do not establish causality. The main risks are historical geographic mismatch, unknown contractual event definition, selection bias, proxy discrimination and absent losses. Before real banking use, obtain bank-specific outcome definitions, temporal cohorts, affordability and profitability inputs, independent validation and governance.\n")
    write('reports/model_card.md',f"""# Model card

## Intended / non-intended use

Educational retrospective existing-cardholder review prioritization. Not production underwriting, adverse-action notices, Indian regulatory compliance, pricing, provisioning or real applicant advice.

## Data, model and evaluation

{empirical}

{table(partitions)}

Input schema: `src.features.BASE`. Demographics and ID are excluded; engineering and train-fitted preprocessing live in the saved pipeline. Training/calibration/selection/test profiles are disjoint. Selection minimizes validation Brier, then log loss. Fixed XGBoost parameters appear in `src/modeling.py`. The fitted logistic benchmark uses regularization, unlike the separate exploratory inferential logit.

## Calibration and boundaries

Selected mapping: `{selected}`. Calibration is fitted on calibration clients; validation selects the model/map and review capacity. Test Brier {metrics['brier']:.6f}. See the full calibration comparison and reliability intervals. Boundaries `{s['band_boundaries']}`; review threshold {s['review_threshold']:.6f}. Band policy is a relative-risk/review-capacity scenario, not an economically optimal approval rule.

## Fairness and limitations

{table(sub[['attribute','group','n','event_rate','roc_auc','brier','calibration_gap','false_positive_rate','false_negative_rate','selection_rate']])}

Demographic omission does not prove fairness. Subgroup metrics have sampling uncertainty and reflect different observed event rates. No protected-attribute mitigation optimization or legal fairness certification is claimed. Taiwan historical clients and random holdouts cannot justify Indian lending transfer. LGD and EAD are unobserved. Reject inference is not implemented because rejected applicants are unavailable.

## Explainability and monitoring

TreeSHAP explains raw XGBoost log-odds, not calibrated probabilities, and is non-causal. Monitoring compares random partitions plus a labelled synthetic shift; no temporal production history exists. Real monitoring would use scored cohorts, feature/missingness checks, delayed outcome maturation, calibration and escalation owners. Recalibration/refitting requires independent validation; monitoring alone must not automatically change lending policy.
""")
    write('reports/statistical_findings.md',f"""# Statistical findings (training partition only)

{table(stats[['feature','test','statistic','p_value','p_holm','effect_size','p_value_underflow']])}

{statistical_details}

Mann–Whitney tests compare distributions; median-shift interpretation requires similarly shaped distributions. Ties are corrected by the asymptotic implementation. The sign of rank-biserial tells whether the event group tends higher. Holm-adjusted p-values control multiplicity within this explicitly declared family. An exported p-value of zero denotes floating-point numerical underflow, not an exactly zero probability; the executed table flags this explicitly. Small p-values can accompany weak effects; practical value must be assessed using effect size and held-out decision trade-offs. Sparse education/marriage categories are pooled as other-or-undocumented only for the association test; every expected cell must pass the enforced adequacy check. Demographic associations are descriptive and do not justify using protected attributes in scoring. Pearson point-biserial relationships may be weak even when nonlinear relationships exist.

Delay segment Wilson intervals:

{table(read('delay_event_rates.csv'))}

Reduced logit, odds ratios per training standard deviation:

{table(read('logistic_inference.csv'))}

Exploratory reduced model: limit, utilization, delay count and payment-to-limit with intercept; robust HC1 covariance. H0 for each coefficient is zero conditional log-odds association; H1 is nonzero. Assumptions include adequate logit specification, no unmodeled client dependence and no perfect separation. Robust covariance does not correct omitted-variable bias or selection. Odds ratios describe associations conditional on these predictors; they are not intervention effects. Inference is distinct from regularized predictive-logit coefficients. VIF is reported on this reduced design, not the redundant full engineered matrix.

{table(read('vif.csv'))}

Spearman correlations are in `artifacts/tables/spearman_correlation.csv`; they diagnose monotone redundancy rather than causation.
""")
    write('reports/eda_interpretation.md',"""# Decision-oriented chart guide

- `delinquency_event_rate.png`: whether recent reported delay distinguishes review groups, with training-only Wilson intervals. It does not establish why payment events occur.
- `eda_LIMIT_BAL.png`: whether limit quantiles show different observed event rates; limits reflect lender selection as well as borrower characteristics, and are not EAD.
- `eda_utilization_latest.png`: whether statement-to-limit bands provide useful review segmentation. Negative/over-limit values are retained; utilization is not affordability.
- `eda_payment_to_limit.png`: whether historical payment intensity adds behavioural differentiation. The measure is not income or contractual installment coverage.
- `roc.png`: rank discrimination compared with the naive benchmark. Ranking alone cannot set a cost-optimal policy.
- `precision_recall.png`: how event capture trades against review precision under imbalance; prevalence is the reference.
- `calibration.png`: whether scored groups' outcomes track mean predicted event probabilities; finite-bin uncertainty and geographic transfer remain concerns.
- `risk_bands.png`: whether validation-defined bands produce differentiated holdout outcomes, useful for simulated review allocation.
- `shap_summary.png` and local waterfalls: which encoded predictors the raw tree associates with score differences. Attribution is not causal and does not explain calibrated probabilities additively.
""")
    original=(ROOT/'docs/data_dictionary.md').read_text(encoding='utf-8-sig').split('\n## Executed')[0]
    write('docs/data_dictionary.md',original+f"""\n## Executed source audit

{empirical.split('The predeclared')[0]}

`ID`: unique source client identifier, excluded. `LIMIT_BAL`: granted credit limit in NT dollars, positive. `SEX`: source codes male/female; audit only. `EDUCATION`: documented graduate/university/high-school/other categories, with additional undocumented codes; audit only. `MARRIAGE`: source marital categories with an undocumented zero code; audit only. `AGE`: years; audit only.

`PAY_0`, `PAY_2` … `PAY_6`: September through April repayment statuses. Source documents -1 as duly paid and positive codes as months delayed; -2 and 0 are present but their precise semantic distinctions are not sufficiently documented here. All statuses enter predictive models as categorical codes, not invented numeric delay semantics. Engineered delay counts use only positive statuses, and maximum delay clips nonpositive codes to zero.

`BILL_AMT1` … `BILL_AMT6`: statement amounts, September through April, NT dollars. Negative statements can be account credits and are retained. `PAY_AMT1` … `PAY_AMT6`: recorded historical payment amounts in NT dollars. Their within-month alignment does not establish contractual due amounts. `event`: renamed source default-payment-next-month target, zero/one. Observed positive target rate is in the audit above.

Undocumented education records: {s['undocumented_education_rows']}; undocumented marriage records: {s['undocumented_marriage_rows']}; repayment -2 cells: {s['repayment_minus_two_cells']}. These are not converted into missing observations. Education/marriage are excluded from modeling; unknown repayment vocabulary at inference is handled by the train-fitted encoder. Full code frequencies are in `categorical_codes.csv`.

Full schema, extrema/cardinalities/missingness: `data_audit.csv`. Numerical quantiles: `numerical_distributions.csv`. Outliers are retained with regularized shallow trees and scaling for logistic; no arbitrary winsorization or negative-bill deletion. Duplicate full rows: {s['duplicate_rows']}; duplicate profiles are grouped, not silently removed. Dataset lacks income, employment, new requested-loan amount, LGD, EAD and recovery data.
""")
    write('reports/resume_bullets.md',f"""# Retail Credit Risk & Explainable Loan Decision Engine

Technology: Python, pandas, SciPy, statsmodels, scikit-learn, XGBoost, SHAP, DuckDB, SQL, Streamlit, pytest.

- Built a reproducible credit-risk analysis pipeline for {s['rows']:,} public cardholder records, with duplicate-profile holdouts and train-only preprocessing.
- Compared baseline, logistic and XGBoost models with independent calibration; selected {selected}, achieving held-out ROC-AUC {metrics['roc_auc']:.4f}, average precision {metrics['average_precision']:.4f} and Brier {metrics['brier']:.4f}.
- Developed SQL-backed segmentation, SHAP explanations and a review simulation capturing {policy.event_capture:.2%} of holdout events at a {policy.review_rate:.2%} review rate, with clearly assumed loss scenarios.

Traceability: `artifacts/tables/run_summary.json` for source/model metrics; `artifacts/tables/threshold_tradeoffs.csv` at the summary's review threshold for simulation rates. These are portfolio experiments, not bank deployment or profit impact.
""")
    from .interview import write_guide
    write_guide(s)
