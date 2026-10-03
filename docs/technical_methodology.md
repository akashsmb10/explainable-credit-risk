# Retail Credit Risk & Explainable Loan Decision Engine

An educational banking analytics portfolio: historical repayment data → statistical understanding → risk ranking → probability calibration → explainability → review-capacity simulation → assumed-loss scenarios → monitoring demonstrations.

This project asks how a risk analyst can prioritize existing cardholders for review while distinguishing predictive evidence from business assumptions. It does not recommend ICICI lending policy.

## Verified results

Executed source audit: 30,000 clients, 6,636 positive events (22.12%), 0 missing cells, 56 duplicate predictor profiles, 3932 negative bill cells. Source file SHA256: `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`.

The predeclared validation-Brier selection rule selected **xgboost:sigmoid**. Test ROC-AUC **0.7808**, average precision **0.5591**, KS **0.4302**, Brier **0.1325**. At the validation-derived review threshold **0.283990**, precision is **0.5204**, recall **0.5731**, F1 **0.5455**.

The simulated acceptance fraction on the untouched test is **75.92%**; accepted clients' observed event rate is **12.29%**. Review captures **57.31%** of observed events. These are retrospective same-sample simulations, not realized approval or profit impacts.

Calibration is not claimed to improve every holdout metric: the selected model's raw test Brier is **0.132463**, versus selected-mapping Brier **0.132508**. Selection remains based on the independent validation result; test differences are reported without switching winners.


![Risk bands](../artifacts/figures/risk_bands.png)
![Calibration](../artifacts/figures/calibration.png)
![SHAP](../artifacts/figures/shap_summary.png)

## Reproduce (Windows PowerShell, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m src.pipeline --clean
.\.venv\Scripts\python.exe -m src.notebooks
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.final_audit
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Run commands from the repository root. First pipeline run downloads the official UCI Excel archive; later runs reuse the source and regenerate all downstream models/tables/figures without loading old fitted artifacts. Unix equivalents use `.venv/bin/python`. Network access is necessary for dependencies and first data acquisition. The final audit checks the app with Streamlit AppTest and launches a temporary HTTP server for a health check; the interactive command above starts your own dashboard.

## Dataset and target

[UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), Yeh (2009), [DOI](https://doi.org/10.24432/C55S3H), CC BY 4.0. Source documentation describes existing Taiwan cardholders, April–September 2005 histories, and a next-month default-payment label. The exact contractual definition and horizon beyond this label are unavailable. Scores estimate this dataset event, not bank-grade PD. Home Credit's access requirements and large relational download make it less reproducible for this self-contained project.

## Validation design

| partition | n | events | event_rate |
| --- | --- | --- | --- |
| test | 6010 | 1314 | 0.2186 |
| validation | 6015 | 1359 | 0.2259 |
| calibration | 6049 | 1322 | 0.2185 |
| train | 11926 | 2641 | 0.2214 |

StratifiedGroupKFold assigns identical predictor profiles to one fold; one fold is calibration, one validation, one test, and the remaining folds training. Preprocessing, imputation, scaling and category vocabularies are fitted only on training. Calibration maps are learned on the dedicated calibration partition. Candidate selection and score quantiles use validation. Test is scored after these are frozen. No time-based generalization can be tested because cohort dates are absent. Group stratification is approximate. Bootstrap intervals are ordinary client resamples, conditional on the fitted model and split, and can understate dependence/selection uncertainty.

## Model hierarchy and calibration

Base-rate benchmark, regularized logistic regression and shallow regularized XGBoost. Fixed model hyperparameters avoid an extensive search on a small validation set. All nontrivial models compare raw, sigmoid and isotonic probabilities. Selection minimizes validation Brier, with log loss as tie-breaker. Classification metrics in the comparison table use the conventional threshold; the policy results use the validation-derived threshold. Average precision is reported explicitly and should not be confused with trapezoidal PR-AUC.

Validation results:

| candidate | roc_auc | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- |
| xgboost:sigmoid | 0.7782 | 0.5369 | 0.1401 | 0.4405 |
| xgboost:raw | 0.7782 | 0.5369 | 0.1403 | 0.4412 |
| xgboost:isotonic | 0.7777 | 0.5177 | 0.1407 | 0.4427 |
| logistic:isotonic | 0.7574 | 0.4916 | 0.1433 | 0.4629 |
| logistic:sigmoid | 0.7589 | 0.5080 | 0.1433 | 0.4524 |
| logistic:raw | 0.7589 | 0.5080 | 0.1437 | 0.4541 |
| base_rate:raw | 0.5000 | 0.2259 | 0.1749 | 0.5344 |

Test results (descriptive, never used for selection):

| candidate | roc_auc | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- |
| base_rate:raw | 0.5000 | 0.2186 | 0.1708 | 0.5252 |
| logistic:raw | 0.7634 | 0.5373 | 0.1355 | 0.4355 |
| logistic:sigmoid | 0.7634 | 0.5373 | 0.1356 | 0.4354 |
| logistic:isotonic | 0.7614 | 0.5128 | 0.1355 | 0.4340 |
| xgboost:raw | 0.7808 | 0.5591 | 0.1325 | 0.4242 |
| xgboost:sigmoid | 0.7808 | 0.5591 | 0.1325 | 0.4242 |
| xgboost:isotonic | 0.7799 | 0.5365 | 0.1328 | 0.4260 |

Confidence intervals:

| metric | ci_low | ci_high |
| --- | --- | --- |
| roc_auc | 0.7680 | 0.7945 |
| average_precision | 0.5347 | 0.5928 |
| brier | 0.1269 | 0.1372 |

Good calibration means that a group scored at a given event probability has roughly that event frequency under the same population and target horizon. It does not guarantee an individual outcome. Rank performance and probability quality can differ; isotonic's ties may alter ranking. Test reliability bins include Wilson event-rate intervals and deciles compare scores with outcomes.

## Decision assumptions

Risk boundaries are the configured validation median, upper quartile and upper-tail quantile: `[0.14707765807900644, 0.2839901475140817, 0.5722191552185503]`. Labels describe relative model risk. Manual review starts at the upper-quartile boundary, reflecting a scenario with approximately one quarter of validation cases prioritized. This is an assumed review-capacity rule, not an optimal underwriting cutoff. Ties can change the realized review fraction. No lending benefit, rejection cost, affordability, legal rule or profit function is observed; a bank would require all of these before determining acceptance. Threshold tables retain false positives, false negatives, event capture, acceptance fraction and accepted event rate. Low score does not ensure eligibility or affordability.

## Expected-loss scenarios

Banking EL = PD × LGD × EAD. Here the probability is a model-estimated dataset event score; LGD and EAD are explicitly assumed. LGD is a fraction of exposure lost conditional on default; EAD is outstanding exposure at the default time, not the observed card limit. Scenario currency is NT dollars. The default display assumes LGD 0.45 and EAD 100000; the configuration and scenario table vary both. These scenario values are not empirical findings. Doubling assumed exposure at the same score and LGD doubles scenario loss. No real provision, IFRS 9 ECL or capital estimate is made.

## Statistics, features and explanations

Training-only Mann–Whitney comparisons, point-biserial associations, Spearman correlations, demographic chi-square/Cramer's V, Holm correction, Wilson intervals, VIF and a reduced logit with robust covariance are executed. Rank-biserial effect sizes assess practical strength separately from p-values. Inference is exploratory, conditional on specification and approximate independent clients; it is not causal or a separate confirmatory experiment. The reduced inferential model is distinct from the penalized predictive logistic model. See [statistical findings](../reports/statistical_findings.md).

Behavioural features quantify utilization, payment-to-limit, bill change and delay frequency. No income or employment ratio is manufactured. Latest/mean bills divided by limit can be negative or exceed unity; values are retained and documented. Payment-to-positive-bill is a history proxy, not a matched statement repayment or affordability measure.

Exact TreeSHAP in XGBoost explains raw log-odds. SHAP plots use the SHAP plotting package; contributions are checked against raw margins. Global importance is computed on a reproducible held-out sample including representative clients. Local contributions are saved for every test client; waterfalls cover low, medium, high and boundary clients. Calibration changes the output scale; TreeSHAP does not add to the calibrated score. If a different model wins, tree explanations remain benchmark-specific. Correlated features can share attribution; model association does not imply causality.

## Responsible use and monitoring

Sex, age, education and marriage are excluded from scoring, retained for audits. Exclusion does not remove proxy discrimination. Test subgroup tables report counts, event rates, AUC, Brier, calibration-in-the-large, errors and simulated selection rates. These are limited descriptive checks, not fairness certification. AUC needs both classes and uncertainty matters in small groups. No current Indian regulatory compliance claim is made.

PSI compares train-derived bins with random holdouts; a synthetic credit-limit shift checks sensitivity. There is no actual production or temporal drift evidence. Outcome-rate and Brier/missingness comparisons demonstrate future delayed-label monitoring. Train-score comparisons can include overfit effects. PSI cutoffs require context and bin sensitivity. Real deployment needs independent temporal validation, applicant selection-bias assessment, governance and human review.

## Architecture and artifacts

`src/`: reusable audit, feature, modeling, evaluation, calibration, explainability, decision, statistics and orchestration modules. `notebooks/`: executed narrative notebooks covering the analysis. `sql/`: executable DuckDB portfolio, segmentation, monitoring, decile and exposure queries. `app/`: seven-page Streamlit dashboard loading saved artifacts. `tests/`: core logic, leakage and dashboard tests. `docs/`: dataset and scope. `reports/`: executive summary, model card, findings, interview guide, verified resume and execution audit. `config/`: explicit assumptions. `artifacts/tables/run_summary.json`: numerical source of truth. Models, raw/processed data, database and virtual environment are ignored; small tables, reports, figures and notebook outputs are intentionally shareable. Review data attribution before redistributing.

Useful reading: [model card](../reports/model_card.md), [interview guide](../reports/interview_guide.md), [resume bullets](../reports/resume_bullets.md), [final audit](../reports/final_audit.md). All empirical headline metrics are rendered by code from the executed artifacts; the final audit verifies consistency and saved prediction replay.
