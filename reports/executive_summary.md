# Executive summary

Executed source audit: 30,000 clients, 6,636 positive events (22.12%), 0 missing cells, 56 duplicate predictor profiles, 3932 negative bill cells. Source file SHA256: `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`.

The predeclared validation-Brier selection rule selected **xgboost:sigmoid**. Test ROC-AUC **0.7808**, average precision **0.5591**, KS **0.4302**, Brier **0.1325**. At the validation-derived review threshold **0.283990**, precision is **0.5204**, recall **0.5731**, F1 **0.5455**.

The simulated acceptance fraction on the untouched test is **75.92%**; accepted clients' observed event rate is **12.29%**. Review captures **57.31%** of observed events. These are retrospective same-sample simulations, not realized approval or profit impacts.

Calibration is not claimed to improve every holdout metric: the selected model's raw test Brier is **0.132463**, versus selected-mapping Brier **0.132508**. Selection remains based on the independent validation result; test differences are reported without switching winners.


Priority is reliable review triage, not automatic rejection. Utilization and repayment-history associations support segmentation; they do not establish causality. The main risks are historical geographic mismatch, unknown contractual event definition, selection bias, proxy discrimination and absent losses. Before real banking use, obtain bank-specific outcome definitions, temporal cohorts, affordability and profitability inputs, independent validation and governance.
