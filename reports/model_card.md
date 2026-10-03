# Model card

## Intended / non-intended use

Educational retrospective existing-cardholder review prioritization. Not production underwriting, adverse-action notices, Indian regulatory compliance, pricing, provisioning or real applicant advice.

## Data, model and evaluation

Executed source audit: 30,000 clients, 6,636 positive events (22.12%), 0 missing cells, 56 duplicate predictor profiles, 3932 negative bill cells. Source file SHA256: `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`.

The predeclared validation-Brier selection rule selected **xgboost:sigmoid**. Test ROC-AUC **0.7808**, average precision **0.5591**, KS **0.4302**, Brier **0.1325**. At the validation-derived review threshold **0.283990**, precision is **0.5204**, recall **0.5731**, F1 **0.5455**.

The simulated acceptance fraction on the untouched test is **75.92%**; accepted clients' observed event rate is **12.29%**. Review captures **57.31%** of observed events. These are retrospective same-sample simulations, not realized approval or profit impacts.

Calibration is not claimed to improve every holdout metric: the selected model's raw test Brier is **0.132463**, versus selected-mapping Brier **0.132508**. Selection remains based on the independent validation result; test differences are reported without switching winners.


| partition | n | events | event_rate |
| --- | --- | --- | --- |
| test | 6010 | 1314 | 0.2186 |
| validation | 6015 | 1359 | 0.2259 |
| calibration | 6049 | 1322 | 0.2185 |
| train | 11926 | 2641 | 0.2214 |

Input schema: `src.features.BASE`. Demographics and ID are excluded; engineering and train-fitted preprocessing live in the saved pipeline. Training/calibration/selection/test profiles are disjoint. Selection minimizes validation Brier, then log loss. Fixed XGBoost parameters appear in `src/modeling.py`. The fitted logistic benchmark uses regularization, unlike the separate exploratory inferential logit.

## Calibration and boundaries

Selected mapping: `xgboost:sigmoid`. Calibration is fitted on calibration clients; validation selects the model/map and review capacity. Test Brier 0.132508. See the full calibration comparison and reliability intervals. Boundaries `[0.14707765807900644, 0.2839901475140817, 0.5722191552185503]`; review threshold 0.283990. Band policy is a relative-risk/review-capacity scenario, not an economically optimal approval rule.

## Fairness and limitations

| attribute | group | n | event_rate | roc_auc | brier | calibration_gap | false_positive_rate | false_negative_rate | selection_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SEX | 1 | 2346 | 0.2327 | 0.7767 | 0.1392 | -0.0037 | 0.1583 | 0.4249 | 0.7447 |
| SEX | 2 | 3664 | 0.2096 | 0.7831 | 0.1282 | 0.0035 | 0.1412 | 0.4284 | 0.7686 |
| age_band | up to 30 | 2200 | 0.2109 | 0.7822 | 0.1291 | 0.0205 | 0.1682 | 0.3966 | 0.7400 |
| age_band | 31 to 45 | 2907 | 0.2133 | 0.7760 | 0.1317 | -0.0091 | 0.1277 | 0.4710 | 0.7867 |
| age_band | above 45 | 903 | 0.2547 | 0.7947 | 0.1435 | -0.0162 | 0.1634 | 0.3696 | 0.7176 |

Demographic omission does not prove fairness. Subgroup metrics have sampling uncertainty and reflect different observed event rates. No protected-attribute mitigation optimization or legal fairness certification is claimed. Taiwan historical clients and random holdouts cannot justify Indian lending transfer. LGD and EAD are unobserved. Reject inference is not implemented because rejected applicants are unavailable.

## Explainability and monitoring

TreeSHAP explains raw XGBoost log-odds, not calibrated probabilities, and is non-causal. Monitoring compares random partitions plus a labelled synthetic shift; no temporal production history exists. Real monitoring would use scored cohorts, feature/missingness checks, delayed outcome maturation, calibration and escalation owners. Recalibration/refitting requires independent validation; monitoring alone must not automatically change lending policy.
