# Dataset scope before acquisition

Source: UCI Default of Credit Card Clients, Yeh (2009), DOI https://doi.org/10.24432/C55S3H, CC BY 4.0. Official page: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients.

One row represents an existing Taiwanese credit-card client. The source reports 30000 clients and 23 predictors; these are source metadata, to be checked after acquisition. Target: default payment next month, binary. Repayment histories cover April to September 2005. The target is a next-month payment event, not a documented Basel default, lifetime default, Indian personal-loan default or new-applicant outcome.

Features: credit limit (NT dollars), age, sex, education, marital status, six repayment-status codes, six monthly bill amounts and six monthly payment amounts. No income, employment length, loss severity, recoveries, actual exposure at default or customer acquisition timestamps are available. No income-based features will be invented. The source reports no missing values; actual missingness, undocumented category codes, duplicates and event balance will be computed in the audit.

Home Credit is richer but requires competition access and multiple large relational files. UCI enables an unauthenticated, small, fully reproducible public pipeline and retains meaningful behavioural credit features.

## Executed source audit

Executed source audit: 30,000 clients, 6,636 positive events (22.12%), 0 missing cells, 56 duplicate predictor profiles, 3932 negative bill cells. Source file SHA256: `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`.



`ID`: unique source client identifier, excluded. `LIMIT_BAL`: granted credit limit in NT dollars, positive. `SEX`: source codes male/female; audit only. `EDUCATION`: documented graduate/university/high-school/other categories, with additional undocumented codes; audit only. `MARRIAGE`: source marital categories with an undocumented zero code; audit only. `AGE`: years; audit only.

`PAY_0`, `PAY_2` … `PAY_6`: September through April repayment statuses. Source documents -1 as duly paid and positive codes as months delayed; -2 and 0 are present but their precise semantic distinctions are not sufficiently documented here. All statuses enter predictive models as categorical codes, not invented numeric delay semantics. Engineered delay counts use only positive statuses, and maximum delay clips nonpositive codes to zero.

`BILL_AMT1` … `BILL_AMT6`: statement amounts, September through April, NT dollars. Negative statements can be account credits and are retained. `PAY_AMT1` … `PAY_AMT6`: recorded historical payment amounts in NT dollars. Their within-month alignment does not establish contractual due amounts. `event`: renamed source default-payment-next-month target, zero/one. Observed positive target rate is in the audit above.

Undocumented education records: 345; undocumented marriage records: 54; repayment -2 cells: 24415. These are not converted into missing observations. Education/marriage are excluded from modeling; unknown repayment vocabulary at inference is handled by the train-fitted encoder. Full code frequencies are in `categorical_codes.csv`.

Full schema, extrema/cardinalities/missingness: `data_audit.csv`. Numerical quantiles: `numerical_distributions.csv`. Outliers are retained with regularized shallow trees and scaling for logistic; no arbitrary winsorization or negative-bill deletion. Duplicate full rows: 0; duplicate profiles are grouped, not silently removed. Dataset lacks income, employment, new requested-loan amount, LGD, EAD and recovery data.
