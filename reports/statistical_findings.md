# Statistical findings (training partition only)

| feature | test | statistic | p_value | p_holm | effect_size | p_value_underflow |
| --- | --- | --- | --- | --- | --- | --- |
| LIMIT_BAL | Mann-Whitney U | 9219051.5000 | 9.341e-85 | 5.604e-84 | -0.2481 | False |
| utilization_latest | Mann-Whitney U | 13583993.5000 | 2.326e-17 | 1.163e-16 | 0.1079 | False |
| months_delayed | Mann-Whitney U | 17970970.5000 | 0.0000 | 0.0000 | 0.4657 | True |
| payment_to_limit | Mann-Whitney U | 11839888.5000 | 0.0070 | 0.0140 | -0.0343 | False |
| SEX | Chi-square | 17.5305 | 2.827e-05 | 8.482e-05 | 0.0383 | False |
| EDUCATION | Chi-square | 74.4168 | 4.832e-16 | 1.933e-15 | 0.0790 | False |
| MARRIAGE | Chi-square | 8.4242 | 0.0148 | 0.0148 | 0.0266 | False |

### LIMIT_BAL

H0: Outcome groups have identical feature distributions. H1: Outcome groups differ in feature distribution. Assumptions: Independent grouped clients; ordinal/continuous measure; ties corrected; not a causal test. Test statistic 9.21905e+06; raw p-value 9.3405e-85; Holm p-value 5.6043e-84; effect size -0.24809 (rank-biserial: positive means event group tends higher). Interpretation: evidence against H0 at the declared conventional significance level; this does not measure causal effects. Business relevance: Granted-limit association may reflect lender selection and borrower differences; it is not affordability or a causal limit effect.

### utilization_latest

H0: Outcome groups have identical feature distributions. H1: Outcome groups differ in feature distribution. Assumptions: Independent grouped clients; ordinal/continuous measure; ties corrected; not a causal test. Test statistic 1.3584e+07; raw p-value 2.32605e-17; Holm p-value 1.16302e-16; effect size 0.107917 (rank-biserial: positive means event group tends higher). Interpretation: evidence against H0 at the declared conventional significance level; this does not measure causal effects. Business relevance: Statement-to-limit differences can guide behavioural review segments; they cannot establish income burden.

### months_delayed

H0: Outcome groups have identical feature distributions. H1: Outcome groups differ in feature distribution. Assumptions: Independent grouped clients; ordinal/continuous measure; ties corrected; not a causal test. Test statistic 1.7971e+07; raw p-value 0; Holm p-value 0; effect size 0.465721 (rank-biserial: positive means event group tends higher). Interpretation: evidence against H0 at the declared conventional significance level; this does not measure causal effects. Business relevance: Historical delay frequency is a directly interpretable behavioural review signal; its out-of-sample value is tested by the modeling layer.

### payment_to_limit

H0: Outcome groups have identical feature distributions. H1: Outcome groups differ in feature distribution. Assumptions: Independent grouped clients; ordinal/continuous measure; ties corrected; not a causal test. Test statistic 1.18399e+07; raw p-value 0.00700759; Holm p-value 0.0140152; effect size -0.0343332 (rank-biserial: positive means event group tends higher). Interpretation: evidence against H0 at the declared conventional significance level; this does not measure causal effects. Business relevance: The small rank association must not be oversold merely because the p-value is below the conventional significance cutoff.

### SEX

H0: Feature and outcome are independent. H1: Feature and outcome are associated. Assumptions: Independent clients; minimum expected cell 1052.547; none. Test statistic 17.5305; raw p-value 2.82736e-05; Holm p-value 8.48207e-05; effect size 0.0383398 (Cramer's V). Interpretation: evidence against H0 at the declared conventional significance level; this does not measure causal effects. Business relevance: Audit-only association supports investigation of population differences, not demographic scoring or fairness certification.

### EDUCATION

H0: Feature and outcome are independent. H1: Feature and outcome are associated. Assumptions: Independent clients; minimum expected cell 43.625; Source other and undocumented education codes pooled; no semantic meaning assigned to undocumented codes. Test statistic 74.4168; raw p-value 4.83155e-16; Holm p-value 1.93262e-15; effect size 0.0789929 (Cramer's V). Interpretation: evidence against H0 at the declared conventional significance level; this does not measure causal effects. Business relevance: Pooled audit categories describe sampled differences; neither undocumented category meaning nor causal education effects are inferred.

### MARRIAGE

H0: Feature and outcome are independent. H1: Feature and outcome are associated. Assumptions: Independent clients; minimum expected cell 31.003; Other and undocumented marital codes pooled to avoid sparse expected cells. Test statistic 8.42422; raw p-value 0.0148151; Holm p-value 0.0148151; effect size 0.0265777 (Cramer's V). Interpretation: evidence against H0 at the declared conventional significance level; this does not measure causal effects. Business relevance: A small audit association is not a basis for demographic underwriting; evaluate proxies and subgroup uncertainty.

Mann–Whitney tests compare distributions; median-shift interpretation requires similarly shaped distributions. Ties are corrected by the asymptotic implementation. The sign of rank-biserial tells whether the event group tends higher. Holm-adjusted p-values control multiplicity within this explicitly declared family. An exported p-value of zero denotes floating-point numerical underflow, not an exactly zero probability; the executed table flags this explicitly. Small p-values can accompany weak effects; practical value must be assessed using effect size and held-out decision trade-offs. Sparse education/marriage categories are pooled as other-or-undocumented only for the association test; every expected cell must pass the enforced adequacy check. Demographic associations are descriptive and do not justify using protected attributes in scoring. Pearson point-biserial relationships may be weak even when nonlinear relationships exist.

Delay segment Wilson intervals:

| recent_delay | size | sum | mean | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| no positive delay | 9234 | 1250 | 0.1354 | 0.1285 | 0.1425 |
| reported delay | 2692 | 1391 | 0.5167 | 0.4978 | 0.5356 |

Reduced logit, odds ratios per training standard deviation:

| feature | coefficient | odds_ratio_per_training_sd | or_ci_low | or_ci_high | p_value |
| --- | --- | --- | --- | --- | --- |
| const | -1.4204 | 0.2416 | 0.2300 | 0.2538 | 0.0000 |
| LIMIT_BAL | -0.2619 | 0.7696 | 0.7228 | 0.8194 | 2.720e-16 |
| utilization_latest | 0.0243 | 1.0246 | 0.9703 | 1.0819 | 0.3821 |
| months_delayed | 0.8029 | 2.2320 | 2.1289 | 2.3401 | 6.614e-243 |
| payment_to_limit | -0.1499 | 0.8608 | 0.8038 | 0.9218 | 1.788e-05 |

Exploratory reduced model: limit, utilization, delay count and payment-to-limit with intercept; robust HC1 covariance. H0 for each coefficient is zero conditional log-odds association; H1 is nonzero. Assumptions include adequate logit specification, no unmodeled client dependence and no perfect separation. Robust covariance does not correct omitted-variable bias or selection. Odds ratios describe associations conditional on these predictors; they are not intervention effects. Inference is distinct from regularized predictive-logit coefficients. VIF is reported on this reduced design, not the redundant full engineered matrix.

| feature | vif |
| --- | --- |
| const | 1.0000 |
| LIMIT_BAL | 1.2246 |
| utilization_latest | 1.1889 |
| months_delayed | 1.0725 |
| payment_to_limit | 1.0629 |

Spearman correlations are in `artifacts/tables/spearman_correlation.csv`; they diagnose monotone redundancy rather than causation.
