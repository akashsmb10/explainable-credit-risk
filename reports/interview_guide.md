# Interview guide: defend the actual project

## Your opening explanation

I built a historical existing-cardholder decision-support project using 30,000 UCI records. I compared a base-rate predictor, logistic regression and XGBoost, then independently fitted probability calibration. The predeclared validation rule selected xgboost:sigmoid. Held-out AUC is 0.7808; Brier is 0.1325. My contribution is the separation of statistical evidence, model scores, review-capacity assumptions and hypothetical losses rather than a claim of production loan approval.

## Beginner to advanced foundations

Credit risk is uncertainty about meeting contractual payment obligations. A model predicts a defined outcome over a defined horizon. Bank PD is default probability for that contract/horizon; LGD is conditional loss severity; EAD is default-time exposure. Their product forms an elementary expected-loss framework. This project uses an event probability and assumed severity/exposure, so its loss is a scenario proxy.

Logistic regression maps a linear predictor through the sigmoid function; log-odds coefficients are conditional associations. XGBoost sequentially adds regularized trees to reduce a differentiable loss, allowing nonlinearities. Class imbalance means most observations may be non-events; accuracy can hide poor event capture. ROC-AUC evaluates ranking; average precision weights precision by recall increments. KS measures separation of conditional score CDFs. Brier evaluates squared probability errors. Reliability curves compare average scores with event rates; ranking and calibration are different properties.

Leakage occurs when training/selection sees information unavailable at scoring or evaluation. Overfitting fits idiosyncrasies rather than stable patterns. Separate grouped holdouts, fixed model settings, train-only preprocessing and independent calibration/selection reduce these risks. A threshold converts scores into an action under stated capacity/cost constraints; it is not determined by AUC. SHAP attributes the model function relative to a baseline, not causality. PSI summarizes binned distribution change without proving deterioration. Fairness requires context, subgroup uncertainty and governance, not one similar metric. Statistical significance concerns evidence against a null under assumptions; practical significance concerns decision usefulness.

Mathematically, logistic p=1/(1+exp(-xβ)); Brier=mean((p-y)^2); KS=sup_s |F_positive(s)-F_negative(s)|; PSI=sum_i (q_i-r_i) log(q_i/r_i). Inferential odds ratios are exp(β), conditional on the chosen design. Calibration estimates E[Y|score] under the sampled distribution. A proper scoring rule encourages truthful probabilities in expectation, while rank metrics are invariant to strictly monotonic transformations. Isotonic steps can introduce ties. No mathematical technique overcomes mismatched targets or unavailable bank evidence.

## Questions and deeper follow-ups
### Q1. What credit-risk problem did you solve?

**Strong answer:** Retrospective prioritization of existing cardholders for review using a next-month payment-event label.

**Possible follow-up:** Would this score new loan applicants?

**Deeper answer:** New applicants may lack these repayment histories; selection bias and domain shift require new data, outcome definitions and independent validation.

### Q2. Can your probability actually be called PD?

**Strong answer:** It estimates the UCI next-month default-payment event, with held-out calibration checks; it is not bank-grade contractual PD.

**Possible follow-up:** What definition is missing?

**Deeper answer:** The precise contractual default trigger, delinquency cure rules and regulatory horizon are not available. A score cannot be relabelled as lifetime or regulatory default probability.

### Q3. What is LGD?

**Strong answer:** Loss given default is the fraction of exposure lost conditional on default, after recovery and relevant costs.

**Possible follow-up:** Did you estimate it?

**Deeper answer:** No recovery observations exist. LGD is a labelled scenario input rather than a trained target. Real estimation needs defaulted accounts, recovery timing and discounting.

### Q4. What is EAD?

**Strong answer:** Exposure at default is the amount outstanding at the moment of default.

**Possible follow-up:** Is the credit limit EAD?

**Deeper answer:** No. Limits are observed, actual default-time drawdowns are not. Our scenario uses assumed exposure separately from granted limits.

### Q5. What does expected loss mean?

**Strong answer:** EL multiplies event/default probability, conditional severity and default-time exposure; our result is a scenario proxy.

**Possible follow-up:** Why is it a proxy here?

**Deeper answer:** Using a payment-event score in place of contractual PD and assumed LGD/EAD cannot produce a valid provision or actual bank loss estimate.

### Q6. Why choose UCI rather than Home Credit?

**Strong answer:** The official small public download is reproducible without competition credentials and includes repayment behaviour.

**Possible follow-up:** What richness did you give up?

**Deeper answer:** Income, employment, prior applications and richer borrower context are absent. The project makes fewer financial claims rather than inventing missing quantities.

### Q7. Why use logistic regression?

**Strong answer:** It is a regularized interpretable benchmark with a log-odds link and efficient training.

**Possible follow-up:** Is it fully interpretable?

**Deeper answer:** Coefficients depend on encoding, scaling, correlated features and specification. Penalized predictive coefficients are not automatically inferential p-values.

### Q8. Why use XGBoost?

**Strong answer:** Shallow boosted trees capture nonlinear behavioural relationships and interactions.

**Possible follow-up:** How did you control complexity?

**Deeper answer:** Fixed shallow depth, learning rate, subsampling and regularization appear in src/modeling.py. Validation chooses among model/calibration pairs; no test-driven tuning occurs.

### Q9. Why include a base-rate model?

**Strong answer:** It makes discrimination and probability gains relative to a no-information score visible.

**Possible follow-up:** What is its Brier score?

**Deeper answer:** For prevalence pi, a correctly matched constant predictor has expected Brier pi times one minus pi. The pipeline measures actual holdout Brier rather than substituting that expectation.

### Q10. Why is accuracy misleading here?

**Strong answer:** A majority-class rule can look accurate while missing every positive event.

**Possible follow-up:** What did you emphasize?

**Deeper answer:** Average precision, ROC-AUC, KS, calibration and review trade-offs. Precision depends on prevalence; reported comparison classification metrics use a conventional threshold separately from the review policy.

### Q11. What is ROC-AUC?

**Strong answer:** The probability a random positive receives a higher score than a random negative, with ties handled.

**Possible follow-up:** What does it ignore?

**Deeper answer:** It does not encode probability calibration, business costs or review capacity; strong global discrimination may still perform poorly in a critical operating region.

### Q12. What is average precision?

**Strong answer:** A recall-increment weighted average of precision across score thresholds.

**Possible follow-up:** Is it identical to PR-AUC?

**Deeper answer:** No. Trapezoidal integration in precision–recall space differs. We report average precision explicitly, and its reference level is positive prevalence.

### Q13. What is KS?

**Strong answer:** The maximum separation between positive and negative score cumulative distributions.

**Possible follow-up:** Is it a policy threshold?

**Deeper answer:** Our KS statistic summarizes discrimination, not an automatic optimal operating point. Review thresholds are validation-score quantiles under a capacity scenario.

### Q14. What is the Brier score?

**Strong answer:** The mean squared difference between predicted probability and binary event.

**Possible follow-up:** Why select by it?

**Deeper answer:** It is a proper probability score combining calibration and discrimination/resolution. Because the decision layer uses probabilities, the rule predeclares lowest validation Brier, then log loss.

### Q15. What does calibration mean?

**Strong answer:** Among clients with similar estimated event probability, the observed event frequency should be similar.

**Possible follow-up:** Does a calibrated client have a guaranteed outcome?

**Deeper answer:** No. Calibration is a population frequency property dependent on event definition, horizon and distribution; individual outcomes remain uncertain.

### Q16. How did sigmoid calibration work?

**Strong answer:** A logistic mapping was learned on raw probability log-odds in a dedicated calibration partition.

**Possible follow-up:** Why not fit on test?

**Deeper answer:** Fitting the mapping on test would make evaluation optimistic. Calibration and selection use separate partitions and the mapping is frozen before test scoring.

### Q17. Why compare isotonic calibration?

**Strong answer:** A monotone nonparametric map can capture nonlinear probability distortion.

**Possible follow-up:** What are its risks?

**Deeper answer:** Small calibration sets cause stepwise overfitting and score ties. Ranking can change through ties; selection uses independent validation Brier rather than assuming isotonic is best.

### Q18. AUC improved but calibration worsened. Which would you choose?

**Strong answer:** For this probability-based decision simulation, follow the predeclared validation Brier rule and inspect ranking and operating trade-offs.

**Possible follow-up:** Would every bank choose that?

**Deeper answer:** No. A review-only ranking application might emphasize precision at capacity; pricing and loss estimation require reliable probabilities. The objective must be specified before test evaluation.

### Q19. Why use separate partitions?

**Strong answer:** Training learns preprocessing/model parameters, calibration learns maps, validation selects candidates and policy, and test estimates final generalization.

**Possible follow-up:** Why not refit after selection?

**Deeper answer:** This portfolio preserves a frozen, auditable fitted bundle. Refitting would require a new calibration design and fresh evaluation rather than reusing the reported test metrics.

### Q20. How did you prevent duplicate contamination?

**Strong answer:** All identical model-input profiles are grouped into the same fold, excluding identifier, demographics and target from the grouping hash.

**Possible follow-up:** Why exclude demographics from grouping?

**Deeper answer:** Two clients with the same scoring inputs but different audit-only demographics must still share one partition. Identifier disjointness alone does not prevent repeated scoring profiles across partitions.

### Q21. What leakage risks remained?

**Strong answer:** Target leakage is blocked by an explicit predictor allowlist and preprocessing is train-fitted.

**Possible follow-up:** Can historical payment features leak?

**Deeper answer:** They are legitimate only if available at the snapshot before the next-month label. Precise bank timestamps and event-triggered updates would need independent availability checks in production.

### Q22. Why not a time split?

**Strong answer:** All clients share the documented historical feature window and no distinct origination/cohort timestamps exist.

**Possible follow-up:** What can random holdout demonstrate?

**Deeper answer:** Only approximate same-sample generalization. It cannot establish stability across future macroeconomic regimes or lending-policy changes.

### Q23. How did you engineer utilization?

**Strong answer:** Latest and average historical statement amounts divided by granted limit.

**Possible follow-up:** Why keep negative or above-one ratios?

**Deeper answer:** Negative bills may reflect account credits and limits can be exceeded. They are audited rather than automatically deleted; utilization is not a measure of income or affordability.

### Q24. What is payment-to-positive-bill?

**Strong answer:** Mean historical payment divided by mean clipped-positive bills with a protective denominator floor.

**Possible follow-up:** Does it measure repayment coverage?

**Deeper answer:** Only a behavioural proxy. Monthly alignment and contractual due amounts are insufficient to interpret it as exact instalment coverage; large ratios can arise with very small bills.

### Q25. How did you treat repayment status codes?

**Strong answer:** Train-fitted one-hot encoding preserves distinct raw categories; delay aggregates use positive codes only.

**Possible follow-up:** What about undocumented codes?

**Deeper answer:** We do not assign unsupported meanings to nonpositive categories. Unknown inference categories are ignored by the encoder, and schema/availability monitoring is necessary.

### Q26. What is your hypothesis-testing strategy?

**Strong answer:** Training-only prespecified feature comparisons with effect sizes and Holm adjustment across the declared family.

**Possible follow-up:** What is the null?

**Deeper answer:** Mann–Whitney uses identical outcome-group distributions; chi-square uses feature/outcome independence. These are association tests, not causal or separately confirmed experiments.

### Q27. Why Mann–Whitney rather than only t-tests?

**Strong answer:** The chosen financial variables can be skewed and have extreme values, making rank-based comparisons useful.

**Possible follow-up:** Does it always test medians?

**Deeper answer:** No. It tests distributional differences; a location/median-shift interpretation requires comparable shapes. Ties and dependence affect inference.

### Q28. Statistical versus practical significance?

**Strong answer:** A small p-value need not imply a useful review signal; effect size and held-out operating results matter.

**Possible follow-up:** Which effect sizes did you use?

**Deeper answer:** Rank-biserial for continuous/rank comparisons, Cramer’s V for categorical association, and point-biserial correlations. Sample size can make weak effects statistically detectable.

### Q29. What is a Wilson interval?

**Strong answer:** A binomial proportion interval with better boundary behavior than a naive Wald interval.

**Possible follow-up:** What does it assume?

**Deeper answer:** Approximately independent Bernoulli observations with a common proportion within the segment; grouping identical profiles reduces contamination but does not establish real client independence.

### Q30. Why calculate VIF?

**Strong answer:** It checks redundancy in the reduced inferential design before interpreting coefficients.

**Possible follow-up:** Why not the entire feature matrix?

**Deeper answer:** Engineered summaries plus raw components can be strongly redundant. We report VIF on a deliberately reduced model, not falsely claim the whole prediction design lacks multicollinearity.

### Q31. What did the inferential logistic model estimate?

**Strong answer:** Conditional log-odds associations for standardized limit, utilization, delay count and payment intensity.

**Possible follow-up:** What do robust standard errors fix?

**Deeper answer:** HC1 changes covariance estimates under some variance misspecification; it does not correct omitted-variable bias, dependence, selection bias or an incorrect logit functional form.

### Q32. How did you choose thresholds?

**Strong answer:** Validation-score quantiles define relative-risk bands; upper-quartile review is an explicit capacity assumption.

**Possible follow-up:** Are they optimal?

**Deeper answer:** No cost or profitability objective is observed. Ties and population drift alter review fractions; a bank must choose constraints and economics before setting policy.

### Q33. Why not reject everyone above a PD threshold?

**Strong answer:** Scores omit affordability, customer economics, recoverability, uncertainty and legal/governance constraints; review is more defensible than automatic rejection.

**Possible follow-up:** What would economic optimization require?

**Deeper answer:** Incremental lending value, funding/operating costs, expected loss, exposure policy, customer behaviour and constraints, evaluated on representative outcomes with counterfactual selection issues addressed.

### Q34. What is SHAP?

**Strong answer:** A Shapley-value attribution assigning feature contributions relative to a defined model baseline.

**Possible follow-up:** What implementation did you use?

**Deeper answer:** Exact XGBoost TreeSHAP via pred_contribs, plotted with the SHAP package; contributions are checked to sum to raw tree margins, not calibrated probabilities.

### Q35. Why can SHAP not establish causality?

**Strong answer:** It explains the fitted prediction function under attribution assumptions, not the effect of intervening on a borrower feature.

**Possible follow-up:** How do correlated features matter?

**Deeper answer:** Attribution allocation depends on dependence assumptions and encoded representations. Redundant repayment features can split importance; causal inference requires a separate identification design.

### Q36. What are the main risk drivers?

**Strong answer:** The executed shap_global.csv ranks mean absolute raw-tree log-odds contributions on a reproducible held-out sample.

**Possible follow-up:** Can you quote them safely?

**Deeper answer:** Use the generated ranking, not a generic claim about income. Check the selected model: tree benchmark explanations are explicitly separate if another model wins.

### Q37. How did you analyze fairness?

**Strong answer:** Held-out sex and age groups include counts, event rates, AUC, Brier, average calibration gaps, error rates and simulated selection fractions.

**Possible follow-up:** Is the model fair?

**Deeper answer:** No such claim follows. Base-rate differences, uncertainty, proxy effects and competing fairness definitions require context and governance; excluding demographics is insufficient.

### Q38. What is PSI?

**Strong answer:** A binned population divergence summing proportion differences times log proportion ratios.

**Possible follow-up:** Did you observe production drift?

**Deeper answer:** No. Random partition comparison and a labelled synthetic limit shift demonstrate the framework only. Reference-bin choices, smoothing and sample size affect the result.

### Q39. What happens if the applicant population changes?

**Strong answer:** Reassess input availability, feature/score distributions, delayed outcome rates and calibration; escalate to human model governance.

**Possible follow-up:** Should PSI trigger immediate retraining?

**Deeper answer:** No. Identify data issues and changes in policy/population, evaluate with mature labels and independent validation; retraining and recalibration need controlled release decisions.

### Q40. How was SQL genuine?

**Strong answer:** Each query ran against the generated DuckDB portfolio table and exported its results.

**Possible follow-up:** What analyses exist?

**Deeper answer:** Split-level portfolio summary, held-out band segmentation, subgroup monitoring, score deciles and assumed high-risk exposure loss; observed limits are explicitly distinct from scenario EAD.

### Q41. What assumptions did you make?

**Strong answer:** Common source snapshot, same-sample holdout applicability, capacity-based review quantiles and explicitly assumed LGD/EAD.

**Possible follow-up:** Which are most fragile?

**Deeper answer:** Geographic/time transfer, unknown contractual default definition and unobserved loss severity/exposure prevent production underwriting and loss estimation claims.

### Q42. What would differ inside a real bank?

**Strong answer:** Bank-specific data governance, as-of feature availability, contractual targets, temporal validation, affordability and economic policy.

**Possible follow-up:** What operational controls are needed?

**Deeper answer:** Access control, audit logs, model inventory/versioning, independent validation, monitoring ownership, manual override and controlled deployment; this local portfolio does not implement bank infrastructure.

### Q43. How did you test the dashboard?

**Strong answer:** Artifact-only Streamlit pages are exercised with AppTest, and the final audit launches a temporary server and checks its health.

**Possible follow-up:** Does that establish usability?

**Deeper answer:** No. Programmatic rendering and server health detect runtime failures; they do not replace accessibility, browser interaction, security or bank user-acceptance testing.

### Q44. What can you claim on your resume?

**Strong answer:** Executed dataset size, held-out metrics and explicitly retrospective review-simulation results from saved artifacts.

**Possible follow-up:** What must you avoid?

**Deeper answer:** No real bank profit uplift, deployment, bank-grade PD, causal drivers, true production drift or fairness certification. The generated resume points directly to numerical sources.
