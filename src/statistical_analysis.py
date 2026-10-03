import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, pointbiserialr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from .features import engineer
from .evaluation import wilson

def analyze(train, tables):
    x = engineer(train); y = train.event
    chosen = ['LIMIT_BAL', 'utilization_latest', 'months_delayed', 'payment_to_limit']
    rows=[]
    for feature in chosen:
        a = x.loc[y == 1, feature]; b = x.loc[y == 0, feature]
        test = mannwhitneyu(a,b,alternative='two-sided')
        corr = pointbiserialr(y,x[feature])
        rows.append(dict(feature=feature, test='Mann-Whitney U', statistic=test.statistic,
            p_value=test.pvalue, effect_size=2*test.statistic/(len(a)*len(b))-1,
            effect_definition='rank-biserial: positive means event group tends higher',
            point_biserial=corr.statistic, event_median=a.median(), non_event_median=b.median(),
            H0='Outcome groups have identical feature distributions',
            H1='Outcome groups differ in feature distribution',
            assumptions='Independent grouped clients; ordinal/continuous measure; ties corrected; not a causal test',
            relevance='Association informs candidate behavioural segmentation; effect size governs practical relevance'))
    for feature in ['SEX', 'EDUCATION', 'MARRIAGE']:
        categorical=train[feature].copy()
        pooling='none'
        if feature=='EDUCATION':
            categorical=categorical.map({1:'graduate',2:'university',3:'high_school'}).fillna('other_or_undocumented')
            pooling='Source other and undocumented education codes pooled; no semantic meaning assigned to undocumented codes'
        elif feature=='MARRIAGE':
            categorical=categorical.map({1:'married',2:'single'}).fillna('other_or_undocumented')
            pooling='Other and undocumented marital codes pooled to avoid sparse expected cells'
        ct = pd.crosstab(categorical,y)
        chi,p,dof,expected = chi2_contingency(ct)
        if expected.min()<5:
            raise ValueError(f'Sparse chi-square expectations for {feature}; revise audit grouping')
        rows.append(dict(feature=feature,test='Chi-square',statistic=chi,p_value=p,min_expected_cell=float(expected.min()),
            effect_size=np.sqrt(chi/(ct.values.sum()*min(ct.shape[0]-1,ct.shape[1]-1))),
            effect_definition="Cramer's V", H0='Feature and outcome are independent',
            H1='Feature and outcome are associated', assumptions=f'Independent clients; minimum expected cell {expected.min():.3f}; {pooling}',
            relevance='Audit-only demographic association; excluded from scoring'))
    result = pd.DataFrame(rows)
    result['p_value_underflow'] = result.p_value.eq(0)
    result['p_holm'] = multipletests(result.p_value,method='holm')[1]
    result.to_csv(tables/'statistical_tests.csv',index=False)
    x[chosen].corr(method='spearman').to_csv(tables/'spearman_correlation.csv')
    z = (x[chosen]-x[chosen].mean())/x[chosen].std()
    design = sm.add_constant(z)
    pd.DataFrame({'feature':design.columns,
        'vif':[variance_inflation_factor(design.values,i) for i in range(design.shape[1])]}).to_csv(tables/'vif.csv',index=False)
    fit = sm.Logit(y,design).fit(disp=False,cov_type='HC1')
    ci = fit.conf_int()
    pd.DataFrame({'feature':design.columns,'coefficient':fit.params.values,
        'odds_ratio_per_training_sd':np.exp(fit.params.values),
        'or_ci_low':np.exp(ci[0].values),'or_ci_high':np.exp(ci[1].values),
        'p_value':fit.pvalues.values}).to_csv(tables/'logistic_inference.csv',index=False)
    # Train-only descriptive rates; bins intentionally fixed to understandable status categories.
    segment = train.assign(recent_delay=np.where(train.PAY_0 > 0,'reported delay','no positive delay'))
    rates = segment.groupby('recent_delay').event.agg(['size','sum','mean']).reset_index()
    rates[['ci_low','ci_high']] = [wilson(r['sum'],r['size']) for _,r in rates.iterrows()]
    rates.to_csv(tables/'delay_event_rates.csv',index=False)
    return result, rates
