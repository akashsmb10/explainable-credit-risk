"""Streamlit artifact explorer; never trains a model on app startup."""
from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.decision_engine import expected_loss, threshold_table

st.set_page_config(page_title='Credit Risk Studio',page_icon='📊',layout='wide')
st.markdown('''<style>
.stApp {background: #f4f7fb;}
h1, h2, h3 {color: #15354b;}
div[data-testid="stMetric"] {background:white;border-radius:12px;padding:18px;border-left:4px solid #ba6638;}
section[data-testid="stSidebar"] {background:#e7eef5;}
</style>''',unsafe_allow_html=True)
st.title('Retail Credit Risk Studio')
st.caption('Historical evidence · Explainable scoring · Decision trade-offs')
st.warning('Educational portfolio decision-support simulation. Not an actual bank underwriting system or ICICI lending recommendation. Scores estimate the UCI payment-event label; losses use assumed LGD and EAD in NT dollars.')

@st.cache_data
def artifacts():
    path=ROOT/'artifacts/tables'
    return json.loads((path/'run_summary.json').read_text()),pd.read_csv(path/'test_predictions.csv')

if not (ROOT/'artifacts/tables/run_summary.json').exists():
    st.error('Run python -m src.pipeline from the repository root to generate artifacts.')
    st.stop()

summary,clients=artifacts()
pages=['Portfolio Overview','Risk Distribution','Applicant Risk Explorer','Model Performance',
       'Explainability','Decision Threshold Simulator','Expected-Loss Scenario Analysis']
page=st.sidebar.radio('Workspace',pages)
st.sidebar.caption(f"Frozen model: {summary['selected']}")
st.sidebar.caption('Applicant explorer shows held-out existing cardholders. Simulated acceptance is not actual eligibility.')

def csv(name): return pd.read_csv(ROOT/'artifacts/tables'/name)
def picture(name): st.image(str(ROOT/'artifacts/figures'/name),width='stretch')

if page=='Portfolio Overview':
    a,b,c,d=st.columns(4)
    a.metric('Source clients',f"{summary['rows']:,}")
    b.metric('Source event rate',f"{summary['event_rate']:.1%}")
    c.metric('Test ROC-AUC',f"{summary['test_metrics']['roc_auc']:.3f}")
    d.metric('Test Brier',f"{summary['test_metrics']['brier']:.4f}")
    st.subheader('Portfolio evidence')
    st.dataframe(csv('sql_01_portfolio_overview.csv'),hide_index=True,width='stretch')
    st.info('Train/calibration/validation rows are not independent evaluation results. Dashboard applicant, threshold and risk pages use the final test holdout. Credit limits are observed NT-dollar amounts, not exposure at default.')
    left,right=st.columns(2)
    with left: picture('delinquency_event_rate.png')
    with right:
        st.subheader('Scope and governance')
        st.write('Historical Taiwan cardholders; random grouped validation, no future-period evidence. Income, contractual LGD/EAD and rejected applications are unavailable. Demographics are excluded from scoring but may have correlated proxies.')
        st.download_button('Download verified test predictions',clients.to_csv(index=False),'test_predictions.csv','text/csv')
elif page=='Risk Distribution':
    st.subheader('Relative risk bands from validation-score quantiles')
    st.bar_chart(clients.risk_band.value_counts())
    st.dataframe(csv('risk_bands.csv'),hide_index=True,width='stretch')
    picture('risk_bands.png')
    st.caption(f"Frozen boundaries: {summary['band_boundaries']}. Bands allocate review priority; they do not certify loan quality.")
    st.dataframe(csv('deciles.csv'),hide_index=True,width='stretch')
elif page=='Applicant Risk Explorer':
    choices=clients.sort_values('ID').ID.tolist()
    applicant=st.selectbox('Held-out cardholder ID',choices)
    row=clients.loc[clients.ID==applicant].iloc[0]
    a,b,c=st.columns(3)
    a.metric('Estimated payment-event score',f'{row.score:.2%}')
    b.metric('Relative risk band',row.risk_band)
    c.metric('Decision support',row.decision_support)
    st.caption('This is an existing source cardholder, not a new-loan affordability assessment. ID is for retrieval only.')
    st.dataframe(row[['LIMIT_BAL','PAY_0','BILL_AMT1','PAY_AMT1']].rename('Observed source value').to_frame(),width='stretch')
    local=csv('shap_local.csv'); matched=local[local.ID==applicant]
    if len(matched):
        values=matched.iloc[0].drop(labels=['ID','base_log_odds','raw_log_odds'])
        drivers=values.loc[values.abs().nlargest(6).index].rename('raw_tree_shap_log_odds').to_frame()
        st.subheader('Major raw-tree model associations')
        st.bar_chart(drivers); st.dataframe(drivers,width='stretch')
        st.caption('Positive contributions raise raw XGBoost log-odds; negative lower them. These do not sum to the calibrated score and do not establish causes.')
    else:
        st.info('This ID is outside the saved SHAP sample. Select a representative ID on the Explainability page for saved drivers; no explanation is invented.')
    lgd=st.slider('Assumed LGD (fraction)',0.,1.,.45,.05)
    ead=st.number_input('Assumed EAD (NT dollars)',min_value=0.,value=100000.,step=10000.)
    st.metric('Scenario loss (NT dollars)',f'{float(expected_loss(row.score,lgd,ead)):,.0f}')
elif page=='Model Performance':
    st.subheader('Model selection evidence: validation only')
    st.dataframe(csv('validation_models.csv'),hide_index=True,width='stretch')
    st.subheader('Final test comparisons')
    st.dataframe(csv('test_models.csv'),hide_index=True,width='stretch')
    st.caption('Comparison classification metrics use the conventional cutoff; run_summary contains metrics at the frozen review threshold. Lowest validation Brier selects the winner, with log loss as tie-breaker.')
    left,right=st.columns(2)
    with left: picture('roc.png')
    with right: picture('calibration.png')
    st.subheader('Uncertainty and subgroup checks')
    st.dataframe(csv('test_confidence_intervals.csv'),hide_index=True,width='stretch')
    st.dataframe(csv('subgroup_metrics.csv'),hide_index=True,width='stretch')
    st.caption('Descriptive subgroup evidence is not fairness certification. Bootstrap intervals condition on a fixed fitted model and split.')
elif page=='Explainability':
    picture('shap_summary.png')
    st.dataframe(csv('shap_global.csv').head(15),hide_index=True,width='stretch')
    label=st.selectbox('Representative cardholder',list(summary['representative_ids']))
    client=summary['representative_ids'][label]
    st.write(f'Source ID: {client}')
    picture(f'shap_{label}.png')
    st.dataframe(csv('representative_explanations.csv').query('representative == @label'),hide_index=True,width='stretch')
    st.info('Exact TreeSHAP explains raw XGBoost margins. Calibration is a separate mapping. Feature dependence and encoded categories affect attribution; these are model associations, not causal or bank-approved adverse-action reasons.')
elif page=='Decision Threshold Simulator':
    threshold=st.slider('Exploratory review threshold',0.,1.,float(summary['review_threshold']),.005)
    out=threshold_table(clients.event,clients.score.to_numpy(),[threshold]).iloc[0]
    a,b,c=st.columns(3)
    a.metric('Simulated acceptance',f'{out.simulated_acceptance_rate:.1%}')
    b.metric('Event capture in review',f'{out.event_capture:.1%}')
    c.metric('Accepted event rate','N/A' if np.isnan(out.accepted_event_rate) else f'{out.accepted_event_rate:.1%}')
    st.dataframe(out.to_frame('Holdout simulation'),width='stretch')
    tradeoffs=csv('threshold_tradeoffs.csv')
    st.line_chart(tradeoffs.set_index('threshold')[['simulated_acceptance_rate','event_capture','false_positive_rate']])
    st.dataframe(tradeoffs,hide_index=True,width='stretch')
    st.caption('Changing this slider is retrospective exploration, not re-selection of the reported policy. It does not alter frozen model metrics. Real approval requires affordability, economics, governance and additional eligibility constraints.')
else:
    st.subheader('Probability × assumed LGD × assumed EAD')
    probability=st.slider('Illustrative payment-event probability (assumption)',0.,1.,float(clients.score.median()),.01)
    lgd=st.slider('Assumed LGD',0.,1.,.45,.05)
    ead=st.number_input('Applicant A assumed exposure (NT dollars)',min_value=0.,value=100000.,step=10000.)
    ead_b=st.number_input('Applicant B assumed exposure (NT dollars)',min_value=0.,value=200000.,step=10000.)
    a,b=st.columns(2)
    a.metric('Applicant A scenario loss',f'{float(expected_loss(probability,lgd,ead)):,.0f} NTD')
    b.metric('Applicant B scenario loss',f'{float(expected_loss(probability,lgd,ead_b)):,.0f} NTD')
    st.caption('Same probability and LGD, different exposures. All exposure/severity inputs above are assumptions, not measured defaults or recovery outcomes.')
    st.dataframe(csv('expected_loss_scenarios.csv'),hide_index=True,width='stretch')
