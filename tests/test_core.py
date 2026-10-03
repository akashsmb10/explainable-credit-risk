import json
import joblib
import numpy as np
import pandas as pd
import pytest
from src.data import ROOT, validate
from src.features import engineer
from src.decision_engine import expected_loss, risk_bands, threshold_table
from src.evaluation import psi, wilson
from src.calibration import ProbabilityMap

@pytest.fixture(scope='session')
def bundle(): return joblib.load(ROOT/'artifacts/models/decision_bundle.joblib')

def test_feature_values():
    x=pd.DataFrame({'LIMIT_BAL':[100.],'PAY_0':[2],'PAY_2':[0],'PAY_3':[-1],
                    'PAY_4':[-2],'PAY_5':[1],'PAY_6':[0]})
    for i in range(1,7): x[f'BILL_AMT{i}']=50.; x[f'PAY_AMT{i}']=10.
    out=engineer(x)
    assert out.utilization_latest.iloc[0]==.5
    assert out.payment_to_limit.iloc[0]==.1
    assert out.payment_to_positive_bill.iloc[0]==.2
    assert out.months_delayed.iloc[0]==2
    assert out.maximum_delay.iloc[0]==2

def test_zero_bill_safe():
    x=pd.read_csv(ROOT/'artifacts/tables/test_predictions.csv').head(3)
    x.loc[:,x.columns.str.startswith('BILL_AMT')]=0
    assert np.isfinite(engineer(x).to_numpy()).all()

def test_no_target_or_demographics():
    x=pd.read_csv(ROOT/'artifacts/tables/test_predictions.csv').head()
    first=engineer(x); x.event=1-x.event; x.ID=-x.ID; x.SEX=99; x.AGE=999
    pd.testing.assert_frame_equal(first,engineer(x))
    assert not {'ID','event','SEX','AGE','EDUCATION','MARRIAGE'} & set(first.columns)

def test_profile_disjointness():
    manifest=pd.read_csv(ROOT/'artifacts/tables/split_manifest.csv',dtype={'profile_hash':str})
    assert manifest.ID.is_unique
    assert manifest.groupby('profile_hash').split.nunique().max()==1
    assert set(manifest.split)=={'train','calibration','validation','test'}

def test_preprocessing_train_only(bundle):
    raw=pd.read_csv(ROOT/'data/processed/portfolio.csv')
    x=engineer(raw[raw.split=='train'])
    preprocess=bundle['model'].named_steps['preprocess']
    numeric=preprocess.transformers_[0][2]
    imputer=preprocess.named_transformers_['numeric'].named_steps['impute']
    np.testing.assert_allclose(imputer.statistics_,x[numeric].median().values)
    scaler=preprocess.named_transformers_['numeric'].named_steps['scale']
    np.testing.assert_allclose(scaler.mean_,x[numeric].mean().values)

def test_saved_prediction_replay(bundle):
    x=pd.read_csv(ROOT/'artifacts/tables/test_predictions.csv')
    p=bundle['calibrator'].predict(bundle['model'].predict_proba(x)[:,1])
    np.testing.assert_allclose(p,x.score,atol=1e-12)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    assert (risk_bands(p,bundle['boundaries'])==x.risk_band).all()

def test_risk_boundary_convention():
    assert list(risk_bands([0,.2,.4,.6,1],[.2,.4,.6]))==['LOW','MEDIUM','HIGH','VERY HIGH','VERY HIGH']

@pytest.mark.parametrize('scores,bounds',[([np.nan],[.2,.4,.6]),([-1],[.2,.4,.6]),([.5],[.4,.2,.6]),([1.1],[.2,.4,.6])])
def test_invalid_bands(scores,bounds):
    with pytest.raises(ValueError): risk_bands(scores,bounds)

def test_loss_scaling():
    assert expected_loss(.2,.5,100000)==10000
    np.testing.assert_allclose(expected_loss(.2,.5,[100000,200000]),[10000,20000])

@pytest.mark.parametrize('p,l,e',[(-.1,.5,100),(1.1,.5,100),(.2,1.1,100),(.2,.5,-100),(np.nan,.5,100)])
def test_invalid_loss(p,l,e):
    with pytest.raises(ValueError): expected_loss(p,l,e)

def test_threshold_endpoints():
    out=threshold_table([0,1],np.array([.2,.8]),[0,1])
    assert out.iloc[0].review_rate==1
    assert np.isnan(out.iloc[0].accepted_event_rate)
    assert out.iloc[1].simulated_acceptance_rate==1
    assert out.iloc[1].event_capture==0

def test_monitoring_and_intervals():
    reference=np.linspace(0,100,1000)
    assert psi(reference,reference)==pytest.approx(0)
    assert psi(reference,reference*3)>0
    low,high=wilson(0,100)
    assert low==pytest.approx(0,abs=1e-10) and 0<high<1

def test_unknown_status_prediction(bundle):
    x=pd.read_csv(ROOT/'artifacts/tables/test_predictions.csv').head(1)
    x['PAY_0']=99
    assert np.isfinite(bundle['model'].predict_proba(x)).all()

def test_schema_rejection():
    x=pd.read_csv(ROOT/'data/processed/portfolio.csv').drop(columns=['split','score','risk_band','decision_support','scenario_loss_ntd'])
    with pytest.raises(ValueError): validate(x.drop(columns='LIMIT_BAL'))
    x.loc[0,'event']=5
    with pytest.raises(ValueError): validate(x)

def test_model_selection_no_test_rule(bundle):
    validation=pd.read_csv(ROOT/'artifacts/tables/validation_models.csv')
    selected=validation.sort_values(['brier','log_loss','candidate']).iloc[0].candidate
    assert selected==bundle['winner']
    all_data=pd.read_csv(ROOT/'data/processed/portfolio.csv')
    vp=all_data.loc[all_data.split=='validation','score']
    config=json.loads((ROOT/'artifacts/tables/run_summary.json').read_text())['configuration']
    np.testing.assert_allclose(bundle['boundaries'],np.quantile(vp,config['threshold_quantiles']))

def test_sigmoid_monotonic_on_sample():
    p=np.array([.1,.2,.3,.7,.8,.9]); y=np.array([0,0,1,0,1,1])
    mapped=ProbabilityMap('sigmoid').fit(p,y).predict(p)
    assert ((mapped>0)&(mapped<1)).all() and (np.diff(mapped)>=0).all()

def test_statistical_assumptions():
    tests=pd.read_csv(ROOT/'artifacts/tables/statistical_tests.csv')
    chi=tests[tests.test=='Chi-square']
    assert (chi.min_expected_cell>=5).all()
    assert (tests.p_holm>=tests.p_value).all()
    assert tests.p_value_underflow.equals(tests.p_value.eq(0))

def test_explanations_cover_all_test_clients():
    predictions=pd.read_csv(ROOT/'artifacts/tables/test_predictions.csv')
    explanations=pd.read_csv(ROOT/'artifacts/tables/shap_local.csv')
    assert set(predictions.ID)==set(explanations.ID)
    assert explanations.ID.is_unique
