"""Export a public, static explorer from executed holdout artifacts; no retraining."""
import json
import shutil
import pandas as pd
from .data import ROOT

def build():
    tables=ROOT/'artifacts/tables'; output=ROOT/'site'
    output.mkdir(exist_ok=True); (output/'assets').mkdir(exist_ok=True)
    summary=json.loads((tables/'run_summary.json').read_text())
    clients=pd.read_csv(tables/'test_predictions.csv').sort_values('ID')
    explanations=pd.read_csv(tables/'shap_local.csv').set_index('ID')
    feature_names=[c for c in explanations.columns if c not in ['base_log_odds','raw_log_odds']]
    rows=[]
    for row in clients.itertuples():
        values=explanations.loc[row.ID,feature_names]
        drivers=[{'feature':c,'contribution':float(values[c])} for c in values.abs().nlargest(5).index]
        rows.append({'id':int(row.ID),'score':float(row.score),'event':int(row.event),
                     'band':row.risk_band,'limit':float(row.LIMIT_BAL),'status':int(row.PAY_0),
                     'bill':float(row.BILL_AMT1),'payment':float(row.PAY_AMT1),'drivers':drivers})
    def records(name):
        return json.loads(pd.read_csv(tables/name).to_json(orient='records'))
    payload={'summary':summary,'clients':rows,'models':records('test_models.csv'),
             'validation_models':records('validation_models.csv'),'intervals':records('test_confidence_intervals.csv'),
             'bands':records('risk_bands.csv'),'deciles':records('deciles.csv'),
             'global_drivers':records('shap_global.csv'),'subgroups':records('subgroup_metrics.csv'),
             'monitoring':records('monitoring.csv'),'delay_rates':records('delay_event_rates.csv')}
    (output/'data.json').write_text(json.dumps(payload,separators=(',',':'),allow_nan=False),encoding='utf-8')
    for name in ['risk_bands.png','calibration.png','shap_summary.png','roc.png',
                 'shap_low.png','shap_medium.png','shap_high.png','shap_borderline.png']:
        shutil.copyfile(ROOT/'artifacts/figures'/name,output/'assets'/name)
    (output/'.nojekyll').write_text('')
    print(f'Exported {len(rows)} verified holdout scores and local drivers; {len((output/"data.json").read_bytes())} JSON bytes')
    return payload

if __name__=='__main__': build()
