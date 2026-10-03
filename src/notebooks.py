"""Build and execute narrative notebooks against saved artifacts."""
import os
import sys
import nbformat as nbf
from nbclient import NotebookClient
from .data import ROOT

SECTIONS=[
('01_data_audit','Data audit and target meaning',
 'Existing cardholders, not new applications. Review schema, code frequencies, missingness and profile groups. No identifier enters scoring.',
 "display(pd.read_csv(T/'data_audit.csv'))\ndisplay(pd.read_csv(T/'categorical_codes.csv'))\nprint({k:summary[k] for k in ['rows','events','missing_cells','duplicate_profiles','source_sha256']})"),
('02_eda','Decision-oriented EDA',
 'Charts use training only. Each relationship asks whether behavioural segments warrant differentiated review. Associations do not establish causes; see reports/eda_interpretation.md.',
 "display(pd.read_csv(T/'delay_event_rates.csv'))\nfor f in ['delinquency_event_rate.png','eda_LIMIT_BAL.png','eda_utilization_latest.png','eda_payment_to_limit.png']:\n    display(Image(filename=str(F/f)))"),
('03_statistical_analysis','Statistical evidence and practical effects',
 'H0/H1, assumptions, statistics, raw and Holm-adjusted p-values and effect sizes are in the executed table. Reduced robust logit is exploratory and separate from regularized prediction. VIF examines the reduced inferential design.',
 "display(pd.read_csv(T/'statistical_tests.csv'))\ndisplay(pd.read_csv(T/'logistic_inference.csv'))\ndisplay(pd.read_csv(T/'vif.csv'))\ndisplay(pd.read_csv(T/'spearman_correlation.csv'))"),
('04_feature_engineering','Financially interpretable behavioural features',
 'Utilization, payment intensity, bill change and positive delay aggregates are supported by the data. Income, affordability and matched contractual payment coverage are unavailable. Transformer fit occurs on training only.',
 "from src.features import engineer\nraw=pd.read_csv(ROOT/'data/processed/portfolio.csv')\ndisplay(engineer(raw.head()))\nassert not {'ID','event','SEX','AGE','EDUCATION','MARRIAGE'} & set(engineer(raw.head()).columns)"),
('05_modeling','Base rate, logistic and regularized tree',
 'Fixed model settings plus dedicated calibration and validation prevent test-driven selection. Selection minimizes validation Brier then log loss. Comparison classification cutoffs differ from the policy threshold.',
 "display(pd.read_csv(T/'validation_models.csv'))\ndisplay(pd.read_csv(T/'test_models.csv'))\nprint('Frozen winner:',summary['selected'])\ndisplay(Image(filename=str(F/'roc.png')))\ndisplay(Image(filename=str(F/'precision_recall.png')))"),
('06_calibration_validation','Ranking, calibration and uncertainty',
 'Calibration maps learn only on the calibration partition. Validation selects the mapping. Reliability and decile outcomes are final test descriptions. Bootstrap intervals condition on the fixed split/model.',
 "display(pd.read_csv(T/'calibration_curve.csv'))\ndisplay(pd.read_csv(T/'deciles.csv'))\ndisplay(pd.read_csv(T/'test_confidence_intervals.csv'))\ndisplay(pd.read_csv(T/'subgroup_metrics.csv'))\ndisplay(Image(filename=str(F/'calibration.png')))"),
('07_explainability','Global and representative local model associations',
 'Exact TreeSHAP sums to raw XGBoost log-odds; not calibrated probabilities or causes. Sample-based global importance includes representative clients. Demographic exclusion does not eliminate proxy discrimination.',
 "display(pd.read_csv(T/'shap_global.csv').head(15))\ndisplay(pd.read_csv(T/'representative_explanations.csv'))\nprint('Maximum additivity error:',summary['shap_max_additivity_error'])\ndisplay(Image(filename=str(F/'shap_summary.png')))\nfor label in summary['representative_ids']:\n    display(Image(filename=str(F/f'shap_{label}.png')))"),
('08_decision_engine','Review trade-offs, scenario losses and monitoring',
 'Quantile bands express a capacity assumption. Slider exploration must not tune reported policy on test. LGD/EAD scenarios are unobserved assumptions in NT dollars. Monitoring is random-partition/synthetic, not actual temporal drift.',
 "display(pd.read_csv(T/'threshold_tradeoffs.csv'))\ndisplay(pd.read_csv(T/'expected_loss_scenarios.csv'))\ndisplay(pd.read_csv(T/'monitoring.csv'))\ndisplay(pd.read_csv(T/'monitoring_outcomes.csv'))\nimport duckdb\ncon=duckdb.connect(str(T/'portfolio.duckdb'),read_only=True)\nfor path in sorted((ROOT/'sql').glob('*.sql')):\n    print(path.name)\n    display(con.execute(path.read_text()).df())\ncon.close()"),
]

def run():
    # Explicit interpreter in kernelspec makes execution independent of global Jupyter installs.
    kernel_dir=ROOT/'artifacts/jupyter/kernels/credit-risk'
    kernel_dir.mkdir(parents=True,exist_ok=True)
    import json
    (kernel_dir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Credit Risk Project','language':'python'}),encoding='utf-8')
    os.environ['JUPYTER_PATH']=str(ROOT/'artifacts/jupyter')
    setup="""from pathlib import Path
import sys,json
import pandas as pd
from IPython.display import display,Image
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
T=ROOT/'artifacts/tables'; F=ROOT/'artifacts/figures'
summary=json.loads((T/'run_summary.json').read_text())
"""
    executed=[]
    for name,title,narrative,code in SECTIONS:
        notebook=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(f'# {title}\n\n{narrative}'),
            nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(code)])
        notebook.metadata.kernelspec={'display_name':'Credit Risk Project','language':'python','name':'credit-risk'}
        NotebookClient(notebook,timeout=180,kernel_name='credit-risk',resources={'metadata':{'path':str(ROOT)}}).execute()
        nbf.write(notebook,ROOT/'notebooks'/f'{name}.ipynb')
        executed.append(name); print('Executed',name,flush=True)
    (ROOT/'artifacts/tables/notebook_execution.json').write_text(json.dumps(executed,indent=2))

if __name__=='__main__': run()
