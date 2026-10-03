"""Run with python -m src.pipeline. Regenerates outputs without reading old models."""
import json
import itertools
import platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yaml
import joblib
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from .data import ROOT, load_data, audit, split_data
from .features import engineer, BASE
from .modeling import build_models
from .calibration import ProbabilityMap
from .evaluation import metrics, bootstrap, reliability, psi, wilson
from .decision_engine import risk_bands, threshold_table, expected_loss
from .statistical_analysis import analyze
from .explainability import explain, all_local_contributions

def save_json(value, path):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def chart(name, title, xlabel, ylabel):
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(alpha=.2)
    plt.tight_layout(); plt.savefig(ROOT/'artifacts/figures'/name,dpi=180); plt.close()

def run():
    for relative in ['data/raw','data/processed','artifacts/models','artifacts/tables','artifacts/figures']:
        (ROOT/relative).mkdir(parents=True,exist_ok=True)
    config = yaml.safe_load((ROOT/'config/config.yaml').read_text(encoding='utf-8-sig'))
    seed=config['seed']; tables=ROOT/'artifacts/tables'; figures=ROOT/'artifacts/figures'
    plt.rcParams.update({'figure.figsize':(8,5),'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    df,checksum=load_data(); print('Loaded source',df.shape,flush=True)
    audit(df).to_csv(tables/'data_audit.csv',index_label='feature')
    df.describe(percentiles=[.01,.5,.99]).T.to_csv(tables/'numerical_distributions.csv')
    categories=[]
    for c in ['SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']:
        for value,n in df[c].value_counts().items():
            categories.append({'feature':c,'code':int(value),'n':int(n)})
    pd.DataFrame(categories).to_csv(tables/'categorical_codes.csv',index=False)
    duplicate_profiles=int(df.drop(columns=['ID','event']).duplicated().sum())
    parts,groups=split_data(df,seed)
    split_summary={name:{'n':len(idx),'events':int(df.iloc[idx].event.sum()),
                        'event_rate':float(df.iloc[idx].event.mean())} for name,idx in parts.items()}
    manifest=pd.DataFrame({'ID':df.ID,'profile_hash':groups.astype(str),'split':''})
    for name,idx in parts.items(): manifest.loc[idx,'split']=name
    manifest.to_csv(tables/'split_manifest.csv',index=False)
    train,cal,val,test=[df.iloc[parts[n]].copy() for n in ['train','calibration','validation','test']]
    statistics,rates=analyze(train,tables)
    plt.bar(rates.recent_delay,rates['mean'],color=['#527d96','#bd5a3a'])
    plt.errorbar(rates.recent_delay,rates['mean'],yerr=[rates['mean']-rates.ci_low,rates.ci_high-rates['mean']],fmt='none',color='black',capsize=4)
    chart('delinquency_event_rate.png','Training-only association: recent repayment delay','Repayment status grouping','Observed next-month event rate')
    for col in ['LIMIT_BAL','utilization_latest','payment_to_limit']:
        x=engineer(train)[col]
        bins=pd.qcut(x,5,duplicates='drop')
        eda=pd.DataFrame({'bin':bins,'event':train.event}).groupby('bin',observed=True).event.agg(['size','mean']).reset_index()
        eda.to_csv(tables/f'eda_{col}.csv',index=False)
        plt.bar(np.arange(len(eda)),eda['mean'],color='#527d96')
        plt.xticks(np.arange(len(eda)),[str(v) for v in eda.bin],rotation=25,ha='right')
        chart(f'eda_{col}.png',f'Training-only event association: {col}','Training quantile bin','Observed event rate')
    models=build_models(train,seed); maps={}; selection=[]
    for name,model in models.items():
        print('Fitting',name,flush=True); model.fit(train,train.event)
        for method in (['raw'] if name=='base_rate' else ['raw','sigmoid','isotonic']):
            mapper=ProbabilityMap(method).fit(model.predict_proba(cal)[:,1],cal.event)
            key=f'{name}:{method}'; maps[key]=mapper
            p=mapper.predict(model.predict_proba(val)[:,1])
            selection.append({'candidate':key,**metrics(val.event,p)})
    validation=pd.DataFrame(selection).sort_values(['brier','log_loss','candidate'])
    validation.to_csv(tables/'validation_models.csv',index=False)
    # Fixed selection rule: lowest validation Brier, then log loss; no test input.
    winner=validation.iloc[0].candidate; modelname,method=winner.split(':')
    model=models[modelname]; mapper=maps[winner]
    vp=mapper.predict(model.predict_proba(val)[:,1])
    boundaries=np.quantile(vp,config['threshold_quantiles'])
    if not np.all(np.diff(boundaries)>0): raise ValueError('Tied validation boundaries require policy review')
    threshold=float(boundaries[1])
    print('Selected',winner,'before test scoring',flush=True)
    evaluation=[]
    for row in selection:
        name,cm=row['candidate'].split(':')
        p=maps[row['candidate']].predict(models[name].predict_proba(test)[:,1])
        evaluation.append({'candidate':row['candidate'],**metrics(test.event,p)})
    pd.DataFrame(evaluation).to_csv(tables/'test_models.csv',index=False)
    p=mapper.predict(model.predict_proba(test)[:,1]); final=metrics(test.event,p,threshold)
    intervals=bootstrap(test.event,p,config['bootstrap_replicates'],seed)
    intervals.to_csv(tables/'test_confidence_intervals.csv',index=False)
    curve=reliability(test.event,p); curve.to_csv(tables/'calibration_curve.csv',index=False)
    for cm in ['raw','sigmoid','isotonic']:
        cp=maps[f'{modelname}:{cm}'].predict(models[modelname].predict_proba(test)[:,1]) if modelname!='base_rate' else p
        rc=reliability(test.event,cp)
        plt.plot(rc.predicted,rc.event_rate,'o-',label=cm)
    plt.plot([0,1],[0,1],'k--'); plt.legend()
    chart('calibration.png','Holdout reliability: mapping frozen before test','Mean estimated event probability','Observed event rate')
    for name,m in models.items():
        raw=m.predict_proba(test)[:,1]; fpr,tpr,_=roc_curve(test.event,raw)
        plt.plot(fpr,tpr,label=name)
    plt.plot([0,1],[0,1],'k--'); plt.legend()
    chart('roc.png','Holdout ranking comparison (raw models)','False positive rate','True positive rate')
    precision,recall,_=precision_recall_curve(test.event,p)
    plt.plot(recall,precision); plt.axhline(test.event.mean(),ls='--',color='grey')
    chart('precision_recall.png','Selected model: holdout precision–recall','Recall','Precision')
    portfolio=df.copy(); portfolio['split']=manifest.split; portfolio['score']=mapper.predict(model.predict_proba(df)[:,1])
    portfolio['risk_band']=risk_bands(portfolio.score,boundaries)
    portfolio['decision_support']=np.where(portfolio.score>=threshold,'manual review','simulated acceptance')
    portfolio['scenario_loss_ntd']=expected_loss(portfolio.score,.45,100000)
    portfolio.to_csv(ROOT/'data/processed/portfolio.csv',index=False)
    test_scores=portfolio[portfolio.split=='test'].copy()
    test_scores.to_csv(tables/'test_predictions.csv',index=False)
    threshold_table(test.event,p,np.unique(np.r_[np.quantile(vp,np.linspace(.1,.95,18)),boundaries])).to_csv(tables/'threshold_tradeoffs.csv',index=False)
    threshold_table(val.event,vp,np.unique(np.r_[np.quantile(vp,np.linspace(.1,.95,18)),boundaries])).to_csv(tables/'validation_threshold_tradeoffs.csv',index=False)
    band_table=test_scores.groupby('risk_band').agg(n=('event','size'),events=('event','sum'),event_rate=('event','mean'),mean_score=('score','mean')).sort_values('mean_score')
    band_table.to_csv(tables/'risk_bands.csv'); plt.bar(band_table.index,band_table.event_rate,color='#527d96')
    chart('risk_bands.png','Holdout outcomes by validation-derived band','Risk band','Observed event rate')
    deciles=test_scores.assign(decile=pd.qcut(test_scores.score.rank(method='first'),10,labels=False)+1).groupby('decile').agg(n=('event','size'),events=('event','sum'),event_rate=('event','mean'),mean_score=('score','mean')).sort_index(ascending=False)
    deciles['lift']=deciles.event_rate/test.event.mean(); deciles['cumulative_event_capture']=deciles.events.cumsum()/test.event.sum()
    deciles.to_csv(tables/'deciles.csv')
    subgroup=[]
    test_scores['age_band']=pd.cut(test_scores.AGE,[0,30,45,100],labels=['up to 30','31 to 45','above 45'])
    for group in ['SEX','age_band']:
        for value,segment in test_scores.groupby(group,observed=True):
            met=metrics(segment.event,segment.score,threshold)
            ci=wilson(segment.event.sum(),len(segment))
            subgroup.append({'attribute':group,'group':str(value),'n':len(segment),'events':int(segment.event.sum()),
                'event_rate':float(segment.event.mean()),'mean_score':float(segment.score.mean()),
                'calibration_gap':float(segment.score.mean()-segment.event.mean()),'event_ci_low':ci[0],'event_ci_high':ci[1],
                **met,'false_positive_rate':met['fp']/(met['fp']+met['tn']),
                'false_negative_rate':met['fn']/(met['fn']+met['tp']),
                'selection_rate':float((segment.score<threshold).mean())})
    pd.DataFrame(subgroup).to_csv(tables/'subgroup_metrics.csv',index=False)
    # Random holdout comparison is a framework demonstration, NOT temporal production monitoring.
    engineered_train=engineer(train); engineered_test=engineer(test); monitoring=[]
    train_p=mapper.predict(model.predict_proba(train)[:,1])
    for col in ['LIMIT_BAL','utilization_latest','payment_to_limit','months_delayed']:
        monitoring.append({'feature':col,'comparison':'random train versus test','psi':psi(engineered_train[col],engineered_test[col]),
                          'train_missing':float(engineered_train[col].isna().mean()),'test_missing':float(engineered_test[col].isna().mean())})
    monitoring.append({'feature':'score','comparison':'random train versus test','psi':psi(train_p,p),'train_missing':0.,'test_missing':0.})
    shifted=engineered_test.LIMIT_BAL*1.5
    monitoring.append({'feature':'LIMIT_BAL','comparison':'synthetic 1.5x limit shift versus test','psi':psi(engineered_test.LIMIT_BAL,shifted),'train_missing':0.,'test_missing':0.})
    pd.DataFrame(monitoring).to_csv(tables/'monitoring.csv',index=False)
    pd.DataFrame([{'split':n,'n':len(idx),'event_rate':float(portfolio.iloc[idx].event.mean()),
        'mean_score':float(portfolio.iloc[idx].score.mean()),
        'brier':float(np.mean((portfolio.iloc[idx].score-portfolio.iloc[idx].event)**2)),
        'missing_cells':int(df.iloc[idx].isna().sum().sum())} for n,idx in parts.items()]).to_csv(tables/'monitoring_outcomes.csv',index=False)
    scenarios=[{'probability':float(np.quantile(vp,q)),'validation_score_quantile':q,'assumed_lgd':lgd,'assumed_ead_ntd':ead,
                'scenario_loss_ntd':float(expected_loss(np.quantile(vp,q),lgd,ead))}
               for q,lgd,ead in itertools.product([.25,.5,.9],config['scenario_lgd'],config['scenario_ead_ntd'])]
    pd.DataFrame(scenarios).to_csv(tables/'expected_loss_scenarios.csv',index=False)
    # Explain representative clients plus a reproducible sample, never all production-like outcomes.
    representative={}
    for label,q in [('low',.1),('medium',.6),('high',.95)]:
        representative[label]=int(test_scores.iloc[np.argmin(np.abs(test_scores.score-np.quantile(vp,q)))].ID)
    representative['borderline']=int(test_scores.iloc[np.argmin(np.abs(test_scores.score-threshold))].ID)
    chosen_ids=list(representative.values())
    shap_frame=pd.concat([test[test.ID.isin(chosen_ids)],test.sample(min(500,len(test)),random_state=seed)]).drop_duplicates('ID')
    shap_error=explain(models['xgboost'],shap_frame,shap_frame.ID,figures,tables)
    shap_data=pd.read_csv(tables/'shap_local.csv')
    local_rows=[]
    for label,client in representative.items():
        s=shap_data[shap_data.ID==client].iloc[0].drop(labels=['ID','base_log_odds','raw_log_odds'])
        for feature in s.abs().nlargest(5).index:
            local_rows.append({'representative':label,'ID':client,'feature':feature,'shap_log_odds':float(s[feature])})
        shap_index=int(np.where(shap_frame.ID.to_numpy()==client)[0][0])
        engineered=models['xgboost'].named_steps['features'].transform(shap_frame.iloc[[shap_index]])
        prep=models['xgboost'].named_steps['preprocess']; transformed=prep.transform(engineered)[0]
        import shap
        sv=shap_data.iloc[shap_index]
        ex=shap.Explanation(values=sv.drop(labels=['ID','base_log_odds','raw_log_odds']).to_numpy(dtype=float),
             base_values=float(sv.base_log_odds),data=transformed,feature_names=list(prep.get_feature_names_out()))
        shap.plots.waterfall(ex,max_display=10,show=False)
        plt.gcf().savefig(figures/f'shap_{label}.png',dpi=180,bbox_inches='tight'); plt.close('all')
    pd.DataFrame(local_rows).to_csv(tables/'representative_explanations.csv',index=False)
    all_local_contributions(models['xgboost'],test,tables)
    # Logistic coefficients describe predictive fit, separate from inferential reduced logit.
    lm=models['logistic']; coef=lm.named_steps['model'].coef_[0]
    pd.DataFrame({'feature':lm.named_steps['preprocess'].get_feature_names_out(),'coefficient':coef}).to_csv(tables/'logistic_coefficients.csv',index=False)
    con=duckdb.connect(str(tables/'portfolio.duckdb'))
    con.register('input_frame',portfolio); con.execute('CREATE OR REPLACE TABLE portfolio AS SELECT * FROM input_frame')
    sql_summary={}
    for query in sorted((ROOT/'sql').glob('*.sql')):
        result=con.execute(query.read_text()).df()
        result.to_csv(tables/f'sql_{query.stem}.csv',index=False); sql_summary[query.name]=len(result)
    con.close()
    bundle={'model':model,'calibrator':mapper,'boundaries':boundaries,'review_threshold':threshold,
            'winner':winner,'required_features':BASE,'source_sha256':checksum}
    joblib.dump(bundle,ROOT/'artifacts/models/decision_bundle.joblib')
    joblib.dump(models,ROOT/'artifacts/models/benchmark_models.joblib')
    summary={'source_sha256':checksum,'source_url':'https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients',
        'rows':len(df),'columns':len(df.columns),'events':int(df.event.sum()),'event_rate':float(df.event.mean()),
        'missing_cells':int(df.isna().sum().sum()),'duplicate_profiles':duplicate_profiles,'duplicate_rows':int(df.duplicated().sum()),
        'negative_bill_cells':int((df.filter(regex='^BILL_AMT')<0).sum().sum()),
        'undocumented_education_rows':int(df.EDUCATION.isin([0,5,6]).sum()),
        'undocumented_marriage_rows':int((df.MARRIAGE==0).sum()),'repayment_minus_two_cells':int((df[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']]==-2).sum().sum()),
        'partitions':split_summary,'selected':winner,'selection_rule':'minimum validation Brier, tie break log loss',
        'test_metrics':final,'band_boundaries':boundaries.tolist(),'review_threshold':threshold,
        'representative_ids':representative,'shap_max_additivity_error':shap_error,'sql_rows':sql_summary,
        'execution_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'configuration':config}
    save_json(summary,tables/'run_summary.json')
    from .documentation import write_documentation
    write_documentation(summary)
    print(json.dumps({'selected':winner,'test':final,'rows':len(df)},indent=2),flush=True)
    return summary

if __name__=='__main__':
    import argparse
    import shutil
    parser=argparse.ArgumentParser()
    parser.add_argument('--clean',action='store_true',help='Remove only generated output directories before rerunning; retain downloaded source')
    args=parser.parse_args()
    if args.clean:
        for relative in ['artifacts/models','artifacts/tables','artifacts/figures','data/processed']:
            target=(ROOT/relative).resolve()
            if not target.is_relative_to(ROOT.resolve()): raise ValueError('Output path escaped project')
            if target.exists(): shutil.rmtree(target)
            target.mkdir(parents=True)
    run()
