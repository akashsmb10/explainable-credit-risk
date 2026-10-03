import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

def all_local_contributions(tree_pipeline, frame, tables):
    engineered = tree_pipeline.named_steps['features'].transform(frame)
    prep = tree_pipeline.named_steps['preprocess']
    transformed = prep.transform(engineered)
    tree = tree_pipeline.named_steps['model']
    contribution = tree.get_booster().predict(xgb.DMatrix(transformed), pred_contribs=True)
    margins = tree.predict(transformed, output_margin=True)
    assert np.max(np.abs(contribution.sum(axis=1)-margins)) < 1e-4
    local = pd.DataFrame(contribution[:,:-1],columns=prep.get_feature_names_out())
    local.insert(0,'ID',frame.ID.to_numpy())
    local['base_log_odds']=contribution[:,-1]; local['raw_log_odds']=margins
    local.to_csv(tables/'shap_local.csv',index=False)

def explain(tree_pipeline, frame, ids, figures, tables):
    engineered = tree_pipeline.named_steps['features'].transform(frame)
    prep = tree_pipeline.named_steps['preprocess']
    transformed = prep.transform(engineered)
    names = prep.get_feature_names_out()
    tree = tree_pipeline.named_steps['model']
    # XGBoost's exact TreeSHAP implementation avoids library-version deserialization issues.
    contribution = tree.get_booster().predict(xgb.DMatrix(transformed),pred_contribs=True)
    margins = tree.predict(transformed, output_margin=True)
    error = float(np.max(np.abs(contribution.sum(axis=1)-margins)))
    assert error < 1e-4
    pd.DataFrame({'feature':names,'mean_abs_shap_log_odds':np.abs(contribution[:,:-1]).mean(axis=0)}).sort_values(
        'mean_abs_shap_log_odds',ascending=False).to_csv(tables/'shap_global.csv',index=False)
    explanation = shap.Explanation(values=contribution[:,:-1],base_values=contribution[:,-1],
                                   data=transformed, feature_names=list(names))
    shap.plots.beeswarm(explanation,max_display=15,show=False)
    plt.title('Raw XGBoost: exact TreeSHAP in log-odds units')
    plt.gcf().savefig(figures/'shap_summary.png',dpi=180,bbox_inches='tight'); plt.close('all')
    local = pd.DataFrame(contribution[:,:-1],columns=names)
    local.insert(0,'ID',np.asarray(ids)); local['base_log_odds']=contribution[:,-1]
    local['raw_log_odds']=margins
    local.to_csv(tables/'shap_local.csv',index=False)
    return error
