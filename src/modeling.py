from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from .features import CreditFeatures, STATUS, engineer

def build_models(sample, seed):
    numeric = [c for c in engineer(sample).columns if c not in STATUS]
    prep = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), numeric),
        ('status', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                             ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), STATUS)],
        verbose_feature_names_out=False)
    estimators = {
        'base_rate': DummyClassifier(strategy='prior'),
        'logistic': LogisticRegression(C=0.3, max_iter=2500, random_state=seed),
        'xgboost': XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.04,
                                subsample=0.85, colsample_bytree=0.85, reg_lambda=8,
                                eval_metric='logloss', random_state=seed, n_jobs=2)}
    import copy
    return {name: Pipeline([('features', CreditFeatures()), ('preprocess', copy.deepcopy(prep)),
                             ('model', model)]) for name, model in estimators.items()}
