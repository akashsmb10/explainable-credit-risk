import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, norm
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                            recall_score, f1_score, confusion_matrix, brier_score_loss, log_loss)

def metrics(y, p, threshold=0.5):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(roc_auc=roc_auc_score(y, p), average_precision=average_precision_score(y, p),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred),
                f1=f1_score(y, pred, zero_division=0), ks=ks_2samp(p[y == 1], p[y == 0]).statistic,
                brier=brier_score_loss(y, p), log_loss=log_loss(y, p), tn=int(tn), fp=int(fp),
                fn=int(fn), tp=int(tp), threshold=float(threshold))

def wilson(events, n):
    if n == 0:
        return np.nan, np.nan
    z = norm.ppf(.975); rate = events / n
    centre = (rate + z*z/(2*n)) / (1 + z*z/n)
    half = z*np.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / (1+z*z/n)
    return centre-half, centre+half

def reliability(y, p):
    table = pd.DataFrame({'event': np.asarray(y), 'score': p})
    table['bin'] = pd.cut(table.score, np.linspace(0, 1, 11), include_lowest=True)
    out = table.groupby('bin', observed=True).agg(n=('event','size'), events=('event','sum'),
               event_rate=('event','mean'), predicted=('score','mean')).reset_index()
    out[['ci_low', 'ci_high']] = [wilson(row.events, row.n) for row in out.itertuples()]
    return out

def bootstrap(y, p, repetitions=300, seed=42):
    y = np.asarray(y); rng = np.random.default_rng(seed); rows=[]
    for _ in range(repetitions):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            rows.append([roc_auc_score(y[idx], p[idx]), average_precision_score(y[idx], p[idx]),
                         brier_score_loss(y[idx], p[idx])])
    return pd.DataFrame({'metric':['roc_auc','average_precision','brier'],
                         'ci_low':np.quantile(rows,.025,axis=0), 'ci_high':np.quantile(rows,.975,axis=0)})

def psi(reference, current):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.maximum(np.histogram(reference, edges)[0] / len(reference), 1e-6)
    c = np.maximum(np.histogram(current, edges)[0] / len(current), 1e-6)
    r /= r.sum(); c /= c.sum()
    return float(np.sum((c-r)*np.log(c/r)))
