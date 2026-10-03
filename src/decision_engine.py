import numpy as np
import pandas as pd

BANDS = np.array(['LOW', 'MEDIUM', 'HIGH', 'VERY HIGH'])

def risk_bands(scores, boundaries):
    if len(boundaries) != 3 or not np.all(np.diff(boundaries) > 0):
        raise ValueError('Three strictly ordered boundaries required')
    p = np.asarray(scores, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Scores must be finite probabilities')
    return BANDS[np.searchsorted(boundaries, p, side='right')]

def expected_loss(probability, lgd, ead):
    p, l, e = np.broadcast_arrays(probability, lgd, ead)
    if not np.isfinite(p+l+e).all() or ((p<0)|(p>1)|(l<0)|(l>1)|(e<0)).any():
        raise ValueError('Invalid probability, LGD or EAD')
    return p*l*e

def threshold_table(y, p, thresholds):
    y = np.asarray(y); rows=[]
    for threshold in thresholds:
        review = p >= threshold; accept = ~review
        tp = int((review & (y == 1)).sum()); fp = int((review & (y == 0)).sum())
        fn = int((accept & (y == 1)).sum()); tn = int((accept & (y == 0)).sum())
        rows.append(dict(threshold=float(threshold), simulated_acceptance_rate=float(accept.mean()),
                         review_rate=float(review.mean()), accepted_event_rate=float(y[accept].mean()) if accept.any() else np.nan,
                         event_capture=tp/int(y.sum()), false_positive_rate=fp/int((y==0).sum()),
                         false_negative_rate=fn/int(y.sum()), mean_accepted_score=float(p[accept].mean()) if accept.any() else np.nan,
                         tp=tp,fp=fp,fn=fn,tn=tn))
    return pd.DataFrame(rows)
