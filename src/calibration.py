import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

def logits(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)

class ProbabilityMap:
    def __init__(self, method='raw'):
        self.method = method
    def fit(self, p, y):
        if self.method == 'sigmoid':
            self.estimator = LogisticRegression(C=1e6).fit(logits(p), y)
        elif self.method == 'isotonic':
            self.estimator = IsotonicRegression(out_of_bounds='clip').fit(p, y)
        elif self.method != 'raw':
            raise ValueError('Unknown calibration method')
        return self
    def predict(self, p):
        if self.method == 'raw':
            return np.asarray(p)
        if self.method == 'sigmoid':
            return self.estimator.predict_proba(logits(p))[:, 1]
        return self.estimator.predict(p)
