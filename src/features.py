import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

STATUS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
BASE = ['LIMIT_BAL'] + STATUS + [f'{s}{i}' for s in ['BILL_AMT', 'PAY_AMT'] for i in range(1, 7)]

def engineer(frame):
    x = frame[BASE].copy()
    bills = frame[[f'BILL_AMT{i}' for i in range(1, 7)]]
    payments = frame[[f'PAY_AMT{i}' for i in range(1, 7)]]
    x['utilization_latest'] = frame.BILL_AMT1 / frame.LIMIT_BAL
    x['utilization_mean'] = bills.mean(axis=1) / frame.LIMIT_BAL
    x['payment_to_limit'] = payments.mean(axis=1) / frame.LIMIT_BAL
    # A behavioural proxy, not a contractual instalment coverage ratio.
    x['payment_to_positive_bill'] = payments.mean(axis=1) / bills.clip(lower=0).mean(axis=1).clip(lower=1)
    x['bill_change_to_limit'] = (frame.BILL_AMT1 - frame.BILL_AMT6) / frame.LIMIT_BAL
    x['months_delayed'] = (frame[STATUS] > 0).sum(axis=1)
    x['maximum_delay'] = frame[STATUS].clip(lower=0).max(axis=1)
    return x.replace([np.inf, -np.inf], np.nan)

class CreditFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = np.asarray(engineer(X).columns, dtype=object)
        return self
    def transform(self, X):
        return engineer(X)
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
