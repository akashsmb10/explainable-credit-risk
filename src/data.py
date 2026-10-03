from pathlib import Path
import hashlib
import io
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .features import BASE

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'event'
SOURCE = 'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip'

def load_data():
    path = ROOT / 'data/raw/credit.xls'
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(SOURCE, headers={'User-Agent': 'credit-risk-portfolio/1.0'})
        payload = urllib.request.urlopen(request, timeout=120).read()
        archive = zipfile.ZipFile(io.BytesIO(payload))
        names = [n for n in archive.namelist() if n.endswith('.xls')]
        if len(names) != 1:
            raise ValueError('Unexpected dataset archive')
        path.write_bytes(archive.read(names[0]))
    df = pd.read_excel(path, header=1).rename(columns={'default payment next month': TARGET})
    validate(df)
    return df, hashlib.sha256(path.read_bytes()).hexdigest()

def validate(df):
    required = ['ID', 'LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE', TARGET]
    required += ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    required += [f'{stem}{i}' for stem in ['BILL_AMT', 'PAY_AMT'] for i in range(1, 7)]
    if set(df.columns) != set(required):
        raise ValueError('Unexpected source schema')
    if not df[TARGET].isin([0, 1]).all() or not df.ID.is_unique:
        raise ValueError('Invalid target or duplicate IDs')
    if not (df.LIMIT_BAL > 0).all() or not df.AGE.between(18, 100).all():
        raise ValueError('Invalid limit or age')
    if (df.filter(regex='^PAY_AMT').min() < 0).any():
        raise ValueError('Negative payments require investigation')
    if not df.filter(regex='^PAY_[023456]$').isin(range(-2, 10)).all().all():
        raise ValueError('Unexpected repayment status')

def split_data(df, seed=42):
    # Group identical scoring inputs even when audit-only demographics differ.
    groups = pd.util.hash_pandas_object(df[BASE], index=False)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(splitter.split(df, df[TARGET], groups))
    parts = {'test': folds[0][1], 'validation': folds[1][1], 'calibration': folds[2][1],
             'train': np.concatenate([folds[3][1], folds[4][1]])}
    for name, indices in parts.items():
        for other, indices2 in parts.items():
            if name != other:
                assert not set(groups.iloc[indices]) & set(groups.iloc[indices2])
    return parts, groups

def audit(df):
    return pd.DataFrame({'dtype': df.dtypes.astype(str), 'missing': df.isna().sum(),
                         'unique': df.nunique(), 'minimum': df.min(), 'maximum': df.max()})
