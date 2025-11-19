import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(path):
    df = pd.read_csv(path)
    return df

def basic_preprocess(df, scale_amount_time=True):
    df = df.copy()
    df = df.drop_duplicates()
    df = df.fillna(df.median())

    if scale_amount_time:
        from sklearn.preprocessing import RobustScaler
        rs = RobustScaler()
        df['scaled_amount'] = rs.fit_transform(df['Amount'].values.reshape(-1,1))
        df['scaled_time'] = rs.fit_transform(df['Time'].values.reshape(-1,1))
        df.drop(['Time','Amount'], axis=1, inplace=True)
        cols = df.columns.tolist()
        cols = ['scaled_amount','scaled_time'] + [c for c in cols if c not in ('scaled_amount','scaled_time')]
        df = df[cols]
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y

def resample_smote(X, y, random_state=42):
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res
