import numpy as np
import os
import pandas as pd


def make_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def combine(X1, X2):
    n1, m1 = X1.shape
    n2, m2 = X2.shape
    X_new = np.zeros((n1, m1+m2))
    X_new[:, :m1] = X1
    X_new[:, m1:] = X2
    return X_new


def indices_except(x, n):
    return [i for i in range(n) if i != x]


def remove_nans(X, y):
    df = pd.DataFrame(X)
    df.loc[:, 'y'] = y
    df = df.dropna(axis=0)
    return (df.drop(['y'], axis='columns').values, df.loc[:, 'y'].values.reshape((-1, 1)))
