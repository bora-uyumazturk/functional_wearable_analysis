import numpy as np
import os
import pandas as pd

from models import *

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


def get_factorizer(method, num_components=None):
    """Returns factorizer corresponding to method.

    Args:
        method (str): String specifying factorization method. One of
                      ('dft', 'nmf', 'pca', 'kmeans'). 
        num_components (int): number of components to use for factorizer,
                              default None.
    Returns:
        instance of factorizer corresponding to method.
    """
    if method == "dft":
        return DFTFactorizer(num_components)
    if method == "nmf":
        return NMFFactorizer(num_components)
    if method == "pca":
        return PCAFactorizer(num_components)
    if method == "kmeans":
        return KMeansFactorizer(num_components)
    raise(Exception("method: {} not supported".format(method)))


def transform_time_series(df, method, num_components=None):
    """Transforms time series using matrix factorization methods.

    Args:
        df (dataframe): Dataframe holding time series data. Rows
                        are individuals, columns are samples.
        method (str): String specifying factorization method. One of
                      ('dft', 'nmf', 'pca', 'kmeans'). 
        num_components (int): number of components to use for factorizer,
                              default None.
    Returns:
        Dataframe with same index as df, each column is the latent
        component of the factorization.
    """
    factorizer = get_factorizer(method, num_components)
    X = df.values
    factorizer.fit(X)
    X_tnsfmed = factorizer.transform(X)
    out_df = pd.DataFrame(X_tnsfmed)
    out_df.index = df.index
    return out_df
    
