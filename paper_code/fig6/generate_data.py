"""
Generate data for figure 6, csvs with components for PCA and NMF
    - nmf_components.csv
    - pca_components.csv
when run on data/week_hour.csv
"""
import pandas as pd

from utils import get_factorizer

if __name__ == '__main__':
    # read data
    X = pd.read_csv('../../data/week_hour.csv', index_col=0)
    X.dropna(axis='rows', inplace=True)
    X = X.to_numpy()

    # fit pca, get components
    pm = get_factorizer('pca')
    pm.fit(X)
    pcomp = pd.DataFrame(pm.model.components_)
    pcomp = pcomp.T

    # fit nmf, get components
    nm = get_factorizer('nmf', 6)
    nm.fit(X)
    ncomp = pd.DataFrame(nm.model.components_)
    ncomp = ncomp.T

    # save
    pcomp.to_csv('pca_components.csv')
    ncomp.to_csv('nmf_components.csv')
    
