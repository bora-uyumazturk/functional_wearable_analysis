import numpy as np

from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

class LinearFactorizationModel():
    def __init__(self, num_components, method, num_static):
        self.num_components = num_components
        self.num_static = num_static
        self.factorizer = None
        self.lm = LinearRegression()
        if num_components == 0:
            return
        if method == 'dft':
            self.factorizer = DFTFactorizer(self.num_components)
        if method == 'nmf':
            self.factorizer = NMFFactorizer(self.num_components)
        if method == 'pca':
            self.factorizer = PCAFactorizer(self.num_components)
        if method == 'kmeans':
            self.factorizer = KMeansFactorizer(self.num_components)

    def fit(self, X, y):
        X_static = X[:, :self.num_static]
        if self.num_components == 0:
            self.lm.fit(X_static, y)
            return
        X_ts = X[:, self.num_static:]
        self.factorizer.fit(X_ts)
        X_tsfmed = self.factorizer.transform(X_ts)
        X_new = self.combine(X_static, X_tsfmed)
        self.lm.fit(X_new, y)

    def predict(self, X):
        X_static = X[:, :self.num_static]
        if self.num_components == 0:
            return self.lm.predict(X_static)
        X_ts = X[:, self.num_static:]
        X_tsfmed = self.factorizer.transform(X_ts)
        X_new = self.combine(X_static, X_tsfmed)
        return self.lm.predict(X_new)

    def combine(self, X1, X2):
        n1, m1 = X1.shape
        n2, m2 = X2.shape
        X_new = np.zeros((n1, m1+m2))
        X_new[:, :m1] = X1
        X_new[:, m1:] = X2
        return X_new


class DFTFactorizer():

    def __init__(self, num_components=None):
        self.num_components = num_components

    def fit(self, X):
        return

    def transform(self, X):
        n, m = X.shape
        X = X / np.std(X, axis=1).reshape((-1, 1))
        X_f = np.abs(np.fft.fft(X, axis=1))
        X_f = X_f[:, 1:m//2]
        if self.num_components is None:
            return X_f
        return X_f[:, :self.num_components]


class NMFFactorizer():

    def __init__(self, num_components):
        self.model = NMF(n_components=num_components, init='random')
        self.num_components = num_components

    def fit(self, X):
        X = X / np.std(X, axis=1).reshape((-1, 1))
        self.model.fit(X)

    def transform(self, X):
        X = X / np.std(X, axis=1).reshape((-1, 1))
        return self.model.transform(X)


class PCAFactorizer():

    def __init__(self, num_components):
        self.model = PCA(n_components = num_components)
        self.num_components = num_components

    def fit(self, X):
        X = X / np.std(X, axis=1).reshape((-1, 1))
        self.model.fit(X)

    def transform(self, X):
        X = X / np.std(X, axis=1).reshape((-1, 1))
        return self.model.transform(X)


class KMeansFactorizer():

    def __init__(self, num_components):
        self.model = KMeans(n_clusters = num_components)
        self.num_components = num_components

    def fit(self, X):
        X = X / np.std(X, axis=1).reshape((-1, 1))
        self.model.fit(X)

    def transform(self, X):
        X = X / np.std(X, axis=1).reshape((-1, 1))
        labels = self.model.predict(X)
        return self.to_one_hot(labels)

    def to_one_hot(self, labels):
        one_hot = np.zeros((labels.size, self.num_components))
        one_hot[np.arange(labels.size), labels] = 1.0
        return one_hot

