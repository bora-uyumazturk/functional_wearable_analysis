"""Linear model on top of matrix factorization methods"""
import numpy as np

from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

class LinearFactorizationModel():
    """Linear model on top of matrix factorization.

    Example Usage:

    lfm = LinearFactorizationModel()
    lfm.fit(X_train, y_train)
    y_pred = lfm.predict(X_test)

    Attributes:
        num_components (int): Number of components to include in matrix factorization.
        num_static (int): Number of static features (features that won't be factorized).
        factorizer (object): Factorizer class which takes in matrix of examples and outputs
                             lower dimensional examples.
        lm (sklearn.linear_model.LinearRegression): Linear regression model.
    """
    def __init__(self, num_components, method, num_static):
        """Initializer.

        Args:
            num_components (int): Number of components to include in matrix factorization.
            method (str): determines factorization method, must be one of ('dft', 'nmf',
                          'pca', 'kmeans').
            num_static (int): Number of static features in model input.
        """
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
        """Fit model to training data.

        Input consists of matrix of shape (num_examples, num_features).
        First num_static features should be static, the rest are dynamic (i.e.
        a time series) and should be factorized. Fits factoriation as well
        as linear model coefficients.

        Args:
            X (np.array): Matrix of shape (num_examples, num_features). First num_static features
                          should be scalars, after that the features are samples from a single time
                          series.
            y (np.array): Column vector of shape (num_examples, 1) holding targets for each example.
        """
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
        """Returns predictions for each row of input.

        Args:
            X (np.array): Matrix of shape (num_examples, num_features). First num_static features
                          should be scalars, after that the features are samples from a single time
                          series.
        Returns:
            Array of predictions of shape (num_examples,).
        """
        X_static = X[:, :self.num_static]
        if self.num_components == 0:
            return self.lm.predict(X_static)
        X_ts = X[:, self.num_static:]
        X_tsfmed = self.factorizer.transform(X_ts)
        X_new = self.combine(X_static, X_tsfmed)
        return self.lm.predict(X_new)

    def combine(self, X1, X2):
        """Helper function to combine two matrices into one."""
        n1, m1 = X1.shape
        n2, m2 = X2.shape
        X_new = np.zeros((n1, m1+m2))
        X_new[:, :m1] = X1
        X_new[:, m1:] = X2
        return X_new


class DFTFactorizer():
    """Factorizer using discrete fourier transform.

    Reduces dimension of input using magnitudes of the
    first num_components frequency components of the discrete
    fourier transform (omitting the first term, which is just
    the average value).

    Note: normalizes input before fitting.

    Attributes:
        num_components (int): Number of components of dft to use.
    """
    def __init__(self, num_components=None):
        """Initializer.

        Args:
            num_components (int): Number of components of dft to use.
        """
        self.num_components = num_components

    def fit(self, X):
        """Fits factorizer (does nothing for dft since it is not adaptive)."""
        return

    def transform(self, X):
        """Transforms input using discrete fourier transform.

        Args:
            X (np.array): Matrix of shape (num_examples, num_samples).

        Returns:
            Transformed matrix of shape (num_examples, num_components).
        """
        n, m = X.shape
        X = X / np.std(X, axis=1).reshape((-1, 1))
        X_f = np.abs(np.fft.fft(X, axis=1))
        X_f = X_f[:, 1:m//2]
        if self.num_components is None:
            return X_f
        return X_f[:, :self.num_components]


class NMFFactorizer():
    """Factorizer using nonnegative matrix factorization.

    Note: normalizes input before fitting.

    Attributes:
        model (NMF): sklearn nonnegative factorization model
        num_components (int): Number of components (i.e. inner dimension)
                              to use for nonnegative matrix factorization.
    """
    def __init__(self, num_components):
        """Initializer.

        Args:
            num_components (int): Number of components (i.e. inner dimension)
                                  to use for nonnegative matrix factorization.
        """
        self.model = NMF(n_components=num_components, init='random')
        self.num_components = num_components

    def fit(self, X):
        """Fit factorizer to data.

        Computer nonnegative factorization of input data X = W*H,
        where the inner dimension is num_components.

        Args:
            X (np.array): Data matrix, of shape (num_examples, original_dimension).
        """
        X = X / np.std(X, axis=1).reshape((-1, 1))
        self.model.fit(X)

    def transform(self, X):
        """Transforms input using previously learned nonnegative matrix factorization.

        Args:
            X (np.array): Matrix of shape (num_examples, num_samples).

        Returns:
            Transformed matrix of shape (num_examples, num_components).
        """
        X = X / np.std(X, axis=1).reshape((-1, 1))
        return self.model.transform(X)


class PCAFactorizer():
    """Factorizer using principal component analysis.

    Reduces dimension of input by projecting onto first
    num_components principal components.

    Note: normalizes input before fitting.

    Attributes:
        model (PCA): sklearn PCA model
        num_components (int): Number of principal components to use.
    """
    def __init__(self, num_components):
        """Initializer.

        Args:
            num_components (int): Number of principal components to use.
        """
        self.model = PCA(n_components = num_components)
        self.num_components = num_components

    def fit(self, X):
        """Computes principal components of data.

        Args:
            X (np.array): Data matrix, of shape (num_examples, original_dimension).
        """
        X = X / np.std(X, axis=1).reshape((-1, 1))
        self.model.fit(X)

    def transform(self, X):
        """Transforms input using previously learned principal components.

        Args:
            X (np.array): Matrix of shape (num_examples, num_samples).

        Returns:
            Transformed matrix of shape (num_examples, num_components).
        """
        X = X / np.std(X, axis=1).reshape((-1, 1))
        return self.model.transform(X)


class KMeansFactorizer():
    """Factorizer using kmeans algorithm.

    Converts input to one-hot representation indicating
    cluster membership.

    Note: normalizes input before fitting.

    Attributes:
        model (kmeans): sklearn kmeans model
        num_components (int): Number of clusters to use.
    """
    def __init__(self, num_components):
        """Initializer.

        Args:
            num_components (int): Number of clusters to use.
        """
        self.model = KMeans(n_clusters = num_components)
        self.num_components = num_components

    def fit(self, X):
        """Computer cluster centroids using kmeans algorithm.

        Args:
            X (np.array): Data matrix, of shape (num_examples, original_dimension).
        """
        X = X / np.std(X, axis=1).reshape((-1, 1))
        self.model.fit(X)

    def transform(self, X):
        """Transforms input using previously learned cluster centroid.

        Converts input examples to one hot representations indicating
        which cluster they belong to.

        Args:
            X (np.array): Matrix of shape (num_examples, num_samples).

        Returns:
            Transformed matrix of shape (num_examples, num_components).
        """
        X = X / np.std(X, axis=1).reshape((-1, 1))
        labels = self.model.predict(X)
        return self.to_one_hot(labels)

    def to_one_hot(self, labels):
        """Helper function to convert numeric labels to one hot vectors."""
        one_hot = np.zeros((labels.size, self.num_components))
        one_hot[np.arange(labels.size), labels] = 1.0
        return one_hot

