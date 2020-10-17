import numpy as np
from ml.core.math import covariance_matrix


class PCA():
    def __init__(self, n_components=None, exp_variance=None):
        self.n_components = n_components
        self.exp_variance = exp_variance
        self.explained_variances_ratio = None

    def fit(self, X):
        cov_mat = covariance_matrix(X.T)
        values, vectors = np.linalg.eig(cov_mat)
        explained_variances = []
        for value in values:
            explained_variances.append(value / np.sum(values))
        self.explained_variances = np.array(explained_variances)
        self.vectors = vectors

    def transform(self, X):
        pc = np.zeros_like(X)
        for i, v in enumerate(self.vectors.T):
            if self.n_components and i == self.n_components:
                break
            if self.exp_variance and np.sum(self.explained_variances[:i]) >= self.exp_variance:
                break
            pc[:, i] = X @ v
        self.explained_variance_ratio = self.explained_variances[:i]
        return pc[:, :i]
