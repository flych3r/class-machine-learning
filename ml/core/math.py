import numpy as np


def variance(X):
    avg = np.mean(X, axis=0)
    var = np.mean((X-avg)**2, axis=0)
    return var

def covariance_matrix(X):
    n = X.shape[1] - 1
    mean = np.mean(X, axis=1)
    X = X - mean[:, None]
    cov_mat = (X @ X.T) * (1 / n)
    return cov_mat

def mode(a):
        (_, idx, counts) = np.unique(a, return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        return a[index]
