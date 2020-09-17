import numpy as np
from ml.core.losses import rss

def mse(y_true, y_pred):
    y_true = y_true.reshape(y_pred.shape)
    return np.mean(np.square(y_true - y_pred))


def r2_score(y_true, y_pred):
    y_true = y_true.reshape(y_pred.shape)
    SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
    SS_res = rss(y_true, y_pred)
    return 1 - SS_res / SS_tot


def accuracy(y_true, y_pred):
    y_pred = y_pred.reshape(y_true.shape)
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    y_pred = y_pred.reshape(y_true.shape)
    cm = []
    classes = np.unique(y_true)
    for i in classes:
        r = []
        for j in classes:
            r.append(((y_true == i) & (y_pred == j)).sum())
        cm.append(r)
    return np.array(cm)


def euclidean_distance(a, b):
        return  np.sqrt(np.sum(np.square(a - b)))


def k_fold(X, y, k, model, metric):
    N = len(X)
    shuffle = np.random.permutation(N)
    X, y = X[shuffle], y[shuffle]
    k_size = N // k
    metrics = np.zeros(k)

    for j, i in enumerate(range(0, N, k_size)):
        val = np.zeros(len(X), dtype=bool)
        val[range(i, i + k_size)] = True
        model.fit(X[~val], y[~val])
        y_pred = model.predict(X[val])
        y_val = y[val]
        metrics[j] = metric(y_val, y_pred)

    model.fit(X, y)
    return metrics
