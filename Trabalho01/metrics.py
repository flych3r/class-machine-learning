import numpy as np
from losses import rss


def mse(y_true, y_pred):
    y_true = y_true.reshape(y_pred.shape)
    return np.mean(np.square(y_true - y_pred))


def r2_score(y_true, y_pred):
    y_true = y_true.reshape(y_pred.shape)
    SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
    SS_res = rss(y_true, y_pred)
    return 1 - SS_res / SS_tot
