import numpy as np


def rss(y_true, y_pred):
    return np.sum(np.square(y_true - y_pred))
