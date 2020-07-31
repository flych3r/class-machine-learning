import numpy as np


def l2_penalty(penalty, W):
    return penalty * (np.sum(W*W) / 2)
