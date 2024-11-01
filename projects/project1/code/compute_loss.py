import math

import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    n = np.shape(y)[0]
    e = y - tx @ w
    return (1 / (2 * n)) * (np.transpose(e) @ e)


def compute_logistic_loss(y, tx, w):
    n = np.shape(y)[0]
    g = tx @ w
    loss = (1 / n) * -(y.T @ g) + (1 / n) * np.sum(np.vectorize(lambda x: math.log(1 + math.exp(x)))(g), 0)
    return loss


def compute_loss_logistic_two(y, tx, w):
    def log_function(gi):
        return math.log(1 + math.exp(gi))

    n = np.shape(y)[0]

    return ((1 / n) * np.sum(np.vectorize(log_function)(-y @ (tx @ w))))
