import numpy as np

from labs.ex02.template.costs import compute_loss


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    n = np.shape(y)[0]
    e = y - tx @ w
    grad = -(1 / n) * (np.transpose(tx) @ e)

    return grad


