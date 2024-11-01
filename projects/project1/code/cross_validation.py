import numpy as np
from dask.array.random import logistic

from implementations import *
from build_poly import *
from projects.project1.code.compute_gradient import sigmoid


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)
    """
    x_train = np.copy(x)
    y_train = np.copy(y)

    x_test = np.zeros(k_indices.shape[1])
    y_test = np.zeros(x_test.shape)
    x_train = np.delete(x_train, k_indices[k])
    y_train = np.delete(y_train, k_indices[k])

    j = 0
    for i in k_indices[k]:
        x_test[j] = x[i]
        y_test[j] = y[i]
        j += 1

    x_train = build_poly(x_train, degree)
    x_test = build_poly(x_test, degree)

    w = ridge_regression(y_train, x_train, lambda_)
    loss_tr = np.sqrt(2 * compute_loss(y_train, x_train, w))
    loss_te = np.sqrt(2 * compute_loss(y_test, x_test, w))

    return loss_tr, loss_te

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def logistic_cross_validation(y, x, k_indices, k, max_iter, gamma):
    """return the loss of logistic regression for a fold corresponding to k_indices

        Args:
            y:          shape=(N,)
            x:          shape=(N,)
            k_indices:  2D array returned by build_k_indices()
            k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
            lambda_:    scalar, cf. ridge_regression()
            degree:     scalar, cf. build_poly()

        Returns:
            f1: f1 score
            loss: the training loss
            loss_te: the test loss
        """
    x_train = np.copy(x)
    y_train = np.copy(y)
    x_test = np.zeros((k_indices.shape[1], x_train.shape[1]))
    y_test = np.zeros(x_test.shape[0])
    x_train = np.delete(x_train, k_indices[k], 0)
    y_train = np.delete(y_train, k_indices[k], 0)
    j = 0
    for i in k_indices[k]:
        x_test[j] = x[i]
        y_test[j] = y[i]
        j += 1
    w = np.zeros(x.shape[1])

    w, loss = logistic_regression(y_train, x_train,w,max_iter, gamma)

    loss_te = compute_logistic_loss(y_test, x_test, w)
    r = x_test @ w
    output = sigmoid(r)
    prediction = np.where(output > 0.5, 1, 0)
    f1 = compute_f1_score(y_test, prediction)
    return f1, loss, loss_te

def get_train_test(y, x, k_indices, k):

    x_train = np.copy(x)
    y_train = np.copy(y)

    x_test = np.zeros((k_indices.shape[1], x_train.shape[1]))
    y_test = np.zeros(x_test.shape[0])
    x_train = np.delete(x_train, k_indices[k], 0)
    y_train = np.delete(y_train, k_indices[k], 0)
    print("la aussi")
    j = 0
    for i in k_indices[k]:
        x_test[j] = x[i]
        y_test[j] = y[i]
        j += 1

    return x_train, y_train, x_test, y_test

def compute_f1_score(y_test, predictions):
    """
        Computes the F1 score

        Parameters:
        y_test : numpy array
            The true labels for the test set.
        predictions : numpy array
            The predicted labels from the model.

        Returns:
        f1_score : float
            The computed F1 score.
        """

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for i in range(len(y_test)):
        if y_test[i] == predictions[i] and predictions[i] == 1:
            true_positive += 1
        if y_test[i] == predictions[i] and predictions[i] == 0:
            true_negative += 1
        if y_test[i] != predictions[i] and predictions[i] == 0:
            false_negative += 1
        if y_test[i] != predictions[i] and predictions[i] == 1:
            false_positive += 1

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    return 2*precision*recall/(precision+recall)