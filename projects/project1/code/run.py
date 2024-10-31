import math

import numpy as np
from dask.array.random import logistic

from clean_data import clean_data
from compute_loss import compute_loss, compute_logistic_loss, compute_loss_logistic_two
from cross_validation import build_k_indices, cross_validation, general_cross_validation, \
    logistic_cross_validation
from projects.project1.code.implementations import logistic_regression, ridge_regression, reg_logistic_regression
from projects.project1.helpers import load_csv_data, create_csv_submission


def part1():
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("/Users/deschryver/PycharmProjects/ML_course/projects/project1/data/dataset")
    x_train = clean_data(x_train)
    x_test = clean_data(x_test)
    n = x_train.shape[0]
    ones = np.ones((n, 1))
    x_train = np.concatenate((ones, x_train), axis=1)
    ones = np.ones((x_test.shape[0], 1))
    x_test = np.concatenate((ones, x_test), axis=1)
    return (x_train, x_test, y_train, train_ids, test_ids)
    
    
def part2(data):
    x_train, x_test, y_train, train_ids, test_ids = data
    y_train = np.nan_to_num(y_train)
    y_train = np.vectorize(lambda x: 1 if x == 1 else 0)(y_train)  # map y vers les bons labels
    k = 5
    k_indices = build_k_indices(y_train, k, 304920)
    max_iter = 1000
    gamma = 0.1
    """
    loss_tr, loss_te = logistic_cross_validation(y_train, x_train, k_indices, 3, max_iter, gamma)
    print("max iter : " + str(max_iter))
    print("gamma : " + str(gamma))
    print(loss_tr)
    print(loss_te)
    """

    print("calcul de w")
    w, loss = reg_logistic_regression(y_train, x_train, 1,np.zeros(x_train.shape[1]), max_iter, gamma)
    #w, loss = cross_validation(y_train, x_train, k_indices, 3, max_iter, gamma)
    print(w)
    print(loss)
    #w, loss = ridge_regression(y_train, x_train, 0.1)
    r = x_test @ w
    y_sigmoid_output = np.vectorize(lambda x: (math.exp(x)/(1 + math.exp(x))))(r)
    y_test = np.vectorize(lambda x: 1 if x > 0.5 else -1)(y_sigmoid_output)
    create_csv_submission(test_ids, y_test, "submission.csv")



    """
    w, loss = reg_logistic_regression(y_train, x_train, np.zeros(x_train.shape[1]), 1, 100, 0.1)
    print(loss)
    print(x_train.shape)
    x_train = clean_data(x_train)
    x_test = clean_data(x_test)
    """
    """
    #on ajoute le biais------------------------------------------------------------------------------
    n = x_train.shape[0]
    ones = np.ones((n, 1))
    x_train = np.concatenate((ones, x_train), axis=1)
    n = x_test.shape[0]
    ones = np.ones((n, 1))
    x_test = np.concatenate((ones, x_test), axis=1)
    #------------------------------------------------------------------------------------------------
    """
    #-------------------------------------------------------------------------------------------------
    """
    k = 5
    k_indices = build_k_indices(y_train, k, 304920)
    loss_tr, loss_te = logistic_cross_validation(y_train, x_train, k_indices, 3, 100, 0.1)
    print(loss_tr)
    print(loss_te)

    w, loss = reg_logistic_regression(y_train, x_train, np.zeros(x_train.shape[1]), 1, 100, 0.1)
    print(loss)
    """