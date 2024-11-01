import math

import numpy as np
from dask.array.random import logistic

from clean_data import clean_data
from compute_loss import compute_loss, compute_logistic_loss, compute_loss_logistic_two
from cross_validation import build_k_indices, cross_validation, \
    logistic_cross_validation, get_train_test
from implementations import mean_squared_error_sgd, reg_logistic_regression
from projects.project1.code.compute_gradient import sigmoid
from projects.project1.code.implementations import logistic_regression
from projects.project1.helpers import load_csv_data, create_csv_submission
from sklearn.linear_model import LogisticRegression


def part1():
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("/Users/deschryver/PycharmProjects/ML_course/projects/project1/data/dataset")
    print("x_train.shape : " +str(x_train.shape))
    x_train = clean_data(x_train)
    print(" new x_train.shape : " + str(x_train.shape))
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

    initial_w = np.zeros(x_train.shape[1])

    w, loss = logistic_regression(y_train, x_train, initial_w, max_iter, gamma)

    loss_test = compute_logistic_loss(y_train, x_train, w)
    print("loss_test : " + str(loss_test))
    r = x_test @ w
    r = sigmoid(r)
    y_test_output = np.where(r > 0.5, 1, -1)

    create_csv_submission(test_ids, y_test_output, "submission-3.csv")
