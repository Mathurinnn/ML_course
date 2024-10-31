import math

import numpy as np
from dask.array.random import logistic

from clean_data import clean_data
from compute_loss import compute_loss, compute_logistic_loss, compute_loss_logistic_two
from cross_validation import build_k_indices, cross_validation, general_cross_validation, \
    logistic_cross_validation
from implementations import logistic_regression, reg_logistic_regression
from helpers import load_csv_data, create_csv_submission

def part1():
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("C:/Users/mathu/PycharmProjects/ML_course/projects/project1/data/dataset/dataset")
    x_train = clean_data(x_train)
    x_test = clean_data(x_test)

    return (x_train, x_test, y_train, train_ids, test_ids)
    
    
def part2(data):
    x_train, x_test, y_train, train_ids, test_ids = data
    
    y_train = np.nan_to_num(y_train)
    y_train = np.vectorize(lambda x: 1 if x == 1 else 0)(y_train)  # map y vers les bons labels
    k = 5
    k_indices = build_k_indices(y_train, k, 304920)
    loss_tr, loss_te = logistic_cross_validation(y_train, x_train, k_indices, 3, 100, 0.1)
    print(loss_tr)
    print(loss_te)
    w, loss = reg_logistic_regression(y_train, x_train, np.zeros(x_train.shape[1]), 1, 100, 0.1)
    print(loss)
    print(x_train.shape)
    x_train = clean_data(x_train)
    x_test = clean_data(x_test)
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
    print(x_train.shape)
    y_train = np.nan_to_num(y_train)
    y_train = np.vectorize( lambda x : 1 if x == 1 else 0)(y_train)#map y vers les bons labels
    #-------------------------------------------------------------------------------------------------
    k = 5
    k_indices = build_k_indices(y_train, k, 304920)
    loss_tr, loss_te = logistic_cross_validation(y_train, x_train, k_indices, 3, 100, 0.1)
    print(loss_tr)
    print(loss_te)

    w, loss = reg_logistic_regression(y_train, x_train, np.zeros(x_train.shape[1]), 1, 100, 0.1)
    print(loss)
