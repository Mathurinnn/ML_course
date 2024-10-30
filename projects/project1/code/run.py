import math

import numpy as np
from dask.array.random import logistic

from projects.project1.code.clean_data import clean_data
from projects.project1.code.compute_loss import compute_loss, compute_logistic_loss, compute_loss_logistic_two
from projects.project1.code.cross_validation import build_k_indices, cross_validation, general_cross_validation, \
    logistic_cross_validation
from projects.project1.code.implementations import logistic_regression
from projects.project1.helpers import load_csv_data, create_csv_submission

x_train, x_test, y_train, train_ids, test_ids = load_csv_data("/Users/deschryver/PycharmProjects/ML_course/projects/project1/data/dataset")
print(x_train.shape)
#x_train = clean_data(x_train)
#x_test = clean_data(x_test)
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
x_train= np.nan_to_num(x_train)
print(x_train.shape)
y_train = np.nan_to_num(y_train)
y_train = np.vectorize( lambda x : 1 if x == 1 else 0)(y_train)#map y vers les bons labels
#-------------------------------------------------------------------------------------------------
"""
k = 5
k_indices = build_k_indices(y_train, k, 304920)
loss_tr, loss_te = logistic_cross_validation(y_train, x_train, k_indices, 3, 100, 0.1)
print(loss_tr)
print(loss_te)
"""
w, loss = logistic_regression(y_train, x_train, np.zeros(x_train.shape[1]), 100, 0.1)
print(loss)












































"""k = 29
seed = 3024920
max_iter = 1000
gamma = 0.1

k_indices = build_k_indices(y_train, k, seed)
losses_tr = []
losses_te = []
"""

"""
test_number = math.floor(0.8 * x_test.shape[0])
x_ntrain = x_train[:test_number]
print(x_ntrain.shape)
y_ntrain = y_train[:test_number]
print(y_ntrain.shape)
y_ntest = y_train[test_number:]
print(y_ntest.shape)
x_ntest = x_train[test_number:]
print(x_ntest.shape)
initial_w = np.zeros(x_ntrain.shape[1])
print(initial_w.shape)
w, loss = logistic_regression(y_ntrain, x_ntrain,initial_w, 1000, 1)


print(compute_loss_logistic_two(y_ntest, x_ntest, w))
"""
"""
def compute_logistic(tx, w):
    def logistic(n):
        r = 1 if (math.exp(n)/(math.exp(n)+1)) > 0.5 else - 1
        print((math.exp(n)/(math.exp(n)+1)))
        return  r
    return np.vectorize(logistic)(tx @ w)

create_csv_submission(range(test_number, 328135) ,(compute_logistic(x_ntest, w)), "submission-test.csv")
"""
"""
for fold in range(k):
    print("tour de boucle : " + str(fold))
    ls_tr, ls_te = logistic_cross_validation(y_train, x_train,k_indices, fold, max_iter, gamma)
    losses_tr.append(ls_tr)
    losses_te.append(ls_te)
    print("fin du tour de boucle : " + str(fold))
ls_tr, ls_te = logistic_cross_validation(y_train, x_train,k_indices, 23, max_iter, 0.1)
print("----------0.1--------")
print(ls_tr)
print(ls_te)
ls_tr, ls_te = logistic_cross_validation(y_train, x_train,k_indices, 23, max_iter, 0.2)
print("----------0.2--------")
print(ls_tr)
print(ls_te)
ls_tr, ls_te = logistic_cross_validation(y_train, x_train,k_indices, 23, max_iter, 0.3)
print("----------0.3--------")
print(ls_tr)
print(ls_te)

#prepare la cross validation
k = 29
seed = 3024920

k_indices = build_k_indices(y_train, k, seed)

max_iter = 1000

build_poly_losses_tr = [] #la liste qui contient toutes les train loss pour les degrees de la feature expansion
build_poly_losses_te = [] #la liste qui contient toutes les train loss pour les degrees de la feature expansion

gamma_losses_tr = [] #la liste qui contient toutes les train loss pour les valeurs de gamma
gamma_losses_te = [] #la liste qui contient toutes les test loss pour les valeurs de gamma

#on essaie plein de valeurs pour gamma et on les cross validate
for i in range(10):

    gamma = i / 10
    print("************GAMMA : " + str(gamma)+ "************")
    losses_tr = []
    losses_te = []

    #rajouter une autre for loop pour la cross validation
    for fold in range(k):
        print("tour de boucle : " + str(fold))
        ls_tr, ls_te = logistic_cross_validation(y_train, x_train,k_indices, fold, max_iter, gamma)
        losses_tr.append(ls_tr)
        losses_te.append(ls_te)
        print("fin du tour de boucle : " + str(fold))

    gamma_losses_tr.append(np.average(losses_tr))
    gamma_losses_te.append(np.average(losses_te))

print("gamme losses te")
print(gamma_losses_te)
print("gamme losses tr")
print(gamma_losses_tr)
"""
