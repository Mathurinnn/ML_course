import numpy as np
from dask.array.random import logistic

from projects.project1.code.clean_data import clean_data
from projects.project1.code.cross_validation import build_k_indices, cross_validation, general_cross_validation, \
    logistic_cross_validation
from projects.project1.code.implementations import logistic_regression
from projects.project1.helpers import load_csv_data

x_train, x_test, y_train, train_ids, test_ids = load_csv_data("/Users/deschryver/PycharmProjects/ML_course/projects/project1/data/dataset")

x_train = clean_data(x_train)
x_test = clean_data(x_test)

#on remplace les nan par des 0
x_train = np.nan_to_num(x_train)
x_test = np.nan_to_num(x_test)
y_train = np.nan_to_num(y_train)

"""
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