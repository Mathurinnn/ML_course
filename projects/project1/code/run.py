import numpy as np
from dask.array.random import logistic

from projects.project1.code.cross_validation import build_k_indices, cross_validation, general_cross_validation, \
    logistic_cross_validation
from projects.project1.code.implementations import logistic_regression
from projects.project1.helpers import load_csv_data

x_train, x_test, y_train, train_ids, test_ids = load_csv_data("/Users/deschryver/PycharmProjects/ML_course/projects/project1/data/dataset")

columnnsToRemove = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,52,53,54,55,56,57,58,59,62,63,64,119,120,121,123,122,124,125,126,127,128,129,130,131,132,133,134,135,136,178,195,197,199,200,201,202,203,204,217,218,219,220,221,222,223,224,225,226,227,228,229,230, 240,241,242,243,244,245,246,251]
not_seve_or_nine = [0, 51, 60, 103, 107, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 281,282,283,284,286,287,290,291,292,293,296,297.298,300,301,302,303,304,305]

number_of_columns = x_train.shape[1]
number_of_lines = x_train.shape[0]


for l in range(0, number_of_lines - 1):
    for r in range(0, number_of_columns - 1):
        if (not (r in not_seve_or_nine)):
            if(x_train[l,r] == 7 or x_train[l,r] == 9):
                x_train[l,r] = 0

x_train = np.delete(x_train, columnnsToRemove, axis=1)
x_test = np.delete(x_test, columnnsToRemove, axis=1)

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