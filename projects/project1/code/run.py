import numpy as np

from projects.project1.code.cross_validation import build_k_indices, cross_validation, general_cross_validation
from projects.project1.code.implementations import logistic_regression
from projects.project1.helpers import load_csv_data

x_train, x_test, y_train, train_ids, test_ids = load_csv_data("/Users/deschryver/PycharmProjects/ML_course/projects/project1/data/dataset")
columnnsToRemove = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,52,53,54,55,56,57,58,59,62,128,129,130,131,132,133,134,135,136,178,195,197,199,200,201,202,203,204,217,218,219,220,221,222,223,224,225,226,227,228,229,230, 240,241,242,243,244,245,246,251]

x_train = np.delete(x_train, columnnsToRemove, axis=1)
x_test = np.delete(x_test, columnnsToRemove, axis=1)

#on remplace les nan par des 0
x_train = np.nan_to_num(x_train)
x_test = np.nan_to_num(x_test)
y_train = np.nan_to_num(y_train)

#prepare la cross validation
k = 5
seed = 3024920
k_indices = build_k_indices(y_train, k, seed)

max_iter = 1000
gamma = 0.1
#cross validation
losses_tr = []
losses_te = []
for fold in range(k):
    ls_tr, ls_te = general_cross_validation(y_train, x_train,k_indices, fold, logistic_regression,[max_iter, gamma])
    losses_tr.append(ls_tr)
    losses_te.append(ls_te)

print(losses_tr)
print(losses_te)