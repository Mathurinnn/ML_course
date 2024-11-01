import numpy as np

def clean_data(x_train):
    columnnsToRemove = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 51,
                        52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 102, 106, 118, 119, 120, 122, 121, 123, 124, 125,
                        126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 177, 194, 196, 198, 199, 200, 201, 202, 203,
                        216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 239, 240, 241, 242, 243,
                        244, 245, 250]
    not_seve_or_nine = [27, 28, 14, 50, 59, 88, 91, 106, 22, 251, 252, 253, 254, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 280, 281, 282, 283, 285, 286, 289, 290, 291, 292, 295, 296.298, 299, 300, 301, 302, 303, 304]
    seven_nine_hundred =[77, 81, 82, 83, 84, 85, 86, 89, 90, 92, 93, 110, 143, 150]

    seventy_seven = [60, 75, 78, 79, 80, 98, 112, 168, 206, 207, 208, 209, 210, 211, 212, 213]
    ninety_thousands = [262, 264, 287, 288, 293, 294, 298]
    number_of_columns = x_train.shape[1]
    number_of_lines = x_train.shape[0]
    special_case = {
        101 :  lambda x : x >= 777777,
        105: lambda x: x >= 777777,
        49: lambda x: x >= 98,
        110: lambda x: x >= 777,
        77: lambda x: x == 777 or x == 999,
        143: lambda x: x >= 777,
        145: lambda x: x >= 98,
        147: lambda x: x >= 88,
        148: lambda x: x >= 88,
        149: lambda x: x >= 88,
        150: lambda x: x >= 777,
        195: lambda x: x >= 97,
        262: lambda x: x == 900,
    }
    for l in range(0, number_of_lines - 1):
        for c in range(0, number_of_columns - 1):
            if (not (c in not_seve_or_nine)):
                if (c in seven_nine_hundred):
                        if(x_train[l,c] == 777 or x_train[l,c] == 999):
                            x_train[l,c] = 0
                        if(c != 77 and c != 93):
                            if(x_train[l,c] == 888):
                                x_train[l,c] = 0
                else:
                    if(c in seventy_seven):
                        if(x_train[l,c] >=  77):
                            x_train[l,c] = 0
                    else:
                        if(c in ninety_thousands):
                            if(x_train[l,c] == 99900):
                                x_train[l,c] = 0
                        else:
                            sc = special_case.get(c)
                            if(sc is not None):
                                if(sc(x_train[l,c])):
                                    x_train[l,c] = 0
                            else:
                                if (x_train[l, c] == 7 or x_train[l, c] == 9 or x_train[l, c] == 8):
                                    x_train[l, c] = 0

    x_train = np.delete(x_train, columnnsToRemove, axis=1)
    x_train = np.nan_to_num(x_train)

    min_colums = np.min(x_train,axis=0)
    max_colums = np.max(x_train,axis=0)

    for i in range(x_train.shape[1]):
        x_train[:,i] = np.vectorize(lambda x : (x - min_colums[i]) / (max_colums[i] - min_colums[i]))(x_train[:,i])

    return x_train