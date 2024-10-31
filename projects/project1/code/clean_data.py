import numpy as np

def clean_data(x_train):
    columnnsToRemove = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,52,53,54,55,56,57,58,59,60,62,63,64,103,107,119,120,121,123,122,124,125,126,127,128,129,130,131,132,133,134,135,136,178,195,197,199,200,201,202,203,204,217,218,219,220,221,222,223,224,225,226,227,228,229,230, 240,241,242,243,244,245,246,251]
    not_seve_or_nine = [0, 28,29,15, 51, 60, 89,92, 107, 23,252,253,254,255,267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 281,282,283,284,286,287,290,291,292,293,296,297.298,300,301,302,303,304,305]
    #78, 94 garder 888,
    seven_nine_hundred =[78, 82,83,84,85,86,87,90,91, 93, 94, 111, 144, 151]

    seventy_seven = [61, 76, 79, 80,81, 99, 113,169, 207, 208, 209, 210, 211, 212, 213, 214]
    ninety_thousands = [263, 265, 288, 289, 294, 295, 299]
    number_of_columns = x_train.shape[1]
    number_of_lines = x_train.shape[0]
    special_case = {
        102 :  lambda x : x >= 777777,
        106: lambda x: x >= 777777,
        50: lambda x: x >= 98,
        111: lambda x: x >= 777,
        78: lambda x: x == 777 or x == 999,
        144: lambda x: x >= 777,
        146: lambda x: x >= 98,
        148: lambda x: x >= 88,
        149: lambda x: x >= 88,
        150: lambda x: x >= 88,
        151: lambda x: x >= 777,
        196: lambda x: x >= 97,
        263: lambda x: x == 900,
    }
    for l in range(0, number_of_lines - 1):
        for c in range(0, number_of_columns - 1):
            if (not (c in not_seve_or_nine)):
                if (c in seven_nine_hundred):
                        if(x_train[l,c] == 777 or x_train[l,c] == 999):
                            x_train[l,c] = 0
                        if(c != 78 and c != 94):
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

    min_colums = np.min(x_train,axis=1)
    max_colums = np.max(x_train,axis=1)

    for i in range(x_train.shape[1]):
        x_train[:,i] = np.vectorize(lambda x : (x - min_colums[i]) / (max_colums[i] - min_colums[i]))(x_train[:,i])

    return x_train