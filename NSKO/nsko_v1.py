

import numpy as np

# default_intro
def additional_constructions(ar_x, ar_cl):
    array_of_y = np.ones((1, len(ar_x)))
    # print(array_of_y[0][2])

    # ar_x_with_class = []
    for i in range(len(ar_x)):
        ar_x[i].append(1 if ar_cl[i] == 0 else -1)
    print(ar_x)

    for i in range(len(ar_x)):
        if i == 0:
            matrix_v = np.array(ar_x[i])
        else:
            matrix_v = np.vstack((matrix_v, ar_x[i]))
    matrix_v = matrix_v.transpose()
    print(matrix_v)

    pass

if __name__ == '__main__':

    array_of_X, array_of_classes = [
        [1, 2],
        [0, 2],
        [1, 3],
        [3, 2]
    ], [
        0,
        0,
        1,
        1
    ]

    additional_constructions(ar_x=array_of_X, ar_cl=array_of_classes)

