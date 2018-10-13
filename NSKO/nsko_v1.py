import numpy as np

# default_intro
def additional_constructions(ar_x, ar_cl):
    array_of_Y = np.ones((len(ar_x), 1))
    print(array_of_Y)
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

