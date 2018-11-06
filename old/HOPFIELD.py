import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1
h = 0.1

E = np.sign(np.load('E.npy'))
X = np.sign(np.load('X.npy'))
X = X + np.random.random(X.shape)*0.1
N = X.shape[1]

ans = []
for x in X:
    # step 1
    y = x
    y_prev = y*100
    while np.linalg.norm(y - y_prev) > EPSILON:
        y.reshape((N, 1))
        y_prev = y
        # step 2
        W = np.dot(E.transpose(), E)
        Q = np.eye(N, N) + h*W
        y = np.sign(np.dot(Q, y))

    ans.append(y)

rate_of_good_map = np.array([any([np.array_equal(e, a) for e in E]) for a in ans]).mean()
print(rate_of_good_map)