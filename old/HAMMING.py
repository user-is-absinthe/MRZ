import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

E = np.sign(np.load('E.npy'))
X = np.sign(np.load('X.npy'))
X = X + np.random.random(X.shape)*0.1
# const
N = X.shape[1]
EPSILON = 1/N
M = E.shape[0]

X = np.hstack((np.ones((X.shape[0], 1)), X))
Q = np.eye(M, M) - EPSILON*(np.ones((M, M)) - np.eye(M, M))

ans = list()
for x in X:  # KO
    x.reshape((N+1, 1))
    # step 1
    W = np.hstack((np.ones((E.shape[0], 1))*(N/2), E/2))
    alpha = np.dot(W, x)
    y = alpha
    y_prev = y * 100
    while not np.array_equal(y, y_prev):
        y_prev = y
        # step 2
        y = np.maximum(np.zeros(y.shape), np.dot(Q, y))
    ans.append(np.argmax(y))

val = list(Counter(ans).items())
val.sort(key=lambda x: x[0])
val = [v for k, v in val]
rang = list(set(ans))
rang.sort()
plt.bar(rang, val)
plt.savefig('hamming.png')