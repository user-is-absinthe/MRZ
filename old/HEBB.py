import matplotlib.pyplot as plt
import numpy as np


X, y = np.load('X.npy'), np.load('y.npy')

h = .02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# HEBB
def learn_HEBB(X, Y):
    np.place(Y, Y == 0, [-1])
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    # step 1
    w = np.zeros((3,1))
    pr_Y = np.sign(np.dot(X, w))
    pr_Y = np.ravel(pr_Y)
    count_iter = 0
    while ((np.abs(pr_Y - Y)).sum() > 500) or (count_iter < 1000):
        # step 2
        for x, y in zip(X, Y):
            x = np.reshape(x, w.shape)
            w = w + x*y
        pr_Y = np.sign(np.dot(X, w))
        pr_Y = np.ravel(pr_Y)
        count_iter += 1
        print(count_iter)
    return w

def LRF(w, x):
    return np.heaviside(w[0] + w[1]*x[0] + w[2]*x[1], 1)

w = learn_HEBB(X, y)
print(w)
np.place(y, y == -1, [0])
# plot HEBB
plt.scatter(X[:, 0], X[:, 1], c=y,
                   alpha=0.9, edgecolors='black', s=25)
Z = np.array([LRF(w, x) for x in (np.c_[xx.ravel(), yy.ravel()])])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.3)
plt.savefig('hebb.png')
plt.clf()