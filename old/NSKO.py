import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy import optimize

X, y = np.load('X.npy'), np.load('y.npy')


h = .02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# NSKO
def MSE(w0, w1, w2,  X, Y):
    error = sum([(y - (w0 + w1*x[0] + w2*x[1]))**2 for x, y in zip(X, Y)])
    return error

def LRF(w, x):
    return np.heaviside(w[0] + w[1]*x[0] + w[2]*x[1], 1)

#bnds = ((-100, 100), (-5, 5))
fun = lambda w: MSE(w[0], w[1], w[2], X, y)
res = optimize.minimize(fun, (0, 0, 0), method='L-BFGS-B')
w0, w1, w2 = res.x
# plot NSKO
plt.scatter(X[:, 0], X[:, 1], c=y,
                   alpha=0.9, edgecolors='black', s=25)
Z = np.array([LRF(res.x, x) for x in (np.c_[xx.ravel(), yy.ravel()])])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.3)
plt.savefig('nsko.png')
plt.clf()