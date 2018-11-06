import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron

X, y = np.load('X.npy'), np.load('y.npy')

# learn PERCEPTRON
clf = Perceptron()
clf.fit(X, y)
# plot PERCEPTRON
h = .02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


plt.scatter(X[:, 0], X[:, 1], c=y,
                   alpha=0.9, edgecolors='black', s=25)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.3)
plt.savefig('perceptron1.png')
plt.clf()