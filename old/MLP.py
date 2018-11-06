from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

X, y = np.load('X.npy'), np.load('y.npy')

# LEARN MLP
clf = MLPClassifier()
clf.fit(X, y)
# PLOT MLP
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
plt.savefig('MLP.png')
plt.clf()