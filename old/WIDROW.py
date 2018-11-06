import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy import special

X, y = np.load('X.npy'), np.load('y.npy')

h = .02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

def LRF(w, x):
    return np.heaviside(w[0] + w[1]*x[0] + w[2]*x[1], 1)

def stochastic_gradient_step(X, y, w, train_ind, eta=0.6):
    l = len(y)
    x = X[train_ind, :]
    dot_res = np.dot(w, x)
    sigm_res = special.expit(dot_res)
    return w - eta * (sigm_res - y[train_ind]) * (sigm_res*(1-sigm_res)) * x

def MSE(w, X, Y):
    error = sum([(y - sc.special.expit(w[0] + w[1]*x[0] + w[2]*x[1]))**2 for x, y in zip(X, Y)])
    return error

def stochastic_gradient_descent(X, y, w_init, eta=0.2, max_iter=1e4,
                                min_weight_dist=1e-8, seed=42, verbose=False):
    # Инициализируем расстояние между векторами весов на соседних
    # итерациях большим числом.
    weight_dist = np.inf
    # Инициализируем вектор весов
    w = w_init
    # Сюда будем записывать ошибки на каждой итерации
    errors = []
    # Счетчик итераций
    iter_num = 0
    # Будем порождать псевдослучайные числа
    # (номер объекта, который будет менять веса), а для воспроизводимости
    # этой последовательности псевдослучайных чисел используем seed.
    np.random.seed(seed)

    # Основной цикл
    while weight_dist > min_weight_dist and iter_num < max_iter:
        # порождаем псевдослучайный
        # индекс объекта обучающей выборки
        random_ind = np.random.randint(X.shape[0])
        w_new = stochastic_gradient_step(X, y, w, random_ind, eta=0.2)
        # print(w_new)
        error = MSE(w_new, X,y)
        errors.append(error)
        weight_dist = np.linalg.norm(w - w_new)
        w = w_new
        if verbose:
            print('error %f' % error)
    return w, errors

X = np.hstack((np.ones((X.shape[0], 1)), X))
w, stoch_errors_by_iter = stochastic_gradient_descent(X, y, np.zeros(3), eta=1e-2, max_iter=1e2,
                                                      min_weight_dist=1e-4, seed=42, verbose=True)
print(w)
# plot WIDROW
X = X[:, 1:]
plt.scatter(X[:, 0], X[:, 1], c=y,
                   alpha=0.9, edgecolors='black', s=25)
Z = np.array([LRF(w, x) for x in (np.c_[xx.ravel(), yy.ravel()])])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.3)
plt.savefig('widrow.png')
plt.clf()