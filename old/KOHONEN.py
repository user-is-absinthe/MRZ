import numpy as np
import matplotlib.pyplot as plt

COUNT_CLUSTER = 8
H = 0.9
E = 0.1
X = np.load('X.npy')
N = X.shape[1]
# step 1
W = np.array([np.random.random((N, )) for i in range(COUNT_CLUSTER)])
W_prev = np.array([np.random.random((N, ))*100 for _ in range(COUNT_CLUSTER)])
iter_count = 0
diff = np.linalg.norm(W_prev - W)
is_diff = (diff > E)
# step 2
while is_diff and H > 0:  # KO
    W_prev = W
    for x in X:
        # step 3
        i_curr_update_index = np.argmin([np.linalg.norm(w-x) for w in W])
        # step 4
        W[i_curr_update_index, :] = W[i_curr_update_index, :] + H*(x - W[i_curr_update_index, :])
    iter_count += 1
    diff = np.linalg.norm(W_prev - W)
    print((iter_count, diff))
    is_diff = (diff > E)
    # decrease H
    if iter_count%10 == 0:
        H -= 0.05

print(W)
np.save('E', W)
plt.scatter(X[:, 0], X[:, 1], alpha=0.9, edgecolors='black', s=25)
plt.scatter(W[:, 0], W[:, 1], c='r', s=45)
plt.savefig('kohonen4.png')
plt.clf()
