import numpy as np

np.random.seed(0)

x = np.array(((0,0), (0,1), (1,0), (1,1)), dtype=np.float32)
t = np.array(((-1), (-1), (-1), (1)), dtype=np.float32)

# perceptron
w = np.random.normal(0, 0.1, (3, 1))

# add bias
_x = np.r_[x[3], [1]]

y = np.dot(_x, w)


print(y)

# activation sigmoid
y = 1. / (1 + np.exp(-y))

print(y)
