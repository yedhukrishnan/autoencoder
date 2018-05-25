# INCOMPLETE SOLUTION
# Still working on it and it's not accurate yet

# Deep Neural Network for MNIST

import numpy as np
from keras.datasets import mnist

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_backward(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(x, 0)

def relu_backward(x):
    return x > 0

def one_hot(y):
    y_ = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        y_[i][y[i]] = 1
    return y_

def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return {
        'x_train': x_train.reshape(60000, 28 * 28).T[:, :1200] / 255.0,
        'y_train': one_hot(y_train).T[:, :1200],
        'x_test': x_test.reshape(10000, 28 * 28).T[:, :100] / 255.0,
        'y_test': one_hot(y_test).T[:, :100]
    }

def initialize_params(num_nodes):
    mu, sigma = 0, 0.1
    w = [0]
    b = [0]
    for i in range(len(num_nodes) - 1):
        wi = np.random.normal(mu, sigma, (num_nodes[i + 1], num_nodes[i]))
        bi = np.random.normal(mu, sigma, (num_nodes[i + 1], 1))
        # wi = np.random.randn(num_nodes[i + 1], num_nodes[i]) * np.sqrt(2.0 / num_nodes[i])
        # bi = np.random.rand(num_nodes[i + 1], 1)
        w.append(wi)
        b.append(bi)
    return {
        'w': w,
        'b': b
    }

data = get_mnist_data()

num_nodes = [784, 1000, 1000, 1000, 500, 100, 10]

params = initialize_params(num_nodes)
w = params['w']
b = params['b']

y_train = data['y_train']
a = [data['x_train']]
z = [0]
l = len(num_nodes)
m = 600

# Test values
tx = data['x_train'].T[5].T.reshape(784, 1)
ty = data['y_train'].T[5].reshape(10, 1)
print(ty)

for epoch in range(100):
    for i in range(1, len(num_nodes)):
        # Forward propagation
        zi = np.dot(w[i], a[i - 1]) + b[i]
        # if i == len(num_nodes) - 1:
        ai = sigmoid(zi)
        # else:
            # ai = relu(zi)
        z.append(zi)
        a.append(ai)

    # da for the final layer
    dal = -(y_train / a[l - 1]) + (1 - y_train) / (1 - a[l - 1] + 0.00001)
    da = [dal]
    dw = []
    dz = []
    db = []
    for i in range(len(num_nodes) - 1, 0, -1):
        # if i == len(num_nodes) - 1:
        dzi = da[l - i - 1] * sigmoid_backward(z[i])
        # else:
            # dzi = da[l - i - 1] * relu_backward(z[i])
        dwi = (1.0 / m) * np.dot(dzi, a[i - 1].T)
        dbi = (1.0 / m) * np.sum(dzi, axis = 1, keepdims = True)
        dai_1 = np.dot(w[i].T, dzi)
        dz.append(dzi)
        dw.append(dwi)
        db.append(dbi)
        da.append(dai_1)

    dw.append(0)
    db.append(0)
    dz.append(0)
    dw.reverse()
    db.reverse()
    dz.reverse()

    for i in range(1, l):
        w[i] = w[i] - 0.1 * dw[i]
        b[i] = b[i] - 0.1 * db[i]

    # Test part
    zt = []
    at = [tx]
    for i in range(1, len(num_nodes)):
        # Forward propagation
        zi = np.dot(w[i], at[i - 1]) + b[i]
        ai = sigmoid(zi)
        zt.append(zi)
        at.append(ai)

    print('y^ = ', ai)
