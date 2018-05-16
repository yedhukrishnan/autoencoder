# XOR using Simple Neural Network with one hidden layer

import numpy as np

LEARNING_RATE = 0.1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Number of training samples
m = 4

# train x
x = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]).T

# train y
y = np.array([[0, 1, 1, 0]])

# Initialize weight and bias for hidden layer
W1= np.random.rand(2, 2)
b1 = np.random.rand(2, 1)

# Initialize weight and bias for output layer
W2 = np.random.rand(1, 2)
b2 = np.random.rand(1, 1)

for i in range(10000):
    # Hidden Layer
    z1 = np.dot(W1, x) + b1
    a1 = sigmoid(z1)

    # Output Layer
    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)

    # Calculate the loss/error, using log
    loss = -(y * np.log(a2) + (1 - y) * np.log(1 - a2))

    # Cost or average loss
    J = loss.mean()

    # Calculate derivatives
    dz2 = a2 - y
    dW2 = (1.0 / m) * np.dot(dz2, a1.T)
    db2 = (1.0 / m) * np.sum(dz2, axis = 1, keepdims = True)
    dz1 = np.dot(W2.T, dz2) * (1 - np.power(a1, 2))
    dW1 = (1.0 / m) * np.dot(dz1, x.T)
    db1 = (1.0 / m) * np.sum(dz1, axis = 1, keepdims = True)

    # Update the parameters
    W2 = W2 - LEARNING_RATE * dW2
    b2 = b2 - LEARNING_RATE * db2
    W1 = W1 - LEARNING_RATE * dW1
    b1 = b1 - LEARNING_RATE * db1

    print("Average Loss %g" % J)
