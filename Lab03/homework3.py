import numpy as np
from torchvision.datasets import MNIST


def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)

    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)

    return np.array(mnist_data), np.array(mnist_labels)


train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

# Dividing by 255 because the max possible value is 255. This way, all values will be in the [0, 1] interval.
train_X = train_X / 255.0
test_X = test_X / 255.0


def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    print(one_hot.shape)
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


train_Y = one_hot_encode(train_Y, 10)
test_Y = one_hot_encode(test_Y, 10)


input_size = 784
hidden_size = 100
output_size = 10


np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# hyperparametri
learning_rate = 0.1
batch_size = 128
epochs = 50
l2_lambda = 0.01


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def forward_propagation(X):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def back_propagation(X, y, Z1, A1, A2):
    global W1, b1, W2, b2
    m = X.shape[0]

    # Calcularea gradientelor
    dZ2 = A2 - y
    dW2 = (np.dot(A1.T, dZ2) / m) + (l2_lambda / m) * W2
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = (np.dot(X.T, dZ1) / m) + (l2_lambda / m) * W1
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1


# antrenare
for epoch in range(epochs):
    permutation = np.random.permutation(train_X.shape[0])
    X_shuffled = train_X[permutation]
    y_shuffled = train_Y[permutation]

    for i in range(0, train_X.shape[0], batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]

        Z1, A1, Z2, A2 = forward_propagation(X_batch)
        back_propagation(X_batch, y_batch, Z1, A1, A2)

    _, _, _, A2_train = forward_propagation(train_X)
    train_loss = -np.mean(np.sum(train_Y * np.log(A2_train + 1e-8), axis=1)) + (l2_lambda / (2 * train_X.shape[0])) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    train_accuracy = np.mean(np.argmax(A2_train, axis=1) == np.argmax(train_Y, axis=1))

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_accuracy * 100:.2f}%")

_, _, _, A2_test = forward_propagation(test_X)
test_accuracy = np.mean(np.argmax(A2_test, axis=1) == np.argmax(test_Y, axis=1))
print(f"Acc on testing set: {test_accuracy * 100:.2f}")
