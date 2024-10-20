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


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward(X, W, b):
    return softmax(np.dot(X, W) + b)


def compute_loss(Y_true, Y_pred):
    m = Y_true.shape[0]
    return -np.sum(Y_true * np.log(Y_pred + 1e-9)) / m


def backward(X, Y_true, Y_pred, W, b, learning_rate):
    m = X.shape[0]
    dz = Y_pred - Y_true
    dW = np.dot(X.T, dz) / m
    db = np.sum(dz, axis=0) / m

    W -= learning_rate * dW
    b -= learning_rate * db
    return W, b


def train_model(X, Y, W, b, epochs, batch_size, learning_rate):
    num_batches = X.shape[0] // batch_size
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X[start:end]
            Y_batch = Y[start:end]

            Y_pred = forward(X_batch, W, b)

            loss = compute_loss(Y_batch, Y_pred)
            epoch_loss += loss

            W, b = backward(X_batch, Y_batch, Y_pred, W, b, learning_rate)

        epoch_loss /= num_batches
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}')
    return W, b


def compute_accuracy(X, Y_true, W, b):
    Y_pred = forward(X, W, b)
    predictions = np.argmax(Y_pred, axis=1)
    labels = np.argmax(Y_true, axis=1)
    return np.mean(predictions == labels)


W = np.random.randn(784, 10) * 0.01
b = np.zeros(10)

W, b = train_model(train_X, train_Y, W, b, epochs=50, batch_size=100, learning_rate=0.1)

test_accuracy = compute_accuracy(test_X, test_Y, W, b)
print(f'Test Accuracy: {test_accuracy * 100}%')
