import math
import numpy as np
import pandas as pd
from matplotlib import pyplot


# np.random.seed(42)


def batch_generator(x, y, batch_size):
    assert len(x) == len(y)
    np.random.seed(42)
    X = np.array(x)
    y = np.array(y)
    perm = np.random.permutation(len(X))

    for batch_start in range(0, X.shape[0] // batch_size):
        yield X[perm][batch_start * batch_size: (batch_start + 1) * batch_size],\
              y[perm][batch_start * batch_size: (batch_start + 1) * batch_size]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_for_el(arr):
    return np.array([sigmoid(el) for el in arr])


def softmax(arr):
    arr = [arr[i]-max(arr[i]) for i in range(len(arr))]
    exps = np.exp(arr)
    sum_exps = np.expand_dims(np.sum(exps, axis=1), -1)
    return exps/(sum_exps + 1e-5)


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + math.exp(-x))

    def backward(self, arr):
        return np.array([self.__call__(el) for el in arr])


class BCELoss:
    def forward(self, predict, y):
        p = sigmoid_for_el(predict)
        return -1 * (y * np.log(p + 1e-6) + (1 - y) * np.log(1 - p + 1e-6))

    def backward(self, predict, y):
        p = sigmoid_for_el(predict)
        return np.expand_dims(p - y, -1)

class CELoss:
    def forward(self, x, y):
        proba = softmax(x)
        # print(y.shape, proba.shape)
        out = np.multiply(y, np.log(proba + 1e-6))
        return -1 * np.mean(out)

    def backward(self, x, y):
        proba = softmax(x)
        return (proba + 1e-6) - y


class Relu:
    def __call__(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return (x > 0).astype("float")


def create_activation(activation):
    if not activation:
        return None
    if activation.lower() == "relu":
        return Relu()

    elif activation.lower() == "sigmoid":
        return Sigmoid()

    elif activation.lower() == "softmax":
        return softmax

    else:
        print("Тут пока точ только relu and sigmoid работает")


class LinearLayer:
    def __init__(self, w, b, activation=None):
        self.w = w
        self.b = b
        self.grad_w = None
        self.grad_b = None
        self.activation = create_activation(activation)
        self.last_input = None
        self.last_out = None

    def calc(self, x):
        self.last_input = x
        # print(x.shape, self.w.shape)
        out = x @ self.w + self.b
        if self.activation:
            self.last_out = out
            out = self.activation(out)

        return out

    def backward(self, grad):
        if self.activation:
            grad = np.multiply(grad, self.activation.backward(self.last_out))

        self.grad_w = np.transpose(self.last_input) @ grad
        self.grad_b = np.sum(grad, axis=0)
        return grad @ np.transpose(self.w)

    def weights_update(self, alpha=1e-6):
        self.w = self.w.astype("float64")
        self.w -= alpha * self.grad_w
        self.b = self.b.astype("float")
        self.b -= alpha * self.grad_b


class Network:
    def __init__(self, x_input, layer_options, loss_function=None):
        """
        TODO: Сделать нормальную инициализацию параметров
        :param x_input:
        :param layers_count:
        :param loss_function:
        """
        self.loss_function = loss_function
        self.input_size = len(x_input[0])
        self.layer_options = layer_options
        self.x_input = x_input
        self.hidden_layer = []
        for option in layer_options:
            input = option["input_size"]
            output = option["output_size"]
            activation = option["activation"]
            weight, bias = self.initialize_parameters(input, output)
            self.hidden_layer.append(LinearLayer(weight, bias, activation))

    def initialize_parameters(self, input, output):
        w = np.random.normal(loc=0, scale=0.05, size=(input, output))
        b = np.random.normal(loc=0, scale=0.05, size=(1, output))
        return w, b

    def forward_pass(self, x):
        layer_output = x
        for layer in self.hidden_layer:
            layer_output = layer.calc(layer_output)
        return layer_output

    def backward_pass(self, p, y, loss_function):
        loss = loss_function()
        loss_value = loss.forward(p, y)
        grad = loss.backward(p, y)
        for layer in reversed(self.hidden_layer):
            grad = layer.backward(grad)
            layer.weights_update()

        return loss_value

# N = 200
# N2 = 25
#
# n_pos = int(N // 2 + np.random.randint(-N2, N2))
# n_neg = int(N // 2 + np.random.randint(-N2, N2))
#
# pos_x = 1
# pos_y = 1
#
# neg_x = -1
# neg_y = -1
#
# pos_pairs = np.array([np.array(
#     [pos_x + np.random.normal(scale=0.2), pos_y + np.random.normal(scale=0.2)])
#     for i in range(0, n_pos)])
#
# pos_answers = np.array([1] * n_pos)
#
# neg_pairs = np.array([np.array(
#     [neg_x + np.random.normal(scale=0.2), neg_y + np.random.normal(scale=0.2)])
#     for i in range(0, n_neg)])
# neg_answers = np.array([0] * n_neg)
#
# x = np.vstack([pos_pairs, neg_pairs])
# print(len(x))
# y = np.hstack([pos_answers, neg_answers])
# print(x)




#
# print("forward_pass =", sigmoid_for_el(network.forward_pass([[-1, -1], [1, 1]])))

train_data = pd.read_csv("mnist_train.csv")

train_cols = train_data.columns[1:]
label_col = train_data.columns[0]

labels = train_data[label_col].to_numpy()
train = train_data[train_cols].to_numpy()

ohe_labels = np.zeros((labels.shape[0], 10))
for idx, el in enumerate(labels):
    ohe_labels[idx, el] = 1

# print(len(train[0]))
layer_info = [{"input_size": 784, "output_size": 1500, "activation": "relu"},
              {"input_size": 1500, "output_size": 750, "activation": "relu"},
              {"input_size": 750, "output_size": 375, "activation": "relu"},
              {"input_size": 375, "output_size": 10, "activation": None}]

network = Network(train, layer_info)
loss_w = CELoss
test_test = None
print(train[0].shape)
for epoch in range(5):
    print(epoch)
    for n_iter, batch in enumerate(batch_generator(train, ohe_labels, 256)):
        x_batch, y_batch = batch
        proba = network.forward_pass(x_batch)
        loss_val = network.backward_pass(proba, y_batch, loss_function=loss_w)
        test_test = x_batch


a = softmax(network.forward_pass(test_test))
print(a)
for i in range(9):
    pyplot.subplot(360 + 1 + i)
    print(train[0])
    pyplot.imshow(test_test[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
    pyplot.show()



