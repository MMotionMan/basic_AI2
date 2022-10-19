import math
import random

import numpy as np

# np.random.seed(42)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype("float")


class LinearLayer:
    def __init__(self, w, b, activation=None):
        self.w = w
        self.b = b
        self.grad_w = None
        self.grad_b = None
        self.activation = activation
        # print("activation =", self.activation)
        self.last_input = None
        self.last_out = None

    def calc(self, x):
        self.last_input = x
        out = x @ self.w + self.b
        if self.activation:
            self.last_out = out
            out = self.activation(out)

        return out

    def backward(self, grad):
        # print("grad in output backward =", grad)
        # print("last_out =", self.last_out)
        # print("relu_derivative last_out =", relu_derivative(self.last_out))
        if self.activation:
            grad = np.multiply(grad, relu_derivative(self.last_out))
        # print("grad under multiply", grad)

        self.grad_w = np.transpose(self.last_input) @ grad
        print("grad_w =", self.grad_w)
        self.grad_b = np.sum(grad, axis=0)
        # print("grad =", grad, "transpose_w =", np.transpose(self.w))
        return grad @ np.transpose(self.w)

    def weights_update(self, alpha=1e-4):
        self.w = self.w.astype("float64")
        self.w += alpha * self.grad_w
        self.b = self.b.astype("float")
        self.b += alpha * self.grad_b


class Loss:
    def __init__(self, predict, y):
        self.p = predict
        self.y = y

    def forward(self):
        # // for each el in predict do sigmoid
        # print("p =", self.p, "y =", self.y)
        return -1 * (self.y * np.log(self.p) + (1 - self.y) * np.log(1 - self.p))
        # reg = np.sum(self.p*np.log(self.p))
        # l -= reg
        # print(l)
        # return l

    def backward(self):
        return np.expand_dims(self.p - self.y, -1)


class Network:
    def __init__(self, x_input, layers_count, activation, layer_activation=None, loss_function=None):
        self.layers_count = layers_count
        self.loss_function = loss_function
        self.layer_activation = layer_activation
        self.input_size = len(x_input[0])
        self.act = [relu, relu, sigmoid]
        self.layer_dims = [self.input_size, self.input_size*2, self.input_size, 1]
        self.weights, self.bias = self.initialize_parameters(self.layer_dims)
        self.x_input = x_input
        self.hidden_layer = np.array(
            [LinearLayer(self.weights[i], self.bias[i], self.act[i]) for i in range(len(self.weights))]
        )
        # print("len hidden_layer =", len(self.hidden_layer))
        self.output_activation = activation

    def initialize_parameters(self, layer_dims):
        length = len(layer_dims)
        w = np.array([np.random.normal(loc=0.1, scale=0.1, size=(layer_dims[i-1], layer_dims[i])) for i in range(1, length)])
        # print("w =", w)
        b = np.array([np.random.normal(loc=0, scale=0.05, size=(layer_dims[i])) for i in range(1, length)])
        # print("b =", b)
        return w, b

    def forward_pass(self, x):
        layer_output = x
        # print("hidde layer =", self.hidden_layer)
        for layer in self.hidden_layer:
            layer_output = layer.calc(layer_output)
            # print("activation =", layer.activation)
            # print("layer_output =", layer_output)
        # print("layer_output =", layer_output)
        return layer_output.ravel()

    def backward_pass(self, p, y, loss_function):
        loss = loss_function(p, y)
        loss_value = loss.forward()
        # print("loss =", loss_value)
        # print("sigmoid = ", sigmoid_derivative(p))
        grad = loss.backward()
        # print("grad =", grad, p, y)
        # print("grad in back_nn =", grad)
        for layer in reversed(self.hidden_layer):
            grad = layer.backward(grad)
            layer.weights_update()

        # print("loss_value =", loss_value)
        return loss_value


input_array = np.array([[10, 50],
                        [20, 30],
                        [25, 30],
                        [20, 60],
                        [15, 70],
                        [40, 40],
                        [30, 45]])
real = np.array([1, 0, 0, 1, 1, 0, 0])
network = Network(input_array, 1, activation=sigmoid, layer_activation=relu)

for i in range(20):
    a = network.forward_pass(input_array)
    # print("a =", a)
    loss_w = Loss
    back = network.backward_pass(a, real, loss_function=loss_w)
    print("loss =", back)

print("forward_pass =", network.forward_pass([[20, 30]]))



