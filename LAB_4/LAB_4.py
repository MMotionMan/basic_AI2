import math
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(x):
    print("relu x =", x)
    return max(0, x)


def relu_derivative(x):
    print(x)
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j] < 0:
                x[i][j] *= 0.01
            else:
                x[i][j] = 1
    return x


class LinearLayer:
    def __init__(self, w, b, activation=None):
        self.w = w
        self.b = b
        self.grad_w = None
        self.grad_b = None
        self.activation = activation
        self.last_input = None
        self.last_out = None

    def calc(self, x):
        self.last_input = x
        out = x @ self.w + self.b
        if self.activation:
            self.last_out = out
            out = np.stack(np.vectorize(self.activation)(out))

        return out

    def backward(self, grad):
        print("last_out =", self.last_out)
        if self.activation:
            grad = np.multiply(grad, relu_derivative(self.last_out))
        print("w =", self.w)

        self.grad_w = np.transpose(self.last_input) @ grad
        print("grad_w =", self.grad_w)
        self.grad_b = np.sum(grad, axis=0)
        print("grad =", grad)
        return np.transpose(self.w) @ grad

    def weights_update(self, alpha=0.1):
        self.w -= alpha * self.grad_w


class Loss:
    def __init__(self, predict, y):
        self.p = predict
        self.y = y

    def forward(self):
        return [(self.y[i] * math.log(1-self.p[i])) - self.y[i] * math.log(self.p[i]) for i in range(len(self.p))]

    # def backward(self):
    #     t = [-1 * ((self.y[i] - self.y[i]**2 - self.p[i] - self.p[i]**2) / (self.y[i] * (1 - self.p[i]))) for i in range(len(self.p))]
    #     return t * sigmoid_derivative(t)


class Network:
    def __init__(self, x_input, layers_count, activation, layer_activation=None, loss_function=None):
        self.layers_count = layers_count
        self.loss_function = loss_function
        self.layer_activation = layer_activation
        self.input_size = len(x_input[0])
        self.layer_dims = [self.input_size, self.input_size*2, round(4/2), 1]
        self.weights, self.bias = self.initialize_parameters(self.layer_dims)
        self.x_input = x_input
        self.hidden_layer = np.array(
            [LinearLayer(self.weights[i], self.bias[i], self.layer_activation) for i in range(len(self.weights))]
        )
        self.output_activation = activation

    def initialize_parameters(self, layer_dims):
        length = len(layer_dims)
        w = np.array([np.ones((layer_dims[i-1], layer_dims[i]), dtype=int) for i in range(1, length)])
        print(w)
        b = np.array([np.zeros(layer_dims[i], dtype=int) for i in range(1, length)])
        return w, b

    def forward_pass(self):
        layer_output = self.x_input
        # print("hidde layer =", self.hidden_layer)
        for layer in self.hidden_layer:
            layer_output = layer.calc(layer_output)
            print(layer_output)
        # print("layer_output =", layer_output)
        return self.output_activation(layer_output.ravel())

    def backward_pass(self, p, y, loss_function):
        loss = loss_function(p, y)
        loss_value = loss.forward()
        print("loss =", loss_value)
        print("sigmoid = ", sigmoid_derivative(p))
        grad = loss_value * sigmoid_derivative(p)
        for layer in reversed(self.hidden_layer):
            grad = layer.backward(grad)
        print("weigths = ", self.weights)

        return loss_value


input_array = np.array([[2, -4], [-6, 8]])
real = [1, 0, 1]
network = Network(input_array, 1, activation=sigmoid, layer_activation=relu)

for i in range(10):
    a = network.forward_pass()
    loss_w = Loss
    back = network.backward_pass(a, real, loss_function=loss_w)
# print(network.backward_pass(a, real, loss))

# print(loss())


