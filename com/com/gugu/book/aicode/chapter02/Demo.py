from math import exp
from random import random


# 定义神经网络
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    hidden_layer2 = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_hidden)]
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(hidden_layer)
    network.append(hidden_layer2)
    network.append(output_layer)
    return network

# 每个神经元的网路输入
def net_input(weights, inputs):
    total_input = weights[-1]
    for i in range(len(weights) - 1):
        total_input += weights[i] * inputs[i]
    return total_input


# 激活函数
def activation(total_input):
    return 1.0 / (1.0 + exp(-total_input))


# 前向传播
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        outputs = []
        for neuron in layer:
            total_input = net_input(neuron['weights'], inputs)
            neuron['output'] = activation(total_input)
            outputs.append(neuron['output'])
        inputs = outputs
    return inputs


################################## 反向传播
# 损失函数，反向传播不需要损失函数，而是需要损失函数的求导函数，但是损失函数可以帮助我们的理解
def cost_function(expected, outputs):
    n = len(expected)
    total_error = 0.0
    for i in range(n):
        total_error += (expected[i] - outputs[i]) ** 2
    return total_error


def transfer_derivative(output):
    return output * (1.0 - output)


def backward_propagate(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()

        if i == len(network) - 1:
            for j in range(len(layer)):
                neuron = layer[j]
                error = -2 * (expected[j] - neuron['output'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])



def update_weight(network, row, learning_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= learning_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= learning_rate * neuron['delta']


def train_network(network, training_data, learning_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in training_data:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += cost_function(expected, outputs)
            backward_propagate(network, expected)
            update_weight(network, row, learning_rate)
            print('>epoch: %d,learning rate: %.3f, error: %.3f' % (epoch, learning_rate, sum_error))


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


if __name__ == '__main__':
    dataset = [[2.78, 4.55, 0],
               [3.39, 4.40, 0],
               [1.38, 1.85, 0],
               [3.06, 3.00, 0],
               [7.62, 2.75, 1],
               [5.33, 2.08, 1],
               [6.92, 1.77, 1]]
    test_data = [[1.46, 2.36, 0],
                 [8.67, -0.24, 1],
                 [7.67, 3.05, 1]]

    n_inputs = 2
    n_outputs = 2

    network = initialize_network(n_inputs, 2, n_outputs)
    train_network(network, training_data=dataset, learning_rate=0.5, n_epoch=2000, n_outputs=n_outputs)

    for row in test_data:
        result = predict(network, row)
        print('expected: %d, predicted: %d\n' % (row[-1], result))
