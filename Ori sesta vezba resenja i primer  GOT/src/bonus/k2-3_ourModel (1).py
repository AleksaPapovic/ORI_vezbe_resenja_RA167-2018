from __future__ import print_function

from abc import abstractmethod
import math
import random
import copy
from keras import backend as K
import numpy as np
import pandas as pd
import pickle
import keras
import types
import tempfile

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import trange

import sklearn
from sklearn import preprocessing
from matplotlib import pyplot
# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


class ComputationalNode(object):

    @abstractmethod
    def forward(self, x):  # x is an array of scalars
        pass

    @abstractmethod
    def backward(self, dz):  # dz is a scalar
        pass


class MultiplyNode(ComputationalNode):

    def __init__(self):
        self.x = [0., 0.]  # x[0] is input, x[1] is weight

    def forward(self, x):
        self.x = x
        return self.x[0] * self.x[1]

    def backward(self, dz):
        return [dz * self.x[1], dz * self.x[0]]


class SumNode(ComputationalNode):

    def __init__(self):
        self.x = []  # x is in an array of inputs

    def forward(self, x):
        self.x = x
        return sum(self.x)

    def backward(self, dz):
        return [dz for xx in self.x]


class SigmoidNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x is an input

    def forward(self, x):
        self.x = x
        return self._sigmoid(self.x)

    def backward(self, dz):
        return dz * self._sigmoid(self.x) * (1. - self._sigmoid(self.x))

    def _sigmoid(self, x):
        try:
            temp = 1. / (1. + math.exp(-x))
        except OverflowError:
            temp = float('inf')
        return temp


class ReluNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x is an input

    def forward(self, x):
        self.x = x
        return self._relu(self.x)

    def backward(self, dz):
        return dz * (1. if self.x > 0. else 0.)

    def _relu(self, x):
        return max(0., x)


class NeuronNode(ComputationalNode):

    def __init__(self, n_inputs, activation):
        self.n_inputs = n_inputs
        self.multiply_nodes = []  # for inputs and weights
        self.sum_node = SumNode()  # for sum of inputs*weights

        for n in range(n_inputs):  # collect inputs and corresponding weights
            mn = MultiplyNode()
            mn.x = [1., random.gauss(0., 0.1)]  # init input weights
            self.multiply_nodes.append(mn)

        mn = MultiplyNode()  # init bias node
        mn.x = [1., random.gauss(0., 0.01)]  # init bias weight
        self.multiply_nodes.append(mn)

        if activation == 'sigmoid':
            self.activation_node = SigmoidNode()
        elif activation == 'relu':
            self.activation_node = ReluNode()
        else:
            raise RuntimeError('Unknown activation function "{0}".'.format(activation))

        self.previous_deltas = [0.] * (self.n_inputs + 1)
        self.gradients = []

    def forward(self, x):  # x is a vector of inputs
        x = copy.copy(x)
        x.append(1.)  # for bias

        for_sum = []
        for i, xx in enumerate(x):
            inp = [x[i], self.multiply_nodes[i].x[1]]
            for_sum.append(self.multiply_nodes[i].forward(inp))

        summed = self.sum_node.forward(for_sum)
        summed_act = self.activation_node.forward(summed)
        return summed_act

    def backward(self, dz):
        dw = []
        dx = []
        b = dz[0] if type(dz[0]) == float else sum(dz)

        b = self.activation_node.backward(b)
        b = self.sum_node.backward(b)
        for i, bb in enumerate(b):
            dw.append(self.multiply_nodes[i].backward(bb)[1])
            dx.append(self.multiply_nodes[i].backward(bb)[0])

        self.gradients = dw
        return dx

    def update_weights(self, learning_rate, momentum):
        for i, multiply_node in enumerate(self.multiply_nodes):
            mean_gradient = self.gradients[i]
            delta = learning_rate * mean_gradient + momentum * self.previous_deltas[i]
            self.previous_deltas[i] = delta
            self.multiply_nodes[i].x[1] -= delta

        self.gradients = []


class NeuralLayer(ComputationalNode):

    def __init__(self, n_inputs, n_neurons, activation):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = activation

        self.neurons = []
        # construct layer
        for _ in range(n_neurons):
            neuron = NeuronNode(n_inputs, activation)
            self.neurons.append(neuron)

    def forward(self, x):  # x is a vector of "n_inputs" elements
        layer_output = []
        for neuron in self.neurons:
            neuron_output = neuron.forward(x)
            layer_output.append(neuron_output)

        return layer_output

    def backward(self, dz):  # dz is a vector of "n_neurons" elements
        b = []
        for idx, neuron in enumerate(self.neurons):
            neuron_dz = [d[idx] for d in dz]
            neuron_dz = neuron.backward(neuron_dz)
            b.append(neuron_dz[:-1])

        return b  # b is a vector of "n_neurons" elements

    def update_weights(self, learning_rate, momentum):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate, momentum)


class NeuralNetwork(ComputationalNode):

    def __init__(self):
        # construct neural network
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):  # x is a vector which is an input for neural net
        prev_layer_output = None
        for idx, layer in enumerate(self.layers):
            if idx == 0:  # input layer
                prev_layer_output = layer.forward(x)
            else:
                prev_layer_output = layer.forward(prev_layer_output)

        return prev_layer_output  # actually an output from last layer

    def backward(self, dz):
        next_layer_dz = None
        for idx, layer in enumerate(self.layers[::-1]):
            if idx == 0:
                next_layer_dz = layer.backward(dz)
            else:
                next_layer_dz = layer.backward(next_layer_dz)

        return next_layer_dz

    def update_weights(self, learning_rate, momentum):
        for layer in self.layers:
            layer.update_weights(learning_rate, momentum)

    def fit(self, X, Y, learning_rate, momentum, nb_epochs, shuffle=True, verbose=0):
        assert len(X) == len(Y)

        hist = []
        for epoch in trange(nb_epochs):
            if shuffle:
                random.seed(epoch)
                random.shuffle(X)
                random.seed(epoch)
                random.shuffle(Y)

            total_loss = 0.0
            for x, y in zip(X, Y):
                # forward pass to compute output
                pred = self.forward(x)
                # compute loss
                grad = 0.0
                for o, t in zip(pred, y):
                    total_loss += (t - o) ** 2.
                    grad += -(t - o)
                # backward pass to compute gradients
                self.backward([[grad]])
                # update weights with computed gradients
                self.update_weights(learning_rate, momentum)

            hist.append(total_loss)
        if verbose == 1:
            print('Epoch {0}: loss {1}'.format(epoch + 1, total_loss))
        print('Loss: {0}'.format(total_loss))
        return hist

    def predict(self, x):
        return self.forward(x)

if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.add(NeuralLayer(359, 20, 'sigmoid'))
    nn.add(NeuralLayer(20, 3, 'sigmoid'))
    nn.add(NeuralLayer(3, 1, 'sigmoid'))

    # obucavajuci skup
    X = pd.read_csv(r'dataset.csv',
                    usecols=["male", "popularity", "book1", "book2", "book3", "book4", "book5", "isNoble",
                             "numDeadRelations", "house"])
    Y = pd.read_csv(r'dataset.csv', usecols=["isAlive"])

    X.house.fillna(value='unknown', inplace=True)
    # X['house'] = OneHotEncoder().fit_transform(X.house.values.reshape(-1, 1))

    #one hot encoding
    X = pd.concat([X, pd.get_dummies(X['house'], prefix='house', dummy_na=True)], axis=1).drop(['house'], axis=1)

    X.numDeadRelations = (X.numDeadRelations - X.numDeadRelations.min()) / (
                X.numDeadRelations.max() - X.numDeadRelations.min())
    X.popularity = (X.popularity - X.popularity.min()) / (X.popularity.max() - X.popularity.min())
    # X.male = (X.male - X.male.min()) / (X.male.max() - X.male.min())
    # X.isNoble = (X.isNoble - X.isNoble.min()) / (X.isNoble.max() - X.isNoble.min())
    X['isAlive'] = Y
    X.to_csv("tryagain3.csv", index=False)

    X = pd.read_csv(r'tryagain3.csv')
    Y = pd.read_csv(r'tryagain3.csv', usecols=["isAlive"])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train = X_train.values.tolist()
    X_test = X_test.values.tolist()
    y_train = y_train.values.tolist()
    y_test = y_test.values.tolist()

    # history = nn.fit(X_train, y_train, learning_rate=0.1, momentum=0.9, nb_epochs=3, shuffle=False, verbose=0)
    # with open('treci_3epohe.pkl', 'wb') as f:
    #     pickle.dump(nn, f)

    with open('treci.pkl', 'rb') as f:
        nn = pickle.load(f)

    acc = 0
    list = []
    for x, y in zip(X_test, y_test):
        print(nn.predict(x), y)
        if -0.5 < y[0] - nn.predict(x)[0] < 0.5:
            acc += 1
        list.append(y[0] - nn.predict(x)[0])
    accuracy = (acc / len(X_test)) * 100
    print('Accuracy: ' + str(round(accuracy, 2)) + '%')
    # y_pred = [nn.predict(x) for x in X_test]
    # acc = sklearn.metrics.r2_score(y_test, y_pred)
    # print('Accuracy: ' + str(acc))
    # # plotovanje funkcije greske
    pyplot.plot(list)
    pyplot.show()
