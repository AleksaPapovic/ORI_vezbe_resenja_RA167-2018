from __future__ import print_function

from abc import abstractmethod
import math
import random
import copy

import pandas as pd

from tqdm import trange


from sklearn import preprocessing
from matplotlib import pyplot
from numpy.random import RandomState
from sklearn import preprocessing


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

class TanhNode(ComputationalNode):

    def init(self):
        self.x = 0.

    def forward(self, x):
        self.x = x
        return self._tanh(self.x)

    def backward(self, dz):
        return dz * (1-self._tanh(self.x)**2)

    def _tanh(self, x):
        try:
            return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        except:
            return math.inf

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
        elif activation == 'tanh':
            self.activation_node = TanhNode()
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
                for p, t in zip(pred, y):
                    total_loss += (t - p) ** 2.
                    grad += -(t - p)
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


def encodinghot(fajl, kolone):
    ulaz = fajl
    for kolona in kolone:
        ulaz  = pd.concat([ulaz, pd.get_dummies(ulaz[kolona], prefix=kolona, dummy_na=False)], axis=1).drop([kolona], axis=1)
    return  ulaz

def normalizovanje(dataframe):
    vrednosti = dataframe.values
    min_max_skaler = preprocessing.MinMaxScaler()
    vrednosti_skalirane = min_max_skaler.fit_transform(vrednosti)
    return pd.DataFrame(vrednosti_skalirane)

if __name__ == '__main__':

    # pod a)
    smokes = smoked = never_smoked = 0
    male = female = other = 0
    maried = not_married = 0

    private = selfemployed = govt = children = 0
    pyplot.style.use('ggplot')
    vrednosti = pd.read_csv('../data/dataset.csv').values
    for k in vrednosti:

        if k[11] == 1:
            if k[10] == 'smokes':
                smokes += 1
            elif k[10] == 'formerly smoked':
                smoked += 1
            elif k[10] == 'never smoked':
                never_smoked += 1
    for k in vrednosti:
        if k[11] == 1:
            if k[1] == 'Male':
                male += 1
            elif k[1] == 'Female':
                female += 1
            elif k[1] == 'Other':
                other += 1
    for k in vrednosti:
        if k[11] == 1:
            if k[5] == 'Yes':
                maried += 1
            elif k[5] == 'No':
                not_married += 1
    for k in vrednosti:
        if k[11] == 1:
            if k[6] == 'children':
                children += 1
            elif k[6] == 'Govt_job':
                govt += 1
            elif k[6] == 'Self-employed':
                selfemployed += 1
            elif k[6] == 'Private':
                private += 1

    types_smokers = ['Smoker', 'Non Smoker', 'Former Smoker']
    smokers_strokes = [smokes, never_smoked, smoked]

    types_smokers_pos = [i for i, _ in enumerate(types_smokers)]

    pyplot.bar(types_smokers_pos, smokers_strokes, color='green')
    pyplot.xlabel("Types of smokers")
    pyplot.ylabel("Strokes")
    pyplot.title("Strokes in correlation to smokers")

    pyplot.xticks(types_smokers_pos, types_smokers)
    pyplot.show()

    genders = ['Male', 'Female', 'Other']
    genders_strokes = [male, female, other]
    genders_pos = [i for i, _ in enumerate(genders)]
    pyplot.bar(genders_pos, genders_strokes, color='green')
    pyplot.xlabel("Genders")
    pyplot.ylabel("Strokes")
    pyplot.title("Strokes in correlation to genders")
    pyplot.xticks(genders_pos, genders)
    pyplot.show()

    marrieds = ['Married', 'Not married']
    marrieds_strokes = [maried, not_married]
    married_pos = [i for i, _ in enumerate(marrieds)]
    pyplot.bar( marrieds,  marrieds_strokes, color='green')
    pyplot.xlabel("Marrieds")
    pyplot.ylabel("Strokes")
    pyplot.title("Strokes in correlation to marrieds")
    pyplot.xticks( married_pos, marrieds)
    pyplot.show()

    posao = ['Private', 'Self-employed', 'Govt_job','chlidren']
    posao_strokes = [private, selfemployed, govt ,children]
    posao_pos = [i for i, _ in enumerate(posao)]
    pyplot.bar(posao_pos, posao_strokes, color='green')
    pyplot.xlabel("Works")
    pyplot.ylabel("Strokes")
    pyplot.title("Strokes in correlation to work")
    pyplot.xticks(posao_pos, posao)
    pyplot.show()


    #B deo
    nn = NeuralNetwork()

    col_list = ["gender", "age","hypertension","heart_disease","ever_married",
                "work_type","Residence_type","avg_glucose_level","bmi","smoking_status"]


    papo_ulaz  = pd.read_csv('../data/dataset.csv',usecols=col_list)
    papo_izlaz = pd.read_csv('../data/dataset.csv',usecols=["stroke"])


    papo_ulaz.fillna(0,inplace =True)
    papo_izlaz.fillna(0,inplace=True)

    hotovane = ["gender", "ever_married", "Residence_type","work_type","smoking_status"]

    papo_ulaz = encodinghot(papo_ulaz,hotovane)

    papo_ulaz = normalizovanje(papo_ulaz)

    normalizovan_ulaz =  papo_ulaz
    normalizovan_ulaz.to_csv(path_or_buf='../data/ulaz.csv', index=False)
    papo_izlaz.to_csv(path_or_buf='../data/izlaz.csv', index=False)
    print("duzina izlaza")
    print(len(papo_izlaz.columns))
    nn.add(NeuralLayer(len(papo_ulaz.columns), len(papo_ulaz.columns), 'sigmoid'))
    nn.add(NeuralLayer(len(papo_ulaz.columns), 10, 'tanh'))
    nn.add(NeuralLayer(10, 1, 'sigmoid'))


    #ravnomerna raspodela ulaz
    ulazi_1 = []
    ulazi_0 = []

    for row in papo_ulaz.values.tolist():
       if row[20] >= 0.5:
           ulazi_1.append(row)
       elif row[20] < 0.5:
           ulazi_0.append(row)
    df_ulaz1 = pd.DataFrame(ulazi_1)
    df_ulaz0 = pd.DataFrame(ulazi_0)

    df_ulaz1.to_csv(path_or_buf='../data/ulazijedinice.csv', index=False)
    df_ulaz0.to_csv(path_or_buf='../data/ulaznula.csv', index=False)

    ptrain_ulaz_1_7_1 =  df_ulaz1.sample(frac=0.7, random_state=RandomState())
    ptrain_ulaz_1_3_1 =  df_ulaz1.loc[~df_ulaz1.index.isin(ptrain_ulaz_1_7_1.index)]

    ptrain_ulaz_1_7_0 = df_ulaz0.sample(frac=0.7, random_state=RandomState())
    ptrain_ulaz_1_3_0 = df_ulaz0.loc[~df_ulaz0.index.isin(ptrain_ulaz_1_7_0.index)]

    ptrain_ulaz_frames = [ptrain_ulaz_1_7_1,ptrain_ulaz_1_7_0]
    ptest_ulaz_frames = [ptrain_ulaz_1_3_1,ptrain_ulaz_1_3_0]

    ptrain_ulaz = pd.concat(ptrain_ulaz_frames)
    ptest_ulaz = pd.concat(ptest_ulaz_frames)

    # racnomerna raspodela izlaz
    izlaz_1 = []
    izlaz_0 = []

    for row in papo_izlaz.values.tolist():
        if row[0] == 1:
            izlaz_1.append(row)
        elif row[0] == 0:
            izlaz_0.append(row)
    df_izlaz1 = pd.DataFrame(izlaz_1)
    df_izlaz0 = pd.DataFrame(izlaz_0)

    ptrain_izlaz_1_7_1 = df_izlaz1.sample(frac=0.7, random_state=RandomState())
    ptrain_izlaz_1_3_1 = df_izlaz1.loc[~df_izlaz1.index.isin(ptrain_izlaz_1_7_1.index)]

    ptrain_izlaz_1_7_0 = df_izlaz0.sample(frac=0.7, random_state=RandomState())
    ptrain_izlaz_1_3_0 = df_izlaz0.loc[~df_izlaz0.index.isin(ptrain_izlaz_1_7_0.index)]

    ptrain_izlaz_frames = [ptrain_izlaz_1_7_1, ptrain_izlaz_1_7_0]
    ptest_izlaz_frames = [ptrain_izlaz_1_3_1, ptrain_izlaz_1_3_0]

    ptrain_izlaz = pd.concat(ptrain_izlaz_frames)
    ptest_izlaz = pd.concat(ptest_izlaz_frames)

    ptrain_ulaz.to_csv(path_or_buf='../data/trainingulaz.csv', index=False)
    ptrain_izlaz.to_csv(path_or_buf='../data/trainingizlaz.csv', index=False)


    ptest_ulaz.to_csv(path_or_buf='../data/testulaz.csv', index=False)
    ptest_izlaz.to_csv(path_or_buf='../data/testizlaz.csv', index=False)
    # ptrain_ulaz  = papo_ulaz.sample(frac=0.7, random_state=RandomState())
    # ptrain_izlaz = papo_izlaz.sample(frac=0.7, random_state=RandomState())
    # ptest_ulaz   = papo_ulaz.loc[~papo_ulaz.index.isin(ptrain_ulaz.index)]
    # ptest_izlaz  = papo_izlaz.loc[~papo_izlaz.index.isin(ptrain_izlaz.index)]

    X= ptrain_ulaz.values.tolist()
    Y= ptrain_izlaz.values.tolist()
    if input("Treniraj mrezu?(y/n)") == 'y':
         # plotovanje funkcije greske
        history = nn.fit( X, Y, learning_rate=0.1, momentum=0.9, nb_epochs=100, shuffle=True, verbose=1)
        pyplot.plot(history)

        tp = tn = fp = fn = 0
        precision = []
        test_X = ptest_ulaz.values.tolist()
        test_Y = ptest_izlaz.values.tolist()
        #matrica konfuzije
        for i, j in zip(test_X, test_Y):
            print([j[0],nn.predict(i)[0]])
            if j[0] == 1:
                if nn.predict(i)[0] > 0.5:
                    tp += 1
                elif nn.predict(i)[0] < 0.5:
                    fn += 1
            elif j[0] == 0:
                if nn.predict(i)[0] < 0.5:
                    tn += 1
                elif nn.predict(i)[0] > 0.5:
                    fp += 1

            # if j[0] == 1 and 0.0 < j[0] - nn.predict(i)[0] < 0.5:
            #    fp += 1
            # elif j[0] == 0 and -0.5 < j[0] - nn.predict(i)[0] < 0.0:
            #     tn += 1
            # elif j[0] == 1 and 0.5 < j[0] - nn.predict(i)[0] < 1.0:
            #     tp += 1
            # elif j[0] == 0 and -1 < j[0] - nn.predict(i)[0] < -0.5:
            #     fn += 1
        print( tp)
        print(tn)
        print( fp)
        print( fn)
        try:
            accuracy = (tp + tn) / (tp + fp + fn + tn)
        except:
            accuracy = 0
        try:
            precision = tp / (tp + fp)
        except:
            precision = 0
        try:
            recall = tp / (tp + fn)
        except:
            recall = 0
        try:
            f1 = (precision * recall) / (precision + recall)
        except:
            f1 = 0

        print('Accuracy', round(accuracy * 100, 2), '%')
        print('Precision:', round(precision, 2))
        print('Recall:', round(recall, 2))
        print('F1 je', round(2*f1, 2))
        pyplot.show()