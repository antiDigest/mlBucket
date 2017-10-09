import random
import time
import numpy as np


class Network():

    """
        Stores a list of layers. Contains the Layer class to store layers.
        Initializes the size of the network using the hidden layers sizes and the input and output layers


    """

    class Layer():

        def __init__(self, name, layerType, size):
            self.name = name
            self.layerType = layerType
            self.nodes = []
            self.nodeCount = 0
            self.initNodes(size)

        def __str__(self):
            return str(self.nodes)

        def __repr__(self):
            return str(self.nodes)

        class Node():

            def __init__(self, name):
                self.name = name
                self.outgoing = []
                self.value = 0.
                self.error = 0.

            def __str__(self):
                return str(np.array(self.outgoing))

            def __repr__(self):
                return str(np.array(self.outgoing))

            def addToOutgoing(self, layer):
                for node in layer.nodes:
                    self.outgoing.append(Network.Layer.Edge(self, node))

            def forward(self):
                for edge in self.outgoing:
                    edge.forward()

            def backward(self, output):
                self.error = output * (1 - output) * \
                    sum([edge.backward() for edge in self.outgoing])

        class Edge():

            def __init__(self, de, a):
                """
                # stores coming from (de)
                # and going to (a) edges
                # and weight of the edge
                """
                self.de = de
                self.a = a
                self.weight = random.random()

            def __str__(self):
                return str(self.weight)

            def __repr__(self):
                return str(self.weight)

            def updateWeight(self, delta):
                self.weight = self.weight + delta

            def forward(self):
                self.a.to = self.de.value * self.weight

            def backward(self):
                return self.a.error * self.weight

        def addNode(self, name):
            self.nodes.append(self.Node(name))

        def initNodes(self, size):
            for i in xrange(0, size):
                self.addNode(i)

        def forward(self):
            for node in self.nodes:
                node.forward()

    def __init__(self, hidden, inNodes, output, alpha=0.01, eeta=0.01):
        assert type(hidden) == list, "Give a list of hidden layer sizes."
        self.alpha = alpha
        self.eeta = eeta

        self.layers = []
        self.initLayers(hidden, inNodes, output)
        self.initEdges()

    def __str__(self):
        for layer in self.layers:
            print layer
        return ""

    def addLayer(self, name, layerType, size):
        self.layers.append(self.Layer(name, layerType, size))

    def initLayers(self, hidden, inNodes, output):

        assert inNodes != 0, "No input nodes, no network formed."
        assert output != 0, "No output nodes, no network formed."

        self.addLayer(0, 'input', inNodes)

        for name, size in enumerate(hidden):
            self.addLayer(name, 'hidden', size)

        self.addLayer(name + 1, 'output', output)

    def initEdges(self):
        for name, layer in enumerate(self.layers[:-1]):
            for node in layer.nodes:
                node.addToOutgoing(self.layers[name + 1])

    def forward(self):
        for layer in self.layers:
            layer.forward()

    # def train(self, Dataset):
