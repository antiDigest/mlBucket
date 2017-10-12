import random
import time
import numpy as np
import sys


class Network():

    """
        A single complete NEURAL NETWORK

        Runs Backpropagation algorithm
    """

    class Layer():
        """
            # A single layer of the neural network
            # Information Stored:
            #     name - Unique layer number (0-n)
            #     layerType - Type of layer (hidden/input/output)
            #     nodes - (list) all the nodes generated in the layer
            #
            # Initializations:
            #     initNodes(size) - creates new nodes for the layer,
            #                 size number of nodes are generated
        """

        class Node():
            """
                # Single node of a neural network
                # Stored attributes:
                #     name,
                #     nodeType - hidden layer node, input layer node or output layer node
                #     outgoing - (list) edges going from this node to any nodes in the next layer
                #     value - output from this node
                #     unsigval - non-sigmoid value output from this node
                # error - delta error calculated for this node during
                # back-propagation
            """

            def __init__(self, name, nodeType):
                self.name = name
                self.nodeType = nodeType
                self.outgoing = []
                self.value = 0.
                self.unsigval = 0.
                self.error = 0.

            def __str__(self):
                return str(np.array(self.outgoing))

            def __repr__(self):
                if self.nodeType == 'output':
                    return str(self.value)

                print str(np.array(self.outgoing))

                return ""

            """
                HELPER FUNCTIONS
            """

            def addToOutgoing(self, layer):
                """
                    # Adds outgoing edges from this node to every node of the next layer
                    # (skips the output layer)
                """
                if not layer.layerType == 'output':
                    for node in layer.nodes[1:]:
                        newEdge = Network.Layer.Edge(self, node)
                        self.outgoing.append(newEdge)
                else:
                    for node in layer.nodes:
                        newEdge = Network.Layer.Edge(self, node)
                        self.outgoing.append(newEdge)

            """
                USEFUL FUNCTIONS
            """

            def forward(self):
                """
                    # forward pass on this node
                    # calls the forward pass on all the edges outgoing from
                    # this node
                """
                for edge in self.outgoing:
                    edge.forward()

            def backward(self, example, target):
                """
                    # backward pass on this node
                    # calculates error
                    # if node is output node then:
                    #     error calculation uses the target value
                    # else:
                    #     for a node, the error is the sum of errors being back propagated
                    #         from all the outgoing edges from the node
                """
                if self.nodeType == 'output':
                    self.error = self.value * \
                        (1 - self.value) * (target - self.value)
                else:
                    self.error = self.value * (1 - self.value) * \
                        sum([edge.backward() for edge in self.outgoing])

            def updateWeights(self, eeta, momentum):
                """
                    # Updating the weights
                    # of all the outgoing edges from this node
                """
                for edge in self.outgoing:
                    edge.updateWeight(eeta, momentum)

        class Edge():

            def __init__(self, fromNode, toNode):
                """
                    # Single edge in the complete neural network
                    # Information Stored:
                    #     fromNode - the start / coming from node
                    #     toNode - the end / going to node
                    #     weight - weight of the edge
                """
                self.fromNode = fromNode
                self.toNode = toNode
                n = 1
                self.weight = np.random.randn(n) * np.sqrt(2.0 / n)
                self.prevDelta = 0.

            def __str__(self):
                return str(self.fromNode.name) + " -- " + str(self.toNode.name) + " :" + str(self.weight)

            def __repr__(self):
                return str(self.weight)

            """
                USEFUL FUNCTIONS
            """

            def updateWeight(self, eeta, momentum):
                """
                    # Calculates the value of delta
                    # Updates edge weight
                """
                delta = self.toNode.error * self.fromNode.value * eeta
                self.weight = self.weight + delta

            def forward(self):
                """
                    # Forward pass on a single edge
                    # takes the incoming value and multiplies it by weight,
                    # sets the input of the outgoing node to the sigmoid of the
                    # value
                """
                self.toNode.unsigval += self.fromNode.value * self.weight
                self.toNode.value = sigmoid(self.toNode.unsigval)

            def backward(self):
                """
                    # Backward pass on a single edge
                    # Takes the error from the outgoing node
                    # multiplies it by weight
                    # essentially this need to be summed up with all the other error values
                    # of all the other outgoing nodes from a single node
                """
                return self.toNode.error * self.weight

        def __init__(self, name, layerType, size):
            self.name = name
            self.layerType = layerType
            self.nodes = []
            self.initNodes(size)

        def __str__(self):
            for node in self.nodes:
                print "Neuron", node.name, "weights:", node
            return ""

        def __repr__(self):
            if self.layerType == 'output':
                return str(self.output)

            for node in self.nodes:
                print "Neuron", node.name, "weights:", node
            return ""

        """
            HELPER FUNCTIONS
        """

        def addNode(self, name):
            self.nodes.append(self.Node(name, self.layerType))

        def initNodes(self, size):
            """
                # Generates size number of nodes in the layer and
                # adds all the nodes to the list of nodes at the layer
            """
            if self.layerType == 'output':
                for i in xrange(1, size + 1):
                    self.addNode(i)
            else:
                for i in xrange(0, size + 1):
                    self.addNode(i)

        """
            USEFUL FUNCTIONS
        """

        def forward(self, example):
            """
                FORWARD PASS
                # for each node in the network, do:
                #     run forward pass on the layer
            """

            if self.layerType == 'input':
                assert len(
                    self.nodes) - 1 == example.shape[0], "Input size needs to match input layer size"
                for node, value in zip(self.nodes[1:], example):
                    node.value = value
                    node.forward()
                return

            if not self.layerType == 'output':
                for node in self.nodes:
                    # print node.value, node.name, node.nodeType
                    node.forward()

        def backward(self, example, target):
            """
                BACKWARD PASS
                # for each node in the network, do:
                #     run backward pass on the layer
            """
            for node, t in zip(self.nodes, target):
                node.backward(example, t)

        def updateWeights(self, eeta, momentum):
            for node in self.nodes:
                node.updateWeights(eeta, momentum)

    def __init__(self, hidden, inNodes, output, alpha=0.01, eeta=1, momentum=0.5):
        assert type(hidden) == list, "Give a list of hidden layer sizes."
        self.alpha = alpha
        self.eeta = eeta
        self.momentum = momentum

        self.inNodes = inNodes
        self.outNodes = output
        self.error = 0.

        self.layers = []
        self.initLayers(hidden, inNodes, output)
        self.initEdges()

    def __str__(self):
        for layer in self.layers[:-1]:
            print "Layer", layer.name, "(" + layer.layerType + ")"
            print layer
        return ""

    """
        HELPER FUNCTIONS
    """

    def addLayer(self, name, layerType, size):
        self.layers.append(self.Layer(name, layerType, size))

    def initLayers(self, hidden, inNodes, output):

        assert inNodes != 0, "No input nodes, no network formed."
        assert output != 0, "No output nodes, no network formed."

        self.addLayer(0, 'input', inNodes)
        self.layers[0].nodes[0].value = 1.

        for name, size in enumerate(hidden):
            self.addLayer(name + 1, 'hidden', size)
            self.layers[name + 1].nodes[0].value = 1.

        self.addLayer(name + 2, 'output', output)

    def initEdges(self):
        for name, layer in enumerate(self.layers[:-1]):
            for node in layer.nodes:
                node.addToOutgoing(self.layers[name + 1])

    def clean(self, layerName):
        for node in self.layers[layerName].nodes:
            node.unsigval = 0.

    """
        USEFUL FUNCTIONS
    """

    def forward(self, example):
        """
            FORWARD PASS
            # for each layer in the network, do:
            #     run forward pass on the layer
        """
        for layer in self.layers[:-1]:
            self.clean(layer.name + 1)
            layer.forward(example)

        x = np.array([node.value for node in self.layers[-1].nodes])
        x = np.transpose(x / x.sum())

        return x

    def backward(self, example, target):
        """
            BACKWARD PASS
            # for each layer in the network, do:
            #     run backward pass on the layer
        """
        for layer in self.layers[1:][::-1]:
            layer.backward(example, target)

    def updateWeights(self, eeta, momentum):
        """
            Update Weights using eeta (n)
        """
        for layer in self.layers[:-1]:
            layer.updateWeights(eeta, momentum)

    def getError(self, examples, targets):
        """
            Calculates the sum of the error for the final layer
        """
        error = 0.
        for example, target in zip(examples, targets):
            self.forward(example)
            for node, t in zip(self.layers[-1].nodes, target):
                error += np.square(t - node.value) / 2.

        return np.abs(error)

    def checkError(self, target):
        """
            Calculates the sum of the error for the final layer
        """
        error = 0.
        for node, t in zip(self.layers[-1].nodes, target):
            error += np.square(t - node.value) / 2.

        return np.abs(error)

    def train(self, examples, targets, iterations=1000):
        """
            BACKPROPAGATION ALGORITHM
            # for each example, do:
            #     run the forward pass
            #     run the backward pass over the output layer
            #     run the backward pass over all the hidden layers
            #     run the weight updation procedure
        """
        assert targets.shape[1] == self.outNodes, "Output size not matching"
        check = iterations / 50.
        sys.stdout.write(str(iterations) + " Iterations : [")
        for iteration in xrange(1, iterations + 1):
            for example, target in zip(examples, targets):
                self.forward(example)
                self.backward(example, target)
                self.updateWeights(self.eeta, self.momentum)

            if np.abs(self.checkError(target)) < 0.0000000001:
                print "]"
                print "Error is less than 1e-10, breaking."
                break

            if iteration % check == 0:
                sys.stdout.write("#")
                sys.stdout.flush()
        print "]"
        print

    def predict(self, examples):
        """
            PREDICTION
            # for each example in the test set, do:
            #     run forward pass over the complete network
        """

        predictions = np.zeros(shape=(examples.shape[0], self.outNodes))
        for index, example in enumerate(examples):
            predictions[index:index + 1] = np.array(self.forward(example))

        return predictions


def sigmoid(z):
    """
        ACTIVATION FUNCTION: Sigmoid
    """
    return 1. / (1. + np.exp(-z))


def threshold(z):
    """
        STEP FUNCTION
    """
    return int((z - 0.5) > 0)
