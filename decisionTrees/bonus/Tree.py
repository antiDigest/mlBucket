##
# Tree class
# for implementing decision trees
#
# @author antriksh
# Version 1: 09/16/2017

from Gain import Gain
from Dataset import Dataset
# from Dataset.Dataset import *

import pandas as pd
import numpy as np
from copy import deepcopy

import sys
sys.setrecursionlimit(100000)


class Tree():

    class Node():

        def __init__(self, name, attribute, parentNode, parentValue=None, dataset=None):
            self.name = name
            self.attribute = attribute
            self.parentNode = parentNode
            self.parentValue = parentValue
            if parentNode == None:
                self.depth = 0
            else:
                self.depth = parentNode.depth + 1
            if dataset != None:
                self.dataset = dataset
            elif attribute != None and parentValue != None:
                self.dataset = Dataset(attribute=self.attribute, parentNode=self.parentNode,
                                       value=self.parentValue)
            else:
                self.dataset = None
            self.pointers = {0: None, 1: None}
            self.label = None
            self.entropy = 0.0
            self.isLeaf = False

        def __str__(self):
            """
                print node
            """
            return str(self.attribute) + " <0:" + str(self.pointers[0]) + \
                ", 1:" + str(self.pointers[1]) + \
                ", Label:" + str(self.label) + ">"

        def assignNext(self, category, pointer):
            """
                Assign which node the category takes you to.
            """
            assert category in self.pointers.keys(), "Category not found in dataset"
            self.pointers[category] = pointer
            return True

        def next(self, category):
            """
                Move to the next node if category found

            """
            assert category in self.dataset.getCategories(), "Category not found in dataset"
            return self.pointers[category]

        def setLabel(self, Class):
            """
                set this node as the label node
            """
            self.pointers[0] = self.pointers[1] = None
            self.label = float(Class)
            self.isLeaf = True

        def assignDataset(self, attribute, parentNode, parentValue):
            """
                Assign a Dataset instance to the node, for ease of use further on
            """
            self.dataset = Dataset(attribute=self.attribute, parentNode=self.parentNode,
                                   value=self.parentValue)
            self.entropy = Gain().entropy(self.dataset)

    def __init__(self, dataset):
        self.dataset = dataset
        self.root = None
        self.nodeCount = 0
        self.leafCount = 0
        self.pruneFactor = 0.
        self.requiredPruning = 0.
        self.pruneCount = 0.
        self.nodes = []
        self.toBePruned = []
        posCount, negCount, totalCount = dataset.getCount()
        if (negCount < posCount):
            mostCommon = 0.
        else:
            mostCommon = 1.
        self.mostCommon = mostCommon

    def __str__(self):
        """
            print decision tree in the required format
        """

        root = self.root

        sys.stdout.write(str(root.attribute))

        if(root.pointers[0] != None):
            sys.stdout.write(" = 0 :")
            self.printTree(root.pointers[0], 1)

        print
        print str(root.attribute),
        if(root.pointers[1] != None):
            sys.stdout.write(" = 1 :")
            self.printTree(root.pointers[1], 1)

        return ""

    def setPruning(self, pruneFactor):
        """
            Set Pruning factor and number of max nodes to be pruned
        """
        self.pruneFactor = float(pruneFactor)
        self.requiredPruning = int(float(pruneFactor) *
                                   (self.nodeCount - self.leafCount))

    def newNode(self, attribute, parentNode, dataset=None, parentValue=None):
        """
            Creates a new node under parentNode
        """
        newNode = self.Node(self.nodeCount, attribute,
                            parentNode, parentValue, dataset)
        if parentNode == None:
            self.root = newNode

        self.nodeCount += 1
        self.nodes.append(newNode.name)

        return newNode

    def addLabelNode(self, outClass, parentNode, value):
        """
            Creates a new Label node under parentNode
        """
        node = self.newNode(parentNode.attribute, parentNode)
        self.nodes.remove(node.name)
        parentNode.assignNext(value, node)
        node.setLabel(outClass)
        self.leafCount += 1

        return node

    def buildTree(self, data=None, parentNode=None, value=None, choice="info_gain"):
        """
            # if pure data then make a label node out of it and return label node
            # A = best attribute selected for next node
            # for value in allowed values (0,1):
            #     if selected dataset for attribute is empty:
            #         make this labelNode with splitting outputs
            #     else:
            #         next of root is assigned recursively
        """
        if data is None:
            data = self.dataset

        posCount, negCount, totalCount = data.getCount()
        if posCount == totalCount:
            root = self.addLabelNode(1, parentNode, value)
            return root
        elif negCount == totalCount:
            root = self.addLabelNode(0, parentNode, value)
            return root

        if(choice == "info_gain"):
            A = Gain().bestInfoGain(data)
        elif choice == "random":
            A = Gain().randomSelect(data)

        root = self.newNode(A, parentNode, dataset=data)
        for value in [0, 1]:
            mostCommon = root.dataset.mostCommon()

            if root.dataset.selectAndRemove(A, value).isEmpty():
                self.addLabelNode(mostCommon, root, value)
                if value == 0:
                    self.addLabelNode(mostCommon ^ 1, root, 1)
                return root
            else:
                root.assignNext(value, self.buildTree(
                    data=data.selectAndRemove(A, value), parentNode=root, value=value, choice=choice))

        return root

    def testTree(self, data):
        """
            # Tests Tree:
            #     makes calls to Test Instance
        """
        attributes = list(data.x.columns)
        classified = []
        for index, instance in data.iterate():
            classified.append(int(self.testInstance(self.root, instance)))
        return classified

    def testInstance(self, root, example):
        """
            # Test Instance:
        """
        attr = root.attribute
        if root.isLeaf:
            return root.pointers[example[attr]].label
        return self.testInstance(root.pointers[example[attr]], example)

    def validateTree(self, data):
        """
            # Validates Tree:
            #     makes calls to Validate Instance
        """
        attributes = list(data.x.columns)
        total = float(data.size)
        classified = 0.
        for index, instance in data.iterate():
            classified += float(self.validateInstance(self.root, instance))

        accuracy = float("{0:.5f}".format(classified / total)) * 100.
        return accuracy

    def validateInstance(self, root, example):
        """
            Validates Instance
        """
        attr = root.attribute
        if root.isLeaf:
            prediction = root.label
            actual = example['Class']
            if prediction == actual:
                return 1
            else:
                return 0
        return self.validateInstance(root.pointers[example[attr]], example)

    def pruneTree(self, nodes, valSet, bestScore):

        newScore = 0.
        while (newScore < bestScore):
            self.toBePruned = selectNRandom(self.nodes, self.requiredPruning)
            self.pruneNodes(self.root)
            newScore = self.validateTree(valSet)
            if (newScore < bestScore):
                self.setBackPruned(self.root)

        return newScore

    def pruneNodes(self, node):
        """
            # Prune Decision Tree:
            # if node is to be pruned:
            #     make node's parent output the mostCommon label of the data
            # if not in to be pruned:
            #     search child at value 0 for to be pruned
            #     search child at value 1 for to be pruned
        """
        if not node.isLeaf:
            if node.name in self.toBePruned:
                mostCommon = node.dataset.mostCommon()
                node.label = mostCommon
                node.isLeaf = True
            else:
                newScore = self.pruneNodes(node.pointers[0])
                newScore = self.pruneNodes(node.pointers[1])

        return

    def setBackPruned(self, node):
        """
            # Prune Decision Tree:
            # if node is to be pruned:
            #     set node's parent back
            # if not in to be pruned:
            #     search child at value 0 for to be pruned
            #     search child at value 1 for to be pruned
        """
        if node is not None:
            if node.name in self.toBePruned:
                node.label = None
                node.isLeaf = False
            else:
                newScore = self.setBackPruned(node.pointers[0])
                newScore = self.setBackPruned(node.pointers[1])

        return

    def printTree(self, root, space):
        """
            Print Tree starting from the root in format
            Format example:
            # X3 = 0 :
            # | X2 = 0 :
            # | | X1 = 0 : 0
            # | | X1 = 1 : 1
            # | X2 = 1 : 1
            # X3 = 1 : 0
        """

        if root.isLeaf:
            sys.stdout.write(" " + str(int(root.label)))
            return

        print
        for i in range(space):
            sys.stdout.write("| ")

        print str(root.attribute),

        if(root.pointers[0] != None):
            sys.stdout.write(" = 0 :")
            self.printTree(root.pointers[0], space + 1)

        print
        for i in range(space):
            sys.stdout.write("| ")

        print str(root.attribute),
        if(root.pointers[1] != None):
            sys.stdout.write(" = 1 :")
            self.printTree(root.pointers[1], space + 1)

    def avgDepth(self, node, avgDepth):
        """
            Calculates and returns the sum of the depths of all the leaf nodes
        """
        if node.isLeaf:
            avgDepth += node.depth
        else:
            avgDepth = self.avgDepth(node.pointers[0], avgDepth)
            avgDepth = self.avgDepth(node.pointers[1], avgDepth)

        return avgDepth


def selectNRandom(nodes, N):
    """
        Selects N random nodes from a list of nodes and returns the list
    """
    import random
    random.shuffle(nodes)

    return nodes[:N]
