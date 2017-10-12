##
# Tree class
# for implementing decision trees
#
# @author antriksh
# Version 1: 09/16/2017

from Gain.Gain import entropy, informationGain
from Dataset import Dataset
from Dataset.Dataset import *

import pandas as pd
import numpy as np


class Tree():

    class Node():

        def __init__(self, name, attribute, parentAttribute, parentValue):
            self.name = name
            self.attribute = attribute
            self.parentAttribute = parentAttribute
            self.parentValue = parentValue
            self.dataset = Dataset(attribute=self.parentAttribute,
                                   value=self.parentValue)
            self.pointers = {0: None, 1: None}
            self.label = None
            self.entropy = entropy(self.dataset)

        def __str__(self):
            return self.name + " <+:" + str(self.countPosClass) + ", -:" +\
                str(self.countNegClass) + ", H:" + str(self.entropy) + ">"

        def assignNext(self, category, pointer):
            """
            Assign which node the category takes you to.

            @param category: one of the branches of the tree
            @param pointer: reference to the next node

            @returns True -- successful
            @throws AssertionError: category not found in dataset

            """
            assert category in self.dataset.getCategories(), "Category not found in dataset"

            self.pointers[category] = pointer
            return True

        def next(self, category):
            """
            Move to the next node if category found

            @param category: the division of the tree is into categories
            @preturns pointer to the next Node

            """
            assert category in self.dataset.getCategories(), "Category not found in dataset"
            return self.pointers[category]

        def setLabel(self, Class):
            self.pointers[0] = self.pointers[1] = None
            self.label = Class
            return

    def __init__(self, dataset):
        self.dataset = dataset
        self.root = None
        self.nodeCount = 0
        pass

    def addNode(attribute, parentAttribute, parentValue):
        if parentAttribute == None:
            newNode = self.root = self.Node(self.nodeCount, attribute)

        # elif self.root = parentAttribute:
        #     if(self.root.pointers[0] == None):
        #         newNode = self.Node(self.nodeCount, attribute,
        #                             parentAttribute, parentValue)
        #         self.root.assignNext(0, newNode)
        #     elif(self.root.pointers[1] == None):
        #         newNode = self.Node(self.nodeCount, attribute,
        #                             parentAttribute, parentValue)
        #         self.root.assignNext(1, newNode)

        self.nodeCount += 1

        return newNode

        # def entropy(self):
    def buildTree(self):
        data = self.dataset
        H = entropy(data)
        newNode =

        posCount, negCount, totalCount = data.getCount()

        if negCount == 0:

        parentAttribute = None
        parentValue = None
        newNode = None
        while(H != 0):
            for value in [0, 1]:
                igMax = informationGain(data)
                newNode = self.addNode(igMax, parent, value)
                parentAttribute = newNode.attribute

            data = newNode.dataset
            parent = newNode
            H = entropy(newNode.dataset)


if __name__ == '__main__':

    X2 = Tree.Node("D1", "X2", None, None)
    X1 = Tree.Node("D2", "X1", X2.attribute, 0)

    print X2.dataset.data
    print X1.dataset.data
