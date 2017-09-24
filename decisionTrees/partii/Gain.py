##
# Gain Calculation
# for implementing decision trees
#
# @author antriksh
# Version 1: 09/16/2017

from Dataset import Dataset

import pandas as pd
import numpy as np

import random


class Gain():

    def __init__(self):
        pass

    def entropy(self, dataset):
        """
            Find the degree of randomness of the dataset
        """
        if dataset.isEmpty():
            return 0.0
        posCount, negCount, totalCount = dataset.getCount()

        # Taking care of divide-by-zero error
        if posCount == 0.0:
            probPos = 0.0
        else:
            probPos = posCount / totalCount

        if negCount == 0.0:
            probNeg = 0.0
        else:
            probNeg = negCount / totalCount

        # Taking care of divide-by-zero error
        if probPos == 0.0:
            posTerm = 0.0
        else:
            posTerm = (probPos * np.log2(probPos))

        if probNeg == 0.0:
            negTerm = 0.0
        else:
            negTerm = (probNeg * np.log2(probNeg))

        H = - (posTerm + negTerm)

        return float("{0:.8f}".format(H))

    def informationGain(self, dataset, attribute):
        """
            Find the information gain of an attribute over the dataset
        """

        HS = self.entropy(dataset)

        infoGain = HS

        for value in [0, 1]:
            data0 = Dataset(data=dataset.select(
                attribute, value).data, attribute=attribute)
            if not data0.isEmpty():
                pos, neg, total = data0.getAttributeCount(attribute)
                H = self.entropy(data0)
                term = (pos / total) * (H)
                infoGain -= term

        return infoGain

    def bestInfoGain(self, dataset):
        """
            Select and return an attribute with best information gain.
        """

        attributes = list(dataset.x.columns)

        maxGain = -9999
        maxGainAttr = None
        for attribute in attributes:
            gain = self.informationGain(dataset, attribute)
            if gain >= maxGain:
                maxGain = gain
                maxGainAttr = attribute

        return maxGainAttr

    def randomSelect(self, dataset):
        """
            Select and return a random attribute of the left out atttributes.
        """

        attributes = list(dataset.x.columns)

        return random.choice(attributes)
