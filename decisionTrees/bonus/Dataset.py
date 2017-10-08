##
# Category
# for implementing decision trees
#
# @author antriksh
# Version 1: 09/16/2017

import pandas as pd
import numpy as np


class Dataset():
    """
        # A utility class for ease of access to dataset items.

        # All dataset items are loaded in pandas DataFrame and all operations in each of the function in the class
        # are performed using the pandas library
    """

    def __init__(self, FILE=None, attribute=None, parentNode=None, value=None, data=pd.DataFrame()):

        if data.empty and parentNode == None:
            try:
                assert FILE != None, "Cannot give nothing to make a dataset when data is empty and parentNode is None"
                self.file = FILE
                self.data = pd.read_csv(self.file, header=0)
            except AssertionError:
                self.data = pd.DataFrame()

        if not data.empty:
            self.data = data

        if parentNode != None:
            assert value != None, "Need a value with attribute to select data"
            assert attribute != None, "Need a attribute to select data"
            self.data = parentNode.dataset.selectData(value).data

        self.attribute = attribute
        if not self.data.empty:
            self.y = self.data[['Class']]
            self.x = self.data[self.data.columns[:-1]]
            self.size = len(self.data)

    def isEmpty(self):
        return self.data.empty

    def getCount(self):
        if 'Class' not in self.data.columns:
            print self.data
        posCount = float(len(self.data[self.data["Class"] == 1]))
        negCount = float(len(self.data[self.data["Class"] == 0]))
        totalCount = float(len(self.data))

        return posCount, negCount, totalCount

    def select(self, attribute, value):
        data = self.data[self.data[attribute] == value]
        return Dataset(data=data, attribute=attribute)

    def selectAndRemove(self, attribute, value):
        data = self.data[self.data[attribute] == value]
        del data[attribute]
        return Dataset(data=data, attribute=attribute)

    def selectData(self, value):
        if self.attribute == None:
            return Dataset(data=self.data, attribute=self.attribute)
        return Dataset(data=self.data[self.data[self.attribute] == value], attribute=self.attribute)

    def iterate(self):
        return self.data.iterrows()

    def mostCommon(self):
        pos, neg, total = self.getCount()

        return int(pos > neg)
