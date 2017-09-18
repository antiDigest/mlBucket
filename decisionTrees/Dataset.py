##
# Category
# for implementing decision trees
#
# @author antriksh
# Version 1: 09/16/2017

import pandas as pd
import numpy as np


class Dataset():

    def __init__(self, attribute=None, value=None, data=None, FILE='../data/data.csv'):
        if data == None:
            self.data = pd.read_csv(FILE, header=0)
        else:
            self.data = data
        self.attribute = attribute

        if attribute != None:
            assert value != None, "Need a value with attribute to select data"
            self.data = self.data[self.data[attribute] == value]

        self.y = self.data[['Class']]
        self.x = self.data[self.data.columns[:-1]]

    def getCategories(self):
        if self.attribute == None:
            return self.y.unique()
        return self.x[self.attribute].unique()

    def getColumnValues(self):
        return self.data[self.attribute].values

    def getClassValues(self):
        return self.y.values

    def getCount(self):
        posCount = float(len(self.data[self.data["Class"] == 1]))
        negCount = float(len(self.data[self.data["Class"] == 0]))
        totalCount = float(len(self.data))

        return posCount, negCount, totalCount

    def getAttributeCount(self, attribute, value, classValue):
        return float(len(self.data[(self.data[attribute] == value) and
                                   (self.data["Class"] == classValue)]))

    def selectData(self, value):
        return self.data[self.data[self.attribute] == value]

if __name__ == '__main__':
    dataset = Dataset('../data/data.csv')
    print dataset.x
    print dataset.y.get('Class') == 0
