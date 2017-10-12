import pandas as pd
import numpy as np

from pandas.api.types import is_numeric_dtype


class Dataset():
    """
        Preprocess and store the reference to the dataset
    """

    def __init__(self, FILE):
        if FILE is None:
            raise "Cannot give no data !"
            exit()

        self.data = pd.read_csv(FILE, header=None)

    def getXY(self, data):
        X = data[list(data.columns)[:-1]]
        y = data[list(data.columns)[-1]]

        return X, y

    def removeNull(self):
        """
            Drops the row if a NULL/Nan is present in the row
        """
        self.data.dropna(axis=0, how='any')

    def toNumeric(self):
        """
            # Parses through each attribute.
            # if the attribute is not numeric type then
            #     extracts unique elements from the attribute column and sorts them
            #     uses the indexes to replace each categorical value
        """
        for column in list(self.data.columns):
            if not is_numeric_dtype(self.data[column]):
                values = list(sorted(self.data[column].unique()))
                indices = [index for index, value in enumerate(values)]
                self.data[column] = self.data[column].replace(
                    to_replace=values, value=indices)

    def normalize(self):
        X, y = self.getXY(self.data)
        X = X - X.mean()
        self.data[list(self.data.columns)[:-1]] = X

    def getCategoricalY(self, y):
        categories = list(sorted(y.unique()))

        k = y.reset_index(drop=True)

        new_y = np.zeros((len(y), len(categories)))

        for index, instance in k.iteritems():
            new_y[index: index + 1, categories.index(instance)] = 1

        return new_y

    def save(self, location):
        self.data.to_csv(location, header=False, index=False)

    def trainTestSplit(self, percent):
        """
            Split train and test split by percentage
        """
        percent = float(percent)

        trainX, trainy = self.getXY(self.data.sample(
            frac=((100. - percent) / 100.), random_state=200))

        testX, testy = self.getXY(self.data.drop(trainX.index))

        return trainX.as_matrix(), testX.as_matrix(), \
            self.getCategoricalY(trainy), self.getCategoricalY(testy)
