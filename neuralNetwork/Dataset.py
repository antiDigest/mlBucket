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

        columns = ["class"]

        self.data = pd.read_csv(FILE, header=None)
        self.X = self.data[list(self.data.columns)[:-1]]
        self.y = self.data[list(self.data.columns)[-1]]

    def removeNull(self):
        self.data.dropna(axis=0, how='any')

    def toNumeric(self):
        for column in list(self.data.columns):
            if not is_numeric_dtype(self.data[column]):
                values = self.data[column].unique()
                indices = [index for index, value in enumerate(values)]
                self.data[column] = self.data[column].replace(
                    to_replace=values, value=indices)

        self.X = self.data[list(self.data.columns)[:-1]]

    def normalize(self):
        self.X = self.X - self.X.mean()
        self.data.loc[:, :-1] = self.X

    def save(self, location):
        self.data.to_csv(location, header=True, index=False)
