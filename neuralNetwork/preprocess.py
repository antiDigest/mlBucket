from Dataset import Dataset

import argparse


def preprocess(dataset, store, header):


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pre Processing Dataset.')
    parser.add_argument('dataset', action='store', required=True,
                        help='Supply a dataset to pre-process.')
    parser.add_argument('store', action='store', required=True,
                        help='Supply a location to store the processed dataset.')
    parser.add_argument('header', action='store_true', required=True, default=False,
                        help='Supply information of header in the file')
    args = parser.parse_args()

    if args.dataset is None:
        print "Please supply a data set to preprocess."
        print parser.print_help()
        exit()

    if args.store is None:
        print "No storage location given, replacing old data location."

    data = Dataset(FILE="https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
                   columns=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"])

    data.save("../data/car/car_unprocessed.csv")
    data.removeNull()
    data.toNumeric()
    data.normalize()
    data.save("../data/car/car.csv")

    data = Dataset(FILE="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                   columns=["sepal length", "sepal width", "petal length", "petal width", "class"])

    data.save("../data/iris/iris_unprocessed.csv")
    data.removeNull()
    data.toNumeric()
    data.normalize()
    data.save("../data/iris/iris.csv")

    data = Dataset(FILE="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                   columns=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                            "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
                            "native-country", "class"])

    data.save("../data/adult/adult_unprocessed.csv")
    data.removeNull()
    data.toNumeric()
    data.normalize()
    data.save("../data/adult/adult.csv")
