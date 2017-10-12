from Network import Network
from Dataset import Dataset
import numpy as np
import argparse


def run(hidden, inNodes, output, dataset, iterations, percent):

    d = Dataset(dataset)

    # print d.trainTestSplit(percent)

    trainX, testX, trainy, testy = d.trainTestSplit(percent)

    # print trainy

    inNodes = trainX.shape[1]
    output = trainy.shape[1]

    network = Network(hidden, inNodes, output)

    network.train(trainX, trainy, iterations)

    print network

    predictionsTrain = network.predict(trainX)
    predictionsTest = network.predict(testX)

    print "Total Training Error:", network.getError(trainX, trainy)
    print "Total Testing Error:", network.getError(testX, testy)

    # trainError = network.checkError()
    # for index, tab in enumerate(predictionsTrain):
    #     trainAcc += int(np.argmax(tab) == np.argmax(testy[index]))

    # print ""


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pre Processing Dataset.')
    parser.add_argument('dataset', action='store',
                        help='Complete path of the post-processed input dataset.')
    parser.add_argument('percent', action='store', type=int,
                        help='Percentage of the dataset to be used for training')
    parser.add_argument('iterations', action='store', type=int,
                        help='Maximum number of iterations that your algorithm will run. This parameter is used so that your program terminates in a reasonable time.')
    parser.add_argument('hiddenLayers', type=int,
                        help='Number of hidden layers')
    parser.add_argument('hidden', type=str,
                        help='number of neurons in each hidden layer')
    args = parser.parse_args()

    if args.dataset is None:
        print "Please supply a data set to train on."
        print parser.print_help()
        exit()

    hidden = [int(val) for val in args.hidden.split(' ')]

    assert len(
        hidden) == args.hiddenLayers, "Command Line arguments need to match the number of hidden layers"
    run(hidden, 2, 1, args.dataset, args.iterations, args.percent)
