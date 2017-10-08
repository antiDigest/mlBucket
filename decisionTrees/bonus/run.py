from Tree import Tree
from Dataset import Dataset
import argparse


def run(train, val, test, choice, prune):
    decisionTree = Tree(Dataset(FILE=train))
    decisionTree.buildTree(choice=choice)
    print decisionTree

    # randomTree = Tree(Dataset(FILE=train))
    # randomTree.buildTree(choice='random')
    # print randomTree

    trainSet = Dataset(FILE=train)
    valSet = Dataset(FILE=val)
    testSet = Dataset(FILE=test)

    trainingAcc = decisionTree.validateTree(trainSet)
    valAcc = decisionTree.validateTree(valSet)
    testAcc = decisionTree.validateTree(testSet)

    print
    print "*" * 50
    print "Pre-Pruned Accuracy"
    print "*" * 50
    print "Number of training instances:", trainSet.size
    print "Number of training attributes:", len(list(decisionTree.dataset.x.columns))
    print "Number of nodes in the tree:", int(decisionTree.nodeCount)
    print "Number of leaf nodes in the tree:", int(decisionTree.leafCount)
    print "Accuracy of the model on the training set:", str(trainingAcc) + "%"
    print
    print "Number of validation instances:", valSet.size
    print "Number of validation attributes:", len(list(decisionTree.dataset.x.columns))
    print "Accuracy of the model on the validation set:", str(valAcc) + "%"
    print
    print "Number of test instances:", testSet.size
    print "Number of test attributes:", len(list(decisionTree.dataset.x.columns))
    print "Accuracy of the model on the test set:", str(testAcc) + "%"
    print "*" * 50

    decisionTree.setPruning(prune)
    decisionTree.pruneTree(decisionTree.root, valSet, valAcc)

    trainingAcc = decisionTree.validateTree(trainSet)
    valAcc = decisionTree.validateTree(valSet)
    testAcc = decisionTree.validateTree(testSet)

    print
    print "*" * 50
    print "Post-Pruned Accuracy"
    print "*" * 50
    print "Number of training instances:", trainSet.size
    print "Number of training attributes:", len(list(decisionTree.dataset.x.columns))
    print "Number of nodes in the tree:", int(decisionTree.nodeCount)
    print "Number of leaf nodes in the tree:", int(decisionTree.leafCount)
    print "Accuracy of the model on the training set:", str(trainingAcc) + "%"
    print
    print "Number of validation instances:", valSet.size
    print "Number of validation attributes:", len(list(decisionTree.dataset.x.columns))
    print "Accuracy of the model on the validation set:", str(valAcc) + "%"
    print
    print "Number of test instances:", testSet.size
    print "Number of test attributes:", len(list(decisionTree.dataset.x.columns))
    print "Accuracy of the model on the test set:", str(testAcc) + "%"
    print "*" * 50


def runBonus(train, val, test):
    decisionTree = Tree(Dataset(FILE=train))
    decisionTree.buildTree(choice='info_gain')
    # print decisionTree

    randomTree = Tree(Dataset(FILE=train))
    randomTree.buildTree(choice='random')
    # print randomTree

    print "Tree constructed using ID3:: ", "Average Depth=",\
        decisionTree.avgDepth(decisionTree.root, 0) / decisionTree.leafCount,\
        "Number of nodes=", decisionTree.nodeCount

    print "Tree constructed using random attribute selection:: ", "Average Depth=",\
        randomTree.avgDepth(randomTree.root, 0) / randomTree.leafCount,\
        "Number of nodes=", randomTree.nodeCount

    trainSet = Dataset(FILE=train)
    valSet = Dataset(FILE=val)
    testSet = Dataset(FILE=test)
    print "\n\nRun#\t|\tAccuracy"
    for i in xrange(0, 5):
        trainingAcc = randomTree.validateTree(trainSet)
        valAcc = randomTree.validateTree(valSet)
        testAcc = randomTree.validateTree(testSet)
        print "+" * 70
        print "\t|\tAccuracy of the model on the training set: " + str(trainingAcc) + "%"
        print "  ", i + 1, "\t|\tAccuracy of the model on the validation set:", str(valAcc) + "%"
        print "\t|\tAccuracy of the model on the test set:", str(testAcc) + "%"
        print "+" * 70
        randomTree = Tree(Dataset(FILE=train))
        randomTree.buildTree(choice='random')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training Decision Trees.')
    parser.add_argument('--train', action='store', required=True,
                        help='Supply a training dataset.')
    parser.add_argument('--validation', action='store', required=True,
                        help='Supply a validation dataset.')
    parser.add_argument('--test', action='store', required=True,
                        help='Supply a test dataset.')
    # parser.add_argument('--prune', action='store', default=0.0,
    #                     help='If pruning, supply a prune factor. (0 by default)')
    # parser.add_argument('--choice', action='store', default="info_gain",
    #                     choices=['info_gain', 'random'],
    # help='For bonus assignment, "info_gain" and "random" implemented')
    args = parser.parse_args()

    if args.train == None or args.validation == None or args.test == None:
        print "Please supply all three data sets."
        print parser.print_help()
        exit()

    runBonus(args.train, args.validation, args.test)
