from glob import glob
import numpy as np
from read import *


class NaiveBayes():

    def __init__(self, FOLDER):
        folders = []
        for folder in glob(FOLDER + "*"):
            folders.append(folder)

        self.basePath = "/".join(folder.split('/')[:-1])
        self.classes = ['talk.politics.mideast', 'sci.crypt', 'comp.os.ms-windows.misc',
                        'comp.sys.ibm.pc.hardware', 'rec.sport.baseball']
        self.allWords = []
        self.vocab = 0.
        self.docList = {}

        print self.classes

    def train(self):
        docList = {}
        allwords = []
        for folder in self.classes:
            print "Reading files in:", folder
            docList[folder], words = read(folder, self.basePath)
            allwords += words

        self.allWords = list(set(allwords))
        self.vocab = len(self.allWords)
        self.docList = docList

        print "Data Loaded."

    def test(self, files):
        assert type(files) == list, "Please enter a list of test data"
        for file in files:
            self.testFile(file)

    def testFile(self, file):
        words = readFile(file)
        print words
        probs = [(self.probability(words, outClass), outClass)
                 for outClass in self.classes]
        for prob in probs:
            print prob
        print np.argmax(probs, lambda x: x[1])

    def probability(self, words, outClass):
        return sum([doc.probability(words, self.allWords)
                    for doc in self.docList[outClass].keys()]) /
            float(len(self.docList[outClass].keys()))

        # def randomChoice(folders):
        #     return np.random.choice(folders, 5)
