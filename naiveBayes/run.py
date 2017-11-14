from naiveBayes import NaiveBayes
from glob import glob

r = NaiveBayes('../data/nb/20news-bydate-train/')

r.train()

folders = ['talk.politics.mideast', 'sci.crypt', 'comp.os.ms-windows.misc',
           'comp.sys.ibm.pc.hardware', 'rec.sport.baseball']

for folder in folders:
    for file in glob('../data/nb/20news-bydate-test/' + folder + "/*"):
        print file
        r.testFile(file)
        break
    break
