stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
              "on", "once", "only", "or", "other", "ought", "our", "ours	ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

from nltk import word_tokenize
from document import Document
from glob import glob


def readFiles(Files):
    assert type(Files) == list, "Enter a list of files"
    for file in Files:
        readFile(file)


def readFile(file):
    file = open(file).read()
    tokens = word_tokenize(file.decode('utf-8', errors="ignore"))
    from collections import Counter
    tokens_dict = dict(Counter(tokens))
    wordlist = []
    for token in tokens:
        if token not in stop_words:
            word = Document.Word(token, tokens_dict[token])
            wordlist.append(word)

    return wordlist


def read(folder, basePath):

    if type(folder) == list:
        readFiles(folder)

    docLists = {}
    allwords = []
    for file in glob(basePath + "/" + folder + "/*"):
        fileName = file.split('/')[-1]
        wordList = readFile(file)
        allwords += list(set([word.name for word in wordList]))
        docLists[fileName] = Document(fileName, wordList, folder)

    return docLists, list(set(allwords))
