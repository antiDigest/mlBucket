class Directory():

	class Document():

	    class Word():

	        def __init__(self, name, count):
	            self.count = float(count)
	            self.name = name

	        def __str__(self):
	            return "(" + self.name + ", " + str(self.count) + ")"

	        def __repr__(self):
	            return self.name

	        def getCount(self):
	            return self.count

	    def __init__(self, FILE, wordList, outCome):
	        self.fileName = FILE
	        self.words = wordList
	        self.outClass = outCome
	        self.wordCount = sum([word.count for word in wordList])

	    def __repr__(self):
	        return self.outClass + "/" + self.fileName

	    def probability(self, word, vocab):
	        for w in self.words:
	            if w.name == word:
	                return (w.count + 1.) / (self.wordCount + float(len(vocab)))

	        return 1.0 / (self.wordCount + float(len(vocab)))

    def __init__(self, documents, className):
    	self.documents = documents
    	self.className = className

    	def __repr__(self):
    		return self.className

    	def probability(self, word, vocab):
    		([document.probability(word, vocab) for document in documents])

