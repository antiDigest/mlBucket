##
# Gain Calculation
# for implementing decision trees
#
# @author antriksh
# Version 1: 09/16/2017

from Dataset import Dataset
from Dataset.Dataset import *

import pandas as pd
import numpy as np


class Gain():

	def __init__():
		pass

    def entropy(dataset):
		posCount, negCount, totalCount = dataset.getCount()

		probPos = posCount/totalCount
		probNeg = negCount/totalCount
		H = - (probPos * np.log2(1./probPos)) - (probNeg * np.log2(1./probNeg))

		return H

	def informationGain(dataset, attribute):

		HS = self.entropy(dataset)

		infoGain = HS

		for value in dataset.getCategories():
			data0 = Dataset(data=dataset.selectData(value))
			pos, neg, total = data0.getCount()
			infoGain -= pos/total * (self.entropy(data0))


		return infoGain




