from __future__ import division
import numpy as np
import pickle
import nltk

targets = pickle.load( open( "targets.p", "rb" ) )
dataset = pickle.load( open( "dataset.p", "rb" ) )


#Dataset structure: [[[sentence 1 in review1], [sentence 2 in review 1] ...], [[sentence 1 in review 2], [sentence 2 in review 2] ...] ...]
# Dataset is shuffled and balanced, 1000 each of pos, neu and neg

X_train = dataset[:2500]
y_train = targets[:2500]

X_test = dataset[2500:]
y_test = targets[2500:]


def ngram(data):
	bigrams = []
	for review in data:
		for sentence in review:
			print sentence
			bgs = nltk.bigrams(sentence)
			for b in bgs:
				bigrams.append(b)

	#compute frequency distribution for all the bigrams in the text
	fdist = nltk.FreqDist(bigrams)
	#for k,v in fdist.items():
	 #   print k,v

ngram(X_train)
