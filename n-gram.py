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


def ngram(data, targets, ngr):
	neg = []
	neu = []
	pos = []

	for i in xrange(len(targets)):
		if targets[i] == 0:
			neg.append(dataset[i])
		if targets[i] == 1:
			neu.append(dataset[i])
		if targets[i] == 2:
			pos.append(dataset[i])

	bigrams_neg = getngram(neg, ngr)
	bigrams_neu = getngram(neu, ngr)
	bigrams_pos = getngram(pos, ngr)		

	all_bigrams = (set(bigrams_neg+bigrams_neu+bigrams_pos))

	return bigrams_neg, bigrams_neu, bigrams_pos, all_bigrams

def getngram(data, ngr):
	ngrams = []
	for review in data:
		for sentence in review:
			if 'bi' in ngr:
				bgs = nltk.bigrams(sentence)
				for b in bgs:
					ngrams.append(b)
			if 'tri' in ngr:
				trg = nltk.trigrams(sentence)
				for t in trg: 
					ngrams.append(t)
			else: 
				print "n-gram not specified properly. Your word should contain either 'bi' or 'tri'"
	return ngrams


neg_bigr, neu_bigr, pos_bigr, all_bigr = ngram(X_train, y_train, 'bigram')

