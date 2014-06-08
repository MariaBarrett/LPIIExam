from __future__ import division
import numpy as np
import pickle

targets = pickle.load( open( "targets.p", "rb" ) )
dataset = pickle.load( open( "dataset.p", "rb" ) )


#Dataset structure: [[[sentence 1 in review1], [sentence 2 in review 1] ...], [[sentence 1 in review 2], [sentence 2 in review 2] ...] ...]
# Dataset is shuffled and balanced, 1000 each of pos, neu and neg

X_train = dataset[:2500]
y_train = targets[:2500]

X_test = dataset[2500:]
y_test = targets[2500:]




""" List of words expressing polarity and subjectivity """

def posnegfeatures(document):
	word_features = ["good","wonderful", "perfect","fabulous","best", "marvelous",
					"love", "positive","absolutely", "amazing", "awesome",
					"beautiful", "empowered", "excellent", "exciting", "extraordinary",
					"fabulous","fantastic", "great", "happy", "incredible", "outstanding", 

					"bad", "miserable", "disgusting", "terrible", "wrong", "worst"
					"ridiculous", "dissatisfied", "stupid", "negative", "horrible",
					"confused", "dislike", "afraid", "boring", "sucks", "suck",
					"waste", "awful", "unwatchable", "emo", "but", "however", ]

	document_words = set(document)
	features = {}
	for word in word_features:
		features['contains(%s)' % word] = (word in document_words)
	return features