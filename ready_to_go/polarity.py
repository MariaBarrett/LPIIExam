from __future__ import division
import numpy as np
import pickle
import nltk
from collections import Counter
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfTransformer

"""
shifter expects an entire feature set. 
It contains 3 lists of words:
It runs through all reviews and all sentences.  
if intensifier is found, then the subsequent word gets MORE_ or LESS_ glued to it.
If a negation is found then the all the subsequent words in the sentence get NOT_ attached
The new dataset is returned.
"""
def shifter(dataset):
	not_list = ['n\'t', 'not']
	intensify_list = ['very', 'deeply', 'really']
	deintensify = ['barely', 'rather', 'hardly', 'rarely'] 
	dataset_c = np.copy(dataset)
	for review in dataset_c:
		for sentence in review:
			for i in xrange(len(sentence)):						
				if sentence[i] in not_list:
					sentence[i+1:] = ['NOT_'+sentence[k] for k in xrange(i+1, len(sentence)) ]
	return dataset_c

""" 
This function takes the test set, its labels, the positive and negative words as inputs.
It then checks for the appearance of positive and negative words, and classify the test set according to the words.
It returns the classification accuracy.
"""
def simpleclassification(x_test,y_test,pos_words,neg_words):
	labels = []
	for doc in x_test:
		pos,neg = 0,0,

		for sent in doc:
			for i in xrange(len(pos_words)):
				if pos_words[i] in sent:
					pos +=1
				if neg_words[i] in sent:
					neg +=1

		if pos > neg:
			labels.append(2)
		elif neg > pos:
			labels.append(0)
		else:
			labels.append(1)

	#measuring accuracy
	score = 0
	for i in xrange(len(labels)):
		if labels[i] == y_test[i]:
			score +=1
	fscore = score/len(y_test)

	return fscore



""" This function takes a single document as input and returns the document .
	The output depends on the selected mode:
		If mode == "nltk" : returns a dictionary of Boolean statements for each feature
		If mode == "sklearn" : returns a single list of binary values for each feature
"""
def polarityfeatures(document,word_features,mode="sklearn",calc="binary"):
	if mode == "nltk":
		document_words = set(document)
		features = {}
		for word in word_features:
			features['contains(%s)' % word] = (word in document_words)


	elif mode == "sklearn":
		features = []
		doc = np.array(document).flatten()
		features = np.zeros(len(word_features))
		
		if calc == "binary":
			for sent in doc:
				for i in xrange(len(word_features)):
					if word_features[i] in sent:
						if features[i] != 1:
							features[i] = 1


		if calc == "freq":
			features = np.zeros(len(word_features))
			for sent in doc:
				for i in xrange(len(word_features)):
					if word_features[i] in sent:
						features[i] += 1

	else: return "Unknown mode"

	return features

""" This function takes the labels and returns the prediction score of the most common label in the train set """
def baseline(y_train,y_test):
	score = 0
	base = Counter(y_train).most_common(1)[0][0]
	for elem in y_test:
		if elem == base:
			score += 1
	return score/len(y_test)


"""check(X_train,features) 
This function requires the transformed dataset as input.
	It checks whether the words used as features are even appearing in the train set. 
	It returns the unused features if there are any.
"""
def check(X_train,features):
	unused_features = []
	check = np.zeros(len(features))
	for doc in X_train:
		for i in range(len(doc)):
			if doc[i] == 1:
				check[i] += 1
	for i in xrange(len(check)):
		if check[i] == 0.:
			unused_features.append(features[i])

	if unused_features == []:
		return "No unused features"
	else:
		return unused_features
