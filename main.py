from __future__ import division
import numpy as np
import pickle
import nltk
from collections import Counter
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

SVC = SVC(kernel="linear")
GNB = GaussianNB()
KNN = KNeighborsClassifier(n_neighbors=3)


targets = pickle.load( open( "targets.p", "rb" ) )
dataset = pickle.load( open( "dataset.p", "rb" ) )


#Dataset structure: [[[sentence 1 in review1], [sentence 2 in review 1] ...], [[sentence 1 in review 2], [sentence 2 in review 2] ...] ...]
# Dataset is shuffled and balanced, 1000 each of pos, neu and neg

X_train = dataset[:2500]
y_train = targets[:2500]

X_test = dataset[2500:]
y_test = targets[2500:]



""" List of words expressing polarity and subjectivity """

word_features = ["good","wonderful", "perfect","best", "love", "satisfied",
					 "positive", "amazing", "awesome","beautiful",
					 "excellent","fantastic", "great", "happy",
					"cheap", "quality", "beautiful", "friendly", "comfortable",
					"friendly", "clean", "close", "delicious", "big", "spacious", 

					"bad", "disgusting", "terrible", "wrong", "worst",
					"ridiculous", "unsatisfied", "stupid", "negative", "horrible",
					"dislike", "sucks", "suck", "small", "uncomfortable", "waiting", "wait",
					"dark", "long", "difficult", "bad","expensive", "dirty"]



""" This function takes a single document as input and returns the document .
	The output depends on the selected mode:
		If mode == "nltk" : returns a dictionary of Boolean statements for each feature
		If mode == "sklearn" : returns a single list of binary values for each feature
"""
def binary_polarityfeatures(document,word_features,mode="nltk"):
	if mode == "nltk":
		document_words = set(document)
		features = {}
		for word in word_features:
			features['contains(%s)' % word] = (word in document_words)


	elif mode == "sklearn":
		features = []

		doc = np.array(document).flatten()
		for word in word_features:
				if word in doc:
					features.append(1)
				else: features.append(0)


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
		print "No unused features"
	else:
		return unused_features



bpol_Xtrain = [binary_polarityfeatures(d,word_features,mode="sklearn") for d in X_train]
bpol_Xtest = [binary_polarityfeatures(d,word_features,mode="sklearn") for d in X_test]

print "Unused features: \t",check(bpol_Xtrain,word_features)

print "Baseline result: \t",baseline(y_train,y_test)

SVC.fit(bpol_Xtrain,y_train)
print "Linear SVM score: \t",SVC.score(bpol_Xtest,y_test)

GNB.fit(bpol_Xtrain,y_train)
print "Gaussian Naive Bayes score: \t", GNB.score(bpol_Xtest,y_test)

KNN.fit(bpol_Xtrain,y_train)
print "K-NearestNeighbor score: \t", KNN.score(bpol_Xtest,y_test)