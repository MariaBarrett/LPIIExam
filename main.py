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

SVC = SVC(kernel="rbf")
parameters = {'C': [ 8,10,12,14,16,]	, 'gamma':[0.001,0.01,0.1,1]}

GNB = GaussianNB()
KNN = KNeighborsClassifier(n_neighbors=3)
transformer = TfidfTransformer()


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
					 "positive", "amazing", "awesome","beautiful", "helpful",
					 "excellent","fantastic", "great", "accomodating",
					"cheap", "quality", "beautiful", "friendly", "comfortable",
					"friendly", "clean", "close", "big", "spacious", "nice",

					"bad", "disgusting", "terrible", "wrong", "worst", "terrible",
					"ridiculous", "hate", "noisy", "negative", "horrible",
					"annoying","small", "uncomfortable", "waiting", "wait",
					"dark", "difficult", "bad","expensive", "dirty",
					 "stained", "old", "used","smelly", "rude"]

pos_words = ["good","wonderful", "perfect","best", "love", "satisfied",
					 "positive", "amazing", "awesome","beautiful", "helpful",
					 "excellent","fantastic", "great", "accomodating",
					"cheap", "quality", "beautiful", "friendly", "comfortable",
					"friendly", "clean", "close", "big", "spacious", "nice"]

neg_words = ["bad", "disgusting", "terrible", "wrong", "worst", "terrible",
					"ridiculous", "hate", "noisy", "negative", "horrible",
					"annoying","small", "uncomfortable", "waiting", "wait",
					"dark", "difficult", "bad","expensive", "dirty",
					 "stained", "old", "used","smelly", "rude"]


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


print "\n","*"*50,"\n", " Method: Polarity and Subjectivity \n"


#Extracting the binary feature set
bpol_Xtrain = [polarityfeatures(d,word_features,mode="sklearn") for d in X_train]
bpol_Xtest = [polarityfeatures(d,word_features,mode="sklearn") for d in X_test]

#Extracting the tf-idf feature set
pol_Xtrain = [polarityfeatures(d,word_features,mode="sklearn",calc="freq") for d in X_train]
pol_Xtest = [polarityfeatures(d,word_features,mode="sklearn",calc="freq") for d in X_test]
tfidf_Xtrain = transformer.fit_transform(pol_Xtrain).toarray()
tfidf_Xtest = transformer.transform(pol_Xtest).toarray()


""" 
------------------------------------------------
Results !
"""

print "Unused features: \t",check(bpol_Xtrain,word_features)

print "Baseline result: \t",baseline(y_train,y_test)

print "Simple Classification: \t", simpleclassification(X_test,y_test,pos_words,neg_words)

SVCC = GridSearchCV(SVC, parameters)
SVCC.fit(bpol_Xtrain,y_train)
print "Radial SVM score: \t",SVCC.score(bpol_Xtest,y_test), SVCC.best_params_

GNB.fit(bpol_Xtrain,y_train)
print "Gaussian Naive Bayes score: \t", GNB.score(bpol_Xtest,y_test)

SVR = GridSearchCV(SVC, parameters)
SVR.fit(tfidf_Xtrain,y_train)
print "TFIDF Radial SVM score: \t",SVR.score(tfidf_Xtest,y_test), SVR.best_params_

GNB.fit(tfidf_Xtrain,y_train)
print "TFIDF Gaussian Naive Bayes score: \t", GNB.score(tfidf_Xtest,y_test)

print "*" * 50