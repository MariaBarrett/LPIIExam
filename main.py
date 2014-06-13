from __future__ import division
import numpy as np
import pickle
import nltk
import polarity
from collections import Counter
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfTransformer
import bow_unigram_Keith as keithbow
from n-gram import *

SVC = SVC(kernel="rbf")
parameters = {'C': [ 8,10,12,14,16,]	, 'gamma':[0.001,0.01,0.1,1]}

GNB = GaussianNB()
KNN = KNeighborsClassifier(n_neighbors=3)
transformer = TfidfTransformer()


targets = pickle.load( open( "targets_easy.p", "rb" ) )
dataset = pickle.load( open( "dataset_easy.p", "rb" ) )

#Dataset structure: [[[sentence 1 in review1], [sentence 2 in review 1] ...], [[sentence 1 in review 2], [sentence 2 in review 2] ...] ...]
# Dataset is shuffled and balanced, 1000 each of pos, neu and neg

X_train = dataset[:2500]
y_train = targets[:2500]

X_test = dataset[2500:]
y_test = targets[2500:]


""" --------------------------- Polarity and Subjectivity -------------------------- """

""" List of words expressing polarity and subjectivity """

word_features = ["good","wonderful", "perfect","best", "love", "satisfied",
					 "positive", "amazing", "awesome","beautiful", "helpful",
					 "excellent","fantastic", "great", "accomodating",
					"cheap", "quality", "beautiful", "friendly", "comfortable",
					"friendly", "clean", "close", "big", "spacious", "nice",

					"bad", "disgusting", "terrible", "wrong", "worst", "awful",
					"ridiculous", "hate", "noisy", "negative", "horrible",
					"annoying","small", "uncomfortable", "waiting", "wait",
					"dark", "difficult", "bad","expensive", "dirty",
					 "stained", "old", "used","smelly", "rude"]

pos_words = ["good","wonderful", "perfect","best", "love", "satisfied",
					 "positive", "amazing", "awesome","beautiful", "helpful",
					 "excellent","fantastic", "great", "accomodating",
					"cheap", "quality", "beautiful", "friendly", "comfortable",
					"friendly", "clean", "close", "big", "spacious", "nice"]

neg_words = ["bad", "disgusting", "terrible", "wrong", "worst", "awful",
					"ridiculous", "hate", "noisy", "negative", "horrible",
					"annoying","small", "uncomfortable", "waiting", "wait",
					"dark", "difficult", "bad","expensive", "dirty",
					 "stained", "old", "used","smelly", "rude"]



print "\n","*"*50,"\n", " Method: Polarity and Subjectivity \n"


#Extracting the binary feature set
new_xtrain = polarity.shifter(X_train)
new_xtest = polarity.shifter(X_test)

vpol_Xtrain = [polarity.polarityfeatures(d,word_features,mode="sklearn",calc="freq") for d in new_xtrain]
vpol_Xtest = [polarity.polarityfeatures(d,word_features,mode="sklearn",calc="freq") for d in new_xtest]

#Extracting the tf-idf feature set
pol_Xtrain = [polarity.polarityfeatures(d,word_features,mode="sklearn",calc="freq") for d in X_train]
pol_Xtest = [polarity.polarityfeatures(d,word_features,mode="sklearn",calc="freq") for d in X_test]
tfidf_Xtrain = transformer.fit_transform(pol_Xtrain).toarray()
tfidf_Xtest = transformer.transform(pol_Xtest).toarray()

#extracting with valence shifter



""" 
Results !
"""

print "Unused features: \t",polarity.check(vpol_Xtrain,word_features)

print "Baseline result: \t",polarity.baseline(y_train,y_test)

print "Simple Classification: \t", polarity.simpleclassification(X_test,y_test,pos_words,neg_words)

print "Simple Classification with valence shifters: \t", polarity.simpleclassification(new_xtest,y_test,pos_words,neg_words)

SVCC = GridSearchCV(SVC, parameters)
SVCC.fit(vpol_Xtrain,y_train)
print "Valence Radial SVM score: \t",SVCC.score(vpol_Xtest,y_test), SVCC.best_params_

GNB.fit(vpol_Xtrain,y_train)
print "Valence Gaussian Naive Bayes score: \t", GNB.score(vpol_Xtest,y_test)

SVR = GridSearchCV(SVC, parameters)
SVR.fit(tfidf_Xtrain,y_train)
print "Radial SVM score: \t",SVR.score(tfidf_Xtest,y_test), SVR.best_params_

GNB.fit(tfidf_Xtrain,y_train)
print "Gaussian Naive Bayes score: \t", GNB.score(tfidf_Xtest,y_test)

print "*" * 50

""" ---------------------------------------- End of Polarity and Subjectivity ----------------------------------- """

### running bag of words model
keithbow.main()