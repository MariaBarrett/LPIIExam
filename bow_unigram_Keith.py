from __future__ import division
import numpy as np
from cPickle import load, dump
import nltk
from nltk import wordpunct_tokenize as punct
from sklearn import svm, naive_bayes, grid_search
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
import pylab as pl
import operator
from collections import Counter as C
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer


targets = load( open( "targets_easy.p", "rb" ) )
dataset = load( open( "dataset_easy.p", "rb" ) )

''' Create bag of words '''
''' 
	Extract all unique words in the dataset. 
	Then go through all sentences and create an array of 1's and 0's where a 1 corresponds
	to whether a word appears or not in that particular sentence.

	Then dump them into a pickle file to use later
'''
def CreateBOW(counter, dataset, targets, presfreq = 'freq'):
	bow = []

	## Create bag of words
	for i in range(len(dataset)):
		rep = [targets[i]] #add polarity

		bag = C()
		for d in dataset[i]:
			bag += C(d)

   		for w in counter:
   			if w in bag:
   				if presfreq == 'freq': #if we want the frequency of words, retrieve it from counter
   					rep.append(bag[w])
   				elif presfreq == 'pres': #else just write 1
   					rep.append(1)
			else:
				rep.append(0)

		bow.append(rep) #append polarity + text as 1 row

	return bow
    #dump the data structure to binary file
	# dump(bow, open ("bow.p", "wb") )

''' Create Unigram for NLTK '''
''' 
	Extract all unique words in the dataset. 
	Then go through all sentences and create a set of words contained in the document.

'''
def CreateUnigramNLTK(dataset):
	ug = []
	counter = C()

	### Get a count of all words, so we only use the words appearing more than once
	for sents in dataset:
		for words in sents:
			counter += C(words) #create counter of words

	counter = {k: counter[k] for k in counter if counter[k]>1} #take only words appearing more than once

	## Create unigram set
	for i in range(len(dataset)):
		bag = C()
		for d in dataset[i]:
			bag += C(d)

		features = {}
		for word in counter:
			features['contains(%s)' % word.lower()] = (word.lower() in bag)

		ug.append([features, targets[i]])

	return ug

""" 
inspect_tree_selection expects a dataset and its target labels as well as all bigrams from the train set. 
I sorts the features in decending order of importance.
It prints and visualizes them in a plot.
"""

def inspect_tree_selection(train_data,train_labels, task):
	
	print "Starting tree selection"
	forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
	forest.fit(train_data, train_labels)
	print "Fitted..."
	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
	indices = np.argsort(importances)[::-1]
	print "Finished"

	# Print the feature ranking
	print "-"*45
	print("\nFeature ranking for %s task:" %(task))

	#for f in range(100):
	#  print("%d. feature, name: %s, importance: %f" % (f + 1, all_bigrams[indices[f]], importances[indices[f]]))

	# Plot the feature importances of the forest
	print "plotting the graph..."
	pl.figure()
	n = train_data.shape[1]
	pl.title("%s: Sorted tree selection feature importance" %(task))
	pl.bar(range(n), importances[indices][:n], color="black", align="center")
	pl.xlim([-1, (n)])
	pl.savefig('bigram.pdf', bbox_inches='tight')
	print "plot saved"
	

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
				"""
				if sentence[i] in intensify_list:
					sentence[i+1] = 'MORE_'+sentence[i+1]
				if sentence[i] in deintensify:
					sentence[i+1] = 'LESS_'+sentence[i+1]
				"""
				if sentence[i] in not_list:
					sentence[i+1:] = ['NOT_'+sentence[k] for k in xrange(i+1, len(sentence)) ]
	return dataset_c



def ReadBOW():
	f = open('bow_nltk.p', 'rb')
	return np.array(load(f))


''' TRAINING A NAIVE BAYES CLASSIFIER ''' 
def RunNaiveBayes(X_train, y_train, X_test, y_test):
	gnb = GaussianNB()
	y_pred = gnb.fit(X_train, y_train).predict(X_test)
	correct = (y_test == y_pred).sum()

	print "*" * 10
	print "Naive Bayes Accuracy :", correct / len(y_test) * 100, "%"
	print "*" * 10
	print "\nDetailed classification report:"
	print ""
	y_pred = gnb.predict(X_test)
	print classification_report(y_test, y_pred)



''' TRAINING AN SVM CLASSIFIER '''
def RunSVM(X_train, y_train, X_test, y_test):
	parameters = {'C': [ 8,10,12,14,16]	, 'gamma':[0.001,0.01,0.1,1]}

	clf = svm.SVC(C=14, gamma=0.1)

	#svc = GridSearchCV(clf, parameters)
	#svc.fit(X_train, y_train)
	clf.fit(X_train, y_train)

	''' print accuracy '''
	print "*" * 10
	print "SVM Accuracy:: \t%",(clf.score(X_test, y_test))*100#, svc.best_params_
	print "*" * 10


	print "\nDetailed classification report:"
	print ""
	y_pred = clf.predict(X_test)
	print classification_report(y_test, y_pred)


#####################################################
#### MAIN ###########################################
#####################################################
def main():
	print "Running Bag-Of-Words method"
	print "*" * 20
	X_train = dataset[:2500]
	y_train = targets[:2500]

	X_test = dataset[2500:]
	y_test = targets[2500:]


	''' create a set of words from the reviews in the train set '''
	counter = C()

	for sents in X_train:
		for words in sents:
			counter += C(words) #create counter of words


	negdataset = shifter(dataset)
	X_train_neg = negdataset[:2500]
	X_test_neg = negdataset[2500:]

	''' create set of feature words from the shifted dataset '''
	negCounter = C()
	for sents in X_train_neg:
		for words in sents:
			negCounter += C(words) 


	print "No. of features (all):", len(counter)
	print "No. of features (all + valence shifters):", len(negCounter)




	''' Create Bag of Words from training words ''' 
	X_train_binarized = CreateBOW(counter, X_train, y_train, "pres")
	X_test_binarized = CreateBOW(counter, X_test, y_test, "pres")

	print "Naive Bayes, All Features, Presence Only"
	RunNaiveBayes(X_train_binarized, y_train, X_test_binarized, y_test)
	print
	print "SVM, All Features, Presence Only"
	transformer = TfidfTransformer()
	X_train_binarized = transformer.fit_transform(X_train_binarized).toarray()
	X_test_binarized = transformer.fit_transform(X_test_binarized).toarray()

	RunSVM(X_train_binarized, y_train, X_test_binarized, y_test)
	print



	X_train_binarized = CreateBOW(counter, X_train, y_train, "freq")
	X_test_binarized = CreateBOW(counter, X_test, y_test, "freq")

	print "Naive Bayes, All Features, Word Frequency"
	RunNaiveBayes(X_train_binarized, y_train, X_test_binarized, y_test)
	print
	print "SVM, All Features, Word Frequency"
	transformer = TfidfTransformer()
	X_train_binarized = transformer.fit_transform(X_train_binarized).toarray()
	X_test_binarized = transformer.fit_transform(X_test_binarized).toarray()

	RunSVM(X_train_binarized, y_train, X_test_binarized, y_test)
	print


	X_train_binarized = CreateBOW(negCounter, X_train_neg, y_train, "pres")
	X_test_binarized = CreateBOW(negCounter, X_test_neg, y_test, "pres")

	print "Naive Bayes, All Features + Negation/intensifier List, Presence Only"
	RunNaiveBayes(X_train_binarized, y_train, X_test_binarized, y_test)
	print
	print "SVM, All Features + Negation/intensifier List, Presence Only"
	transformer = TfidfTransformer()
	X_train_binarized = transformer.fit_transform(X_train_binarized).toarray()
	X_test_binarized = transformer.fit_transform(X_test_binarized).toarray()
	RunSVM(X_train_binarized, y_train, X_test_binarized, y_test)
	print 



	X_train_binarized = CreateBOW(negCounter, X_train_neg, y_train, "freq")
	X_test_binarized = CreateBOW(negCounter, X_test_neg, y_test, "freq")

	print "Naive Bayes, All Features + Negation/intensifier List, Word Frequency"
	RunNaiveBayes(X_train_binarized, y_train, X_test_binarized, y_test)
	print
	print "SVM, All Features + Negation/intensifier List, Word Frequency"
	transformer = TfidfTransformer()
	X_train_binarized = transformer.fit_transform(X_train_binarized).toarray()
	X_test_binarized = transformer.fit_transform(X_test_binarized).toarray()
	RunSVM(X_train_binarized, y_train, X_test_binarized, y_test)
	print 



	counter = {k: counter[k] for k in counter if counter[k]>1} #take only words appearing more than once
	print "No. of features (reduced):", len(counter)

	negCounter = {k: negCounter[k] for k in negCounter if negCounter[k] > 1}
	print "No. of features (reduced + valence shifters):", len(negCounter)



	''' Create Bag of Words from training words ''' 
	X_train_binarized = CreateBOW(counter, X_train, y_train, "pres")
	X_test_binarized = CreateBOW(counter, X_test, y_test, "pres")

	print "Naive Bayes, Reduced Features, Presence Only"
	RunNaiveBayes(X_train_binarized, y_train, X_test_binarized, y_test)
	print
	print "SVM, Reduced Features, Presence Only"
	transformer = TfidfTransformer()
	X_train_binarized = transformer.fit_transform(X_train_binarized).toarray()
	X_test_binarized = transformer.fit_transform(X_test_binarized).toarray()
	RunSVM(X_train_binarized, y_train, X_test_binarized, y_test)
	print


	X_train_binarized = CreateBOW(counter, X_train, y_train, "freq")
	X_test_binarized = CreateBOW(counter, X_test, y_test, "freq")

	print "Naive Bayes, Reduced Features, Word Frequency"
	RunNaiveBayes(X_train_binarized, y_train, X_test_binarized, y_test)
	print
	print "SVM, Reduced Features, Word Frequency"
	transformer = TfidfTransformer()
	X_train_binarized = transformer.fit_transform(X_train_binarized).toarray()
	X_test_binarized = transformer.fit_transform(X_test_binarized).toarray()
	RunSVM(X_train_binarized, y_train, X_test_binarized, y_test)
	print


	X_train_binarized = CreateBOW(negCounter, X_train_neg, y_train, "pres")
	X_test_binarized = CreateBOW(negCounter, X_test_neg, y_test, "pres")

	print "Naive Bayes, Reduced Features + Negation/intensifier List, Presence Only"
	RunNaiveBayes(X_train_binarized, y_train, X_test_binarized, y_test)
	print
	print "SVM, Reduced Features + Negation/intensifier List, Presence Only"
	transformer = TfidfTransformer()
	X_train_binarized = transformer.fit_transform(X_train_binarized).toarray()
	X_test_binarized = transformer.fit_transform(X_test_binarized).toarray()
	RunSVM(X_train_binarized, y_train, X_test_binarized, y_test)
	print 


	X_train_binarized = CreateBOW(negCounter, X_train_neg, y_train, "freq")
	X_test_binarized = CreateBOW(negCounter, X_test_neg, y_test, "freq")

	print "Naive Bayes, Reduced Features + Negation/intensifier List, Word Frequency"
	RunNaiveBayes(X_train_binarized, y_train, X_test_binarized, y_test)
	print
	print "SVM, Reduced Features + Negation/intensifier List, Word Frequency"
	transformer = TfidfTransformer()
	X_train_binarized = transformer.fit_transform(X_train_binarized).toarray()
	X_test_binarized = transformer.fit_transform(X_test_binarized).toarray()
	RunSVM(X_train_binarized, y_train, X_test_binarized, y_test)
	print 

