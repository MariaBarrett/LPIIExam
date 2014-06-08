from __future__ import division
import numpy as np
import pickle
import nltk
from sklearn import svm, naive_bayes, grid_search
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
import pylab as pl


targets = pickle.load( open( "targets_easy.p", "rb" ) )
dataset = pickle.load( open( "dataset_easy.p", "rb" ) )

#Dataset structure: [[[sentence 1 in review1], [sentence 2 in review 1] ...], [[sentence 1 in review 2], [sentence 2 in review 2] ...] ...]
# Dataset is shuffled and balanced, 1000 each of pos, neu and neg

X_train = dataset[:2500]
y_train = targets[:2500]

X_test = dataset[2500:]
y_test = targets[2500:]

"""
getngram expects a dataset without labels and a string specifing the n of n-gram. 
That string must contain either 'bi' or 'tri'. 
It calls nltk to either make bigrams or trigrams of an entire dataset"""
def ngram(data, ngr, n):
	all_ngrams = []
	for review in data:
		for sentence in review:
			if 'bi' in ngr:
				bgs = nltk.bigrams(sentence)
				for b in bgs:
					all_ngrams.append(b)
			elif 'tri' in ngr:
				trg = nltk.trigrams(sentence)
				for t in trg: 
					all_ngrams.append(t)

	fdist_all = nltk.FreqDist(all_ngrams)

	print "Using %s most frequent %ss out of %s" %(n, ngr, len(fdist_all))
	return fdist_all.keys()[:n] #taking n most frequent from sorted by decreasing frequency

"""
getngram_singlereview expects a review and type of ngram.
It returns a list of all ngrams of the specified type in the review. 
"""
def getngram_singlereview(review, ngr):
	ngrams = []
	for sentence in review:
		if 'bi' in ngr:
			bgs = nltk.bigrams(sentence)
			for b in bgs:
				ngrams.append(b)
		if 'tri' in ngr:
			trg = nltk.trigrams(sentence)
			for t in trg: 
				ngrams.append(t)
	return ngrams


def tree_selection(train_data,train_labels,number_of_features):
	""" Returns the indices for the best parameters of a given dataset
	and it's target labels. The number_of_features parameter should be
	choosen by visual inspection using the inspect_tree_selection function. """ 

	forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
	forest.fit(train_data, train_labels)
	importances = forest.feature_importances_
	indices = np.argsort(importances)[::-1]

	return indices[:number_of_features]


def inspect_tree_selection(train_data,train_labels, all_bigrams, task):
	""" Given a dataset and its target labels, this
	function sorts the best features and prints and visualize them """

	forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
	forest.fit(train_data, train_labels)
	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print "-"*45
	print("\nFeature ranking for %s task:" %(task))

	for f in range(100):
	  print("%d. feature, name: %s, importance: %f" % (f + 1, all_bigrams[indices[f]], importances[indices[f]]))

	# Plot the feature importances of the forest
	pl.figure()
	n = train_data.shape[1]
	pl.title("%s: Sorted tree selection feature importance" %(task))
	pl.bar(range(n), importances[indices][:n], color="black", align="center")
	pl.xlim([-1, (n)])
	pl.show()



"""
binarize binarizes the dataset from all features found in the train set. 
First it calls getngram_singlereview to get all ngrams in a review.
For every review it checks for all ngrams from the train set and appends 1 if the ngram appers in the review and 0 if not.
it returns the binarized dataset as a numpy array. 
"""
def binarize(dataset, all_ngrams, ngr):
	binary_dataset = []
	for review in dataset:
		review_ngrams = getngram_singlereview(review, ngr)
		temp = []	
		for ngram in all_ngrams:
			if ngram in review_ngrams:
				temp.append(1)
			elif ngram not in review_ngrams:
				temp.append(0)
		binary_dataset.append(temp)
	return(np.asarray(binary_dataset))

def supvecmac(X_train, y_train, X_test, y_test):
	parameters = {'C': [ 4, 8, 16, 32, 64], 'gamma':[0.001, 0.01, 0.1, 1,]}
	svr = svm.SVC()
	clf = grid_search.GridSearchCV(svr, parameters)

	clf.fit(X_train, y_train)
	print "Done finding parameter settings, now starting to predict"
	print "Best parameters: ", clf.best_params_
	print "\nSVM Accuracy", clf.score(X_test, y_test)

	print "\nDetailed classification report:"
	print ""
	y_true, y_pred = y_test, clf.predict(X_test)
	print classification_report(y_true, y_pred)

def document_features(review, all_ngrams, ngr): 
    review_ngrams = set(getngram_singlereview(review, ngr)) 
    features = {}
    for ngram in all_ngrams:
        features[ngram] = (ngram in review_ngrams)
    return features
#######################################################################################
#
#					Calling
#
#######################################################################################



#Bigram
print "*"*45
print "n-gram"
print "*"*45

print "-"*45
print "Bigram"
print "-"*45

# NLTK

joint_for_nltk = zip(X_train + X_test, y_train + y_test)

"""
all_bigr_nltk = ngram(X_train, 'bigram', 400)
bi_featuresets = [(document_features(d, all_bigr_nltk, 'bigram'), c) for (d,c) in joint_for_nltk]
bi_train_for_nltk = bi_featuresets[:2500]
bi_test_for_nltk = bi_featuresets[2500:]

clf_bi = nltk.NaiveBayesClassifier.train(bi_train_for_nltk)

print "NTLK Naive Bayes bigram accuracy:", nltk.classify.accuracy(clf_bi, bi_test_for_nltk)
clf_bi.show_most_informative_features(20)
print ""
"""
#Sklearn
all_bigr = ngram(X_train, 'bigram', 3000)

print "Done making bigrams from train set"
print "Making bigrams and binarizing train set..."
X_train_bigram = binarize(X_train, all_bigr, 'bigram')
print "Done"
print "Making bigrams and binarizing test set..."
X_test_bigram = binarize(X_test, all_bigr, 'bigram')

inspect_tree_selection(X_train_bigram, y_train, all_bigr, 'Bigram')
indices_important_feats = tree_selection(X_train_bigram, y_train, 1000)
X_train_bigram_feat_sel = X_train_bigram[:,indices_important_feats]
X_test_bigram_feat_sel = X_test_bigram[:,indices_important_feats]

"""
clf = naive_bayes.GaussianNB()
clf.fit(X_train_bigram, y_train)

score = clf.score(X_test_bigram, y_test)
print "GaussianNB accuracy:", score
""" 

print "Done"
print "Beginning SVM with %s most important features" %(len(indices_important_feats))
supvecmac(X_train_bigram_feat_sel, y_train, X_test_bigram_feat_sel, y_test)


#Trigram

print "-"*45
print "Trigram"
print "-"*45

#Naive Bayes
all_trigr_nltk = ngram(X_train, 'trigram', 10000)
tri_featuresets = [(document_features(d, all_trigr_nltk, 'trigram'), c) for (d,c) in joint_for_nltk]
tri_train_for_nltk = tri_featuresets[:2500]
tri_test_for_nltk = tri_featuresets[2500:]

clf_tri = nltk.NaiveBayesClassifier.train(tri_train_for_nltk)

print "Naive Bayes trigram accuracy:", nltk.classify.accuracy(clf_tri, tri_test_for_nltk)
clf_tri.show_most_informative_features(20)
print ""

#Sklearn
all_trigr = ngram(X_train, 'trigram', 70)
print "Done making trigrams from train set"
print "Making trigrams and binarizing train set..."

X_train_trigram = binarize(X_train, all_trigr, 'trigram')
print "Done"
print "Making trigrams and binarizing test set..."
X_test_trigram = binarize(X_test, all_trigr, 'trigram')

print "Done"
print "Beginning Support Vector Machines"
acc = supvecmac(X_train_trigram, y_train, X_test_trigram, y_test)
print "SVM Accuracy", acc




