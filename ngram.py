from __future__ import division
import numpy as np
import pickle
import nltk
from sklearn import svm, naive_bayes, grid_search
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import Binarizer
import pylab as pl
import operator
from collections import Counter

targets = pickle.load( open( "targets_easy.p", "rb" ) )
dataset = pickle.load( open( "dataset_easy.p", "rb" ) )

#Dataset structure: [[[sentence 1 in review1], [sentence 2 in review 1] ...], [[sentence 1 in review 2], [sentence 2 in review 2] ...] ...]
# Dataset is shuffled and balanced, 1000 each of pos, neu and neg

X_train = dataset[:2500]
y_train = targets[:2500]

X_test = dataset[2500:]
y_test = targets[2500:]

##########################################################################
#
#				Making n-grams
#
#########################################################################

"""
ngram expects a dataset without labels and a string specifing the n of n-gram. 
That string must contain either 'bi' or 'tri'. 
It calls nltk to either make bigrams or trigrams of an entire dataset, sentence per sentence
It returns the features sorted in decreasing frequency"""
def ngram(data, ngr):
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

	return fdist_all.keys() #taking n most frequent from sorted by decreasing frequency

"""
getngram_singlereview expects a review and type of ngram as a string.
It returns an unsorted list of all ngrams of the specified type in the review. 
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

"""
count function makes vectors from dataset from all features found in the train set. 
First it calls getngram_singlereview to get all ngrams in a review.
For every review it checks for all ngrams from the train set and appends 1 if the ngram appers in the review and 0 if not.
it returns the counted dataset as a numpy array. 
"""
def count(dataset, all_ngrams, ngr):
	counted_dataset = []
	for review in dataset:
		review_ngrams = getngram_singlereview(review, ngr)	
		temp = []
		for ngram in all_ngrams:
			temp.append(review_ngrams.count(ngram))
		counted_dataset.append(temp)
	return(np.asarray(counted_dataset))


"""
tfidf function expects X_train, and X_test that has already been turned into feature count vectors
it returns tfidf vectors instead
"""
def tfidf(X_train, X_test):
	transf = TfidfTransformer()

	tfidf_train = transf.fit_transform(X_train).toarray()
	tfidf_test = transf.transform(X_test).toarray()
	return tfidf_train, tfidf_test

##########################################################################
#
#				Feature selection
#
#########################################################################


""" 
tree expect a train set and adjacent labels as well as names of all features (for printing)

"""
def tree(train_data, train_labels, all_bigrams, task):
	forest = ExtraTreesClassifier(n_estimators=100, random_state=0)
	forest.fit(train_data, train_labels)
	importances = forest.feature_importances_
	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print "-"*45
	print task

	for f in range(20):
	  print("%d. feature, name: %s, importance: %f" % (f + 1, all_bigrams[indices[f]], importances[indices[f]]))

	# Plot the feature importances of the forest
	pl.figure()
	n = train_data.shape[1]
	n = 2000
	pl.title("Sorted feature importance for %s" %(task))
	pl.bar(range(n), importances[indices][:n], color="black", align="center")
	pl.xlim([0, (n)])
	pl.xticks([num for num  in range(0, n+1, 250)])
	pl.savefig(task+'.pdf', bbox_inches='tight')
	print "plot saved"

	return indices
	

##########################################################################
#
#				Clasifiers
#
#########################################################################


"""
supvecmac expects a train set with labels and a test set with labels.
It runs 5-fold gridsearch on the train set to find best C and gamma.
Then it uses the best parameters on the test set and prints the result 
"""
def supvecmac(X_train, y_train, X_test, y_test):
	print "Starting 5-fold cross validation"
	parameters = {'C': [ 4, 8, 16, 32, 64, 128], 'gamma':[0.0001, 0.001, 0.01, 0.1, 1]}
	#svr = svm.LinearSVC()
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


"""
NaiveBayes just calls the SK-learn gaussian naive bayes and prints the accuracy and the classification report
"""
def NaiveBayes(X_train, y_train, X_test, y_test):
	clf = naive_bayes.GaussianNB()
	clf.fit(X_train, y_train)

	print "\nNaive Bayes Accuracy", clf.score(X_test, y_test)

	print "\nDetailed classification report:"
	print ""
	y_true, y_pred = y_test, clf.predict(X_test)
	print classification_report(y_true, y_pred)


#######################################################################################
#
#					Calling
#
#######################################################################################
n = 1000 #number of most important indices to use

#---------------------------------------------------------------------------------------
#
#	Make section below active if you have already created the pickle files
#
#---------------------------------------------------------------------------------------
"""
print "Loading pickle bigram files..."
X_train_bi_binary = pickle.load( open( "X_train_bi_binary.p", "rb" ) )
X_test_bi_binary = pickle.load( open( "X_test_bi_binary.p", "rb" ) )

X_train_bi_tfidf = pickle.load( open( "X_train_bi_tfidf.p", "rb" ) )
X_test_bi_tfidf = pickle.load( open( "X_test_bi_tfidf.p", "rb" ) )

indices_important_feats_bi_tfidf = pickle.load( open( "indices_important_feats_bi_tfidf.p", "rb" ) )
indices_important_feats_bi_bin = pickle.load( open( "indices_important_feats_bi_bin.p", "rb" ) )
"""
#---------------------------------------------------------------------------------------
#End loading pickle files



#---------------------------------------------------------------------------------------
#
#	Comment section below out if you already have made pickle files
#
#---------------------------------------------------------------------------------------

all_bigr = ngram(X_train, 'bigram') #starting with all features

print "Starting counting bigrams..."
X_train_bi_counted = count(X_train, all_bigr, 'bigram')
print "Done counting train set"
X_test_bi_counted = count(X_test, all_bigr, 'bigram')
print "Done counting test set"

print "Binarizing and dumping files"
bin = Binarizer()
X_train_bi_binary = bin.fit_transform(X_train_bi_counted)
X_test_bi_binary = bin.transform(X_test_bi_counted)
pickle.dump(X_train_bi_binary, open( "X_train_bi_binary.p", "wb" ) )
pickle.dump(X_test_bi_binary, open( "X_test_bi_binary.p", "wb" ) )
print "Done"


print "Starting tfidf vectors..."
X_train_bi_tfidf, X_test_bi_tfidf = tfidf(X_train_bi_counted, X_test_bi_counted)
pickle.dump(X_train_bi_tfidf, open( "X_train_bi_tfidf.p", "wb" ) )
pickle.dump(X_test_bi_tfidf, open( "X_test_bi_tfidf.p", "wb" ) )
print "Done"


print "Starting feature selection using CART random forests on binary files"
indices_important_feats_bi_bin = tree(X_train_bi_binary, y_train, all_bigr, 'Bigram_binary')
pickle.dump(indices_important_feats_bi_bin, open( "indices_important_feats_bi_bin.p", "wb" ) )
print "Done and pickle file created"

print "Starting feature selection using CART random forests on TF-IDF"
indices_important_feats_bi_tfidf = tree(X_train_bi_tfidf, y_train, all_bigr, 'Bigram_TF-IDF')
pickle.dump(indices_important_feats_bi_tfidf, open( "indices_important_feats_bi_tfidf.p", "wb" ) )
print "Done and pickle file created"

#---------------------------------------------------------------------------------------
#End making pickle files


print "*"*45
print "n-gram"
print "*"*45

#Bigram

#only the most important features
indices_important_feats_bi_tfidf = indices_important_feats_bi_tfidf[:n]
indices_important_feats_bi_bin = indices_important_feats_bi_bin[:n]

X_train_bi_tfidf_sel = X_train_bi_tfidf[:,indices_important_feats_bi_tfidf]
X_test_bi_tfidf_sel = X_test_bi_tfidf[:,indices_important_feats_bi_tfidf]

X_train_bi_bin_sel = X_train_bi_binary[:,indices_important_feats_bi_bin]
X_test_bi_bin_sel = X_test_bi_binary[:,indices_important_feats_bi_bin]

print "-"*45
print "Bigram Naive Bayes"
print "-"*45
print "Binary"
NaiveBayes(X_train_bi_bin_sel, y_train, X_test_bi_bin_sel, y_test)
print ""
print "TF-IDF"
NaiveBayes(X_train_bi_tfidf_sel, y_train, X_test_bi_tfidf_sel, y_test)

print "-"*45
print "Bigram SVM" 
print "-"*45
print "Binary"
supvecmac(X_train_bi_bin_sel, y_train, X_test_bi_bin_sel, y_test)

print "TF-IDF"
supvecmac(X_train_bi_tfidf_sel, y_train, X_test_bi_tfidf_sel, y_test)


#Trigram

#---------------------------------------------------------------------------------------
#
#	Make section below active if you have already created the pickle files
#
#---------------------------------------------------------------------------------------
"""
print "Loading pickle trigram files..."
X_train_tri_tfidf = pickle.load( open( "X_train_tri_tfidf.p", "rb" ) )
X_test_tri_tfidf = pickle.load( open( "X_test_tri_tfidf.p", "rb" ) )

X_train_tri_binary = pickle.load( open( "X_train_tri_binary.p", "rb" ) )
X_test_tri_binary = pickle.load( open( "X_test_tri_binary.p", "rb" ) )

indices_important_feats_tri_bin = pickle.load( open( "indices_important_feats_tri_bin.p", "rb" ) )
indices_important_feats_tri_tfidf = pickle.load( open( "indices_important_feats_tri_tfidf.p", "rb" ) )
"""
#---------------------------------------------------------------------------------------
#End loading pickle files

#---------------------------------------------------------------------------------------
#
#	Comment this section out if you already made pickle files
#
#---------------------------------------------------------------------------------------#
all_trigr = ngram(X_train, 'trigram')
print "Done making trigrams from train set"

print "Starting counting trigrams..."
X_train_tri_counted = count(X_train, all_trigr, 'trigram')
print "Done counting train set"
X_test_tri_counted = count(X_test, all_trigr, 'trigram')
print "Done counting test set"


print "Binarizing and dumping files"
bin = Binarizer()
X_train_tri_binary = bin.fit_transform(X_train_tri_counted)
X_test_tri_binary = bin.transform(X_test_tri_counted)
pickle.dump(X_train_tri_binary, open( "X_train_tri_binary.p", "wb" ) )
pickle.dump(X_test_tri_binary, open( "X_test_tri_binary.p", "wb" ) )
print "Done"

print "Starting tfidf vectors..."
X_train_tri_tfidf, X_test_tri_tfidf = tfidf(X_train_tri_counted, X_test_tri_counted)
pickle.dump(X_train_tri_tfidf, open( "X_train_tri_tfidf.p", "wb" ) )
pickle.dump(X_test_tri_tfidf, open( "X_test_tri_tfidf.p", "wb" ) )
print "Done"


print "Starting feature selection using CART random forests on binary files"
indices_important_feats_tri_bin = tree(X_train_tri_binary, y_train, all_trigr, 'Trigram_binary')
pickle.dump(indices_important_feats_tri_bin, open( "indices_important_feats_tri_bin.p", "wb" ) )
print "Done and pickle file created"


print "Starting feature selection using CART random forests on TF-IDF" 
indices_important_feats_tri_tfidf = tree(X_train_tri_tfidf, y_train, all_trigr, 'Trigram_TF-IDF')
pickle.dump(indices_important_feats_tri_tfidf, open( "indices_important_feats_tri_tfidf.p", "wb" ) )
print "Done and pickle file created"

#---------------------------------------------------------------------------------------
#End making pickle files

#Only taking the most important features
indices_important_feats_tri_bin = indices_important_feats_tri_bin[:n]
indices_important_feats_tri_tfidf = indices_important_feats_tri_tfidf[:n]

X_train_tri_feat_sel_bin = X_train_tri_binary[:,indices_important_feats_tri_bin]
X_test_tri_feat_sel_bin = X_test_tri_binary[:,indices_important_feats_tri_bin]

X_train_tri_feat_sel_tfidf = X_train_tri_tfidf[:,indices_important_feats_tri_tfidf]
X_test_tri_feat_sel_tfidf = X_test_tri_tfidf[:,indices_important_feats_tri_tfidf]

print "-"*45
print "Trigram Naive Bayes "
print "-"*45
print "Binary"
NaiveBayes(X_train_tri_feat_sel_bin, y_train, X_test_tri_feat_sel_bin, y_test)
print "TF_IDF"
NaiveBayes(X_train_tri_feat_sel_tfidf, y_train, X_test_tri_feat_sel_tfidf, y_test)
print ""

print "-"*45
print "Trigram SVM "
print "-"*45
print "Binary"
supvecmac(X_train_tri_feat_sel_bin, y_train, X_test_tri_feat_sel_bin, y_test)
print "TF_IDF"
supvecmac(X_train_tri_feat_sel_tfidf, y_train, X_test_tri_feat_sel_tfidf, y_test)
