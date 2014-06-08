from __future__ import division
import numpy as np
import pickle
import nltk
from sklearn import svm, naive_bayes, grid_search
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
import pylab as pl
import operator

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
	pl.savefig('test.pdf', bbox_inches='tight')
	print "plot saved"
	


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
	print "Starting 5-fold cross validation"

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

##########################################################################
#
#				Crossvalidation
#
#########################################################################

"""
This function splits the train set in s equal sized splits. 
It expects the features and number of slices. 
It starts by making a copy 
It returns a list of s slices containg lists of datapoints belonging to s.
"""
def sfold(features, s):
	featurefold = np.copy(features)
	feature_slices = [featurefold[i::s] for i in xrange(s)]
	return feature_slices



"""
The function expects a train set, a 1D list of train labels and number of folds. 
The function has dicts of all C's and gammas and number of features. For each combination it runs 5 fold crossvalidation: 
For every test-set for as many folds as there are: use the remaining as train sets (exept if it's the test set.) 
Then we sum up the test and train result for every run and average it. The average performances per combination is stored.
The lowest test average and the combination that produced it is returned with the train error rate.   
"""
def crossval(X_train, y_train, folds, ngr):
	# Set the parameters by cross-validatiom
	tuned_parameters = {'Num_features': [100, 500, 1000, 2000, 4000, 8000, 16000]}
	accuracy = []

	#gridsearch
	print "Starting %s fold cross validation for finding best number of features" %(folds)
	for num in tuned_parameters['Num_features']:
		joint = zip(X_train, y_train)
		all_bigr_nltk = ngram(X_train, ngr, num)
		featuresets = [(document_features(d, all_bigr_nltk, ngr), c) for (d,c) in joint]
		features_slices = sfold(featuresets, folds)

		temp = []
		#crossvalidation
		for f in xrange(folds):
			crossvaltrain = []
			#define test-set for this run
			crossvaltest = np.array(features_slices[f])
			
			#define train set for this run
			for i in xrange(folds): #putting content of remaining slices in the train set 
				if i != f: # - if it is not the test slice: 
					for elem in features_slices[i]:
						crossvaltrain.append(elem) #making a list of trainset for this run

			crossvaltrain_c = np.copy(crossvaltrain)
			crossvaltest_c = np.copy(crossvaltest)

			#taking only the num first features
			[crossvaltrain_c[i][:num] for i in range(len(crossvaltrain))]
			[crossvaltest_c[i][:num] for i in range(len(crossvaltest))]

			#Classifying using library function
			clf = nltk.NaiveBayesClassifier.train(crossvaltrain_c)
			acc = nltk.classify.accuracy(clf, crossvaltest_c)			

			temp.append(acc)

		#for every num, get the average performance of the 5 runs:
		testmean = np.array(np.mean(temp))
		print "Average accuracy of %s features: %.6f" %(num, testmean)
		accuracy.append([num, testmean])

	#After all combinations have been tried get the best performance and the hyperparam pairs for that:
	accuracy.sort(key=operator.itemgetter(1)) #sort by error - lowest first
	bestperf = accuracy[0][-1]
	bestnum = accuracy[0][0]
	print "\nBest number of features = %s: test error = %.6f" %(bestnum, bestperf)
	return bestnum


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
print "Bigram Naive Bayes"
print "-"*45

# NLTK

joint_for_nltk = zip(X_train + X_test, y_train + y_test)

best_num_feats = crossval(X_train, y_train, 5, 'bigram')

#Extracting only best number of features (starting from most frequent in decending order)
#NLTK wants the label in the last position
all_bigr_nltk = ngram(X_train, 'bigram', best_num_feats)
bi_featuresets = [(document_features(d, all_bigr_nltk, 'bigram'), c) for (d,c) in joint_for_nltk]
bi_train_for_nltk = bi_featuresets[:2500]
bi_test_for_nltk = bi_featuresets[2500:]

bi_train_for_nltk_optimized = [bi_train_for_nltk[i][:best_num_feats] for i in range(len(bi_train_for_nltk))]
bi_test_for_nltk_optimized = [bi_test_for_nltk[i][:best_num_feats] for i in range(len(bi_test_for_nltk))]

clf_bi = nltk.NaiveBayesClassifier.train(bi_train_for_nltk_optimized)
print "NTLK Naive Bayes bigram accuracy using best number of features:", nltk.classify.accuracy(clf_bi, bi_test_for_nltk_optimized)
clf_bi.show_most_informative_features(20)
print ""

#Sklearn
print "-"*45
print "Bigram SVM"
print "-"*45

all_bigr = ngram(X_train, 'bigram', 5000)

print "Done making bigrams from train set"
print "Making bigrams and binarizing train set..."
X_train_bigram = binarize(X_train, all_bigr, 'bigram')
print "Done"
print "Making bigrams and binarizing test set..."
X_test_bigram = binarize(X_test, all_bigr, 'bigram')

print "Starting feature selection using CART random forests"
inspect_tree_selection(X_train_bigram, y_train, all_bigr, 'bigram')
indices_important_feats = tree_selection(X_train_bigram, y_train, 750)
X_train_bigram_feat_sel = X_train_bigram[:,indices_important_feats]
X_test_bigram_feat_sel = X_test_bigram[:,indices_important_feats]
print "Done"

"""
clf = naive_bayes.GaussianNB()
clf.fit(X_train_bigram, y_train)

score = clf.score(X_test_bigram, y_test)
print "GaussianNB accuracy:", score
""" 
supvecmac(X_train_bigram_feat_sel, y_train, X_test_bigram_feat_sel, y_test)


#Trigram
print "-"*45
print "Trigram Naive Bayes "
print "-"*45
best_num_feats = crossval(tri_train_for_nltk, y_train, 5, 'trigram')

all_tri_nltk = ngram(X_train, 'trigram', best_num_feats)
tri_featuresets = [(document_features(d, all_tri_nltk, 'bigram'), c) for (d,c) in joint_for_nltk]
tri_train_for_nltk = bi_featuresets[:2500]
tri_test_for_nltk = bi_featuresets[2500:]

tri_train_for_nltk_optimized = [tri_train_for_nltk[i][:best_num_feats] for i in range(len(tri_train_for_nltk))]
tri_test_for_nltk_optimized = [tri_test_for_nltk[i][:best_num_feats] for i in range(len(tri_test_for_nltk))]

clf_tri = nltk.NaiveBayesClassifier.train(tri_train_for_nltk_optimized)
print "NTLK Naive Bayes trigram accuracy using best number of features:", nltk.classify.accuracy(clf_tri, tri_test_for_nltk_optimized)
clf_tri.show_most_informative_features(20)
print ""

#Sklearn
print "-"*45
print "Trigram SVM "
print "-"*45
all_trigr = ngram(X_train, 'trigram', 5000)
print "Done making trigrams from train set"
print "Making trigrams and binarizing train set..."

X_train_trigram = binarize(X_train, all_trigr, 'trigram')
print "Done"
print "Making trigrams and binarizing test set..."
X_test_trigram = binarize(X_test, all_trigr, 'trigram')
print "Done"

print "Starting feature selection using CART random forests"
inspect_tree_selection(X_train_trigram, y_train, all_bigr, 'trigram')
indices_important_feats = tree_selection(X_train_trigram, y_train, 750)
X_train_trigram_feat_sel = X_train_trigram[:,indices_important_feats]
X_test_trigram_feat_sel = X_test_trigram[:,indices_important_feats]
print "Done"

supvecmac(X_train_trigram, y_train, X_test_trigram, y_test)




