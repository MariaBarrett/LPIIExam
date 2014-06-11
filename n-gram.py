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


##########################################################################
#
#				Making n-grams
#
#########################################################################

"""
ngram expects a dataset without labels and a string specifing the n of n-gram. 
That string must contain either 'bi' or 'tri'. 
It calls nltk to either make bigrams or trigrams of an entire dataset, sentence per sentence
It returns the n most frequent features sorted in decreasing frequency"""
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


##########################################################################
#
#				Feature selection
#
#########################################################################

""" 
tree_selection returns the indices for the n best features of a given dataset
and its target labels. The n parameter should be
choosen by visual inspection using the inspect_tree_selection function. 
""" 
def tree_selection(train_data,train_labels, n):
	forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
	forest.fit(train_data, train_labels)
	importances = forest.feature_importances_
	indices = np.argsort(importances)[::-1]

	return indices[:n]


""" 
inspect_tree_selection expects a dataset and its target labels as well as all bigrams from the train set. 
I sorts the features in decending order of importance.
It prints and visualizes them in a plot.
"""
def inspect_tree_selection(train_data, train_labels, all_bigrams, task):
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
	pl.savefig('bigram.pdf', bbox_inches='tight')
	print "plot saved"
	

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
	#parameters = {'C': [ 4, 8, 16, 32, 64, 128]}
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



##########################################################################
#
#				Crossvalidation
#
#########################################################################


"""
This function splits the train set and labels in s equal sized splits. 
It expects the features, labels and number of slices. 
It starts by making a copy of the features and labels
It returns a list of s slices containg lists of datapoints belonging to s and a similar list for labels.
"""
def sfold(features, labels, s):
	assert len(features) == len(labels)
	featurefold = np.copy(features)
	labelfolds = np.copy(labels)
	feature_slices = [featurefold[i::s] for i in xrange(s)]
	labels_slices = [labelfolds[i::s] for i in xrange(s)]
	return feature_slices, labels_slices

"""
The function expects a train set, a 1D list of train labels and number of folds. 
The function has a dict of all number of features to be tried. For each num it runs 5 fold crossvalidation with one num from the Num_features.
It copies the crossvaltrain and crossvaltest before cropping them.  
For every test-set for as many folds as there are: use the remaining slices as train sets.  
Then it sums up the test accuracies for every run and averages over them. The average performances per combination is stored.
The highest test average and the num that produced it is returned.   
"""
def crossval(X_train, y_train, folds, ngr):
	# Set the parameters by cross-validatiom
	tuned_parameters = {'Num_features': [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]}
	accuracy = []
	max = 32000
	#Starting by discarding excess features
	X_train_c = np.copy(X_train)
	X_train_c = [X_train_c[i][:max] for i in range(len(X_train_c))]
	features_slices, labels_slices = sfold(X_train_c, y_train, folds)

	#gridsearch
	print "Starting %s fold cross validation for finding best number of features" %(folds)
	for num in tuned_parameters['Num_features']:
		#joint = zip(X_train, y_train)
		#all_bigr_nltk = ngram(X_train, ngr, num)
		#featuresets = [(document_features(d, all_bigr_nltk, ngr), c) for (d,c) in joint]

		temp = []
		#crossvalidation
		for f in xrange(folds):
			crossvaltrain = []
			crossvaltrain_labels = []
			#define test-set for this run
			crossvaltest = np.array(features_slices[f])
			crossvaltest_labels = np.array(labels_slices[f])
			#define train set for this run
			for i in xrange(folds): #putting content of remaining slices in the train set 
				if i != f: # - if it is not the test slice: 
					for elem in features_slices[i]:
						crossvaltrain.append(elem) #making a list of trainset for this run
					for lab in labels_slices[i]:
						crossvaltrain_labels.append(lab)
							
			crossvaltrain_c = np.copy(crossvaltrain)
			crossvaltest_c = np.copy(crossvaltest)

			#taking only the num first features. Features are ordered by freqency
			crossvaltrain_c = [crossvaltrain_c[i][:num] for i in range(len(crossvaltrain))]
			crossvaltest_c = [crossvaltest_c[i][:num] for i in range(len(crossvaltest))]

			#clf = nltk.NaiveBayesClassifier.train(crossvaltrain_c)
			#acc = nltk.classify.accuracy(clf, crossvaltest_c)			
			clf = naive_bayes.GaussianNB()
			clf.fit(crossvaltrain_c, crossvaltrain_labels)

			acc = clf.score(crossvaltest_c, crossvaltest_labels)

			temp.append(acc)

		#for every num, get the average performance of the 5 runs:
		testmean = np.array(np.mean(temp))
		print "Average accuracy of %s features: %.6f" %(num, testmean)
		accuracy.append([num, testmean])

	#After all combinations have been tried get the best performance and the hyperparam pairs for that:
	accuracy.sort(key=operator.itemgetter(1), reverse = True) #sort by error - lowest first
	bestperf = accuracy[0][-1]
	bestnum = accuracy[0][0]
	print "\nBest number of features = %s: test error = %.6f" %(bestnum, bestperf)
	return bestnum


#######################################################################################
#
#					Calling
#
#######################################################################################

#---------------------------------------------------------------------------------------
#
#	Make section below active if you have already created the pickle files
#
#---------------------------------------------------------------------------------------
print "loading pickle bigram files..."
X_train_bigram = pickle.load( open( "X_train_bigram.p", "rb" ) )
X_test_bigram = pickle.load( open( "X_test_bigram.p", "rb" ) )

indices_important_feats_bi = pickle.load( open( "1000_bigram_indices.p", "rb" ) )
#---------------------------------------------------------------------------------------
#End loading pickle files




#---------------------------------------------------------------------------------------
#
#	Comment section below out if you already have made pickle files
#
#---------------------------------------------------------------------------------------

#all_bigr = ngram(X_train, 'bigram', 67052) #starting with all features
#
#print "Done making bigrams from train set"
#print "Making bigrams and binarizing train set..."
#X_train_bigram = binarize(X_train, all_bigr, 'bigram')
#print "Done"
#print "Making bigrams and binarizing test set..."
#X_test_bigram = binarize(X_test, all_bigr, 'bigram')
#
#pickle.dump(X_train_bigram, open( "X_train_bigram.p", "wb" ) )
#pickle.dump(X_test_bigram, open( "X_test_bigram.p", "wb" ) )
#
#print "Starting feature selection using CART random forests"
#inspect_tree_selection(X_train_bigram, y_train, all_bigr, 'bigram')
#print "Getting indices for most important features..."
#indices_important_feats_bi = tree_selection(X_train_bigram, y_train, 1000) #from visual inspection of plot you can set number of most important features
#
#pickle.dump(indices_important_feats_bi, open( "1000_bigram_indices.p", "wb" ) )
#print "Done and pickle file created"
#
#---------------------------------------------------------------------------------------
#End making pickle files



#Bigram
print "*"*45
print "n-gram"
print "*"*45

print "-"*45
print "Bigram Naive Bayes"
print "-"*45
	
best_num_feats = crossval(X_train_bigram, y_train, 5, 'bigram')

#Making a dataset with only the best number of features
bi_train_bestnum = [X_train_bigram[i][:best_num_feats] for i in range(len(X_train_bigram))]
bi_test_bestnum = [X_test_bigram[i][:best_num_feats] for i in range(len(X_test_bigram))]

#... and giving it to the Naive Bayes function

NaiveBayes(bi_train_bestnum, y_train, bi_test_bestnum, y_test)
print ""

print "-"*45
print "Bigram SVM" 
print "-"*45

X_train_bigram_feat_sel = X_train_bigram[:,indices_important_feats_bi]
X_test_bigram_feat_sel = X_test_bigram[:,indices_important_feats_bi]
print "Done"

supvecmac(X_train_bigram_feat_sel, y_train, X_test_bigram_feat_sel, y_test)

#Trigram
print "-"*45
print "Trigram Naive Bayes "
print "-"*45

#---------------------------------------------------------------------------------------
#
#	Make section below active if you have already created the pickle files
#
#---------------------------------------------------------------------------------------


print "loading pickle trigram files..."
X_train_trigram = pickle.load( open( "X_train_trigram.p", "rb" ) )
X_test_trigram = pickle.load( open( "X_test_trigram.p", "rb" ) )
indices_important_feats_tri = pickle.load( open( "1000_trigram_indices.p", "rb" ) )
#---------------------------------------------------------------------------------------
#End loading pickle files



#---------------------------------------------------------------------------------------
#
#	Comment this section out if you already made pickle files
#
#---------------------------------------------------------------------------------------#
#all_trigr = ngram(X_train, 'trigram', 127175) #all trigrams
#print "Done making trigrams from train set"
#print "Making trigrams and binarizing train set..."
#
#X_train_trigram = binarize(X_train, all_trigr, 'trigram')
#print "Done"
#print "Making trigrams and binarizing test set..."
#_test_trigram = binarize(X_test, all_trigr, 'trigram')
#print "Done"
#
#pickle.dump(X_train_trigram, open( "X_train_trigram.p", "wb" ) )
#pickle.dump(X_test_trigram, open( "X_test_trigram.p", "wb" ) )
#
#print "Starting feature selection using CART random forests"
#inspect_tree_selection(X_train_trigram, y_train, all_trigr, 'trigram')
#print "Getting indices for most important features..."
#indices_important_feats_tri = tree_selection(X_train_trigram, y_train, 1000)
#
#pickle.dump(indices_important_feats_tri, open( "1000_trigram_indices.p", "wb" ) )
#print "Done and pickle file created"
#---------------------------------------------------------------------------------------
#End making pickle files



best_num_feats_tri = crossval(X_train_trigram, y_train, 5, 'trigram')

#Making a dataset with only the best number of features
tri_train_bestnum = [X_train_trigram[i][:best_num_feats_tri] for i in range(len(X_train_trigram))]
tri_test_bestnum = [X_test_trigram[i][:best_num_feats_tri] for i in range(len(X_test_trigram))]

#... and giving it to the Naive Bayes function
NaiveBayes(tri_train_bestnum, y_train, tri_test_bestnum, y_test)
print ""

print "-"*45
print "Trigram SVM "
print "-"*45

X_train_trigram_feat_sel = X_train_trigram[:,indices_important_feats_tri]
X_test_trigram_feat_sel = X_test_trigram[:,indices_important_feats_tri]

supvecmac(X_train_trigram_feat_sel, y_train, X_test_trigram_feat_sel, y_test)
"""
