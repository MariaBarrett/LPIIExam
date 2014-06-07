from __future__ import division
import csv
import nltk
from collections import Counter
import random
import pickle
import numpy as np

"""
make_dataset makes a dataset from the csv file. 
it returns two lists: a list with the targets and a list with the tokenized reviews. Each sentence has its own list in the reviews
Dataset structure: [[[sentence 1 in review1], [sentence 2 in review 1] ...], [[sentence 1 in review 2], [sentence 2 in review 2] ...] ...]
The reviews are first sentence-tokenized and then word-tokenized. 

Remarks: 
-newline- is in the tokenized text, to show newlines. Should we remove those?
Punctuation is in separate tokens.
Contractions are tokenized like this: ['did','n't']
"""
def make_dataset():
	dataset = []
	targets = []
	print "Reading csv file"
	with open('three_cities_reviews.csv','r') as f:
	    reader=csv.reader(f,delimiter='\t')
	    for row in reader:
			targets.append(row[3])
			temp = []
			sentences = nltk.sent_tokenize(row[4])
			for sent in sentences:
				temp.append(nltk.word_tokenize(sent))
			dataset.append(temp)
	print "Done making oversampled_dataset"
	return targets, dataset			

"""
This function first makes all rating 1 and 2 to negtative reviews, 0, 3 is turned into 1. 4 and 5 are positive reviews and gets the value 2.
Then it samples the first 1000 occurences of each class into the balanced_targets and balanced_dataset.
I shuffle and return a list og targets and a list of reviews. 
"""
def balancedataset(oversampled_targets, oversampled_dataset):
	n = 1000
	
	# changing targets to 0 for negative, 1 for middle and 2 for positive
	# here we can also chose to take out the rating 2 or 4 to get more defined classes
	new_oversampled_t = [x if x != '1' else 0 for x in oversampled_targets]
	new_oversampled_t = [x if x != '2' else 0 for x in new_oversampled_t]
	new_oversampled_t = [x if x != '3' else 1 for x in new_oversampled_t]
	new_oversampled_t = [x if x != '4' else 2 for x in new_oversampled_t]
	new_oversampled_t = [x if x != '5' else 2 for x in new_oversampled_t]

	print "Done changing all values to 0,1 or 2"
	neg_list = []
	neu_list = []
	pos_list = []

	for i in xrange(len(new_oversampled_t)):
		if new_oversampled_t[i] == 0:
			neg_list.append(oversampled_dataset[i])
		elif new_oversampled_t[i] == 1:
			neu_list.append(oversampled_dataset[i])
		elif new_oversampled_t[i] == 2:
			pos_list.append(oversampled_dataset[i])
	

	balanced_dataset = neg_list[:1000]+neu_list[:1000]+pos_list[:1000]
	balanced_targets = [0]*1000+[1]*1000+[2]*1000

	zipped = zip(balanced_targets, balanced_dataset) #I'm zipping while shuffline because otherwise label and review did not match despite the fact that I seeded. Did I misunderstand something about seeding?
	
	np.random.seed(55)
	np.random.shuffle(zipped)

	balanced_targets_shuf, balanced_dataset_shuf = zip(*zipped)

	print "Done making a balanced, shuffled dataset"
	return list(balanced_targets_shuf), list(balanced_dataset_shuf)


#Calling

oversampled_targets, oversampled_dataset = make_dataset()	
balanced_targets, balanced_dataset = balancedataset(oversampled_targets, oversampled_dataset)

pickle.dump(balanced_targets, open( "targets.p", "wb" ) )
pickle.dump(balanced_dataset, open( "dataset.p", "wb" ) )
