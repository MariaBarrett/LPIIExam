from __future__ import division
import csv
import nltk
from collections import Counter



"""
make_dataset makes a dataset from the csv file. 
it returns two lists: a list with the targets and a list with the tokenized reviews. Each sentence has its own list in the reviews
Dataset structure: [[[sentence 1 in review1], [sentence 2 in review 1] ...], [[sentence 1 in review 2], [sentence 2 in review 2] ...] ...]
The reviews are first sentence-tokenized and then word-tokenized. 

Remarks: 
-newline- is in the tokenized text, to show newlines. Should we remove those?
punctuation is in separate tokens, which I didn't expect.
contractions are tokenized like this: ['did','n't']
"""
def make_dataset():
	dataset = []
	targets = []
	with open('reviews.csv','r') as f:
	    reader=csv.reader(f,delimiter='\t')
	    for row in reader:
			targets.append(row[3])
			temp = []
			sentences = nltk.sent_tokenize(row[4])
			for sent in sentences:
				temp.append(nltk.word_tokenize(sent))
			dataset.append(temp)
	return targets, dataset			

targets, dataset = make_dataset()