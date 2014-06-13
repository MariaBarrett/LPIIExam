README

The following libraries must be installed prior to running this code:
csv
nltk
collections
random
pickle
numpy
sklearn
pylab
operator
cPickle  load, dump

dataset.p and targets.p were created by running make-dataset.py. You don't have to run it. It's already done. 

############################################################
How to
###########################################################

Run main.py from your favorite IDE
Main.py calls the other files as well as dataset.p and targets.p and reproduce all results from the report.

First it runs n-gram.py, which takes quite a while to run
Then it runs word lists (in the main file) with / without valence shifters

Lastly it runs bow_unigram_Keith.py. 
