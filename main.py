from __future__ import division
import numpy as np
import pickle

targets = pickle.load( open( "targets.p", "rb" ) )
dataset = pickle.load( open( "dataset.p", "rb" ) )

#Dataset structure: [[[sentence 1 in review1], [sentence 2 in review 1] ...], [[sentence 1 in review 2], [sentence 2 in review 2] ...] ...]
# Dataset is shuffled and balanced, 1000 each of pos, neu and neg

X_train = dataset[:2500]
y_train = target[:2500]

X_test = dataset[2500:]
y_test = targets[2500:]
