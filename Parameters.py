#!/usr/bin/python

import os
import sys

# Fixed for reproductivity
random_seeds = [150, 950, 84512, 4, 3218487]

# Version Name
classifier_name = ["GBT_Classifier_Example"]

# Splitting Parameters
reserve_fraction=0.20
test_fraction = 0.5

# Hyperparameters For Training
max_depth = 3
min_samples_leaf = 0.1
subsample = 0.8
n_estimators = 500
learning_rate = 0.1