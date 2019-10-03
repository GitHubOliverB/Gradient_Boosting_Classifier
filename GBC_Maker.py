#!/usr/bin/python

# Basic Python Libs
import os
import sys
import importlib
import csv
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
np.set_printoptions(threshold=sys.maxsize)

GB_clfs = {}
def GB_classifier_maker(classifier_name, max_depth=3, min_samples_leaf=0.05, subsample=1, n_estimators=50, learning_rate=0.1):
	if max_depth <= 0:
		print("Error, Max_Depth is negative or zero. Needs to be an integer greater than 0")
		exit()
	elif min_samples_leaf <=0:
		print("Error, Min_Samples_Leaf is negative or zero. Set to a float between 0 and 1!")
		exit()
	elif subsample <=0:
		print("Error, Sub_Sample is negative or zero. Set to a float between 0 and 1!")
		exit()
	elif n_estimators <=0:
		print("Error, number of trees is negative or zero. Needs to be an integer greater than 0!")
		exit()
	elif learning_rate <=0:
		print("Error, Min_Samples_Leaf is negative or zero. Set to a float between 0 and 1!")
		exit()
	else:
		GB_clfs[str(classifier_name[0])] = GradientBoostingClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0, subsample=subsample, n_estimators=n_estimators, learning_rate=learning_rate, verbose=True)	
		