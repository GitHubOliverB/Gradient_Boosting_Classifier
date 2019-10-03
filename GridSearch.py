#!/usr/bin/python
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Reference for the code of this script
# Block 0 - Library Imports
# 	Block 0.1 - Defining functions and globals...
#	Block 0.2 - Paths... 
# Block 1.0 - Data Imports
# Block 2.0 - Splitting Data
# Block 4.0 - Training and Testing
# Block 5.0 - Validation
# Block 6.0 - Plotting Probability On Data
# Block X.0 - Trash code
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Block 0 - Library Imports

# Basic Python Libs
import os
import sys
import importlib
import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
np.set_printoptions(threshold=sys.maxsize)

# Own Imports
from Data_Feature_Imports import *
from Parameters import *
from GBC_Maker import *

GB_clfs = {}

#	Block 0.2 - Paths...
classifier_path = classifier_name[0]
if not os.path.exists(classifier_path):
	os.makedirs(classifier_path)
model_path = classifier_path+"\Model\\"
if not os.path.exists(model_path):
	os.makedirs(model_path)
plot_path = classifier_path+"\Plots"
if not os.path.exists(plot_path):
	os.makedirs(plot_path)
tree_path = plot_path+"\Decision_Trees\\"
if not os.path.exists(tree_path):
	os.makedirs(tree_path)
correlation_path = plot_path+"\Correlations"
if not os.path.exists(correlation_path):
	os.makedirs(correlation_path)
training_path = plot_path+"\Training\\"
if not os.path.exists(training_path):
	os.makedirs(training_path)
validation_path = plot_path+"\Validation\\"
if not os.path.exists(validation_path):
	os.makedirs(validation_path)
visualization_path = plot_path+"\Visualization\\"
if not os.path.exists(visualization_path):
	os.makedirs(visualization_path)	
	
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Block 1 - Data Imports

print("Starting to Import Data...")
# Importing Signal and Background File Names
Signal_List, Background_List = module_import('Data_File', 'Signal_List', 'Background_List', Type='Data')
print("\nFinished Defining Data")
print("---"*42)
# Importing Features
print("Specifying Features...")
Feature_List = module_import('Feature_File', 'Feature_List', Type='Feature')[0]
print("\nFinished Defining Features")
print("---"*42)
# Import CSV File For Each Defined Data
print("Loading in Signal Data: " + str(format(len(Signal_List), ',d')) + " File/s\n")
Signal_Data = data_import(Signal_List, Feature_List, Type = "Signal")
print("\nFinished Loading Signal Data")
print("---"*42)
print("Loading in Background Data: " + str(format(len(Background_List), ',d')) + " File/s\n")
Background_Data = data_import(Background_List, Feature_List, Type = "Background")
print("\nFinished Loading Background Data")
# Make Complete Dataframe
try:
	signal_df = pd.concat(Signal_Data)
	background_df = pd.concat(Background_Data)
	data_df = pd.concat([signal_df, background_df], ignore_index=True)
except Exception:
	print("Error: This line should never be seen. There is something wrong with the concat of the dataframes.")
print("\nData Loading Completed")
print("---"*42)
# Summary For Total Number Of Events
print("Number Of Events To Work With")
print("Total Signal Events: " + str(format(len(signal_df), ',d')))
print("Total Background Events: " + str(format(len(background_df), ',d')))
print("---"*42)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Block 2 - Defining variables and splitting data

print("Defining dependent and independent variables...")
# Define dependent and indepent variables and splitting
X = data_df[Feature_List] # Indepent variables
y = data_df.Signal_Indicator # Dependent Variables
print("Splitting Data for Training and Testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=random_seeds[0])
print("\nUsing " + str(format((1-reserve_fraction)*(1-test_fraction)*100, '.2f')) + "% Of Events For Training: " + str(format(len(X_train), ',.0f')))
print("\nUsing " + str(format((1-reserve_fraction)*test_fraction*100, '.2f')) + "% Of Events For Testing: " + str(format(len(X_test), ',.0f')))
print("\nData Has Been Split...")
print("---"*42)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Block 3.0 - Gridsearch

# specify parameters and distributions to sample from
parameters = {
	"loss":["deviance"],
	"min_samples_split": np.linspace(0.05, 0.2, 4),
	"min_samples_leaf": np.linspace(0.05, 0.2, 4),
	"max_depth": range(2,6),
	"criterion": ["friedman_mse"],
	"subsample":[0.3, 0.5, 0.8, 0.9, 1.0],
	"learning_rate": [0.01, 0.05, 0.1],
	"n_estimators": np.arange(50, 300, 50)
	}


# run randomized search
print("Defining the Gridsearch")
clf = GridSearchCV(GradientBoostingClassifier(), parameters, scoring='roc_auc', cv=3, n_jobs=4, verbose=2)
print("Training on all gridpoints...")
clf.fit(X_train, y_train)
print('Gradient boosting trees best params:', clf.best_params_)
print('Gradient boosting trees score:', clf.best_score_)
