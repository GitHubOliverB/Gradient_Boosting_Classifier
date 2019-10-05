#!/usr/bin/python

"""
Summary:

Two functions to evaluate the number of trees using AUC-ROC/PR as score.
One Function to evalute the training sample size.

Inputs for plot_Ntree_..._curve:									
classifier								- The classifier you want to train. See GBC_Maker file.
train, test								- (X_train,y_train), (X_test,y_test) ; Train and Test sample from splitting.
fig_name								- Name for the png.

Inputs for plot_learning_curve:									
estimator								- Classifier which was used before.
title 									- Title for the plot.
X, y,									- X and y which has to contain more examples than training sample.
cv=None									- k-fold cross-validation. k=10 is recommended for enough data. 
n_jobs=1								- Cores to use.
train_sizes=np.linspace(.1, 1.0, 10)	- Sets 10 equally spaced training samples in terms of size.
scoring='roc_auc'						- Define a score to use. Default is 'roc_auc'.
ax=None									- axis for figure
xlabel=True								- Left this one in if you want to stack plots.
"""

import os
import sys
import importlib
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.externals.six import StringIO  

from Parameters import *

def plot_Ntree_ROC_curve(classifier, train, test, fig_name, n_estimators=n_estimators):
	X_test, y_test = test
	X_train, y_train = train
	#fig = plt.figure(figsize=(12, 8))
	fig, ax = plt.subplots(figsize=(12, 8))
	test_score = np.empty(len(classifier.estimators_))
	train_score = np.empty(len(classifier.estimators_))
	for i, pred in enumerate(classifier.staged_predict_proba(X_test)):
		test_score[i] = 1-roc_auc_score(y_test, pred[:,1])
	for i, pred in enumerate(classifier.staged_predict_proba(X_train)):
		train_score[i] = 1-roc_auc_score(y_train, pred[:,1])
	best_iter = np.argmin(test_score)
	Cutoff_p1_index = np.argmax(test_score < np.min(test_score)/0.99)
	print("Highest ROC-AUC At Tree: %i" %(best_iter+1))
	if best_iter < 0.9*n_estimators:
		print("WARNING: wasting computing power. Lowering the number of trees by at least 10% is adviced!")
	#learn = classifier.get_params()['learning_rate']
	#depth = classifier.get_params()['max_depth']
	test_line = plt.plot(test_score, label='Testing')
	colour = test_line[-1].get_color()
	#plt.plot(train_score, '--', color=colour, label='learn=%.3f depth=%i - Crosstraining - %i (train)'%(learn,depth, n))       
	plt.plot(train_score, '--', color=colour, label='Training')       
	plt.xlabel("Number of boosting iterations")
	plt.ylabel("1 - Area Under ROC-Curve")
	plt.axvline(x=best_iter, color=colour)   
	plt.text(x=best_iter, y=(np.max(test_score)+np.min(test_score))/2.2, s='best iteration')
	plt.axvline(x=Cutoff_p1_index, color=colour)   
	plt.text(x=Cutoff_p1_index, y=(np.max(test_score)+np.min(test_score))/2, s='1% Cutoff')
	plt.legend(loc='best')
	plt.grid()
	plt.savefig(fig_name)
	plt.close()
	
def plot_Ntree_PR_curve(classifier, train, test, fig_name, n_estimators=n_estimators):
	X_test, y_test = test
	X_train, y_train = train
	fig = plt.figure(figsize=(12, 8))
	test_score = np.empty(len(classifier.estimators_))
	train_score = np.empty(len(classifier.estimators_))
	for i, pred in enumerate(classifier.staged_predict_proba(X_test)):
		precision, recall, thresholds_test = precision_recall_curve(np.array(y_test), pred[:,1])
		test_score[i] = 1-auc(recall, precision)
	for i, pred in enumerate(classifier.staged_predict_proba(X_train)):
		precision, recall, thresholds_test = precision_recall_curve(np.array(y_train), pred[:,1])
		train_score[i] = 1-auc(recall, precision)
	best_iter = np.argmin(test_score)
	Cutoff_p1_index = np.argmax(test_score < np.min(test_score)/0.99)
	print("Highest PR-AUC At Tree: %i" %(best_iter+1))
	if best_iter < 0.9*n_estimators:
			print("WARNING: wasting computing power. Lowering the number of trees by at least 10% is adviced!")
	#learn = classifier.get_params()['learning_rate']
	#depth = classifier.get_params()['max_depth']
	test_line = plt.plot(test_score, label='Testing')
	colour = test_line[-1].get_color()
	plt.plot(train_score, '--', color=colour, label='Training')       
	plt.xlabel("Number of boosting iterations")
	plt.ylabel("1 - Area Under PR-Curve")
	plt.axvline(x=best_iter, color=colour)   
	plt.text(x=best_iter, y=(np.max(test_score)+np.min(test_score))/2.2, s='best iteration')
	plt.axvline(x=Cutoff_p1_index, color=colour)   
	plt.text(x=Cutoff_p1_index, y=(np.max(test_score)+np.min(test_score))/2, s='1% Cutoff')		
	plt.legend(loc='best')
	plt.grid()
	plt.savefig(fig_name)
	plt.close()

def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10), scoring='roc_auc', ax=None, xlabel=True):
	# Check is ax is defined
	if ax is None:
		plt.figure()
		ax.title(title)    
	# Check if label is defined
	if xlabel:
		ax.set_xlabel("Training examples")        
	ax.set_ylabel("Area Under ROC-Curve")
	# Learning Curve computation
	train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring, verbose=1)
	# Mean And Standard Deviation For Training and Testing Scores
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	Score_Max = np.max([train_scores_mean,test_scores_mean])
	Score_Min = np.min([train_scores_mean,test_scores_mean])
	# Fill Plot and return
	ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
	ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
	ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training AUC-ROC")
	ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Testing AUC-ROC")
	ax.set_ylim([Score_Min*0.995, Score_Max/0.995])
	ax.grid()
	return plt
	
def plot_Ntree_ROC_curves(classifiers, train, test, fig_name, n_estimators=n_estimators):
	X_test, y_test = test
	X_train, y_train = train
	#fig = plt.figure(figsize=(12, 8))
	fig, ax = plt.subplots(figsize=(12, 8))
	for n, classifier in enumerate(classifiers):
		test_score = np.empty(len(classifier.estimators_))
		train_score = np.empty(len(classifier.estimators_))
		if n == 0:
			print("Validation For Crosstraining - 0")
			for i, pred in enumerate(classifier.staged_predict_proba(X_test)):
				test_score[i] = 1-roc_auc_score(y_test, pred[:,1])
			for i, pred in enumerate(classifier.staged_predict_proba(X_train)):
				train_score[i] = 1-roc_auc_score(y_train, pred[:,1])
		elif n == 1:
			print("\nValidation For Crosstraining - 1")
			for i, pred in enumerate(classifier.staged_predict_proba(X_test)):
				train_score[i] = 1-roc_auc_score(y_test, pred[:,1])
			for i, pred in enumerate(classifier.staged_predict_proba(X_train)):
				test_score[i] = 1-roc_auc_score(y_train, pred[:,1])
		best_iter = np.argmin(test_score)
		Cutoff_p1_index = np.argmax(test_score < np.min(test_score)/0.99)
		print("Highest ROC-AUC At Tree: %i" %(best_iter+1))
		if best_iter < 0.9*n_estimators:
			print("WARNING: wasting computing power. Lowering the number of trees by at least 10% is adviced!")
		#learn = classifier.get_params()['learning_rate']
		#depth = classifier.get_params()['max_depth']
		test_line = plt.plot(test_score, label='Crosstraining - %i (test)'%(n))
		colour = test_line[-1].get_color()
		#plt.plot(train_score, '--', color=colour, label='learn=%.3f depth=%i - Crosstraining - %i (train)'%(learn,depth, n))       
		plt.plot(train_score, '--', color=colour, label='Crosstraining - %i (train)'%(n))       
		plt.xlabel("Number of boosting iterations")
		plt.ylabel("1 - Area Under ROC-Curve")
		plt.axvline(x=best_iter, color=colour)   
		plt.text(x=best_iter, y=(np.max(test_score)+np.min(test_score))/2.2, s='best iteration')
		plt.axvline(x=Cutoff_p1_index, color=colour)   
		plt.text(x=Cutoff_p1_index, y=(np.max(test_score)+np.min(test_score))/2, s='1% Cutoff')
	plt.legend(loc='best')
	plt.grid()
	plt.savefig(fig_name)
	plt.close()
	
def plot_Ntree_PR_curves(classifiers, train, test, fig_name, n_estimators=n_estimators):
	X_test, y_test = test
	X_train, y_train = train
	fig = plt.figure(figsize=(12, 8))
	for n, classifier in enumerate(classifiers):
		test_score = np.empty(len(classifier.estimators_))
		train_score = np.empty(len(classifier.estimators_))
		if n == 0:
			print("\nValidation For Crosstraining - 0")
			for i, pred in enumerate(classifier.staged_predict_proba(X_test)):
				precision, recall, thresholds_test = precision_recall_curve(np.array(y_test), pred[:,1])
				test_score[i] = 1-auc(recall, precision)
			for i, pred in enumerate(classifier.staged_predict_proba(X_train)):
				precision, recall, thresholds_test = precision_recall_curve(np.array(y_train), pred[:,1])
				train_score[i] = 1-auc(recall, precision)
		elif n == 1:
			print("\nValidation For Crosstraining - 1")
			for i, pred in enumerate(classifier.staged_predict_proba(X_test)):
				precision, recall, thresholds_test = precision_recall_curve(np.array(y_test), pred[:,1])
				train_score[i] = 1-auc(recall, precision)
			for i, pred in enumerate(classifier.staged_predict_proba(X_train)):
				precision, recall, thresholds_test = precision_recall_curve(np.array(y_train), pred[:,1])
				test_score[i] = 1-auc(recall, precision)
		best_iter = np.argmin(test_score)
		Cutoff_p1_index = np.argmax(test_score < np.min(test_score)/0.99)
		print("Highest PR-AUC At Tree: %i" %(best_iter+1))
		if best_iter < 0.9*n_estimators:
			print("WARNING: wasting computing power. Lowering the number of trees by at least 10% is adviced!")
		#learn = classifier.get_params()['learning_rate']
		#depth = classifier.get_params()['max_depth']
		test_line = plt.plot(test_score, label='Crosstraining - %i (test)'%(n))
		colour = test_line[-1].get_color()
		plt.plot(train_score, '--', color=colour, label='Crosstraining - %i (train)'%(n))       
		plt.xlabel("Number of boosting iterations")
		plt.ylabel("1 - Area Under PR-Curve")
		plt.axvline(x=best_iter, color=colour)   
		plt.text(x=best_iter, y=(np.max(test_score)+np.min(test_score))/2.2, s='best iteration')
		plt.axvline(x=Cutoff_p1_index, color=colour)   
		plt.text(x=Cutoff_p1_index, y=(np.max(test_score)+np.min(test_score))/2, s='1% Cutoff')		
	plt.legend(loc='best')
	plt.grid()
	plt.savefig(fig_name)
	plt.close()