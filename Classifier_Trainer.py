#!/usr/bin/python

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Reference for the code of this script
# Block 0 - Library Imports
# Block 1.0 - Training Function

"""
Summary:

This script is the heart of our code. Here the given classifier will be trained by the defined function classifier_training.

Inputs:									
X_train, y_train, X_test, y_test		- You split your sample into a training and testing sample. X contains all the independent features, while y will be the dependent feature (class).
clfs									- This is just an empty list which will be filled for each run
cross_index								- Crosstraining, either 0 or 1 to switch training and testing sample.
classifier								- The classifier you want to train. See GBC_Maker file
model_path, tree_path, training_path	- Paths which are defined in main(BDT_Training_Testing).
dt_safe									- Toggles the saving of all decision trees as png on/off. Default is False(off).

Features:
1) Trains a classifier on the specified training or testing sample (depending on cross_index).
2) Can save decision trees as png.
3) Prints and plots variable importance with standard deviation.
4) Using the testing(training) sample, it predicts the signal probability and prints/plots:
	i)  	ROC-Curve and AUC-ROC
	ii) 	Precision-Recall-Curve and AUC of P-R
	iii)	Signal Probability for the training and testing sample. Furthermore does a Kolmogorov-Smirnov test
			which compares the two distributes and answers the question if both follow a common distribution.
			This is the cruical over/undertraining test.
	iv)		Furthermore we compare 3 values of the ROC-Curve which also acts as an indicator for over/undertraining.
	v)		Save trained models.
"""



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Block 0 - Library Imports

# Basic Python Libs
import os
import sys
import importlib
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import pydotplus
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_curve
import joblib
np.set_printoptions(threshold=sys.maxsize)

from Parameters import *
from Feature_File import *

background_efficiencies = [0.01, 0.10, 0.30]
Important_Features_0 = []
Important_Features_1 = []

def classifier_training(X_train, y_train, X_test, y_test, clfs, cross_index, classifier, model_path, tree_path, training_path, dt_safe=False):
	if cross_index == 0:
		R = classifier.fit(X_train, y_train)
	elif cross_index == 1:
		R = classifier.fit(X_test, y_test)
	else:
		print("Wrong cross_index set.")
		exit()
	# Save Trained Model
	print("Safing trained model...")
	Model_File_Name = model_path + str(classifier_name[0]) + "_Crosstraining_" + str(cross_index) + "_Tree_" + str(len(classifier.estimators_))
	joblib.dump(R, Model_File_Name)
	clfs.append(classifier)
	# Save Decision Trees
	if dt_safe == True:
		print("\nSaving Decision Trees...")
		for i, estimator in enumerate(classifier.estimators_):
			if (i+1) % (n_estimators/10) == 0:
				print(i+1)
			sub_tree = classifier.estimators_[i, 0]
			dot_data = export_graphviz(sub_tree, out_file=None, feature_names=Feature_List, filled=True, rounded=True, special_characters=True, proportion=True)
			# Draw graph
			graph = pydotplus.graph_from_dot_data(dot_data)  
			# Save graph
			graph.write_png(tree_path + str(classifier_name[0]) + "_" + str(cross_index) + "_Decision_Tree_" + str(i) + ".png")
	else:
		print("Decision Trees can be saved as a PNG if wanted. Change Input to True.") 
	classifier_Importance = classifier.feature_importances_
	classifier_Importance_std = np.std([tree[0].feature_importances_ for tree in classifier.estimators_], axis=0, ddof=0)
	classifier_Importance_Indices = np.argsort(classifier_Importance)[::-1]
	classifier_Feature_Indices = [Feature_List[i] for i in classifier_Importance_Indices]
	# Print the feature ranking
	print("\nFeature Ranking By Variable Importance")
	print("\nRank	Variable		Variable Importance (+/- STD)")
	print("---------------------------------------------------------------")
	for i, j in enumerate(Feature_List):
		print("   " + str(i+1) + "	" + str(Feature_List[classifier_Importance_Indices[i]]) + "			" + str(round(classifier.feature_importances_[classifier_Importance_Indices][i], 4)) + " +/- " + str(round(classifier_Importance_std[classifier_Importance_Indices][i], 4)))
		if cross_index == 0:
			Important_Features_0.append(Feature_List[classifier_Importance_Indices[i]])
		elif cross_index == 1:
			Important_Features_1.append(Feature_List[classifier_Importance_Indices[i]])
	# Plot the feature importances of the forest
	fig = plt.figure()
	ax = fig.add_subplot(111)
	N = int(len(classifier_Feature_Indices))
	ind = np.arange(N)  # the x locations for the groups
	plt.bar(ind, classifier_Importance[classifier_Importance_Indices],
		color="g", yerr=classifier_Importance_std[classifier_Importance_Indices], align="center")
	plt.xticks(range(len(classifier_Feature_Indices)), classifier_Feature_Indices)
	plt.xlim([-1, len(Feature_List)])
	ax.set_ylabel('Variable Importance')
	plt.title(str(classifier_name[0]) + " Crosstraining - " + str(cross_index))
	fig_name = training_path + str(classifier_name[0]) + "_Crosstraining_" + str(cross_index) + "_Tree_" + str(len(classifier.estimators_)) + "_Feature_Ranking.png"
	plt.savefig(fig_name)
	plt.close()
	# Testing on the other data set
	if cross_index == 0:
		y_predicted = classifier.predict(X_test)
	elif cross_index == 1:
		y_predicted = classifier.predict(X_train)
	# Plotting Test Results
	eff_index = []
	# Signal Probabilities
	Predicted_Proba = []
	Predicted_Data_Proba = []
	for X,y in ((X_train, y_train), (X_test, y_test)):
		Signal_Signal_Proba = classifier.predict_proba(X[y > 0.5])[:,1]
		Background_Signal_Proba = classifier.predict_proba(X[y < 0.5])[:,1]
		Data_Signal_Proba = classifier.predict_proba(X)[:,1]
		Predicted_Proba += [Signal_Signal_Proba, Background_Signal_Proba]
		Predicted_Data_Proba += [Data_Signal_Proba]
	predict_proba_train = Predicted_Data_Proba[0]
	predict_proba_test = Predicted_Data_Proba[1]
	# Compute ROC curve and area under the curve
	if cross_index == 0:
		print("\nClassification For Crosstraining - 0:")
		print(classification_report(y_test, y_predicted, target_names=["background", "signal"]))
		print("Confusion Matrix on Testing Set:")
		print(confusion_matrix(y_test, y_predicted))
		print("\nArea under ROC curve: %.4f"%(roc_auc_score(y_test, predict_proba_test)))
		fpr, tpr, thresholds = roc_curve(y_test, predict_proba_test)
		fpr_train, tpr_train, thresholds_train = roc_curve(y_train, predict_proba_train)
		roc_auc = auc(fpr, tpr)
	elif cross_index == 1:
		print("\nClassification For Crosstraining - 1:")
		print(classification_report(y_train, y_predicted, target_names=["background", "signal"]))
		print("Confusion Matrix on Training Set:")
		print(confusion_matrix(y_train, y_predicted))
		print("\nArea under ROC curve: %.4f"%(roc_auc_score(y_train, predict_proba_train)))
		# Compute ROC curve and area under the curve
		fpr, tpr, thresholds = roc_curve(y_train, predict_proba_train)
		fpr_train, tpr_train, thresholds_train = roc_curve(y_test, predict_proba_test)
		roc_auc = auc(fpr, tpr)
	for eff in background_efficiencies:
		index = np.searchsorted(fpr, eff)
		index_train = np.searchsorted(fpr_train, eff)
		eff_index.extend((index, index_train))
	# Plot ROC-Curve
	fig = plt.figure(figsize=(12, 8))
	plt.plot(tpr, (1-fpr), lw=1, label='ROC-AUC = %0.3f'%(roc_auc))
	plt.plot([1, 0], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.ylabel('Background Rejection')
	plt.xlabel('Signal Efficiency')
	plt.title("Receiver Operating Characteristic - Crosstraining - " + str(cross_index))
	plt.legend(loc="lower left")
	plt.grid()
	fig_name = training_path  + str(classifier_name[0]) + "_Crosstraining_" + str(cross_index) + "_Tree_" + str(len(classifier.estimators_)) + "_ROC_Curve.png"
	plt.savefig(fig_name)
	plt.close()
	# Precision-Recall-Curve and AUC of P-R
	if cross_index == 0:
		precision, recall, thresholds_test = precision_recall_curve(np.array(y_test), predict_proba_test)
	elif cross_index == 1:
		precision, recall, thresholds_train = precision_recall_curve(np.array(y_train), predict_proba_train)
	AUC_PR = auc(recall, precision)
	print("Area under PR curve:  %0.4f"%(AUC_PR))
	# Plot Precision-Recall-Curve
	fig = plt.figure(figsize=(12, 8))
	plt.plot(recall, precision, lw=1, label='PR-AUC = %0.3f'%(AUC_PR))
	if cross_index == 0:
		plt.plot([0, 1], [len(y_test[y_test == 1])/len(y_test), len(y_test[y_test == 1])/len(y_test)], '--', color=(0.6, 0.6, 0.6), label='Luck')
	elif cross_index == 1:	
		plt.plot([0, 1], [len(y_train[y_train == 1])/len(y_train), len(y_train[y_train == 1])/len(y_train)], '--', color=(0.6, 0.6, 0.6), label='Luck')
	plt.xlim([- 0.05, 1.05])
	plt.ylim([(len(y_train[y_train == 1])/len(y_train)-0.15), 1.05])
	plt.ylabel('Precision')
	plt.xlabel('Recall')
	plt.title("Precision-Recall-Curve Crosstraining  - " + str(cross_index))
	plt.legend(loc="lower left")
	plt.grid()
	fig_name = training_path  + str(classifier_name[0]) + "_Crosstraining_" + str(cross_index) + "_Tree_" + str(len(classifier.estimators_)) + "_PR_Curve.png"
	plt.savefig(fig_name)
	plt.close()
	# OVER AND UNVERTRAINING TESTS
	print("\nOver/Underfitting Test:\n")
	# Efficiency comparision on three points between testing and training
	print("Testing signal efficiency compared to (training signal efficiency)")
	print("\n@B=0.01			@B=0.10			@B=0.30")
	print("-------------------------------------------------------------------------")
	print(str(round(tpr[eff_index[0]], 3)) + " ("+ str(round(tpr_train[eff_index[1]], 3)) + ")" + "		" + str(round(tpr[eff_index[2]], 3)) + " ("+ str(round(tpr_train[eff_index[3]], 3)) + ")" + "		" + str(round(tpr[eff_index[4]], 3)) + " ("+ str(round(tpr_train[eff_index[5]], 3)) + ")")
	if abs(round((tpr[eff_index[0]] - tpr_train[eff_index[1]]), 3)) > 0.03:
		print("WARNING: The difference between training and testing efficiency @B=.01 is a bit high.")
		print("The model may be a bit over/undertrained!")
	if abs(round((tpr[eff_index[2]] - tpr_train[eff_index[3]]), 3)) > 0.01:
		print("WARNING: The difference between training and testing efficiency @B=.1 is a bit high.")
		print("The model may be a bit over/undertrained!")
	if abs(round(tpr[eff_index[4]] - tpr_train[eff_index[5]], 3)) > 0.005:
		print("WARNING: The difference between training and testing efficiency @B=.3 is a bit high.")
		print("The model may be a bit over/undertrained!")
	# 2-Sample Kolmogorov-Smirnov-Test
	Signal_KS = stats.ks_2samp(Predicted_Proba[0], Predicted_Proba[2])
	Background_KS = stats.ks_2samp(Predicted_Proba[1], Predicted_Proba[3])
	print("\nComparing BDT Output Distribution for Training and Testing Set")
	print("\n		Signal			Background")
	print("-------------------------------------------------------------------------")
	print("KS p-value 	" + str(round(Signal_KS[1],3)) + "			" + str(round(Background_KS[1],3)))
	print("\nBased on the p-value of the Kolmogorov-Smirnov Test...")
	if round(Signal_KS[1],3) > 0.05:
		print("there seems to be no Over/Undertraining present for the Signal-Distribustions.")
	else:
		print("WARNING: Over/Undertraining present for the Signal-Distribustions!")
	if round(Background_KS[1],3) > 0.05:
		print("\nthere seems to be no Over/Undertraining present for the Background-Distribustions.")
	else:
		print("\nWARNING: Over/Undertraining present for the Background-Distribustions!")
	Proba_Min = min(np.min(p) for p in Predicted_Proba)
	Proba_Max = max(np.max(p) for p in Predicted_Proba)
	Proba_Min_Proba_Max = (Proba_Min,Proba_Max)
	bins=30
	fig = plt.figure(figsize=(12, 8))
	plt.hist(Predicted_Proba[0],
			color='red', alpha=0.5, range=Proba_Min_Proba_Max, bins=bins,
			histtype='stepfilled', density=True,
			label='S (train)')
	plt.hist(Predicted_Proba[1],
			color='blue', alpha=0.5, range=Proba_Min_Proba_Max, bins=bins,
			histtype='stepfilled', density=True,
			label='B (train)')
	hist, bins = np.histogram(Predicted_Proba[2],
								bins=bins, range=Proba_Min_Proba_Max, density=True)
	scale = len(Predicted_Proba[2]) / sum(hist)
	err = np.sqrt(hist * scale) / scale
	width = (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	plt.errorbar(center, hist, yerr=err, fmt='o', c='red', label='S (test)')   
	hist, bins = np.histogram(Predicted_Proba[3],
								bins=bins, range=Proba_Min_Proba_Max, density=True)
	scale = len(Predicted_Proba[2]) / sum(hist)
	err = np.sqrt(hist * scale) / scale
	plt.errorbar(center, hist, yerr=err, fmt='o', c='blue', label='B (test)')
	plt.xlabel("BDT Output - Signal Class Probability")
	plt.ylabel("Arbitrary units")
	plt.plot([], [], ' ', label='KS p-value S(B): ' + str(round(Signal_KS[1],3)) + ' (' + str(round(Background_KS[1],3)) + ')')
	plt.legend(loc='best')
	fig_name = training_path + str(classifier_name[0]) + "_Crosstraining_" + str(cross_index) + "_Tree_" + str(len(classifier.estimators_)) + "_BDT_Output_Prob.png"
	plt.savefig(fig_name)
	plt.close()