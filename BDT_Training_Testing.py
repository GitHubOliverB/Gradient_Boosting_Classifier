#!/usr/bin/python
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Reference for the code of this script
# Block 0 - Library Imports
# 	Block 0.1 - Defining functions and globals...
#	Block 0.2 - Paths... 
# Block 1.0 - Data Imports
# Block 2.0 - Plotting Data
# Block 3.0 - Splitting Data
# Block 4.0 - Training and Testing
# Block 5.0 - Validation
# Block 6.0 - Plotting Probability On Data
# Block X.0 - Unsed code
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Block 0 - Library Imports

# Basic Python Libs
import os
import sys
import importlib
import csv
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, learning_curve
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.externals import joblib
np.set_printoptions(threshold=sys.maxsize)

# Own Imports
from Data_Feature_Imports import *
from Correlation_Plots import *
from Classifier_Validation_Plots import *
from TwoD_Signal_Proba_Plots import *
from Parameters import *
from GBC_Maker import *
from Classifier_Trainer import *


#class SignalEmpty(Exception):
#	pass

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
# Block 2 - Plotting Features

print("Start To Create Plots")

# Boxplots
print("\nBoxplot...")
#data_df.boxplot(by='Background_Indicator', column=Feature_List, sym='x', return_type='axes', layout=(len(Feature_List), 1))
#plt.show()

# Correlation Matrix (Second Version below)
# Signal
print("\nCorrelation Matrix For Signal...")
corr = signal_df.drop(['Signal_Indicator', 'Background_Indicator'], 1).corr()
corr = pd.melt(corr.reset_index(), id_vars='index')
corr.columns = ['x', 'y', 'value']
plot_correlation_matrix(x=corr['x'],y=corr['y'],size=corr['value'].abs(),color=corr['value'], title="Correlation Matrix For Signal", figurename=correlation_path+"\Signal_Correlation_Matrix.png")
# Background
print("\nCorrelation Matrix For Background...")
corr = background_df.drop(['Signal_Indicator', 'Background_Indicator'], 1).corr()
corr = pd.melt(corr.reset_index(), id_vars='index')
corr.columns = ['x', 'y', 'value']
plot_correlation_matrix(x=corr['x'],y=corr['y'],size=corr['value'].abs(),color=corr['value'], title="Correlation Matrix For Background", figurename=correlation_path+"\Background_Correlation_Matrix.png")

# Signal
#print("\nCorrelation Matrix For Signal...")
#correlations(signal_df.drop(['Signal_Indicator', 'Background_Indicator'], 1))
#plt.show()
# Background
#print("\nCorrelation Matrix For Background...")
#correlations(background_df.drop(['Signal_Indicator', 'Background_Indicator'], 1))
#plt.show()
print("\nFinished All Plots")
print("---"*42)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Block 3 - Splitting Data

print("Splitting Data for Training and Testing...")
# Define dependent and indepent variables and splitting
X = data_df[Feature_List] # Indepent variables
y = data_df.Signal_Indicator # Dependent Variables
# Split
X_dev, X_eval, y_dev, y_eval = train_test_split(X, y, test_size=reserve_fraction, random_state=42)
print("\nReserving " + str(format(reserve_fraction*100, '.2f')) + "% Of Events For Later Use: " + str(format(len(X_eval), ',.0f')))
print("\nUsing " + str(format((1-reserve_fraction)*100, '.2f')) + "% Of Events For Splitting: " + str(format(len(X_dev), ',.0f')))
X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=test_fraction, random_state=random_seeds[0])
print("\nUsing " + str(format((1-reserve_fraction)*(1-test_fraction)*100, '.2f')) + "% Of Events For Training: " + str(format(len(X_train), ',.0f')))
print("\nUsing " + str(format((1-reserve_fraction)*test_fraction*100, '.2f')) + "% Of Events For Testing: " + str(format(len(X_test), ',.0f')))
print("\nData Has Been Split...")
print("---"*42)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Block 4.0 - Training, Testing and Overfitting

for i, clf in enumerate(classifier_name):
	GB_classifier_maker(classifier_name, max_depth=max_depth, min_samples_leaf=min_samples_leaf, subsample=subsample, n_estimators=n_estimators, learning_rate=learning_rate)

print(GB_clfs[classifier_name[0]])
# Defining Classifier and Parameters
print("Setting Classifier and Parameters...")
print("\nInitializing Crosstraining...")
clfs=[]    
print("\nCrosstraining - 0")
classifier_training(X_train, y_train, X_test, y_test, clfs, 0, GB_clfs[classifier_name[0]], model_path, tree_path, training_path, True)
print("---"*42)
print("Crosstraining - 1")
classifier_training(X_train, y_train, X_test, y_test,clfs,  1, GB_clfs[classifier_name[0]], model_path, tree_path, training_path, False)
print("\nFinished Training...")
print("---"*42)
clfs[0].set_params(verbose=False)
clfs[1].set_params(verbose=False)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Block 5.0 - Validation

print("Plotting Validation For Number Of Trees...")
Ntree_ROC_Curve_Figname = validation_path + str(classifier_name[0]) + "_Tree_" + str(len(clfs[0].estimators_)) + "_ROC.png"
plot_Ntree_ROC_curve(clfs[0], (X_train,y_train),(X_test,y_test), "GBT_Classifier", n_estimators, Ntree_ROC_Curve_Figname)
Ntree_PR_Curve_Figname = validation_path + str(classifier_name[0]) + "_Tree_" + str(len(clfs[0].estimators_)) + "_PR.png"
plot_Ntree_PR_curve(clfs[0], (X_train,y_train),(X_test,y_test), "GBT_Classifier", n_estimators, Ntree_PR_Curve_Figname)

# Reference For Some Of These Numbers
# P - condition positive, the number of real positive cases in the data
# N - condition negative, the number of real negative cases in the data

# TP - true positive, eqv. with hit
# TN - true negative,  eqv. with correct rejection
# FP - false positive, eqv. with false alarm, Type I error
# FN - false negative,  eqv. with miss, Type II error

# PPV - Precision = TP / (TP + FP) = 1 - FDR, positive predictive value
# TPR - Recall = TP / (TP + FN) = 1 - FNR, true positive rate
# FNR - Miss Rate = FN / (FN + TP) = 1 - TPR, false negative rate
# FDR - False Discovery Rate = FP / (FP + TP)
# ACC - Accuracy = (TP + TN) / (TP + TN + FP + FN)
# F betta - F betta Score = (1+ betta**2) * ( PPV * TPR) / (betta**2 * PPV + TPR), is the harmonic mean of precision and recall

# If I ever want to plot multiple clfs
#fig, axes = plt.subplots(nrows=len(clfs), sharex=True)
#for clf, ax in zip(clfs, axes):  
#	plot_learning_curve(clf, "Learning curves", X_dev, y_dev, scoring='roc_auc', n_jobs=7, cv=4, ax=ax, xlabel=False)
#axes[0].legend(loc="best")
#axes[-1].set_xlabel("Training examples")

print("Plotting Performance vs. Size Of Training Set...")
fig, axis = plt.subplots(nrows=1, sharex=True)
plot_learning_curve(clfs[0], "Learning curves", X_dev, y_dev, scoring='roc_auc', n_jobs=7, cv=10, ax=axis, xlabel=False)
axis.legend(loc="best")
axis.set_xlabel("Training examples")
fig_name = validation_path + str(classifier_name[0]) + "_Score_Validation" + ".png"
plt.savefig(fig_name)
plt.close()
print("\nFinished Validation...")
print("---"*42)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Block 6.0 - Plotting Probability On Data

print("\nFinish things up here with some nice plots for you to look at :).")
models = [('GBT_Classifier_Crosstraining_0', clfs[0]),
		  ('GBT_Classifier_Crosstraining_1', clfs[1])]
if len(Important_Features_0) == 2:
	Sym_Min = data_df[[Important_Features_0[0], Important_Features_0[1]]].min()
	Sym_Max = data_df[[Important_Features_0[0], Important_Features_0[1]]].max()
Sym_Min = (Sym_Min-abs(0.10*Sym_Min)).min()
Sym_Max = (Sym_Max+abs(0.10*Sym_Max)).max()
for mname, model in models:
	if mname == 'GBT_Classifier_Crosstraining_0':
		Important_Feature_List = Important_Features_0
	elif mname == 'GBT_Classifier_Crosstraining_1':
		Important_Feature_List = Important_Features_1
	plt.figure(figsize=(12,12))
	Signal_Proba_Data_Figname = visualization_path + str(mname) + "_Visualization_Signal_Probabilty_XY.png"
	Signal_Probabilty_XY_Space(mname, model, signal_df, background_df, Sym_Min, Sym_Max, Important_Feature_List, True, Signal_Proba_Data_Figname)
	
	plt.figure(figsize=(12,12))
	Signal_Proba_No_Data_Figname = visualization_path + str(mname) + "_Visualization_Signal_Probabilty_XY_Background_No_Data.png"
	Signal_Probabilty_XY_Space(mname, model, signal_df, background_df, Sym_Min, Sym_Max, Important_Feature_List, False, Signal_Proba_No_Data_Figname)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Block X.0 - Unsed code
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax	