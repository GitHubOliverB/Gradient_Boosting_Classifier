#!/usr/bin/python

"""
Summary:

You can ignore this function.
Only works if you use two features, as it plots the signal probability in a 2-dimensional space.
"""

import os
import sys
import importlib
import csv
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.externals.six import StringIO 


n_colors = 256 # Use 256 colors for the diverging color palette
palette = sns.diverging_palette(h_neg=15, h_pos=260, s=90, l=50, n=n_colors) # Create the palette
color_min, color_max = [0, 1] # Range of values that will be mapped to the palette, i.e. min and max possible correlation
my_cm = sns.diverging_palette(h_neg=15, h_pos=260, s=90, l=50, n=n_colors, center='light', as_cmap=True)

def Signal_Probabilty_XY_Space(mname, model, signal_df, background_df, Sym_Min, Sym_Max, Important_Feature_List, Data_Toggle, fig_name, gridpoints=200):
	print("Generating Grid...")
	xx, yy = np.meshgrid(np.linspace(Sym_Min, Sym_Max, gridpoints),
						np.linspace(Sym_Min, Sym_Max, gridpoints))
	print("Getting Predictions For Gridspace..")
	Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
	Z -= np.min(Z)
	print("Creating The Plot..")
	plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1)
	fig = plt.figure(figsize=(12, 10))
	ax = plt.subplot(plot_grid[:,:12]) 
	ax.pcolormesh(xx, yy, Z[:,1].reshape(xx.shape), cmap=my_cm)
	if Data_Toggle == True:
		ax.scatter(signal_df[Important_Feature_List[1]], signal_df[Important_Feature_List[0]], c='blue', marker='x', s=4, label='Signal', alpha=0.8)
		ax.scatter(background_df[Important_Feature_List[1]], background_df[Important_Feature_List[0]], c='darkred', marker='+', s=6, label='Background', alpha=0.3)
	else:
		print("Datapoints can be plotted by setting the Data_Toggle value to True.")
	plt.title("BDT Output - Signal Probability in XY-Space")
	plt.xlabel(str(Important_Feature_List[1]))
	plt.ylabel(str(Important_Feature_List[0]))
	# Add bar
	ax2 = plt.subplot(plot_grid[:,-1])
	col_x = [0]*len(palette) # Fixed x coordinate for the bars
	bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars
	bar_height = bar_y[1] - bar_y[0]
	ax2.barh(y=bar_y, width=[5]*len(palette), left=col_x,  height=bar_height, color=palette, linewidth=0)
	ax2.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
	ax2.grid(False) # Hide grid
	ax2.set_facecolor('white') # Make background white
	ax2.set_xticks([]) # Remove horizontal ticks
	ax2.set_yticks(np.linspace(min(bar_y), max(bar_y), 5)) # Show vertical ticks for min, middle and max
	ax2.yaxis.tick_right() # Show vertical ticks on the right 
	ax.legend(loc='best', title="Groups")
	plt.ylabel("Signal Probability")
	plt.savefig(fig_name)
	plt.close()