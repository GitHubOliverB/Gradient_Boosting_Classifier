#!/usr/bin/python

import os
import sys
import importlib
import csv
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

#	Colors for plot_correlation_matrix, see Block 2.0 - Plotting Data
n_colors = 256 # Use 256 colors for the diverging color palette
palette = sns.diverging_palette(h_neg=15, h_pos=260, s=90, l=50, n=n_colors) # Create the palette
color_min, color_max = [-1, 1] # Range of values that will be mapped to the palette, i.e. min and max possible correlation
def value_to_color(val):
	val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
	ind = int(val_position * (n_colors - 1)) # target index in the color palette
	return palette[ind]

def plot_correlation_matrix(x, y, size, color, title, figurename):
	plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1)
	fig = plt.figure(figsize=(12, 10))
	ax = plt.subplot(plot_grid[:,:13])    
	# Mapping from column names to integer coordinates
	X_labels = [v for v in sorted(x.unique())]
	y_labels = [v for v in sorted(y.unique())]
	X_to_num = {p[1]:p[0] for p in enumerate(X_labels)} 
	y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
	# Fill in Scatter squares
	size_scale = 250000/(len(X_labels)**2)
	ax.scatter(x=x.map(X_to_num),y=y.map(y_to_num),s=size * size_scale,c=color.apply(value_to_color),marker='s')   
	# Show column labels on the axes
	ax.set_xticks([X_to_num[v] for v in X_labels])
	ax.set_xticklabels(X_labels, rotation=45, horizontalalignment='right')
	ax.set_yticks([y_to_num[v] for v in y_labels])
	ax.set_yticklabels(y_labels)
	ax.grid(False, 'major')
	ax.grid(True, 'minor')
	ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
	ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
	ax.set_xlim([-0.5, max([v for v in X_to_num.values()]) + 0.5]) 
	ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])	
	plt.title(title)
	# Add bar
	ax2 = plt.subplot(plot_grid[:,-1])
	col_x = [0]*len(palette) # Fixed x coordinate for the bars
	bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars
	bar_height = bar_y[1] - bar_y[0]
	ax2.barh(y=bar_y, width=[5]*len(palette), left=col_x,  height=bar_height, color=palette, linewidth=0)
	# Axis, Grid and Plot
	ax2.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
	ax2.grid(False) # Hide grid
	ax2.set_facecolor('white') # Make background white
	ax2.set_xticks([]) # Remove horizontal ticks
	ax2.set_yticks(np.linspace(min(bar_y), max(bar_y), 9)) # Show vertical ticks for min, middle and max
	ax2.yaxis.tick_right() # Show vertical ticks on the right 
	plt.ylabel("Correlation Coefficient")
	plt.savefig(figurename)
	plt.close()
# 	Other Correlation Matrix Plot
def plot_correlations(data, **kwds):
    """Calculate pairwise correlation between features.
    
    Extra arguments are passed on to DataFrame.corr()
    """
    # simply call df.corr() to get a table of
    # correlation values if you do not need
    # the fancy plotting
    corrmat = data.corr(**kwds)

    fig, ax1 = plt.subplots(ncols=1, figsize=(6,5))
    
    opts = {'cmap': plt.get_cmap("RdBu"),
            'vmin': -1, 'vmax': +1}
    heatmap1 = ax1.pcolor(corrmat, **opts)
    plt.colorbar(heatmap1, ax=ax1)

    ax1.set_title("Correlations")

    labels = corrmat.columns.values
    for ax in (ax1,):
        # shift location of ticks to center of the bins
        ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
        ax.set_yticklabels(labels, minor=False)
        
    plt.tight_layout()