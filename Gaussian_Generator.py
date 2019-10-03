#!/usr/bin/python
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Reference for the code of this script
# Block 0 - Library Imports
# Block 1.0 - Define And Create Gaussians
# Block 2.0 - Write CSV File
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Block 0 - Library Imports

import os
import sys
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
np.set_printoptions(threshold=sys.maxsize)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Block 1.0 - Define And Create Gaussians

np.random.seed(5) # Set a random seed for reproducible
N_events = 100000 # Number of events per gaussian

# Feature List
Feature_List = ['X', 'Y']

# Signal
signal = np.random.multivariate_normal([0, 0], [[0.5, 0],[0, 0.5]], N_events)
SignalEmptydf = pd.DataFrame()
# Backgrounds
bg_1 = np.random.multivariate_normal([0.5, -1.5], [[0.1, 0],[0, 0.1]], N_events)
bg_2 = np.random.multivariate_normal([-2, 1], [[0.5, 0],[0, 1.0]], N_events)
bg_3 = np.random.multivariate_normal([3, 2], [[3, 0],[0, 2]], N_events)
bg_4 = np.random.multivariate_normal([8, 8], [[0.1, 0],[0, 0.1]], N_events)
# Turn into Dataframes
SignalEmptydf = pd.DataFrame()
Bg1Emptydf = pd.DataFrame()
Bg2Emptydf = pd.DataFrame()
Bg3Emptydf = pd.DataFrame()
Bg4Emptydf = pd.DataFrame()
for k, l in enumerate(Feature_List):
	SignalEmptydf[Feature_List[k]] = signal[:,k]
	Bg1Emptydf[Feature_List[k]] = bg_1[:,k]
	Bg2Emptydf[Feature_List[k]] = bg_2[:,k]
	Bg3Emptydf[Feature_List[k]] = bg_3[:,k]
	Bg4Emptydf[Feature_List[k]] = bg_4[:,k]

# Let's make a quick scatter plot
fig, ax = plt.subplots(figsize=(10, 8))
# Add signal and background events
ax.scatter(signal[:,0], signal[:,1],c='blue', marker='x', s=10, label='Signal', alpha=1.0, edgecolors='none')
ax.scatter(bg_1[:,0], bg_1[:,1],c='red', marker='+', s=8, label='Background 1', alpha=0.1, edgecolors='none')
ax.scatter(bg_2[:,0], bg_2[:,1],c='darkred', marker='+', s=8, label='Background 2', alpha=0.1, edgecolors='none')
ax.scatter(bg_3[:,0], bg_3[:,1],c='black', marker='+', s=8, label='Background 3', alpha=0.1, edgecolors='none')
ax.scatter(bg_4[:,0], bg_4[:,1],c='purple', marker='+', s=8, label='Background 4', alpha=0.1, edgecolors='none')
# Legend, Labels, etc.
ax.legend(loc='best', title="Groups")
ax.axis('equal')
plt.title("Signal and Background Gaussians, " + str(format(N_events, ',d')) + " Events each")
plt.xlabel("X")
plt.ylabel("Y")
# Save Plot
fig_name = "Gaussian_Plot.png"
plt.savefig(fig_name)
plt.close()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Block 2.0 - Write CSV File

# Paths
data_path = "Data\\"
signal_path = data_path + "Signal\\"
background_path = data_path + "Background\\"

# Signal CSV File
SignalEmptydf.to_csv("Gaussian_Signal.csv", sep=",", encoding='utf-8', index=False)
# Background CSV Files
Bg1Emptydf.to_csv(signal_path+"\Gaussian_Bg_1.csv", sep=",", encoding='utf-8', index=False)
Bg2Emptydf.to_csv(background_path+"\Gaussian_Bg_2.csv", sep=",", encoding='utf-8', index=False)
Bg3Emptydf.to_csv(background_path+"\Gaussian_Bg_3.csv", sep=",", encoding='utf-8', index=False)
Bg4Emptydf.to_csv(background_path+"\Gaussian_Bg_4.csv", sep=",", encoding='utf-8', index=False)
