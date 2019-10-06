#!/usr/bin/python

from os import listdir
from os.path import isfile, join

"""
Summary:

Give the names(str) of the dataframes for the Signal and Background Dataframes you want to include for your run.
Signal Dataframes belong in the "Data\Signal" and Background Dataframes in the "Data\Background" directory.

Now automatically adds the data from the directories.
"""

# Paths to data
data_path = "Data\\"
signal_path = data_path + "Signal\\"
background_path = data_path + "Background\\"

# Define the signal and background data files. Only takes strings.
Signal_List = [f for f in listdir(signal_path) if isfile(join(signal_path, f))]
Background_List = [f for f in listdir(background_path) if isfile(join(background_path, f))]

#Signal_List = ['Signal_1.csv']
#Background_List = ['bg_1.csv', "bg_2.csv"]