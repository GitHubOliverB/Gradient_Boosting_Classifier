#!/usr/bin/python

import os
import sys
import importlib
import csv
import numpy as np
import pandas as pd

# Module To Import Lists From File To Get Names Of CSV Files
def module_import(Module, *Packages, Type):
	package_list = []
	for package in Packages:
		try:
			mypackage =  getattr(importlib.import_module(Module), package)
		except ModuleNotFoundError as module_not_found:
			print("Error, couldn't find the Datafile: " + str(module_not_found))
			exit()
		except AttributeError as import_not_found:
			print("Error, couldn't find a list in the Datafile: " + str(import_not_found))
			exit()
		else:
			if len(mypackage) == 0:
				print("Error, " + str(package) + " in Data_File seems to be empty!")
				exit()
			if len(mypackage) > len(np.unique(mypackage)):
				print("Error, in " + str(package) + " a file was specified more than once.")
				exit()
			if all(isinstance(x, str) for x in mypackage) is False:
				print("Error, an element in " + str(package) + " is not a string.")
				exit()
			print("\nDefining " + str(package) + " Data...")
			if Type == 'Data':
				for i, _ in enumerate(mypackage):
					print("Defined " + str(_) + " As " + str(package) + " Type")
				print("Defined " + str(format(len(mypackage), ',d')) + " " + str(package) + " Type/s")
				print("\nData was successfully specified...")
			elif Type == 'Feature':
				print("Features were successfully specified...")
				for i, _ in enumerate(mypackage):
					print("Feature " + str(i) +": " + str(_))
				print("\nDefined " + str(format(len(mypackage), ',d')) + " Feature/s")
			else:
				print("Put either Data or Feature as Type.")
				exit()			
		package_list.append(mypackage)
	return package_list
# Module To Import specified Dataframes And Select Imported Features
def data_import(Data_List, Features, Type):
	Data = []
	data_path = "Data\\"
	if Type == 'Signal':
		path = data_path + "Signal\\"
	elif Type == 'Background':
		path = data_path + "Background\\"
	else:
		print("Put either Signal_List or Background_List as Input.")
		exit()	
	for i, j in enumerate(Data_List):
		Data_Import = pd.read_csv(path+str(j)+".csv", delimiter=',')
		if len(Data_Import.index) == 0:
			print("Error, the dataframe " + str(j) + " seems to be empty.")
			exit()
		if any(".1" in item for item in Data_Import.columns.tolist()):
			print("Error, at least two columns have the same name in the dataframe " + str(j) + ".")
			print("The column was rename with columnname.1 by Pandas.")
			exit()
		if Data_Import.isnull().sum().sum() > 0:
			print("Error, there is a NaN value in the dataframe " + str(j) + ".")
			exit()
		DataEmptydf = pd.DataFrame()
		for k, l in enumerate(Features):
			try:
				DataEmptydf[Features[k]] = Data_Import[Features[k]]
			except KeyError:
				print("Error, the feature " + Features[k] + " was not found in the dataframe " + str(j) + ".")
				exit()
		if Type == 'Signal':
			DataEmptydf['Signal_Indicator'] = 1
			DataEmptydf['Background_Indicator'] = 0
		elif Type == 'Background':
			DataEmptydf['Signal_Indicator'] = 0
			DataEmptydf['Background_Indicator'] = i+1
		else:
			print("Put either Signal_List or Background_List as Input.")
			exit()		
		Data.append(DataEmptydf)
		print(str(j) + " Found: " + str(format(len(Data[i]), ',d')) + " Events")
	return Data
