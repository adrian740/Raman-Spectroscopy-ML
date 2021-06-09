# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import CubicSpline
from os import listdir
from os.path import isfile, join
import pandas as pd


"""
Python file that will process the raw raman data, in files in the form of:
"120 80_0__X_-50__Y_-2.35417__Time_0.txt":
    Raman Shift       Intensity
    1721.000977	      79.361343
    1720.027344	      -137.462051
    ...               ...

This will return an x coordinate (x=-50) paired with a raman intensity 
distribution at the specified raman shifts (domain). This will make inputing
the data into the machine learning algorithm easier.

A cubic spline is used to removed high frequency oscilations that may 
occur in interpolation.

To run the machine learning algorithm, make sure that you have orange installed and opened.
Open the "raman.ows" file. And add the required files on the left. Make sure that all the 
data is loaded in. Next, save the processed data (SVM and RF) using the save files nodes.
You can also save the SVM and random forest algorithms by adding the "Save Model" node.

Read and run the result_processor.py to construct the concentration profiles.


"""

# Directory containing the data
temp = "180"
path = "data analysis project//" + temp + "oC//"

#temp = "multilayer"
#path = "multilAYER//"

# All the files in the directory
allfiles = [f for f in listdir(path) if isfile(join(path, f))]

# Domain, used for interpolation. Between 650 and 1700, evenly space 2000 samples
n = 2000
domain = np.linspace(650, 1700, n)

# List of all samples where the concentration is known (100% or 0% PEI). Used for ML training data
lst_tot = []

# List of all the samples, even when the concentration is unknown. Used to run predictions on the concentration profile
lst_unknown = []

# Create cols titled: x PEI raman650 raman....
cols = ["raman" + str(round(s,1)) for s in domain.tolist()]
cols.insert(0, "PEI")
cols.insert(0, "x")

# Insert all the files from the given path into the lists
for filename in allfiles:
    if filename.endswith(".txt"):
        x = float(filename.split("_")[-7])
        
        cat = None
        
        if x > 20:
            cat = 1 # PEI
        elif x < -5:
            cat = 0 # EPO
        
        df = pd.read_csv(path + filename, delimiter = "\t", header=None)
        cs = CubicSpline(df[0][::-1], df[1][::-1])
        lst = cs(domain).tolist()
        lst.insert(0, cat if cat is not None else "")
        lst.insert(0, x)
        
        if cat is not None:
            lst_tot.append(lst)
        lst_unknown.append(lst)

# Convert list to pandas dataframe, sort for increasing x coordinate, and set the column names to cols
df_train = pd.DataFrame.from_records(lst_tot)
df_train = df_train.sort_values(0)
df_train.columns = cols

# Same as above, just delete the PEI column to make sure that this is not being carried over to the ML algorithm
df_test = pd.DataFrame.from_records(lst_unknown)
df_test = df_test.sort_values(0)
df_test.columns = cols
del df_test["PEI"]

# Save to CSV.
df_train.to_csv("train_" + temp + ".csv", sep=',')
df_test.to_csv("test_" + temp + ".csv", sep=',')
