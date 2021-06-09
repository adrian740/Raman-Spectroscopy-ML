# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Format matplotlib
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.dpi': 300})
plt.rcParams.update({'figure.autolayout': True})

def format_plot():
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='gray', linestyle='-')
    plt.grid(b=True, which='minor', color='lightgray', linestyle='--')
    
    
"""
This code takes in the processed data from the machine learning and plots a
concentration profile according to the data. It will also remove anomalies (hand detected)
and replace them with a cross. 

To use first run the machine learning algorithm, details specified in preprocessor.py
"""

# Read the data from the csv file, and return the x coordinate, the concentration
# according to the SVM and RF model
def read(t):
    path = "res_" + t + ".csv"
    
    df = pd.read_csv(path)
    df = df.drop([0, 1])
    
    x = np.array(df["x"]).astype(np.float32)
    y = np.array(df["SVM (1)"]).astype(np.float16)
    y_RF = np.array(df["Random Forest (1)"]).astype(np.float16)
    
    return x, y, y_RF

# Target temperature to read
path = "120"

#path = "multilayer"

x, y, y_RF = read(path)


# Plot the result from the random forest. Since only 10 trees were used the 
# accuracy of the RF is limited to 1/10=0.1=10% accuracy. Still useful to 
# see what is going on for the SVM model as they pick on very similar patterns
plot_RF = True

# whether to remove the anomaly or not. Specify the location of the anomaly in mask
remove_anomaly = True

# Process so as to remove the anomalies and replace them with a cross (if remove_anomaly is true)
mask = (x < 0.5) | (x > 1.1)

if not remove_anomaly:
    mask = np.ones(len(x), dtype=bool)

# Invert the mask
mask_inv = np.invert(mask)

# Obtain the new data (without anomaly)
x_new = x[mask]
y_new = y[mask]

x_inv = x[mask_inv]
y_inv = y[mask_inv]

# Formatting
plt.xlabel("Distance [microns]")
plt.ylabel("Normalized Peak Intensity [-]")
plt.title(r"Concentration Profile at ${0}\degree C$".format(path))

plt.plot(x_new, 1-y_new, label="% Epoxy (SVM)", color="C0")
plt.plot(x_new, y_new, label="% PEI (SVM)", color="C1")

plt.scatter(x_inv, y_inv, label="Anomaly", color="r", marker="x", s=13, zorder=2)

# Print what was removed
print(x_inv, "\n", y_inv)

# Plot random forest if true
if plot_RF:
    plt.plot(x, y_RF, "--", label="% PEI (RF)", color="C2")
plt.legend()
format_plot()

plt.show()