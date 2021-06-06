# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

# Format matplotlib
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.dpi': 300})
plt.rcParams.update({'figure.autolayout': True})

def format_plot():
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='gray', linestyle='-')
    plt.grid(b=True, which='minor', color='lightgray', linestyle='--')
    

plot_RF = True

def read(t):
    path = "res_" + t + ".csv"
    
    df = pd.read_csv(path)
    df = df.drop([0, 1])
    
    x = np.array(df["x"]).astype(np.float32)
    y = np.array(df["SVM (1)"]).astype(np.float16)
    y_RF = np.array(df["Random Forest (1)"]).astype(np.float16)
    
    return x, y, y_RF

temp = "120"
path = "res_" + temp + ".csv"

#path = "res_multilayer.csv"

df = pd.read_csv(path)
df = df.drop([0, 1])

x = np.array(df["x"]).astype(np.float32)
y = np.array(df["SVM (1)"]).astype(np.float16)
y_RF = np.array(df["Random Forest (1)"]).astype(np.float16)

# Process so as to remove the anomalies and plot them as a dashed line
mask = (x < 0.5) | (x > 1.1)

#mask = x > -100 # Remove to remove anomaly, see plot line as well

mask_inv = np.invert(mask)

x_new = x[mask]
y_new = y[mask]

x_inv = x[mask_inv]
y_inv = x[mask_inv]

plt.xlabel("Distance [microns]")
plt.ylabel("Normalized Peak Intensity [-]")
plt.title(r"Concentration Profile at ${0}\degree C$".format(temp))

plt.plot(x_new, 1-y_new, label="% Epoxy (SVM)", color="C0")
plt.plot(x_new, y_new, label="% PEI (SVM)", color="C1")

plt.scatter(x_inv, y_inv, label="Anomaly", color="r", marker="x", s=13, zorder=2)

print(x_inv, "\n", y_inv)

if plot_RF:
    plt.plot(x, y_RF, "--", label="% PEI (RF)", color="C2")
plt.legend()
format_plot()

plt.show()

if True:
    exit()

# Butterworth Filter
N = 3  # Filter order
Wn = 0.1  # Cutoff frequency
B, A = signal.butter(N, Wn, output='ba')

def smooth(y):
    return signal.filtfilt(B, A, y)

smooth_data = signal.filtfilt(B, A, y)
smooth_data_RF = signal.filtfilt(B, A, y_RF)

plt.xlabel("Distance [micrometers]")
plt.ylabel("Normalized Peak Intensity [-]")
plt.plot(x, smooth_data, label="% PEI (SVM)", color="C1")
if plot_RF:
    plt.plot(x, smooth_data_RF, label="% PEI (RF)", color="C2")
plt.legend()
format_plot()

plt.show()

if True:
    exit()

temps = ["120", "140", "160", "180"]

plt.xlabel("Distance [micrometers]")
plt.ylabel("Normalized Peak Intensity [-]")
for t in temps:
    x, y_SVM, y_RF = read(t)
    y_s = smooth(y_SVM)
    d = np.minimum(abs(1 - y_s), abs(y_s))
    plt.plot(x, d, label="% PEI (SVM) (" + t + r"$^{\circ}$C)")

plt.legend()
format_plot()
plt.show()