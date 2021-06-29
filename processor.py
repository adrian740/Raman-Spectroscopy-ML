# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

"""
This python script is used to understand the random forest approach. It will
plot the intensity distribution of the raman data, along with the intensity
cutoff that was found using the machine learning model, to help visualize the
algorithm. 

This file is not necessary to compute for later procesing.

"""

# Format matplotlib
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.dpi': 300})
plt.rcParams.update({'figure.autolayout': True})

def format_plot():
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='gray', linestyle='-')
    plt.grid(b=True, which='minor', color='lightgray', linestyle='--')

# Plot all the intensities in cols? Useful for making an overview of a certain
# sample
plot_all = True

# Plot a histogram with a normal distribution overlayed on top of it
# Will also plot the intensity cutoff in a green dashed line
def plot_hist(data, color, name, ax, max_=0, min_=100, bins=50):
    # Create histogram
    ax.hist(data, density=True, bins=bins, color=color, alpha=0.3, rwidth=0.5, edgecolor='black', linewidth=0.5)
    
    # Fit a normal distribution on the intensity data
    mu, std = norm.fit(data)
    
    # Hide the axis labels if we are plotting all the raman shifts into one plot
    # This means that the actual values cannot be read of the graph, but rather
    # it will provide some intution behind the process
    if plot_all:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    # Plot the normal distribution
    x = np.linspace(min_, max_, 100)
    p = norm.pdf(x, mu, std)
    
    # Formating, esp. graph legends    
    lbl = name + r": $\mu = {0:10.0f}$, $\sigma = {1:10.0f}$".format(mu, std)
    if plot_all:
        lbl = name
    ax.plot(x, p, 'k', linewidth=2, color=color, label=lbl)

# Obtain the CSV file from the preprocessor.py file.
temp = 180
path = "train_{0}.csv".format(temp)

# Remove index col.
df = pd.read_csv(path)
df = df.drop(df.columns[[0]], axis=1)

# Manually input the intensity cutoffs obtained from machine learning 
intensity_cutoffs = [973, 739, 738, 2302, 1236, 2271, 446, 865, 813, 1633]

# TESTING
intensity_cutoffs = [1811.7052, 1351.9456, 1427.9778, 756.3234, 908.9467, 639.3465, 521.4253, 476.7677, 347.7531, 264.428]

# The raman shifts considered.
cols = ["705.7", "706.2", "951.0", "986.7", "990.4", "1004.6", "1347.0", "1370.1", "1390.6", "1601.8"]

# TESTING
cols = [1004.6, 1380.6, 702.0, 1115.9, 1621.2, 952.6, 1455.2, 1249.8, 784.5, 1166.9]
cols = ["%.1f" % number for number in cols]

cols = ["raman" + c for c in cols]

# Convert the "1" to a 1 (or 0)
class_PEI = np.array(df["PEI"]).astype(np.int8)

# Branch out: first block is if only one plot is to be drawn per raman shift
# the second block (the else statement) if we want to combine all the plots into one
if not plot_all:
    for c, i_c in zip(cols, intensity_cutoffs):
        # Midpoint value, should not even be close to 0.5 (as this is on the training data)
        df_PEI = df[df["PEI"] >= 0.5]
        df_EPO = df[df["PEI"] < 0.5]
        
        # Get the intensity of PEI and epoxy samples at the specified raman shift
        counts_PEI = np.array(df_PEI[c]).astype(np.float64)
        counts_EPO = np.array(df_EPO[c]).astype(np.float64)
        
        # Bounds, used for creating the normal distribtution
        a, b = max(max(counts_EPO), max(counts_PEI)), min(min(counts_EPO), min(counts_PEI))
        
        # Get current axes
        ax = plt.gca()
        
        # Histogram settings, specifically 50 bins, PEI in red and epoxy in blue
        bins = 50
        plot_hist(counts_PEI, "r", "PEI", ax, a, b, bins=bins)
        plot_hist(counts_EPO, "b", "EPO", ax, a, b, bins=bins)
        
        # Intensity cutoff line
        plt.axvline(x=i_c, color="g", label="Intensity Cutoff", linestyle='dashed')
        
        # Formating
        plt.title(r"Raman Spectroscopy Intensity Histogram at ${0}\degree C$".format(temp))
        plt.xlabel("Intensity from Raman Spectroscopy at " + c.replace("raman", "") + r" [$cm^{-1}$]")
        plt.ylabel("Probability Density")
        
        plt.legend()
        plt.show()
else:
    # Plot  
    
    fig, axs = plt.subplots(5, 2)
    axs = axs.flatten()
    
    for c, ax, i_c in zip(cols, axs, intensity_cutoffs):
        df_PEI = df[df["PEI"] >= 0.5]
        df_EPO = df[df["PEI"] < 0.5]
        
        counts_PEI = np.array(df_PEI[c]).astype(np.float64)
        counts_EPO = np.array(df_EPO[c]).astype(np.float64)
        
        a, b = max(max(counts_EPO), max(counts_PEI)), min(min(counts_EPO), min(counts_PEI))
        
        bins = 50
        plot_hist(counts_PEI, "r", "PEI", ax, a, b, bins=bins)
        plot_hist(counts_EPO, "b", "EPO", ax, a, b, bins=bins)
        
        ax.axvline(x=i_c, color="g", label="Intensity Cutoff", linestyle='dashed')
        
        # this is the best way to title the subplot
        ax.text(.5,.72,c.replace("raman", "") + r" [$cm^{-1}$]", horizontalalignment='center', transform=ax.transAxes)
    
    # Get the last legend + format
    lines, labels = axs[-1].get_legend_handles_labels()
    
    fig.legend(lines, labels, loc = 'lower right')

    fig.text(0.5, 0.01, 'Intensity', ha='center')
    fig.text(0.001, 0.5, 'Probability Density', va='center', rotation='vertical')

    fig.suptitle(r"Raman Spectroscopy Intensity Histogram at ${0}\degree C$".format(temp))
    
    plt.show()