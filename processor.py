# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Format matplotlib
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.dpi': 300})
plt.rcParams.update({'figure.autolayout': True})

plot_all = True

def format_plot():
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='gray', linestyle='-')
    plt.grid(b=True, which='minor', color='lightgray', linestyle='--')

def plot_hist(data, color, name, ax, max_=0, min_=100, plot_dashed=False):
    ax.hist(data, density=True, bins=bins, color=color, alpha=0.3, rwidth=0.5, edgecolor='black', linewidth=0.5)
    
    mu, std = norm.fit(data)
    
    if plot_all:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    xmin, xmax = (100000, 0)
    xmin = min(xmin, min_)
    xmax = max(xmax, max_)
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    
    lbl = name + r": $\mu = {0:10.0f}$, $\sigma = {1:10.0f}$".format(mu, std)
    if plot_all:
        lbl = name
    ax.plot(x, p, 'k', linewidth=2, color=color, label=lbl)

temp = 180
path = "train_{0}.csv".format(temp)

df = pd.read_csv(path)
df = df.drop(df.columns[[0]], axis=1)

intensity_cutoffs = [973, 739, 738, 2302, 1236, 2271, 446, 865, 813, 1633]

cols = ["705.7", "706.2", "951.0", "986.7", "990.4", "1004.6", "1347.0", "1370.1", "1390.6", "1601.8"]
cols = ["raman" + c for c in cols]

class_PEI = np.array(df["PEI"]).astype(np.int8)

if not plot_all:
    for c, i_c in zip(cols, intensity_cutoffs):
        df_PEI = df[df["PEI"] >= 0.5]
        df_EPO = df[df["PEI"] < 0.5]
        
        counts_PEI = np.array(df_PEI[c]).astype(np.float64)
        counts_EPO = np.array(df_EPO[c]).astype(np.float64)
        
        a, b = max(max(counts_EPO), max(counts_PEI)), min(min(counts_EPO), min(counts_PEI))
        
        ax = plt.gca()
        
        bins = 50
        plot_hist(counts_PEI, "r", "PEI", ax, a, b)
        plot_hist(counts_EPO, "b", "EPO", ax, a, b)
        
        plt.axvline(x=i_c, color="g", label="Intensity Cutoff", linestyle='dashed')
        
        plt.title(r"Raman Spectroscopy Intensity Histogram at ${0}\degree C$".format(temp))
        plt.xlabel("Intensity from Raman Spectroscopy at " + c.replace("raman", "") + r" [$cm^{-1}$]")
        plt.ylabel("Probability Density")
        
        plt.legend()
        plt.show()
else:
    fig, axs = plt.subplots(5, 2)
    axs = axs.flatten()
    
    for c, ax, i_c in zip(cols, axs, intensity_cutoffs):
        df_PEI = df[df["PEI"] >= 0.5]
        df_EPO = df[df["PEI"] < 0.5]
        
        counts_PEI = np.array(df_PEI[c]).astype(np.float64)
        counts_EPO = np.array(df_EPO[c]).astype(np.float64)
        
        a, b = max(max(counts_EPO), max(counts_PEI)), min(min(counts_EPO), min(counts_PEI))
        
        bins = 50
        plot_hist(counts_PEI, "r", "PEI", ax, a, b, plot_dashed=True)
        plot_hist(counts_EPO, "b", "EPO", ax, a, b)
        
        ax.axvline(x=i_c, color="g", label="Intensity Cutoff", linestyle='dashed')
        
        ax.text(.5,.72,c.replace("raman", "") + r" [$cm^{-1}$]", horizontalalignment='center', transform=ax.transAxes)
    
    
    lines, labels = axs[-1].get_legend_handles_labels()
    
    fig.legend(lines, labels, loc = 'lower right')

    fig.text(0.5, 0.01, 'Intensity', ha='center')
    fig.text(0.001, 0.5, 'Probability Density', va='center', rotation='vertical')

    fig.suptitle(r"Raman Spectroscopy Intensity Histogram at ${0}\degree C$".format(temp))
    
    plt.show()