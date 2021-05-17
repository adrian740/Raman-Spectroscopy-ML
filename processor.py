# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Format matplotlib
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.dpi': 300})
plt.rcParams.update({'figure.autolayout': True})

def format_plot():
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='gray', linestyle='-')
    plt.grid(b=True, which='minor', color='lightgray', linestyle='--')

def plot_hist(data, color, name, max_=0, min_=100):
    plt.hist(data, density=True, bins=bins, color=color, alpha=0.3, rwidth=0.5, edgecolor='black', linewidth=0.5)
    
    mu, std = norm.fit(data)
    
    ax = plt.gca()
    
    xmin, xmax = plt.xlim()
    xmin = min(xmin, min_)
    xmax = max(xmax, max_)
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2, color=color, label=name + r": $\mu = {0:10.0f}$, $\sigma = {1:10.0f}$".format(mu, std))
    
    ax.legend()

path = "train_120.csv"

df = pd.read_csv(path)
df = df.drop(df.columns[[0]], axis=1)

cols = ["987.2", "1370.1"]
cols = ["raman" + c for c in cols]

class_PEI = np.array(df["PEI"]).astype(np.int8)

for c in cols:
    df_PEI = df[df["PEI"] > 0.5]
    df_EPO = df[df["PEI"] < 0.5]
    
    counts_PEI = np.array(df_PEI[c]).astype(np.float64)
    counts_EPO = np.array(df_EPO[c]).astype(np.float64)
    
    bins = 50
    plot_hist(counts_PEI, "r", "PEI", max(max(counts_EPO), max(counts_PEI)), min(min(counts_EPO), min(counts_PEI)))
    plot_hist(counts_EPO, "b", "EPO")
    
    plt.xlabel(r"Intensity at " + c.replace("raman", "") + r" [$cm^{-1}$]")
    plt.ylabel(r"Probability Density")
    
    plt.show()