# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Format matplotlib
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.dpi': 300})
plt.rcParams.update({'figure.autolayout': True})

def format_plot():
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='gray', linestyle='-')
    plt.grid(b=True, which='minor', color='lightgray', linestyle='--')
    
    
temp = "120"
path = "train_" + temp + ".csv"

df = pd.read_csv(path)
df = df.drop(df.columns[[0]], axis=1)

n = 2000
domain = np.linspace(650, 1700, n)

def row(x):
    a = []
    for i in x:
        a.append((df.loc[df['x'] == i]).to_numpy()[0][2::])
    return np.array(a)

spec = row([-50.0, -30.0, 30.0, 70.0])
materials = ["Epoxy", "PEI"]

def peaks(x, y):
    peaks, _ = signal.find_peaks(y, height=0, distance=100, prominence=2500)
    return peaks, x[peaks]

def find_k_means(peaks_comb):
    max_, min_ = -1, 100
    all_ = []
    for p in peaks_comb:
        l = len(p)
        if l > max_:
            max_ = l
        if l < min_:
            min_ = l
        for i in p:
            all_.append(i)
    min_ = max(1, min_ - 2)
    
    all_ = np.array(all_).reshape(-1,1)
    
    sil = []    
    for k in range(max(min_, 2), max_ + 1, 1):
        kmeans = KMeans(n_clusters = k).fit(all_)
        labels = kmeans.labels_
        sil.append(silhouette_score(all_, labels, metric = 'euclidean'))
        
    if max_ != 1:
        k_opt = np.argmax(sil) + min_
        alg = KMeans(n_clusters = k_opt)
        kmeans = alg.fit(all_)
        
        return k_opt, alg.cluster_centers_.T[0]
    
    return 1, np.array([np.mean(all_)])

def find_peak_prominence(x, spec, peaks, width=10):
    prominence = []
    for pi in peaks:
        idx = np.argmin(abs(pi - x))
        prominence.append(max(spec[int(idx-width/2):int(idx+width/2):]))
    return prominence

def find_signatures(spectra):
    materials_idx = np.array(range(len(spectra)))
    materials_means = []
    prominence_means = []
    
    for mat in materials_idx:
        peaks_comb = []
        for spec in spectra[mat]:
            peak_idx, peak_sig = peaks(domain, spec)
            peaks_comb.append(peak_sig)
        k, means = find_k_means(peaks_comb)
        
        tot_prominences = []
        
        for spec in spectra[mat]:
            prominences = find_peak_prominence(domain, spec, means)
            tot_prominences.append(prominences)
        tot_prominences = np.array(tot_prominences)
        prominence_means.append(tot_prominences.mean(axis=0))
        
        materials_means.append(means.tolist())
    
    return materials_means, prominence_means

peaks_, prominences_ = find_signatures([spec[0:2], spec[2:4]])

dec = 2

weights_matrix = []

for i in range(len(peaks_)):
    print(materials[i], "has peaks at:")
    p = peaks_[i]
    promi = prominences_[i]
    tot_prom = sum(promi)
    weights_vector = np.zeros(n)
    for j in range(len(p)):
        idx = np.argmin(abs(p[j] - domain))
        print("\t", round(p[j], dec), "-", round(promi[j]/tot_prom, dec), "(idx:", str(idx) + ")")
        weights_vector[idx] = promi[j]/tot_prom
    weights_matrix.append(weights_vector)
weights_matrix = np.array(weights_matrix)

def conc(m, sample):
    v = np.sum(m*sample, axis=1)
    v_sum = np.sum(v)
    return v / v_sum

colors = ["r", "g"]

def plot(m, x, sample):
    plt.plot(x, sample, label="Unknown Sample: " + materials[np.argmax(conc(m, sample))])
    
    for i in range(m.shape[0]):
        idxs = np.where(m[i] > 0)
        plt.vlines(x[idxs], 0, 4000, color=colors[i])
    
    plt.legend()
    format_plot()
    plt.show()

plot(weights_matrix, domain, spec[3])



































