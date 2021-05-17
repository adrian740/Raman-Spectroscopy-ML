# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import CubicSpline
from os import listdir
from os.path import isfile, join
import pandas as pd

temp = "180"
path = "data analysis project//" + temp + "oC//"

allfiles = [f for f in listdir(path) if isfile(join(path, f))]

n = 2000
domain = np.linspace(650, 1700, n)

lst_tot = []
lst_unknown = []

cols = ["raman" + str(round(s,1)) for s in domain.tolist()]
cols.insert(0, "PEI")
cols.insert(0, "x")

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
        
df_train = pd.DataFrame.from_records(lst_tot)
df_train = df_train.sort_values(0)
df_train.columns = cols

df_test = pd.DataFrame.from_records(lst_unknown)
df_test = df_test.sort_values(0)
df_test.columns = cols
del df_test["PEI"]

df_train.to_csv("train_" + temp + ".csv", sep=',')
df_test.to_csv("test_" + temp + ".csv", sep=',')
