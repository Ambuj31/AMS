# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 20:32:25 2021

@author: ASUS
"""

import matplotlib as plt
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sn

df = pd.read_csv("C:/Users/ASUS/Desktop/tdataset.csv")
del df['Employee_Name']

print(df.head())
#c = (df['Employee_ID'])
#print(c)

#df.columns = c.T

print(df.head())
df =df.T
#x = ['01-Mar','02-Mar','03-Mar','04-Mar','05-Mar','06-Mar','07-Mar','08-Mar','09-Mar','10-Mar','11-Mar']
#df.plot(x = '01-Mar',y = "Employee ID")

print(df.head(20))

#df.T.plot()

#df.plot(x = '101', y = 'Employee Name')


df = pd.read_csv("C:/Users/ASUS/Desktop/tdataset.csv")
c = []
c.append(df['Employee_ID'])
del df['Employee_Name'] 
del df['Employee_ID']
df.head(10)


fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(25, 15))
i = 0
for row, ax in zip(df.index, axes.flatten()):
    ax.plot(df.loc[row].values)

    ax.set_title(c[0][i], fontsize=10)
    i=i+1
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(list(df.columns), rotation=90, fontsize=8)
    fig.tight_layout(pad=3.0)
plt.show()


