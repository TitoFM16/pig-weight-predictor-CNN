#program to plot data

import pandas as pd
import os
import numpy as np
import re
import matplotlib.pyplot as plt

cerdos = input ('Cerdos machos o hembras C1 o C2 ?:\n')
if cerdos == "C1" or cerdos == "c1":
    cerdos = "1"
elif cerdos == "C2" or cerdos =="c2":
    cerdos = "2"

# read weight info from ./data/...
df = pd.read_csv(f'./data/cerdos_pesos_c{cerdos}.csv', sep=',')
print(df)
#plot area vs weight columns
df = df.dropna()
x =df['area']
y=df['weight']
plt.scatter(x,y)

plt.xlabel('area')
plt.ylabel('peso')

# calc the trendline
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")
plt.xlabel(f'Areas C{cerdos}')
plt.ylabel(f'Pesos C{cerdos}')
plt.title("y=%.6fx+(%.6f)"%(z[0],z[1]))
# the line equation:
print("y=%.6fx+(%.6f)"%(z[0],z[1]))

# find scatter plot correlation
corr = np.corrcoef(df['area'],df['weight'])
print(corr)
# plot trendlind

plt.show()




