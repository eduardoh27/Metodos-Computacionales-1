# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 13:41:43 2021

@author: juanp
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial
#plt.style.use('dark_background')
import pandas


df = pandas.read_csv("https://raw.githubusercontent.com/diegour1/CompMetodosComputacionales/main/DataFiles/world_pop.csv")
data= pandas.DataFrame(df)
years=[]
world_p=[]
for i in range(len(df)-1):
    if ((data.loc[i+1,"Entity"] == "Our World In Data") and (((data.loc[i+1,"Year"])>700) and (data.loc[i+1,"Year"]< 1900))): 
        years.append(data.loc[i+1 , "Year"])
        world_p.append(data.loc[i+1, "World Population"])


coeffs=[]
pts=len(years)
np_years = np.array(years)
np_world = np.array(world_p)
amp = 2

x = np_years.reshape(-1, 1)
y = np_world.reshape(-1, 1)

P = np.array([np.ones([pts, 1]), x, x**2 , x**3]).reshape(4, pts).T
v = (np.linalg.inv(P.T @ P) @ P.T) @ y
b, m , m2 , m3= v

coe0=b[0]
coe1=m[0]
coe2=m2[0]
coe3=m3[0]

coeffs.append(coe0)
coeffs.append(coe1)
coeffs.append(coe2)
coeffs.append(coe3)

#plt.rcParams['axes.facecolor'] = 'w'

plt.plot(x, y, 'o')
coeficientes= [ coe0, coe1 , coe2, coe3 ]
polinomio= Polynomial(coeficientes)

plt.plot(x, polinomio(x))
plt.show()
plt.savefig("ApellidoNombre_grafica.png.")

print(f"coeffs 4to grado = {coeffs}")