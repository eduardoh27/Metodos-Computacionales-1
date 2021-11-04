import numpy as np
from scipy.stats import norm
import sympy as sym
from sympy import exp

dataList = [130,122,119,142,136,127,120,152,141,132,127,118,150,141,133,137,129,142]
print(min(dataList), max(dataList))
media, desv = norm.fit(dataList)
#media = np.mean(dataList)
#desv = np.std(dataList)
print(media, desv)

f = lambda x, sigma: (1/(2*sigma**2))*np.exp(-np.abs(x)/(sigma**2))

probaTotal = 0
for dato in dataList:
    probaTotal += f(dato, desv)

print(probaTotal)
lista = f(media, desv)

print(lista)

print(lista.sum())

sigma = desv
#sigma = 1221

x = sym.Symbol('x', Real=True)
#y = sym.Symbol('y', Real=True)
pdf = (1/(2*sigma**2))*exp(-np.abs(x)/(sigma**2))

integral = sym.integrate(pdf, (x, -sym.oo, sym.oo))

print(integral)
