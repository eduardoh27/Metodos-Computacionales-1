import numpy as np
from scipy.stats import norm
import sympy as sym
from sympy import exp

dataList = [130,122,119,142,136,127,120,152,141,132,127,118,150,141,133,137,129,142]

media, desv = norm.fit(dataList)
print(f'media, desv = {media, desv}')


# 1.

sigma = np.random.randint(10) # no importa el valor de sigma

x = sym.Symbol('x', Real=True)
#y = sym.Symbol('y', Real=True)
pdf = (1/(2*sigma**2))*exp(-np.abs(x)/(sigma**2))

integral = sym.integrate(pdf, (x, -sym.oo, sym.oo))
print(integral)


# 2.

print(f'sigma = {np.sqrt(media)}')


