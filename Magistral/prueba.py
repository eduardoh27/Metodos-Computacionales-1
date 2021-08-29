#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

xi, xf, h = -10., 10., 0.05
Npoints = int((xf-xi)/h + 1)
print(Npoints)

x = np.linspace(xi,xf,Npoints)
print(x[0], x[200])

def Function(x):
    return 1/(np.sqrt(1 + np.e**(-x**2)))

def EDerivative(x):
    num = (np.e**(-x**2))*x
    denom = (1 + np.e**(-x**2))**(3/2)
    return num/denom

y=Function(x)
Dy=EDerivative(x)

def CentralDerivative(f,x,h):
    
    d = 0.
    
    if h!=0:
        d = (f(x+h)-f(x-h))/(2*h)
      
    return d

cero = CentralDerivative(Function, 0, h)
print(cero)