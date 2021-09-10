import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

f = lambda x: np.sqrt(1+np.exp(-x**2))

xi, xf, Npoints = -1., 1., 300 # Debe ser múltiplo de 3

X = np.linspace(xi,xf,Npoints+1)
Y = f(X)

def SimpsonMethod(f,X):
   
    h = (X[-1]-X[0])/(len(X)-1)
   
    integral = 0
   
    integral += f(X[0]) + f(X[-1])
   
    for i in range( len(X[1:-1]) ):
       
        if (i+1)%3 == 0:
            integral += 2*f(X[i+1])
       
        else:
            integral += 3*f(X[i+1])
           
    integral *= 3*h/8
   
    return integral, h

Integral, _ = SimpsonMethod(f,X)
print("The integral up to 10 decimals is "+str(round(Integral,10)))
#2.6388571169