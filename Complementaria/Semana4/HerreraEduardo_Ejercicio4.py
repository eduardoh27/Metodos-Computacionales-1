import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


def Lagrange(x,xi,i,n):
    
    prod = 1.0
    
    for j in range(n+1):
        if i!=j:
            prod *= (x-xi[j])/(xi[i]-xi[j])
            
    return prod


def Poly(x,xi,fxi,n):
    
    suma = 0.
    
    for i in range(n+1):
        
        suma += fxi[i]*Lagrange(x,xi,i,n)
        
    return suma


X = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
Y = [0.0, 0.0, 0.0, 0.01, 0.16, 0.91, 1.93, 1.51, 0.43, 0.05, 0.0]

x = np.linspace(-4, 203, 1000)
px = Poly(x, X, Y, 10)


fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(1,1,1)
plt.plot(x,px, c= "dodgerblue")
plt.scatter(X,Y,color='magenta')
ax.set_title(r'Método de Lagrange',fontsize=16)
ax.set_xlabel(r'x',fontsize=12)
ax.set_ylabel(r'y',fontsize=12)
plt.savefig('HerreraEduardo_Grafica.png')
plt.show()
