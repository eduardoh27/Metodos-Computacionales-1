"""Este programa se basa en los algoritmos vistos 
en el curso Metodos Computacionales 1 de la
Universidad de los Andes

Desarrolladores:
-Juan Pablo Idarraga
-Eduardo Jose Herrera Alba
"""

import numpy as np
import matplotlib.pyplot as plt


def NewtonInterpolation(X,Y,x):
    
    sum_ = Y[0]
    
    Diff = np.zeros( (len(X),len(Y)) )
       
    Diff[:,0] = Y 
    
    poly = 1.0
    n=1 # diferencia entre las x
    for i in range(1,len(X)): # i columna
    
        poly *= (x-X[i-1])
        
        for j in range(i, len(X)): # j fila
            Diff[j,i] = (Diff[j,i-1]-Diff[j-1,i-1]) /(X[j]-X[j-n]) # dividir por el cambio en la x
        n=n+1
        sum_ += poly*Diff[i,i]
    
   
    return sum_, np.round(Diff,2)



"""
Puntos equidistantes
"""
Xe = [ 0,  1,  2, 3, 4, 5]
Ye = [-18, -13, 0, 5, 3, 10]

xpe = np.linspace(-2,7,1000)
ype = []

sumae, _ = NewtonInterpolation(Xe,Ye,9)
print(sumae)

for x in xpe:
    ye,_ = NewtonInterpolation(Xe,Ye,x)
    ype.append(ye)

# Grafica de los puntos equidistantes
plt.scatter(Xe,Ye,color='r',label='Data')
plt.plot(xpe,ype,c='b',label='Polinomio Interpolante')
plt.title(r'Interpolación para puntos equiespaciados',fontsize=15)
plt.legend()
plt.show()

 


"""
Puntos no equidistantes
"""
X = [-2, -1, 1, 2, 5, 7]
Y = [157, 20, -13, 0, 10, 178]

xp = np.linspace(-2,7,1000)
yp = []

suma, _ = NewtonInterpolation(X,Y,9)
print(suma)

for x in xp:
    y,_ = NewtonInterpolation(X,Y,x)
    yp.append(y)

# Grafica de los puntos no equidistantes
plt.scatter(X,Y,color='r',label='Data')
plt.plot(xp,yp,c='b',label='Polinomio Interpolante')
plt.title(r'Interpolación para puntos no equiespaciados',fontsize=15)
plt.legend()
plt.show()
