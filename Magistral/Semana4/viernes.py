"""
Este programa se basa en los algoritmos vistos 
en el curso Métodos Computacionales 1 de la
Universidad de los Andes

Desarrolladores:
-Eduardo José Herrera Alba
-Juan Pablo Idarraga
"""

import numpy as np
import matplotlib.pyplot as plt

#X = [ 0,  1,  2, 3, 4, 5]
#Y = [-18, -13, 0, 5, 3, 10]

X = [-2, 0, 2, 4, 6, 8]
Y = [157, -18, 0, 2, 55, 428]

# X = [-6, -3, 0, 3, 6, 9]
# Y = [3662, 472, -18, 5, 55, 861]


#X = [-2, -1, 1, 2, 5, 7]
#Y = [157, 20, -13, 0, 10, 178]

"""
[-6, 3662]
[-3, 472]
[5, 10]
[10,1538]
[]
[7, 178]
[-1, 20]"""


def NewtonInterpolation(X,Y,x):
    
    sum_ = Y[0]
    
    Diff = np.zeros( (len(X),len(Y)) )
    
    #h = X[1]-X[0] 
    
    Diff[:,0] = Y # La derivada 0
    
    poly = 1.0

    for i in range(1,len(X)): # i columna
        h = X[i]-X[i-1]
        #if i < len(X)-1:
        #    h = abs(X[i+1]-X[i])

        poly *= (x-X[i-1])
        
        for j in range(i, len(X)): # j fila
            Diff[j,i] = (Diff[j,i-1]-Diff[j-1,i-1]) #/ (X[i] - X[]#/ h #(X[i]-X[0])
        
        #h = X[i]-X[i-1]
        sum_ += poly*Diff[i,i]/(np.math.factorial(i)*h**i)
        
    return sum_, np.round(Diff,2)


xp = np.linspace(-2,8,1000)
yp = []


bla, _ = NewtonInterpolation(X,Y,9)
print(bla)

"""
X = [-1, 1, 2, 5, 7, 10]
Y = [20, -13, 0, 10, 178, 1538]
"""

for x in xp:
    y,_ = NewtonInterpolation(X,Y,x)
    yp.append(y)

plt.scatter(X,Y,color='r',label='Data')
plt.plot(xp,yp,c='b',label='Polinomio Interpolante')
#plt.scatter(xp,yp,color='b',label='Polinomio Interpolante', s=1)
plt.legend()
plt.show()