import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc



matriz = np.zeros( (10, 3) )
matriz[8,2] = 1
print(matriz)



def Lagrange(x,xi,i,n):
    
    prod = 1.0
    
    for j in range(n+1): # tengo n+1 puntos 
        if i!=j:
            prod *= (x-xi[j])/(xi[i]-xi[j])
            
    return prod


def Poly(x,xi,fxi,n):
    
    summ = 0.
    
    for i in range(n+1):
        
        summ += fxi[i]*Lagrange(x,xi,i,n)
        
    return summ


   
X1 = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
Y1 = [0.0, 0.0, 0.0, 0.01, 0.16, 0.91, 1.93, 1.51, 0.43, 0.05, 0.0]


x = np.linspace(0, 200 ,100)
px2 = Poly(x, X1 , Y1 , 10)

plt.scatter(X1,Y1,color='r')
plt.plot(x,px2)
plt.show()


"""y_max=0
for i in range(0,200):
    y= Poly(i, X1, Y1, 10)
    if y>y_max:
        y_max=y
    else:
        None 

y_mitad=round(y_max/2 , 2 )

print(y_mitad)

x_mitad=[]       
for i in range(0,200):
    y_y= Poly(i , X1, Y1, 10)
    y=round(y_y,2)
    #print(y==y_mitad)
    if (y == y_mitad)==True:
        x_mitad.append(i)
print(x_mitad)"""