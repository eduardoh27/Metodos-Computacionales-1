import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

def Biseccion(a,b,f,n,x1, y1, n1,tolerance=1e-10):
    
    i = 0
    root = 1e6
    error = 0.

    while i < n:
        
        m1 = (a + b)/2.
        
        
        if f(m1,x1,y1,n1) == 0.: # Encontre la raiz
            root = m1
            break
            
        if f(m1,x1,y1,n1)*f(b,x1,y1,n1) < 0:
            a = m1
        elif f(a,x1,y1,n1)*f(m1,x1,y1,n1) < 0:
            b = m1
        else:
            root = 'No-roots'
            break
            
        error = np.abs(root - m1)
        root = m1
        
        if error <= tolerance:
            break
        
        
        i += 1
    
    return root, error


def Poly(x,xi,fxi,n):

    summ = 0.

    for i in range(n+1):

        summ += fxi[i]*Lagrange(x,xi,i,n)

    return summ


def Lagrange(x,xi,i,n):

    prod = 1.0

    for j in range(n+1): #tengo n+1 puntos
        if i!=j:
            prod *= (x-xi[j])/(xi[i]-xi[j])

    return prod


x1 = [1.4000, 3.5000, 5.6000]
y1 = [0.4007, 0.5941, 0.2980]
n = 2

"""
n = 2
x = np.linspace(0, 8, 10000)
px2 = Poly(3, x1, y1, 2)
#print(px2)
"""
# Usamos el método de bisección desde 0.1 para hallar la raíz. 
# La raíz es el punto de mayor alcance
x_max, _ = Biseccion(0.1,10, Poly, 100, x1, y1, n)


plt.scatter(x1, y1, color='r')
plt.plot(x, px2)
plt.show()



def Linearlagrange2(a,b,c,fa,fb,fc):
    return -(b+c)*fa/((a-b)*(a-c)) -(a+c)*fb/((b-a)*(b-c)) -(a+b)*fc/((c-a)*(c-b))

# x1 = [1.4000, 3.5000, 5.6000]
# y1 = [0.4007, 0.5941, 0.2980]

g = 9.8

# x = np.linspace(0,10,10)
tanangle = Linearlagrange2(1.4000, 3.5000, 5.6000,  0.4007, 0.5941, 0.2980)
angle = round(np.degrees(np.arctan(tanangle)),2)
print(f"El ángulo inicial del disparo fue {angle}")


v_0 = round(np.sqrt(x_max*g/np.sin(2*angle)),2)
print(f"La velocidad inicial del disparo fue {v_0}")
