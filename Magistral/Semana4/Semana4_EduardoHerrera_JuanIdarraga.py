import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


"""
Método 1
"""

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

# Usamos el método de bisección desde 0.1 para hallar la raíz.
# La raíz es el punto de mayor alcance

x_max, _ = Biseccion(0.1,10, Poly, 100, x1, y1, n)


x = np.linspace(0,x_max,30)
px2 = Poly(x,x1,y1,2)
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(x,px2, c="gold")
ax.set_title(r'Movimiento de una bala',fontsize=16)
ax.set_xlabel(r'x[m]',fontsize=14)
ax.set_ylabel(r'y[m]',fontsize=14)
plt.show()



"""
Método 2
"""

# Término lineal
def LinearLagrange1(a,b,c,fa,fb,fc):
    return -(b+c)*fa/((a-b)*(a-c)) -(a+c)*fb/((b-a)*(b-c)) -(a+b)*fc/((c-a)*(c-b))

# Término cuadrático
def LinearLagrange2(a,b,c,fa,fb,fc):
    return (fa/((a-b)*(a-c))) + (fb/((b-a)*(b-c))) + (fc/((c-a)*(c-b)))

# x1 = [1.4000, 3.5000, 5.6000]
# y1 = [0.4007, 0.5941, 0.2980]

g = 9.8
tanangle = LinearLagrange1(x1[0], x1[1], x1[2],  y1[0], y1[1], y1[2])
angle = np.arctan(tanangle)

expr_cuad = LinearLagrange2(x1[0], x1[1], x1[2],  y1[0], y1[1], y1[2])
v0 = np.sqrt((-g)/((np.cos(angle)**2)*2*expr_cuad))

# Con el método 1:
# De la distancia horizontal máxima x_max = (v0^2 sin(2x))/g
# v_0 = np.sqrt(x_max*g/np.sin(2*angle))
# Y se obtiene el mismo resultado

print(f"El ángulo inicial del disparo fue {round(np.degrees(angle),2)}")
print(f"La velocidad inicial del disparo fue {round(v0,2)}")