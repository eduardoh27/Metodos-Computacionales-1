import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
import numpy as np
#plt.style.use('dark_background')

print('\nTarea Semana 7 - Métodos Computacionales 1')


# Punto 1a
print('\n\nPunto 1a\n')

A = np.array([[2, -1],
              [1, 2],
              [1, 1]])
b = np.array([2, 1, 4])

At = A.T
Inv = np.linalg.inv(At @ A)
xSol = Inv @ (At @ b)

print(f"El punto solución es ({xSol[0]}, {xSol[1]})")

x = np.linspace(0,8,100)
l1 = lambda x:  2*x- 2
l2 = lambda x: -((x-1)/2)
l3 = lambda x: 4-x
y1 = l1(x)
y2 = l2(x)
y3 = l3(x)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(x, y1)
ax.plot(x, y2)
ax.plot(x, y3)
ax.set_xlabel(r'x',fontsize=10)
ax.set_ylabel(r'y',fontsize=10)
ax.plot(xSol[0], xSol[1], "o", c="r")
plt.show()

print("Como el punto no hace parte del espacio columna\
de la matriz dados por las ecuaciones de las líneas,\
el punto hallado es el punto que minimiza la proyección\
del punto sobre el espacio columna")




# Punto 1b
print('\nPunto 1b\n')

xi, xf, h = -5.,5.,0.03
Npoints = int((xf-xi)/(h))

x = np.linspace(xi,xf,Npoints)
y = np.linspace(xi,xf,Npoints)

X,Y = np.meshgrid(x,y)
Z = np.zeros((len(x), len(y)))

#print(x)
min = [None, None, np.inf]
for i in range(len(x)):
    for j in range(len(x)):
        v=np.array([x[i],y[j]])
        Z[i][j] = np.linalg.norm((A@v)-b)
        if Z[i][j] < min[2]:
            min = [x[i],y[j],Z[i][j]]

print(f"El punto obtenido fue ({min[0]}, {min[1]})")

fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d', elev = 28, azim = 46)
ax.plot_surface(X,Y,Z, cmap=cm.coolwarm)
ax.set_xlabel(r'x',fontsize=10)
ax.set_ylabel(r'y',fontsize=10)
ax.set_zlabel(r'd(x*)',fontsize=10)
plt.show()





# Punto 2
print('\nPunto 2\n')

