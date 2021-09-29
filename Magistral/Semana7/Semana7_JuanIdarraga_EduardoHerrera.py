import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
import os.path as path
import wget
from scipy.optimize import curve_fit
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

"""fig = plt.figure()
ax = fig.add_subplot()
ax.plot(x, y1)
ax.plot(x, y2)
ax.plot(x, y3)
ax.set_xlabel(r'x',fontsize=10)
ax.set_ylabel(r'y',fontsize=10)
ax.plot(xSol[0], xSol[1], "o", c="r")
plt.show()"""

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

"""fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d', elev = 28, azim = 46)
ax.plot_surface(X,Y,Z, cmap=cm.coolwarm)
ax.set_xlabel(r'x',fontsize=10)
ax.set_ylabel(r'y',fontsize=10)
ax.set_zlabel(r'd(x*)',fontsize=10)
plt.show()"""



# Punto 2
print('Disclaimer: En los siguintes puntos, interpretamos que la\
    primera columna es para los valores de "x" y la segunda para \
    los valores de "y"')
print('\nPunto 2\n')

file = 'datos.dat'
url = 'https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/MinimosLineal.txt'

if not path.exists(file):
    Path_ = wget.download(url,file)
    print('File downloaded')

data = np.loadtxt(file)
x = data[:,0]
y = data[:,1]
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
pts = x.size

P = np.array([np.ones([pts, 1]), x]).reshape(2, pts).T
v = (np.linalg.inv(P.T @ P) @ P.T) @ y
b, m = v
coeffs = [m[0], b[0]]
ajuste = lambda x: coeffs[1] + x*coeffs[0]

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(x, y, s=15)
line, = ax.plot(x, ajuste(x), c='r')
line.set_label(f'$a_0$ = {round(coeffs[1],2)}, $a_1$ = {round(coeffs[0],2)}')  
ax.set_xlabel(r'x',fontsize=10)
ax.set_ylabel(r'y',fontsize=10)
ax.set_title(r'Ajuste lineal', fontsize=14)
fig.legend(loc='upper right') 
plt.show()



# Punto 3
print('\nPunto 3\n')

file = 'datos1.dat'
url = 'https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/MinimosCuadratico.txt'

if not path.exists(file):
    Path_ = wget.download(url,file)
    #print('File downloaded')

data1 = np.loadtxt(file)
x = data1[:,0]
y = data1[:,1]
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
pts = x.size

P = np.array([np.ones([pts, 1]), x, x**2]).reshape(3, pts).T
v = (np.linalg.inv(P.T @ P) @ P.T) @ y
b, m, m1 = v
coeffs = [m1[0], m[0], b[0]]
ajuste1 = lambda x: coeffs[2] + x*coeffs[1] + x**2*coeffs[0] 
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(x, y, s=15)
line, = ax.plot(x, ajuste1(x), c='r')
line.set_label(f'$a_0$ = {round(coeffs[2],2)}, $a_1$ = {round(coeffs[1],2)}, $a_2$ = {round(coeffs[0],2)}')  
ax.set_xlabel(r'x',fontsize=10)
ax.set_ylabel(r'y',fontsize=10)
ax.set_title(r'Ajuste cuadrático', fontsize=14)
fig.legend(loc='lower right') 
plt.show()



# Punto 4
print('\nPunto 4\n')

lineal = lambda x, a, b: a*x + b

x = data[:,0]
y = data[:,1]
 
a, b = curve_fit(lineal, x, y)[0]

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(x, y, s=15)
line, = ax.plot(x, lineal(x, a, b), c='r')
line.set_label(f'$a_0$ = {round(b,2)}, $a_1$ = {round(a,2)}')  
ax.set_xlabel(r'x',fontsize=10)
ax.set_ylabel(r'y',fontsize=10)
ax.set_title(r'curve_fit lineal', fontsize=14)
fig.legend(loc='lower right') 
plt.show()



cuadratico = lambda x, a, b, c: a*x**2 + b*x + c

x1 = data1[:,0]
y1 = data1[:,1]

a1, b1, c1 = curve_fit(cuadratico, x1, y1)[0]

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(x1, y1, s=15)
line, = ax.plot(x1, cuadratico(x1, a1, b1, c1), c='r')
line.set_label(f'$a_0$ = {round(c1,2)}, $a_1$ = {round(b1,2)}, $a_2$ = {round(a1,2)}')  
ax.set_xlabel(r'x',fontsize=10)
ax.set_ylabel(r'y',fontsize=10)
ax.set_title(r'curve_fit cuadrático', fontsize=14)
fig.legend(loc='lower right') 
plt.show()
# 'COMPARAR???'
# SÍ LO HICE CON MÍNIMOS CUADRADOS???????


# Punto 5
print('\nPunto 5\n')



# Punto 6
print('\nPunto 6\n')



# Punto 1.2
print('\nPunto 1.2\n')