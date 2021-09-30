import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
import os.path as path
import wget
from scipy.optimize import curve_fit
#plt.style.use('dark_background')

print('\nTarea Semana 7 - Métodos Computacionales 1')


# Punto 1
print('\n\nPunto 1:\n')

#a
print('a)')
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

print("Como el punto no hace parte del espacio columna \
de la matriz dados por las ecuaciones de las líneas, \
el punto hallado es el punto que minimiza la proyección \
del punto sobre el espacio columna")




# b
print('\nb)')

xi, xf, h = -5.,5.,0.03
Npoints = int((xf-xi)/(h))

x = np.linspace(xi,xf,Npoints)
y = np.linspace(xi,xf,Npoints)

X,Y = np.meshgrid(x,y)
Z = np.zeros((len(x), len(y)))

min = [None, None, np.inf]
for i in range(len(x)):
    for j in range(len(x)):
        v=np.array([x[i],y[j]])
        Z[i][j] = np.linalg.norm((A@v)-b)
        if Z[i][j] < min[2]:
            min = [x[i],y[j],Z[i][j]]

print(f"El punto obtenido de la búsqueda iterativa fue ({min[0]}, {min[1]})")

fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d', elev = 28, azim = 46)
ax.plot_surface(X,Y,Z, cmap=cm.coolwarm)
ax.set_xlabel(r'x',fontsize=10)
ax.set_ylabel(r'y',fontsize=10)
ax.set_zlabel(r'd(x*)',fontsize=10)
plt.show()

print('Los puntos son muy similares: iguales con una dos cifras significativas')







print('\n\nDisclaimer: En los siguientes puntos, interpretamos que la \
primera columna de la tabla con los datos tiene los valores de "x" \
y la segunda columna tiene los valores de "y"')
# Punto 2
print('\nPunto 2:\n')
print('Ver gráfica')

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
print('\nPunto 3:\n')
print('Ver gráfica')

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
print('\nPunto 4:\n')

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

print('Los resultados con el método curve_fit son los mismos tanto para \
el caso lineal como para el caso cuadrático')






# Punto 5
print('\nPunto 5:\n')
print('Ver documento PDF')



# Punto 6
print('\nPunto 6:\n')

# a
print('a)')
A = np.array([[3, 1, -1],
              [1, 2, 0],
              [0, 1, 2],
              [1, 1, -1]])

b = np.array([-3, -3, 8, 9])
At = A.T
Inv = np.linalg.inv(At @ A)
xSol = np.round(Inv @ (At @ b))
print(f'La solución de mínimos cuadrados es: {tuple(xSol)}')

print(f'Se multiplica por la matriz con las bases como columnas:\n{A}')
resp = np.round(A@xSol)
print(f'La proyección ortogonal que se obtiene es: {tuple(resp)}')

# b
print('\nb)')
V = np.array([[3, 1, 0, 1],
              [1, 2, 1, 1],
              [-1, 0, 2, -1]])

def obtener_Base(v, normal = True):
    u = np.zeros((v.shape[0], v.shape[1]))
    
    for k in range(v.shape[0]):
        u[k] = v[k]
        for j in range(k):
            u[k] -= np.dot(v[k], u[j])/ np.dot(u[j], u[j]) * u[j]
        
        if normal:
            u[k] /= np.linalg.norm(u[k])
    return u
        
base_orto = obtener_Base(V, True)
print("La matriz que representa la base ortonormal construida a partir de los vectores dados es: \n", base_orto)
coeffs = np.zeros(base_orto.shape[0])
Aux = base_orto.copy()
for i in range (base_orto.shape[0]):
    coeffs[i] = np.dot(b, base_orto[i])
    Aux[i] = base_orto[i] * coeffs[i]
 
proyfinal = np.round(np.sum(Aux, axis = 0))
print(f"La proyección sobre la base ortonormal es: {tuple(proyfinal)}")






# Punto 1.2
print('\nPunto 1.2:\n')


G1 = np.array([lambda x,y: np.log((x**2) + (y**2)) - np.sin(x*y) - np.log(2) - np.log(np.pi),
     lambda x,y: np.exp(x-y) + np.cos(x*y)])

G2 = np.array([lambda x,y,z: 6*x - 2*np.cos(y*z) - 1,
     lambda x,y,z: 9*y + np.sqrt(x**2 + np.sin(z) + 1.06) + 0.9,
     lambda x,y,z: 60*z + 3*np.exp(-1*x*y) + 10*np.pi - 3])

def GetVectorF(G,r):
    
    dim = len(G)
    v = np.zeros(dim)

    if dim == 3:
        for i in range(dim):
            v[i] = G[i](r[0],r[1],r[2])

    elif dim == 2:
        for i in range(dim):
            v[i] = G[i](r[0],r[1])
            
    return v

def GetJacobian(G,r,h=0.001):
    dim = len(G)
    J = np.zeros( (dim,dim) )
    
    if dim == 3:
            
        for i in range(dim):
            J[i,0] = (G[i](r[0]+h,r[1],r[2]) - G[i](r[0]-h,r[1],r[2]) )/(2*h)  
            J[i,1] = (G[i](r[0],r[1]+h,r[2]) - G[i](r[0],r[1]-h,r[2]) )/(2*h)  
            J[i,2] = (G[i](r[0],r[1],r[2]+h) - G[i](r[0],r[1],r[2]-h) )/(2*h)
        
    elif dim == 2:
        for i in range(dim):
            J[i,0] = (G[i](r[0]+h,r[1]) - G[i](r[0]-h,r[1]) )/(2*h)  
            J[i,1] = (G[i](r[0],r[1]+h) - G[i](r[0],r[1]-h) )/(2*h)  

    return J.T

def NewtonRaphson(G,r,error=1e-10,itmax=1000):
    
    it = 0
    d = 1
    dvector = []
    
    while d > error and it < itmax:
        
        it += 1
        # Valor actual
        rc = r # Valor inicial
        
        F = GetVectorF(G,r)
        J = GetJacobian(G,r)
        InvJ = np.linalg.inv(J)
        
        r = rc - np.dot(InvJ,F)
        
        d = np.linalg.norm(r-rc)
        
        dvector.append(d)
        
    return r,it,dvector

r1 = np.array([2,2])
r1,it,distancias = NewtonRaphson(G1,r1)

r2 = np.array([0,0,0])
r2,it,distancias = NewtonRaphson(G2,r2)

print(f'La solución del primer sistema de ecuaciones \
no lineales es {tuple(r1)}')

print(f'La solución del segundo sistema de ecuaciones \
no lineales es {tuple(r2)}\n')
