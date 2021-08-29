#%%
import numpy as np
import matplotlib.axes 
import matplotlib.pyplot as plt
from matplotlib import rc

xi, xf, h = 0, 10, 1 
Npoints = int((xf-xi)/h+1)


x = np.linspace(xi,xf,Npoints)
y = np.linspace(xi,xf,Npoints)

x_1, y_1 = np.meshgrid(np.linspace(xi,xf,Npoints), np.linspace(xi,xf,Npoints))

matriz_ex = np.zeros( (len(x), len(y)))
matriz_ey = np.zeros((len(x), len(y)))




def Function(x, y):
    return x**3 - y**2


#fun_y=Function(x, y)
#plt.scatter(x, y, fun_y)

def CentralDerivative_x(f,x,h, y): 
    d = 0.

    if h!=0:
        d = -(f((x+h), y)-f((x-h), y))/(2*h)
      
    return d

def CentralDerivative_y(f,y,h, x):
    d = 0.

    if h!=0:
        d = -(f(y+h, x)-f(y-h, x))/(2*h)
      
    return d

for i in range(len(matriz_ex)): # y
    for j in range(len(matriz_ex)): # x
        print("i",i)
        print("j",i)
        matriz_ex[i][j] = CentralDerivative_x(Function,x[j],h, y[i])
        matriz_ey[i][j] = CentralDerivative_y(Function,y[i],h, x[j])

print(matriz_ex)

Ex = CentralDerivative_x(Function,x,h, y)
#print(Ex)  
Ey = CentralDerivative_y(Function,y,h, x)

fig, ax = plt.subplots()
ax.quiver(x, y, Ex, Ey)
