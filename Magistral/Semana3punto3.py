#%%
import numpy as np
import matplotlib.axes 
import matplotlib.pyplot as plt
from matplotlib import rc

xi, xf, h = 0., 1., 0.05
Npoints = int((xf-xi)/h+1)
#print(Npoints)

x_1 = np.linspace(xi,xf,Npoints)
y_1 = np.linspace(xi,xf,Npoints)

x, y = np.meshgrid(np.linspace(xi,xf,Npoints), np.linspace(xi,xf,Npoints))

def Function(x, y):
    return (1/(np.sqrt( ((x-0.51)**2) + ((y-0.21)**2) ))) - (1/np.sqrt( ((x-0.51)**2) + ((y+0.21)**2) ))


def CentralDerivative_x(f,x,h, y): 
    d = 0.
    
    if h!=0:
        d = -(f(x+h, y)-f(x-h, y))/(2*h)
      
    return d

def CentralDerivative_y(f,y,h, x):
    d = 0.

    if h!=0:
        d = -(f(y+h, x)-f(y-h, x))/(2*h)
      
    return d


Ex = CentralDerivative_x(Function,x,h, y) 
Ey = CentralDerivative_y(Function,y,h, x)



matriz_ex = np.zeros( (len(x), len(y)))
matriz_ey = np.zeros((len(x), len(y)))

for i in range(len(matriz_ex)): # y
    for j in range(len(matriz_ex)): # x
        #print("i",i)
        #print("j",i)
        matriz_ex[i][j] = CentralDerivative_x(Function,x_1[j],h, y_1[i])
        matriz_ey[i][j] = CentralDerivative_y(Function,y_1[i],h, x_1[j])

print(matriz_ex)

fig, ax = plt.subplots()
ax.quiver(x, y, matriz_ex, matriz_ey)
plt.show()





# for i in range(Npoints):
#     for j in range(Npoints):
#         ax.quiver(x[i], y[j], Ex[i,j], Ey[i,j])