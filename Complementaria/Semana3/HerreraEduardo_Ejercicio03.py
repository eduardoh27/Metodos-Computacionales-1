import numpy as np
import matplotlib.pyplot as plt

def poly(x, c):
    return -x - (np.e**(-c*x)) + 1

def Derivada(f,x,h, c):
    
    d = 0.
    
    if h!=0:
        d = (f(x+h, c)-f(x-h, c))/(2*h)
        
    return d

def NewtonMethod(f,c,df,xn,error,it,precision=1.0e-6,iterations=1000):
    
    h = 1.0e-6
    
    while error > precision and it < iterations:
        
        try:
            xn1 = xn - f(xn,c)/df(f,xn,h, c)
            
            error = np.abs( (xn1- xn)/xn1 )
            
        except ZeroDivisionError:
            print('Hay una division por cero')
            
        xn = xn1
        
        it += 1
    
    return xn1

# Punto 1
a = 2
solution_a = NewtonMethod(poly, a, Derivada, 2, 100, it = 1)
print(f"La solución de la ecuación para c = 2 es {solution_a}")

# Punto 2
xi, xf, h = 0.,3., 0.01
Npoints = int((xf-xi)/float(h)+1)

c = np.linspace(0., 3., Npoints)
x = np.zeros(len(c))
for i in range(len(c)):
    x[i] = NewtonMethod(poly, c[i], Derivada, 2, 100, it=1)

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(1,1,1)
plt.plot(c,x)
ax.set_title(r'x = 1 - e^(-cx)',fontsize=16)
ax.set_xlabel(r'c',fontsize=14)
ax.set_ylabel(r'x',fontsize=14)
plt.savefig('HerreraEduardo_Ejercicio3.png')
plt.show()