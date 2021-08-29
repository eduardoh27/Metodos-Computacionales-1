import numpy as np
import matplotlib.pyplot as plt

def poly(x):
    return x + (np.e**(-2*x)) - 1

xi, xf, Npoints = 0.1,4,20
h = (xf-xi)/float(Npoints)

x = np.linspace(xi,xf,Npoints)
y = poly(x)

plt.plot(x,y)
plt.show()

def NewtonMethod(f,df,xn,error,it,precision=0.000001,iterations=1000):
    
    h = 1.0e-4
    
    while error > precision and it < iterations:
        
        try:
            
            xn1 = xn - f(xn)/df(f,xn,h)
            
            error = np.abs( (xn1- xn)/xn1 )
            #print(error)
            
        except ZeroDivisionError:
            print('Hay una division por cero')
            
        xn = xn1
        
        it += 1
    
    return xn1