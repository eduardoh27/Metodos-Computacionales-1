import sympy as sym
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import rc
from numpy.polynomial import polynomial as Pol

def CreatePoly(n):
    
    x = sym.Symbol('x',Real=True)
    y = sym.Symbol('y',Real=True)
    
    y = (x**2-1)**n
    
    poly = sym.diff( y,x,n  )/( 2**n * np.math.factorial(n))
    
    return poly


def NewtonMethod(f,df,xn,error,it,precision=1.0e-6,iterations=1000):
    #print("xn: ",xn)
    h = 1.0e-6
    
    while error > precision and it < iterations:
        
        try:
            xn1 = xn - f(xn)/df(f,xn,h)
            
            error = np.abs( (xn1- xn)/xn1 )
            
        except ZeroDivisionError:
            print('Hay una division por cero')
            
        xn = xn1
        
        it += 1

        #print("xn: ",xn)
    
    #NewtonMethod(f, df, xn, error, it)
    
    return xn1



def Derivada(f,x,h):
    
    d = 0.
    
    if h!=0:
        d = (f(x+h)-f(x-h))/(2*h)
        
    return d



"""ceros y pesos teoricos"""
deg = 10
x, w = np.polynomial.legendre.leggauss(deg)
for i in range(deg):
    #print(x[i],w[i])
    None
    

"""procedimiento"""
Legendre = []

n = 30
# guardar todos los polinomios
for i in range(n):
    newPoly =  CreatePoly(i)
    #print(newPoly)
    Legendre.append(newPoly)
 
#print(Legendre)    

#x = sym.Symbol('x',Real=True)
# cambiar simbolico a numerico    
funciones=[]
for i in range(n):
    x = sym.Symbol('x',Real=True)
    f = sym.lambdify( [x] , Legendre[i] , 'numpy' )
    funciones.append(f)

derivadas=[]    
# derivadas 
for i in Legendre:
    x = sym.Symbol('x',Real=True)
    derivadas.append(sym.diff(i,x,1))

n = 29
print(Legendre[n])
#encontrar ceros con Newthon rhapson
xCeros = np.linspace(-1, 1, n*3)
print(xCeros)
raices = []
for x in xCeros:
    precision=1.0e-6
    root = NewtonMethod(funciones[n], Derivada, x, 100, it = 0)
    if len(raices) == 0:
        raices.append(root)
    else:
        #print("root: ", root)
        #print("raices: ", raices)
        existe = False
        for cadaRaiz in raices:
            if np.abs(cadaRaiz-root)<precision:
                existe = True
        if not existe:
            raices.append(root)
print(len(raices))



# hayar los pesos

"""
pesos=[]
for i in legendre:
    denominador = ((i(raices))**2)*(1-)
"""
