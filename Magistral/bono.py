import sympy as sym
import numpy as np
import matplotlib.pyplot as plt


# Generar los polinomios de Laguerre
def CreatePoly(n):

    x = sym.Symbol('x', Real = True)
    y = sym.Symbol('y', Real = True)

    y = (sym.exp(-1*x**2))

    poly = (sym.exp(x**2)*sym.diff( y,x,n )) * ((-1)**n)
    return poly

np.seterr(all = 'ignore') 
def NewtonMethod(f,df,xn,error,it,precision=1.0e-6,iterations=1000):

    h = 1.0e-6
    
    while error > precision and it < iterations:
        
        try:
            xn1 = xn - f(xn)/df(f,xn,h)
            error = np.abs( (xn1- xn)/xn1 )
            
        except ZeroDivisionError:
            print('Hay una division por cero')
            
        xn = xn1
        
        it += 1

    return xn1


def Derivada(f,x,h):

    d = 0.

    if h!=0:
        d = (f(x+h)-f(x-h))/(2*h)   

    return d

n=10
Hermite = []
for i in range(n+1):
    Hermite.append(CreatePoly(i))
#print(Hermite)

funciones=[]
for i in range(n+1):
    x = sym.Symbol('x',Real=True)
    fx = sym.lambdify([x], Hermite[i] , 'numpy' )
    funciones.append(fx)

#rint("fun",funciones[1](2))
#print("H1:",Hermite[1])
psi =  lambda x, n : (1/np.sqrt((2**n) * np.math.factorial(n))) * ((1/np.pi)**(1/4)) * (np.exp(-1*(x**2)/2)) * (funciones[n](x)) 
#psi_cuadrado = lambda x, n=1 : (1/((2**n) * np.math.factorial(n))) * ((1/np.pi)**(1/2)) *(2*x)**2 #* funciones[n](x) #* (np.exp(-1*x**2))

#def psi_cuadrado(x, n=1):
#    return (1/(2**n * np.math.factorial(n))) * (1/np.sqrt(np.pi)) *(2*x)
#f = lambda x : (1/2)*(np.sqrt(1/np.pi))*(4*x**2) * (x**2) # * (np.exp(-1*x**2))

#f = lambda x:  psi_cuadrado(x,1) * (x**2) #/ (np.exp(-1*x**2))
f = lambda x:  psi(x,n=1)**2 * (x**2) / (np.exp(-1*x**2))
"""
# Cálculo con los puntos y pesos de numpy
sd
deg = 1
x, w = np.polynomial.hermite.hermgauss(deg)
for i in range(deg):    
    print(x[i],w[i])

print(Hermite)

integral = 0
for i in range(deg):
    integral += w[i] * f(x[i])
#print(integral)
"""



n = 5
def GaussHermite(n):
    try:
        xCeros = np.linspace(-4, 4, n*3) # REVISAR LIMTES
        raices = []
        for x in xCeros:
            precision=1.0e-6
            root = NewtonMethod(funciones[n], Derivada, x, 100, it = 0) # funciones +1 resuelve
            if len(raices) == 0:
                raices.append(root)
            else:
                existe = False
                for cadaRaiz in raices:
                    if np.abs(cadaRaiz-root)<precision:
                        existe = True
                if not existe:
                    raices.append(root)

        pesos = []
        for raiz in raices:
            num = (2**(n-1))*(np.math.factorial(n))*np.sqrt(np.pi)
            den = ((funciones[n-1](raiz))**2) * (n**2)
            newPeso = num / den
            pesos.append(newPeso)

        return raices, pesos
    except:
        print("El grado ingresado no es válido")

print("GaussHermite")
raices_, pesos_ = GaussHermite(6)
for i, j in zip(raices_, pesos_):
    print(i, j)
""""""

def getIntegral(raices, pesos, f):
    integral = 0 
    for i in range(0, len(raices)):
        integral += pesos[i]*f(raices[i])
    return integral

def getZeros(min_, max_):

    for n in range(min_, max_+1):
        raices, pesos = GaussHermite(n)
        print(f"\nLas raíces y sus pesos de la cuadratura de grado {n} son:")
        for i, j in zip(raices, pesos):
            print(f"raíz = {i} // peso = {j}")

getZeros(1, n)

print("\nEl valor de la integral usando cuadratura Gauss-Hermite de "+
    f"grado {n} es: {np.round(getIntegral(raices_, pesos_, f),10)}")
