import sympy as sym
import numpy as np

# Generar los polinomios de Hermite
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

n=8 # grado de polinomios a calcular

Hermite = []
for i in range(n+1):
    Hermite.append(CreatePoly(i))

funciones=[]
for i in range(n+1):
    x = sym.Symbol('x',Real=True)
    fx = sym.lambdify([x], Hermite[i] , 'numpy' )
    funciones.append(fx)

psi =  lambda x, n : (1/np.sqrt((2**n) * np.math.factorial(n))) * ((1/np.pi)**(1/4)) * (np.exp(-1*(x**2)/2)) * (funciones[n](x)) 
f = lambda x:  psi(x,n=1)**2 * (x**2) / (np.exp(-1*x**2))

def GaussHermite(n):
    xCeros = np.linspace(-4, 4, n*3)
    raices = []
    for x in xCeros:
        precision=1.0e-6
        root = NewtonMethod(funciones[n], Derivada, x, 100, it = 0) 
        if len(raices) == 0:
            raices.append(root)
        else:
            existe = False
            for cadaRaiz in raices:
                if np.abs(cadaRaiz-root)<precision:
                    existe = True
            if not existe:
                raices.append(root)
    raices.sort()
    pesos = []
    for raiz in raices:
        num = (2**(n-1))*(np.math.factorial(n))*np.sqrt(np.pi)
        den = ((funciones[n-1](raiz))**2) * (n**2)
        newPeso = num / den
        pesos.append(newPeso)

    return raices, pesos

def getIntegral(raices, pesos, f):
    integral = 0 
    for i in range(0, len(raices)):
        integral += pesos[i]*f(raices[i])
    return integral

def getZeros(min_, max_):

    for n in range(min_, max_+1):
        raices, pesos = GaussHermite(n)
        print(f"\nGrado {n}:")
        for i, j in zip(raices, pesos):
            print(f"raíz = {i} // peso = {j}")




print("\nBono Métodos Computacionales 1 - Cuadratura de Gauss-Hermite\n")

# Raíces y pesos
print(f"\nLas raíces y sus pesos de la cuadratura Gauss-Hermite hasta grado {n} son:")
getZeros(1, n)

# Integral
raices_, pesos_ = GaussHermite(n)
print("\n\nEl valor de la integral usando cuadratura Gauss-Hermite de "+
    f"grado {n} es: {np.round(getIntegral(raices_, pesos_, f),10)}\n")
