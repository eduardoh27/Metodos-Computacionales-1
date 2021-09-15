import sympy as sym
import numpy as np
import matplotlib.pyplot as plt


# Generar los polinomios de Laguerre
def CreatePoly(n):

    x = sym.Symbol('x', Real = True)
    y = sym.Symbol('y', Real = True)

    y = (sym.exp(-1*x)*(x**n))

    poly = (sym.exp(x)*sym.diff( y,x,n )) / np.math.factorial(n)
    return poly


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


n=15
Laguerre = []
for i in range(n+1):
    Laguerre.append(CreatePoly(i))


funciones=[]
for i in range(n+1):
    x = sym.Symbol('x',Real=True)
    fx = sym.lambdify([x], Laguerre[i] , 'numpy' )
    funciones.append(fx)

"""
# Cálculo con los puntos y pesos de numpy

f = lambda x : (x**3)/(np.e**x-1)

deg = 10
x, w = np.polynomial.laguerre.laggauss(deg)
for i in range(deg):
    print(x[i],w[i])

suma = 0
for i in range(deg):
    suma += w[i]*(np.exp(x[i]))*f(x[i])

valorReal = np.pi**4 / 15
"""

f = lambda x : (x**3)/(np.e**x-1)


def GaussLaguerre(n):
    xCeros = np.linspace(0, 30, n*3)
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

    pesos = []
    for raiz in raices:
        newPeso = raiz / ((n+1)**2 * funciones[n+1](raiz)**2)
        pesos.append(newPeso)

    integral = 0 
    for i in range(len(raices)):
        integral += pesos[i]*(np.exp(raices[i]))*f(raices[i])
    
    return integral





print("\nTarea Métodos Computacionales 1 - Semana 5 Punto 2\n")

print("Punto 2a:")
# Parte a
n = 3
integral = GaussLaguerre(n)
print(f"El valor de la integral con n={n} es: {integral}\n") 


# Parte b
min, max = 2, 10
lista_n = [n for n in range(min, max+1)]
valorReal = np.pi**4/15
resultados = []
for n in lista_n:
    resultados.append(GaussLaguerre(n)/valorReal)
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel(r'n',fontsize=14)
ax.set_ylabel(r'$\epsilon$(n)',fontsize=14)
plt.scatter(lista_n, resultados, color='b',label='Laguerre quadrature acurracy')
plt.grid()
plt.legend()
plt.show()
