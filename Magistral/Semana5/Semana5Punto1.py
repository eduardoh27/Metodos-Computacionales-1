import sympy as sym
import numpy as np


def CreatePoly(n):
    
    x = sym.Symbol('x',Real=True)
    y = sym.Symbol('y',Real=True)
    
    y = (x**2-1)**n
    
    poly = sym.diff( y,x,n  )/( 2**n * np.math.factorial(n))
    
    return poly

def NewtonMethod(f,df,xn,error,it,precision=1.0e-6,iterations=1000):
    h = 1.0e-6
    while error > precision and it < iterations:
        
        try:
            
            xn1 = xn - f(xn)/df(f,xn,h)
            
            #error = np.abs( (xn1- xn)/xn1+0.001)
            #print(error)
            
        except ZeroDivisionError:
            None
            
        xn = xn1
        
        it += 1
    
    return xn1


def Derivada(f,x,h):
    
    d = 0.
    
    if h!=0:
        d = (f(x+h)-f(x-h))/(2*h)
        
    return d

def base_legendre( f , Legendre):
    p0=sym.Symbol("p0", Real=True)
    p1=sym.Symbol("p1", Real=True)
    p2=sym.Symbol("p2", Real=True)
    p0=p0
    p1=p1
    p2=p2
    
    poli= f.subs(b, p0).subs(n, p1).subs(m,sym.root(((2*p2+p0)/3),2))
    
    return poli






"""           ///          PUNTO 1.a              ///       """ 
print("\nTarea Métodos Computacionales 1 - Semana 5 Punto 1\n")
print("Punto 1a:")

Legendre = []
n = 31

print("Por favor, espere un momento. Numpy está creando los polinomios simbólicos") 
print("Mientras espera, adivíneme esta: puedo volar, pero no tengo alas. Puedo rugir, pero no tengo boca. Puedo empujar, pero no tengo brazos. ¿Qué soy? c:")

# crea los polinomios simbólicos
for i in range(n):
    Legendre.append(CreatePoly(i))
    

# cambiar simbolico a numerico    
funciones=[]
for i in range(n):
    x = sym.Symbol('x',Real=True)
    f = sym.lambdify( [x] , Legendre[i] , 'numpy' )
    funciones.append(f)

    
derivadas=[]    
fderivada=[] 
for i in Legendre:
    x = sym.Symbol('x',Real=True)
    d=sym.diff(i,x,1)
    df = sym.lambdify( [x] , d , 'numpy' )
    fderivada.append(df)








print("\nLas raíces son:")
raices=[]
for i in range(len(Legendre)):
    n=i
    #encontrar ceros con Newthon rhapson
    xCeros = np.linspace(-1, 1, n*2)

    raices_i = []
    for x in xCeros:
        precision=1.0e-6
        root = NewtonMethod(funciones[n], Derivada, x, 100, it = 0)
        if len(raices_i) == 0:
            raices_i.append(root)
        else:
            existe = False
            for cadaRaiz in raices_i:
                if np.abs(cadaRaiz-root)<precision:
                    existe = True
            if not existe:
                raices_i.append(root)

    raices.append(raices_i)
    print(str(n)+":",raices_i)


# hayar los pesos
print("\nLos pesos son:")
pesos=[]

for i in range(len(raices)):#1
    peso_poly=[]
    (raices[i])
    for j in range(len(raices[i])):#0

        denominador= ((fderivada[i](raices[i][j]))**2)*(1-(raices[i][j])**2)
        peso_raiz= 2/denominador
        peso_poly.append(peso_raiz)
        
    pesos.append(peso_poly)
    print(str(i)+":", peso_poly)
        
print("Sí, yo soy el viento ;)")    
        
      
        




        
"""        ///          PUNTO 1.B        ///         """      

print("\nPunto 1b:")


b = sym.Symbol('b',Real=True)
n = sym.Symbol('n',Real=True)
m = sym.Symbol('m',Real=True)

f= b+2*n+m**2
print("El polinomio: 1 + 2x + x**2 en base de Legendre sería así:")
print(base_legendre(f, Legendre))







        
"""        ///          PUNTO 1.C        ///            """ 

print("\nPunto 1c:")


f = lambda x: 1/((x**4)+1)
g = lambda x: (x**2)/((x**4)+1)

b = 1
a = 0
x = np.array(raices[7])

t = 0.5*((b-a)*x + a + b )

W = np.array(pesos[7])

Integral1 = 0.5*(b-a)*np.sum(W*f(t))
Integral2 = 0.5*(b-a)*np.sum(W*g(t))

Integral = round((Integral1 + Integral2),6)
print("El resultado de la integral es: " + str(Integral)+"\n")     
