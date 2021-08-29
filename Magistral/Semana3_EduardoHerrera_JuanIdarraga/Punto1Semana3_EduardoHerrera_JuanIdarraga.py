import numpy as np

def Function(x):
    return 1/(np.sqrt(1+np.exp(-x**2)))


def Derivative(x):
    return (x*np.exp(-x**2)) / ((np.exp(-x**2)+1)**(3/2))


def CentralDerivative(funcion,x,h):
    d = 0. 

    if h!=0:
        d = (funcion(x+h)-funcion(x-h))/(2*h)

    return d


def Error(Dy, Dcy):
    Sum_num = 0.
    Sum_den = 0.

    ErrorLocal = np.abs(Dcy-Dy)

    for i in range(len(ErrorLocal)):
        Sum_num += ErrorLocal[i]**2
        Sum_den += Dy[i]**2
    
    ErrorGlobal = np.sqrt( Sum_num / Sum_den )

    return ErrorLocal, ErrorGlobal


x_cero = 0.

xi, xf, h = -10. , 10. , 0.05
Npoints = int((xf-xi)/float(h) + 1)

x = np.linspace(xi,xf,Npoints)

Derivada_intervalo = CentralDerivative(Function, x ,h)


def opcion1():
    Derivada_en_cero = CentralDerivative(Function,x_cero,h)
    print("\na)")
    print("La derivada en 0 es: "+str(Derivada_en_cero))

def opcion2():    
    print("\nb)")
    for i in range(len(Derivada_intervalo)):
        print("La derivada en x="+str(round(x[i],2))+" es",Derivada_intervalo[i])

def opcion3 ():
    print("\nc)")
    Dy = Derivative(x)

    error_local, error_global = Error(Dy, Derivada_intervalo)

    for i in range(len(Derivada_intervalo)):
        print("El error local en x="+str(round(x[i],2))+" es",error_local[i])
    
    print("\nEl error global es: "+str(error_global))

def printMenu():
    print("\na) Derivada el punto x = 0")
    print("b) Derivada en el intervalo −10 ≤ x ≤ 10")
    print("c) Error en cada nodo y error global de la aproximación")
    print("d) Salir")

def main():
    print("\nTarea 3 Métodos Computacionales: Punto 1")

    printMenu()
    entrada = input("Ingrese el literal: ")

    while entrada != "d" and entrada != "d)":
        if entrada == "a" or entrada == "a)":
            opcion1()
        elif entrada == "b" or entrada == "b)":
            opcion2()
        elif entrada == "c" or entrada == "c)":
            opcion3()
        printMenu()
        entrada = input("Ingrese el literal: ")

main()