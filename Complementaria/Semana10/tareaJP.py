import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from sklearn.linear_model import LinearRegression
import random
from scipy.stats import norm

#plt.style.use('dark_background')
import pandas

def GetIntercept( p, size= 10):
    
    muestra = random.choices(p, k=size)
    #print(muestra)
    X = []
    Y = []
    for i in muestra:
        X.append(i[0])
        Y.append(i[1])

    X = np.array(X)
    Y = np.array(Y)
    
    reg=LinearRegression()

    v=reg.fit(X.reshape(-1,1), Y)

    return v.intercept_

def GetPendiente( p, size= 10):
    
    muestra = random.choices(p, k=size)
    #print(muestra)
    X = []
    Y = []
    for i in muestra:
        X.append(i[0])
        Y.append(i[1])

    X = np.array(X)
    Y = np.array(Y)
    
    reg=LinearRegression()

    v=reg.fit(X.reshape(-1,1), Y)

    return v.coef_


def GetY( p, size= 10, x_muestra = 5):
    
    muestra = random.choices(p, k=size)
    #print(muestra)
    X = []
    Y = []
    for i in muestra:
        X.append(i[0])
        Y.append(i[1])

    X = np.array(X)
    Y = np.array(Y)
    
    reg=LinearRegression()
    v=reg.fit(X.reshape(-1,1), Y)

    return v.intercept_ + v.coef_*x_muestra 



def GetSample(p, Npoints = 10000):
    
    lInterceptos = []
    lPendientes = []
    lY = []
    
    for i in range(int(Npoints)):
        lInterceptos.append(GetIntercept(p))
        lPendientes.append(GetPendiente(p))
        lY.append(GetY(p))
        
    return lInterceptos, lPendientes, lY

df = pandas.read_csv("https://raw.githubusercontent.com/ComputoCienciasUniandes/MetComp1_202110/main/ejemplos/linear.csv")
data= pandas.DataFrame(df)
#print(data)
X=np.array(data["x"])
Y=np.array(data["y"])






T=[]
for i in range(len(X)):
    T.append( (X[i], Y[i]) )


def error(X: list, Y: list , x_prueba: float)->tuple:
   
    listas = []

    lInterceptos, lPendientes, lY = GetSample(T)

    listas.append(lInterceptos)
    listas.append(lPendientes)
    listas.append(lY)

    muI, sigmaI = norm.fit( listas[0] )   

    muP, sigmaP = norm.fit( listas[1] )   

    muY, sigmaY = norm.fit( listas[2] )   

    print("Valor medio de Intercepto: ",muI)
    print("Incertidumbre de Intercepto: ",sigmaI)

    print("Valor medio de la pendiente: ",muP)
    print("Incertidumbre de la pendiente: ",sigmaP)

    print("Valor medio de Y: ",muY)
    print("Incertidumbre de Y: ",sigmaY)
    
error(X, Y, 5)