import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from scipy.stats import norm


def getBootValues(X, Y, x_prueba, size = 40):
    indices = np.arange(len(X))
    muestra = random.choices(indices, k=size)
    X_muestra = []
    Y_muestra = []
    for i in muestra:
        X_muestra.append(X[i])
        Y_muestra.append(Y[i])
    
    X_muestra = np.array(X_muestra)
    Y_muestra = np.array(Y_muestra)

    reg=LinearRegression().fit(X_muestra.reshape(-1,1), Y_muestra)
    inter = reg.intercept_
    pend = reg.coef_[0]
    valorY_5 = inter + pend*x_prueba

    return inter, pend, valorY_5

def GetSample(X, Y, x_prueba, Npoints = int(1e4)):
    
    listInter = []
    listPend = []
    listValoresY = []
    
    for i in range(int(Npoints)):
        inter, pend, valorY = getBootValues(X, Y, x_prueba)

        listInter.append(inter)
        listPend.append(pend)
        listValoresY.append(valorY)
        
    return listInter, listPend, listValoresY

def error(X, Y, x_prueba = 5):

    listInter, listPend, listValoresY = GetSample(X, Y, x_prueba)

    muI, sigmaI = norm.fit(listInter)   
    muP, sigmaP = norm.fit(listPend)   
    muY, sigmaY = norm.fit(listValoresY)   

    return muI, sigmaI, muP, sigmaP, muY, sigmaY


def main():
    
    url = "https://raw.githubusercontent.com/ComputoCienciasUniandes/MetComp1_202110/main/ejemplos/linear.csv"
    df = pd.read_csv(url)

    X = np.array(df["x"])
    Y = np.array(df["y"])
    
    """
    plt.scatter(X,Y)
    plt.show()
    reg=LinearRegression().fit(X.reshape(-1,1), Y)
    inter = reg.intercept_ # 4.47
    pend = reg.coef_[0] # 1.58
    valorY_5 = inter + pend*5 # 12.38

    print(f'inter = {inter}')
    print(f'pend = {pend}')
    print(f'y (x=5) = {valorY_5}')
    """

    muI, sigmaI, muP, sigmaP, muY, sigmaY = error(X, Y, 5)

    print("Distribución del intercepto:")
    print(f"Valor medio = {muI}")
    print(f"Incertidumbre = {sigmaI}")
 
    print("\nDistribución de la pendiente:")
    print(f"Valor medio = {muP}")
    print(f"Incertidumbre = {sigmaP}")

    print("\nDistribución de y para x_prueba=5:")
    print(f"Valor medio = {muY}")
    print(f"Incertidumbre = {sigmaY}")


if __name__ == "__main__":
    main()
