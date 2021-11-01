import numpy as np
import matplotlib.pyplot as plt
from random import choices
from scipy.stats import norm
import os


def getNumber():
    movs = [1, 2, -1]
    probs = [0.7, 0.1, 0.2]
    resultado = choices(movs, probs)
    return resultado[0]
    
def getRecorrido():
    Nsteps = 100
    recorrido = 0

    for i in range(Nsteps):
        recorrido += getNumber()

    return recorrido

def MonteCarlo():
    N = int(5e4)
    recorridos = []
    for i in range(N):
        recorridos.append(getRecorrido())
    return recorridos

def main():
    recorridos = MonteCarlo()
    recorridos2 = []
    for i in recorridos:
        recorridos2.append(i**2)
    ev_x = np.mean(recorridos)
    ev_x2 = np.mean(recorridos2)
    var_x = ev_x2 - (ev_x**2)
    std_x = np.sqrt( var_x )
    print(f"Ex = {ev_x}, Ex2 = {ev_x2}, StdX = {std_x}")

    mu, sigma = norm.fit( recorridos )
    bins = np.linspace(0, 200, 201)
    y = norm.pdf(bins, mu, sigma )
    plt.plot(bins, y, c='orange')
    plt.hist(recorridos, density=True, bins=bins, color='#1f77b4')
    plt.xlabel('Total Steps')
    plt.ylabel('Probability')
    plt.savefig('HerreraEduardo_grafica.png')
    plt.show()

if __name__ == '__main__':
    main()
