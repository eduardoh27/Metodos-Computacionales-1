"""
Método de Monte Carlo para la obtención de pi
"""

import numpy as np
import random


def MonteCarloMethod (iteraciones):
    """
    Rango de operaciones: cuadrado en x = [0,3] y  y = [0,3]
    Circulo centrado en 1,1
    """
    circulo = 0
    cuadrado = 0

    diametro = 2

    i = 1
    while i < iteraciones:
        aleatoriox = random.random()*3
        aleatorioy = random.random()*3

        if (aleatoriox-1)**2 + (aleatorioy-1)**2 <= (diametro/2)**2:
            circulo += 1
        
        if aleatoriox >= 2 and aleatoriox <= 3:
            if aleatorioy >= 2 and aleatorioy <= 3:
                cuadrado += 1

        """if i % imprimir_cada == 0:

            pi = circulo / cuadrado
            print(pi) """
        i+=1
    print(circulo/cuadrado)

#MonteCarloMethod(100000000)

a = np.zeros(4)
print(a)
