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

G1 = np.array([lambda x,y: np.log(x**2 + y**2) - np.sin(x*y) - np.log(2) - np.log(np.pi),
     lambda x,y: np.exp(x-y) + np.cos(x*y)])
print(G1[1](1.736083,1.804428))

G2 = np.array([lambda x,y,z: 6*x - 2*np.cos(y*z) - 1,
     lambda x,y,z: 9*y + np.sqrt(x**2 + np.sin(z) + 1.06) + 0.9,
     lambda x,y,z: 60*z + 3*np.exp(-1*x*y) + 10*np.pi - 3])
print(G2[2](0.4951447,-0.1996059, -0.5288260))