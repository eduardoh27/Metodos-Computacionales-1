"""

#Método de Monte Carlo para la obtención de pi

import numpy as np
import random


def MonteCarloMethod (iteraciones):
    
    # Rango de operaciones: cuadrado en x = [0,3] y  y = [0,3]
    # Circulo centrado en 1,1
    
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

        # if i % imprimir_cada == 0:

        #     pi = circulo / cuadrado
        #         print(pi) 
        i+=1
    print(circulo/cuadrado)

#MonteCarloMethod(10000000)

G1 = np.array([lambda x,y: np.log(x**2 + y**2) - np.sin(x*y) - np.log(2) - np.log(np.pi),
     lambda x,y: np.exp(x-y) + np.cos(x*y)])
#print(G1[1](1.736083,1.804428))

G2 = np.array([lambda x,y,z: 6*x - 2*np.cos(y*z) - 1,
     lambda x,y,z: 9*y + np.sqrt(x**2 + np.sin(z) + 1.06) + 0.9,
     lambda x,y,z: 60*z + 3*np.exp(-1*x*y) + 10*np.pi - 3])
#print(G2[2](0.4951447,-0.1996059, -0.5288260))
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from tqdm import tqdm

class MyRandom():
    
    def __init__(self, seed = 15, method='simple'):
        
        self.r = seed
        self.method = method
        
        if method=='simple':
            self.a = 57
            self.c = 1
            self.M = 265
        elif method == 'drand48':
            self.a = int('5DEECE66D',16)
            self.c = int('B',16)
            self.M = 2**48
        else:
            print('Generador no reconocido')
            
    def Random(self):
        
        r = (self.a*self.r + self.c)%self.M
        self.r = r
        
        return r/float(self.M)
    
    def TestMethod(self, Npoints, moment, seed_ = 32, method_ = 'simple'):
        
        rand = MyRandom(seed = seed_, method = method_)
        
        array = np.zeros(Npoints)
            
        for i in range(Npoints):
            array[i] = rand.Random()
            
        return np.sqrt(Npoints)* np.abs(  np.mean(array**moment) - 1./(1.+moment) )

def FillPoints(seed_, method_, Npoints):
    
    rand = MyRandom(seed = seed_, method = method_)
    
    points = np.zeros(Npoints)
    
    for i in tqdm(range(Npoints)):
        points[i] = rand.Random()
        
    return points

#print((25214903917*6625+11)%(2**48)/(2**48))

Npoints = 50
Nsimple = FillPoints(165, 'simple', Npoints)
Nrand48 = FillPoints(695, 'drand48', Npoints)
#print(Nsimple)

indices = np.arange(Npoints)
#print(indices)
Even = (indices%2) == 0
#print(Even)

print(Nsimple[~Even])

def Logitech835Function():
    return "mi primera función con mi teclado mecánico Logitech :3"

print(Logitech835Function())
