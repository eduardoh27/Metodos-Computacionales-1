# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 06:43:35 2021

@author: juanpa I Eduardo H
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
        
        array = np.zeros(Npoints)
            
        for i in range(Npoints):
            #array[i] = rand.Random()
            array[i] = np.random.rand()
            
        return np.sqrt(Npoints)* np.abs(  np.mean(array**moment) - 1./(1.+moment) )
    
    
    
    def Calcular_correlacion(self, Npoints, lstnumbers, salto_k):
        productos=[]
        for i in range(Npoints):
            if (i+salto_k) < Npoints: 
                product= lstnumbers[i]*lstnumbers[i+salto_k]
            else:
                None
                
            productos.append(product)
        suma=sum(productos)
        
        return (suma/Npoints) 
    
    
def FillPoints(seed_, method_, Npoints):
    
    rand = MyRandom(seed = seed_, method = method_)
    
    points = np.zeros(Npoints)
    
    for i in tqdm(range(Npoints)):
        points[i] = rand.Random()
        
    return points

Npoints= int(1e6)
Nrand48 = FillPoints(695, 'drand48', Npoints)    



rand2 = MyRandom(seed = 96, method='drand48')

Points = np.logspace(2,6,5)

k=10
Moments = []

for k in tqdm(range(k+1)):
    
    test2 = []
    
    for i in range(len(Points)):
        test2.append(rand2.TestMethod(int(Points[i]),k+1,96,'drand48'))
        

    Moments.append(test2)

    
labelk = []
for i in range(k):
    labelk.append(i+1)
    
    
fig = plt.figure()
ax2 = fig.add_subplot()

for i in range(k):
    ax2.plot(Points,Moments[i], label=r'$k=%.0f$' %(labelk[i]))

ax2.set_title('Generador numpy', fontsize=14, style='italic')
ax2.set_ylabel('k-moment value', fontsize=12)    
ax2.set_xlabel('N Points', fontsize=12)
ax2.set_xscale('log')
ax2.legend(loc=1)
plt.show()


"""   ///   Segundo Punto   \\\   """

N =int( 1e4) # numero de puntos
k2= 30  #numero de vecinos cercanos que vamos a tomar 
kn= np.linspace(0,30,30)
  # numero de numeros que vamos a tomar 

AleN=[]

for i in range(N):
    AleN.append(np.random.rand())
    
    
Cs=[]
for i in range (30):
    Cs.append( rand2.Calcular_correlacion( N, AleN, i+1 ))
    

fig = plt.figure()
ax3 = fig.add_subplot()
ax3.plot(kn, Cs)
ax3.set_ylabel('C(k)', fontsize=12)    
ax3.set_title('Generador numpy', fontsize = 14, style='italic')
ax3.set_xlabel('k-ésimo vecino', fontsize=12)
ax3.set_ylim(0.2,0.3)
plt.grid()
plt.show()

















