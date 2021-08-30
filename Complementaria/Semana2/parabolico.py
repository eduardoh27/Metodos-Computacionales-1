import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

class Particle():
    
    # Constructor
    def __init__(self, r0, v0, a, t):
        # atributo de clase
        self.t = t
        self.r = np.zeros( (len(t),2) )
        self.v = np.zeros( (len(t),2) )
        self.a = np.zeros( (1,2) )
        
        # Condiciones iniciales
        self.r[0] = r0
        self.v[0] = v0
        self.a[0] = a
       
    # Metodos de clase 
    def PrintR(self):
        print(self.r)
        
    def PrintV(self):
        print(self.v)
        
    def PrintA(self):
        print(self.a)
        
    def EvolucionTemporal(self):
        print('Evolucionando la particula')
        
        for i in range( 1, len(self.t) ):
            for j in range(2):
                self.v[i,j] = self.v[0,j] + self.a[0,j]*self.t[i]
                self.r[i,j] = self.r[0,j] + self.v[0,j]*self.t[i] + 0.5*self.a[0,j]*self.t[i]**2
         
    # Getters
    def GetR(self):
        return self.r
    def GetV(self):
        return self.v








#h0 = float(input("Ingrese la altura inicial: "))
h0 = 20
#v0 = float(input("Ingrese la velocidad inicial: "))
v0 =50
g = 9.8

angles = [round(0.1*n,1) for n in range(900)]
#print(angles[0], angles[1], angles[2], angles[-1], angles[-2])

d_mayor = 0
angle_mayor = None

for angle in angles:
    vy = abs(v0*np.sin(angle))
    vx = abs(v0*np.cos(angle))

    t1 = abs(2*vy/g)

    #t1 = abs(np.sqrt(vy/(2*g))) 
    t2 = abs((-vy+np.sqrt((vy)**2+2*g*h0))/g)
    t = t1 + t2 

    #print(angle)
    x = abs(vx*t1)
    #print(x)
    if x > d_mayor:
        d_mayor = x
        angle_mayor = angle

print("El ángulo de mayor alcance horizontal es: "+str(angle_mayor))