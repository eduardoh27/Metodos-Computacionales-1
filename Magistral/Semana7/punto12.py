import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from tqdm import tqdm


# Definamos el sistema usando una lista
G = np.array([lambda x,y,z: 6*x - 2*np.cos(y*z) - 1,
     lambda x,y,z: 9*y + np.sqrt(x**2 + np.sin(z) + 1.06) + 0.9,
     lambda x,y,z: 60*z + 3*np.exp(-1*x*y) + 10*np.pi - 3])

def GetVectorF(G,r):
    
    dim = len(G)
    
    v = np.zeros(dim)
    
    for i in range(dim):
        v[i] = G[i](r[0],r[1],r[2])
        
    return v


def GetJacobian(G,r,h=0.001):
    
    dim = len(G)
    
    J = np.zeros( (dim,dim) )
    
    
    for i in range(dim):
        
        J[i,0] = (G[i](r[0]+h,r[1],r[2]) - G[i](r[0]-h,r[1],r[2]) )/(2*h)  
        J[i,1] = (G[i](r[0],r[1]+h,r[2]) - G[i](r[0],r[1]-h,r[2]) )/(2*h)  
        J[i,2] = (G[i](r[0],r[1],r[2]+h) - G[i](r[0],r[1],r[2]-h) )/(2*h)
        
    return J.T


def NewtonRaphson(G,r,error=1e-10,itmax=1000):
    
    it = 0
    d = 1
    dvector = []
    
    while d > error and it < itmax:
        
        it += 1
        # Valor actual
        rc = r # Valor inicial
        
        F = GetVectorF(G,r)
        J = GetJacobian(G,r)
        InvJ = np.linalg.inv(J)
        
        r = rc - np.dot(InvJ,F)
        
        d = np.linalg.norm(r-rc)
        
        dvector.append(d)
        
    return r,it,dvector



r = np.zeros(len(G))
r,it,distancias = NewtonRaphson(G,r)


print(r,it)


plt.plot(distancias)
#plt.yscale('log')
plt.xlabel(r'$iterations$', fontsize=15)
plt.ylabel(r'$(|| x^{k+1} - x^{k} ||)/x^{k+1} $', fontsize=15)
plt.grid()














G1 = np.array([lambda x,y: np.log(x**2 + y**2) - np.sin(x*y) - np.log(2) - np.log(np.pi),
     lambda x,y: np.exp(x-y) + np.cos(x*y)])

def GetVectorF1(G,r):
    
    dim = len(G)
    
    v = np.zeros(dim)
    
    for i in range(dim):
        v[i] = G[i](r[0],r[1])
        
    return v


def GetJacobian1(G,r,h=0.001):
    
    dim = len(G)
    
    J = np.zeros( (dim,dim) )
    
    
    for i in range(dim):
        
        J[i,0] = (G[i](r[0]+h,r[1]) - G[i](r[0]-h,r[1]) )/(2*h)  
        J[i,1] = (G[i](r[0],r[1]+h) - G[i](r[0],r[1]-h) )/(2*h)  
        
    return J.T


def NewtonRaphson1(G,r,error=1e-10,itmax=1000):
    
    it = 0
    d = 1
    dvector = []
    
    while d > error and it < itmax:
        
        it += 1
        # Valor actual
        rc = r # Valor inicial
        
        F = GetVectorF1(G,r)
        J = GetJacobian1(G,r)
        InvJ = np.linalg.inv(J)
        
        r = rc - np.dot(InvJ,F)
        
        d = np.linalg.norm(r-rc)
        
        dvector.append(d)
        
    return r,it,dvector


r1 = np.zeros(len(G1))
r1,it1,distancias1 = NewtonRaphson1(G1,r1)

print(r1,it1)


# x**(0) ???