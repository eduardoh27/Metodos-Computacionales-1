import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def funcion_probabilidad(r,R,a_0=1):
    constante = 1/np.sqrt(np.pi*(a_0**3))
    return (constante*np.exp(-1*(np.linalg.norm(r-R))/a_0))**2

def posicion_electron(funcion_p, R, seed=0.5,paso_=1, Npuntos = int(1e5)):
    x = np.zeros((Npuntos,3))
    x[0,:]=np.array((seed,seed,seed))
    
    for i in range(1,len(x)):
        xj=x[i-1,0]+np.random.uniform(-paso_,paso_)
        yj=x[i-1,1]+np.random.uniform(-paso_,paso_)
        zj=x[i-1,2]+np.random.uniform(-paso_,paso_)
        r_ = np.array((xj,yj,zj))
        
        min_=np.minimum(1,funcion_p(r_,R)/funcion_p(x[i-1,:],R))
        t = np.random.rand()
        if t < min_:
            x[i,:]= r_
        else:
            x[i,:] = x[i-1,:]
    return x

def U (r1,r2,R1,R2):

    return ( (1/(np.linalg.norm(r1-r2))) + (1/(np.linalg.norm(R1-R2))) - (1/(np.linalg.norm(r1-R1))) - (1/(np.linalg.norm(r2-R1))) - (1/(np.linalg.norm(r2-R2))) - (1/(np.linalg.norm(r1-R2))) ) 

L = 2
electron_1 = posicion_electron(funcion_probabilidad, R=np.array((0,0,L/2)))
electron_2 = posicion_electron(funcion_probabilidad, R=np.array((0,0,-L/2)), seed=2.5) ## FLATA SEED = 2.5
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection = '3d', elev =30, azim=50)
ax.scatter(electron_1[:,0],electron_1[:,1],electron_1[:,2],marker='.')
ax.scatter(electron_2[:,0],electron_2[:,1],electron_2[:,2],marker='.',color = 'r')
print("hola")
plt.show()


def u_promedio(e1, e2, R1 = np.array((0,0,2/2)),R2=np.array((0,0,-2/2))):
    promedio = 0
    
    for i in range(len(e1)):
        promedio += U(e1[i,:],e2[i,:],R1,R2)
        
prom = u_promedio(electron_1, electron_2)
print(prom)