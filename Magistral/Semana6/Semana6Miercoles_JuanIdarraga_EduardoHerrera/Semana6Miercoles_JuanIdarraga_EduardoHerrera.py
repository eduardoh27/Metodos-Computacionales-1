import numpy as np

def getTriangular(M):
    MT = M.copy()

    for i in range(1,MT.shape[0]):
        contador = 0
        for j in range(i):
            delta = - MT[i][j] / MT[contador][j]
            MT[i] = MT[i] + delta*MT[contador]
            contador += 1    
    
    return MT


M = np.array( [[3.0,-1.0,-1.0],[-1.0,3.0,1.0],[2.0,1.0,4.0]] )
print("Matriz:\n",M)

print("\nLa matriz triangular superior es:\n",np.round(getTriangular(M),3))
