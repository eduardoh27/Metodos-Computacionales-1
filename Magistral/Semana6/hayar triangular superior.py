import numpy as np

def sacar_matriz_triangular(M: list):
    l1=M[0].copy()
    d= float(l1[0])
    
    for j in range(M.shape[0]):
        M[0][j]= M[0][j] / d
        
    l2=M[1].copy()
    e= float(l2[0])    
    for j in range(M.shape[1]):
        M[1][j]=M[1][j]-e*M[0][j]
    
    
    l3=M[2].copy()
    f= float(l3[0])
    for j in range(M.shape[1]):
         M[2][j]=float(M[2][j]-f*M[0][j])
    
    l4=M[1].copy()
    g= float(l4[1])
    
    l5=M[2].copy()
    h= float(l5[1])
    
    for j in range(M.shape[1]):
         M[2][j]=M[2][j]-M[1][j]*(h/g)
    
    print("La matriz queda: ")
    print(M)

M = np.array( [[2,-3,5],[6,-1,3],[-4,1,-2]] )
#M = np.array( [[3.0,-1.0,-1.0],[-1.0,3.0,1.0],[2.0,1.0,4.0]] )
print(M)
sacar_matriz_triangular(M)
