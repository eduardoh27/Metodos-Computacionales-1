import numpy as np

def sacar_matriz_triangular(M: list):
    l1=M[0].copy()
    print("l1",l1)
    d= float(l1[0])
    print("d",d)

    print(M.shape[0])
    print(M.shape[1])
    print(M.shape[2])
    
    for j in range(M.shape[0]):
        M[0][j]= M[0][j] / d

    for j in range(M.shape[1]):
         M[1][j]=float(M[1][j])-float(M[1][0])*float(M[0][j])
    print(M)

    for j in range(M.shape[1]):
         M[2][j]=float(M[2][j]-M[2][0]*M[0][j])
    print(M)

    for j in range(M.shape[1]):
         M[2][j]=float(M[2][j]-M[1][j]*(M[2][1]/M[1][1]))
    print(M)








M = np.array( [[3.0,-1.0,-1.0],[-1.0,3.0,1.0],[2.0,1.0,4.0]] )
print("M",M)
sacar_matriz_triangular(M)