import numpy as np

M = np.array( [[4,-1,-1,-1], [-1,3,0,-1], [-1,0,3,-1], [-1,-1,-1,4]])

b = np.array([5,0,5,0])

sol = np.linalg.solve(M,b)
print(f"La solución oficial es {sol}")


def GetGaussJordan(M_,b_):

    A = M_.copy()
    b = b_.copy()
    
    A = np.float_(A)
    b = np.float_(b)
    
    n = len(b)
    
    for i in range(n):
        for j in range(i+1,n):
            
            a = A[j,i]/A[i,i]
            
            A[j,:] -= a*A[i,:]
            b[j] -= a*b[i]

    x = b.copy()
    
    for i in reversed(range(n)):
        for j in range(i+1,n):
            x[i] = (x[i]-A[i,j]*x[j])  
        x[i] /= A[i,i]    
        
    return x,A,b

Xsol1, NewM, Newb = GetGaussJordan(M,b)
print(f"La solución con lo de clase es {Xsol1}")


V_one, V_two, V_three, V_four = 0,0,0,0

print(f"V_one = {V_one}, V_two = {V_two},V_three = {V_three}, V_four = {V_four}")
