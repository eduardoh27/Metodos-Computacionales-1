import numpy as np
import matplotlib.pyplot as plt

def Function(x,y,r=1):
    z = r**2 - x**2 - y**2
    
    if z <= 0:
        return 0
    else:
        return np.sqrt(z)

Function = np.vectorize(Function)

def Integral(f,x,y,r,A):
    integral = 0
    
    for i in range(len(x)-1):
        for j in range(len(y)-1):
  
            a = f(x[i],y[j],r)
            b = f(x[i],y[j+1],r)
            c = f(x[i+1],y[j],r)
            d = f(x[i+1],y[j+1],r)
            
            integral += A*(a+b+c+d)/4

    return integral

def volume_semisphere(n: int)-> float:
    size = 1
    x = np.linspace(-size,size,n+1)
    y = np.linspace(-size,size,n+1)
    dx = (x[-1] - x[0])/(n)
    dy = (y[-1] - y[0])/(n)
    A = dx*dy
    r=1

    return Integral(Function,x,y,r,A)


print(f"Volume semisphere with n = 2: {volume_semisphere(2)}")
print(f"Volume semisphere with n = 3: {volume_semisphere(3)}")
print(f"Volume semisphere with n = 10: {volume_semisphere(10)}")
print(f"Volume semisphere with n = 100: {volume_semisphere(100)}")
