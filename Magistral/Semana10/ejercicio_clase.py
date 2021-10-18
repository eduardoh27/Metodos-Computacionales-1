import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from tqdm import tqdm

def GetFrequencies(n):
    
    NTimes  = 0
    
    for i in range(int(n)):
        t = np.random.randint(1,7)
        t2 = np.random.randint(1,7)
        if t+t2 == 2:
            NTimes += 1
            
    return NTimes / n

def main():

    n = np.logspace(1,5,30)

    freqr = np.zeros(len(n))
    P = freqr.copy()
    P[:] = 1./36.


    for i in range(len(n)):
        freqr[i] = GetFrequencies(n[i])
        
    plt.plot(n,freqr, linewidth=3)
    plt.plot(n,P,color='r')
    plt.xscale('log')
    plt.ylabel(r'$f_{r}$',fontsize=15)
    plt.show()

if __name__ == "__main__":
    main()
    
