import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

"""  /// punto 24 \\\ """
def punto24():
    Ntrials = int(1e5)

    results = np.zeros((Ntrials,3))

    for i in range(Ntrials):
        d1 = np.random.randint(1,7)
        d2 = np.random.randint(1,7)
        d3 = np.random.randint(1,7)
        results[i] = [d1,d2,d3]

    it = 0

    for R in results:
        
        for i in range(len(R)-1):
            
            # Ordenamos la lista
            R.sort()
            
            if R[i] == R[i+1]:
                
                a = R[i]
                
                if a not in R[i+2:] and a not in R[:i]:
                    
                    it += 1
                    
    print(it/Ntrials, 5./12.)




"""  /// punto 28 \\\ """
def punto28():
    Ntrials = int(1e5)

    results = np.zeros((Ntrials,4))
    opciones=[-1,1]

    for i in range(Ntrials):
        l1 = np.random.choice(opciones)
        l2 = np.random.choice(opciones)
        l3 = np.random.choice(opciones)
        l4 = np.random.choice(opciones)
        results[i] = [l1,l2,l3,l4]

    it = 0

    for R in results:
        
        if R[0] + R[1] + R[2] + R[3] == 0:
            it += 1

    print(it/Ntrials, 3/8)

    print("La probabilidad de sacar dos caras y dos sellos en\
    el lanzamiento de cuatro monedas es: ", it/Ntrials)



"""  /// punto 28 \\\ """

def punto29():

    Npoints = 100
    #listap1 = np.arange(0.1,1,0.1)
    listap1 = np.linspace(0.1, 0.9, Npoints)
    print(listap1)
    #listap2 = np.arange(0.1,0.6,0.1)
    listap2 = np.linspace(0.1, 0.6, Npoints)
    print(listap2)

    X,Y = np.meshgrid(listap1,listap2)
    Z = np.zeros((len(listap1), len(listap2)))

    listaEventoA = []

    for i in range(len(listap1)):

    #for p1 in listap1:
        p1 = listap1[i]

        for j in range(len(listap2)):

        #for p2 in listap2:
            p2 = listap2[j]

            ccss = p1*p2*0.5*0.5
            cscs = p1*(1-p2)*0.5*0.5
            cssc = p1*(1-p2)*0.5*0.5

            sccs = (1-p1)*p2*0.5*0.5
            scsc = (1-p1)*p2*0.5*0.5
            sscc = (1-p1)*(1-p2)*0.5*0.5

            Z[i][j] = ccss+cscs+cssc+sccs+scsc+sscc
            listaEventoA.append(ccss+cscs+cssc+sccs+scsc+sscc)

    print(f"El punto máximo obtenido fue ({max(listaEventoA)}, \
         y el mínimo fue {min(listaEventoA)})")

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d', elev = 28, azim = 46)
    ax.plot_surface(X,Y,Z, cmap=cm.coolwarm)
    """
        ax.set_xlabel(r'x',fontsize=10)
    ax.set_ylabel(r'y',fontsize=10)
    ax.set_zlabel(r'd(x*)',fontsize=10)"""
    plt.show()

    print("Disclaimer: los resultados del enunciado y de la Figura1 \
    del pdf son diferentes. Nos guiamos en los valores del enunciado")
    #print(p1)
    #print(listap2) 
    
punto29()