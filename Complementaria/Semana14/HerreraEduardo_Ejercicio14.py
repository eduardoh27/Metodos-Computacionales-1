import numpy as np
import matplotlib.pyplot as plt
import warnings

f = lambda x: (x**-0.5)*((1-x)**-0.5) / np.pi

def metropolis(distribucion, steps=10000, x_ini=0.1):
    muestras = np.zeros(steps)
    old_x = x_ini
    old_fx = distribucion(old_x)

    for i in range(steps):
        new_x = old_x + np.random.normal(0, 0.5)
        new_fx = distribucion(new_x)
        aceptacion = new_fx / old_fx
        if aceptacion.real >= np.random.rand() and new_x > 0 and new_x < 1:
            muestras[i] = new_x
            old_x = new_x
            old_fx = new_fx
        else:
            muestras[i] = old_x

    return muestras
  
def graficar_distribucion(samples, distribucion, bins, x_ini, x_fin):
    x = np.linspace(x_ini, x_fin,1000)
    y = distribucion(x)
    plt.figure(figsize =(10,6) )
    plt.xlim(x_ini, x_fin)
    plt.plot(x,y,'r')
    plt.hist(samples,bins = bins, density = True)
    plt.savefig("HerreraEduardo_grafica.png")
    plt.show()

def main():
    with np.errstate(divide='ignore'):
        samples = metropolis(f, 100000)
        graficar_distribucion(samples, f, 200, 0, 1)

if __name__ == '__main__':
    main()
    