import numpy as np
import matplotlib.pyplot as plt

def getProbability(n: int)->float:
    """
    Halla la probabilidad de que n personas
    cumplan anios en diferentes dias del anio
    """
    diasAnio = 365
    diasDisponibles = diasAnio
    probabilidad = 1
    n = int(n)
    for i in range(1, n+1):
        probabilidad *= diasDisponibles/diasAnio
        diasDisponibles -= 1
    return probabilidad

def main():
    personas = np.linspace(1, 80, 80)
    probabilidades = np.zeros(len(personas))
    for i in range(len(personas)):
        probabilidades[i] = getProbability(personas[i])
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_ylabel('$P_n$', fontsize=14, style='italic')
    ax.set_xlabel('Personas', fontsize=15, style='italic')
    ax.plot(personas, probabilidades, ".-")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
