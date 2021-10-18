import numpy as np
import matplotlib.pyplot as plt

def getRandomPoint():
    """
    Halla un punto aleatorio en un cilindro
    de radio < 1 y 0 < altura < 3
    """
    r = np.sqrt(np.random.rand())
    theta = np.radians(np.random.rand()*360)
    
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = np.random.rand()*3

    return x, y, z

def getVolumenCilindro(r=1, h=3):
    """
    Halla el volumen real del cilindro
    """
    return h*np.pi*(r**2)

def getProporcion():
    """
    Encuentra la proporción de puntos dentro del cilindro
    """
    H = 3
    R = 1
    
    adentro = 0
    N = int(1e5)

    for i in range(N):
        p = getRandomPoint()
        x = p[0]
        y = p[1]
        h = p[2]

        # el radio del círculo del cono a esa altura:
        alpha = H-h
        r = R*alpha / H

        if (x**2)+ (y**2) <= (r**2): # si entra en el círculo del cono
            adentro+=1

    return adentro/N

def main():
    volume_cone = getVolumenCilindro()*getProporcion()
    print(f"Volume of a cone = {volume_cone}")

if __name__ == "__main__":
    main()
