import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from tqdm import tqdm

def getRandomCylinder():
    r = np.random.rand()
    theta = np.radians(np.random.rand()*360)
    
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = np.random.rand()*3
    return [x,y,z]

def main():
    x = np.array([0,0,3]) #the tip of the cone
    dir = np.array([0,0,-1]) # the normalized axis vector, pointing from the tip to the base
    h = 3
    r = 1

    puntosAdentro = 0
    N = int(1e5)

    for i in tqdm(range(N)):
        p = np.array(getRandomList()) # point to test
        cone_dist =  np.dot(p - x, dir)
        cone_radius = (cone_dist / h) * r

        orth_distance = np.linalg.norm((p - x) - cone_dist * dir)
        is_point_inside_cone = (orth_distance < cone_radius)

        if is_point_inside_cone:
            puntosAdentro += 1
    
    print(puntosAdentro/N)

def getRandomParallelepiped():
    """
    obtiene puntos aleatorios dentro de un paralelepido 
    con x, y en [-1,1] X [-1,1]  y  0 < altura < 3
    """
    x = np.random.rand()*2-1
    y = np.random.rand()*2-1
    z = np.random.rand()*3
    return [x,y,z]

def main1():
    """
    Encuentra la proporción de puntos dentro del cilindro
    """
    H = 3
    R = 1
    
    adentro = 0
    N = int(1e5)

    for i in tqdm(range(N)):
        p = getRandomParallelepiped()
        x = p[0]
        y = p[1]
        h = p[2]

        # el radio del círculo del cono a esa altura:
        alpha = H-h
        r = R*alpha / H

        if (x**2)+ (y**2) <= (r**2): # si entra en el círculo del cono
            adentro+=1

    return adentro/N

def getVolumenCilindro(r=1, h=3):
    return h*np.pi*(r**2)


def getVolumenParallelepiped(l=2, h=3):
    return h*l*l

if __name__ == "__main__":
    proporcion = main1()
    volume_cone = getVolumenParallelepiped()*proporcion
    print(f"Volume of a cone = {volume_cone}")
    