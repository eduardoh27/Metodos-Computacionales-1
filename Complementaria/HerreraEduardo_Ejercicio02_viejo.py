import math 

def angulo_ideal(h: float, v0: float)-> float:
    g = 9.8
    theta = to_degrees(math.atan(v0/(v0**2+2*g*h)**0.5))
    return round(theta, 1)

def to_degrees(theta: float)-> float:
    return theta/math.pi*180

altura = float(input("Ingrese la altura: "))
velocidad = float(input("Ingrese la velocidad: "))
print("El ángulo para lograr la distancia máxima es "+str(angulo_ideal(altura, velocidad))+"°")