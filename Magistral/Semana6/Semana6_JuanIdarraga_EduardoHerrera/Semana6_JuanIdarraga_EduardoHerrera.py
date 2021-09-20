import numpy as np
import matplotlib.pyplot as plt

cos = lambda x: np.cos(x)
sin = lambda x: np.sin(x)

m1, m2, M, r1, r2, k, e, u1 = 3, 1, 3, 0.1, 0.2, 0.5, 0.8, 2
I1 = k*m1*(r1**2)
I2 = k*m2*(r2**2)

def getValues(angulo):
    th = np.radians(angulo)

    Matriz = np.array( [[sin(th), cos(th), -sin(th), -cos(th), 1, 1],
                        [0, M, 0, 1, 0, 0], 
                        [M, 0, 1, 0, 0, 0], 
                        [-cos(th), sin(th), cos(th), -sin(th), 0, 0],
                        [sin(th), cos(th), 0, 0, -k, 0],
                        [0, 0, sin(th), cos(th), 0, k]])

    b = np.array([0, 0, M*u1, e*u1*cos(th), u1*sin(th), 0])
    return Matriz, b

angulos = np.linspace(0, 90, 20)

lista_v1x, lista_v1y, lista_v2x, lista_v2y, lista_w1, lista_w2 = [], [], [], [], [], []
lista_pxi, lista_pxf, lista_pyi, lista_pyf, Li, Lf =  [], [], [], [], [], []

for angulo in angulos:
    Matriz, b = getValues(angulo)
    v1x, v1y, v2x, v2y, r1w1, r2w2 = np.linalg.solve(Matriz,b)
    w1, w2 = r1w1/r1, r2w2/r2
    lista_v1x.append(v1x); lista_v1y.append(v1y)
    lista_v2x.append(v2x); lista_v2y.append(v2y)
    lista_w1.append(w1); lista_w2.append(w2)
    lista_pxi.append(m1*u1 + m2*0); lista_pxf.append(m1*v1x + m2*v2x)
    lista_pyi.append(m1*0 + m2*0); lista_pyf.append(m1*v1y + m2*v2y) 
    Li.append(m1*u1*r1*-sin(np.radians(angulo)))    
    Lf.append(w1*I1 + w2*I2 - m1*r1*(v1x*sin(np.radians(angulo))+v1y*cos(np.radians(angulo))) + m2*r2*(v2x*sin(np.radians(angulo))+v2y*cos(np.radians(angulo))) ) 

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(angulos, lista_pxi, '.-', c="r", label="Momento X inicial")
ax.plot(angulos, lista_pxf, '.-', c="b", label="Momento X final")
ax.set_xlabel(r'$\theta$',fontsize=15)
ax.set_ylabel(r'p$_x$[kgm/s]',fontsize=15)
ax.legend(loc='upper right')
ax.set_title("Conservación del momento lineal horizontal antes y después del choque")
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(angulos, lista_pyi, '.-', c="r", label="Momento Y inicial")
ax.plot(angulos, lista_pyf, '.-', c="b", label="Momento Y final")
ax.set_xlabel(r'$\theta$',fontsize=15)
ax.set_ylabel(r'p$_y$[kgm/s]',fontsize=15)
ax.set_ylim(-1,1)
ax.legend(loc='upper right')
ax.set_title("Conservación del momento lineal vertical antes y después del choque")
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(angulos, Li, '.-', c="r", label=r"Momento Angular Inicial")
ax.plot(angulos, Lf, '.-', c="b", label=r"Momento Angular Final")
ax.set_xlabel(r'$\theta$',fontsize=15)
ax.set_ylabel(r'L$_z$[kgm$^2$/s]',fontsize=15)
ax.legend(loc='upper right')
ax.set_title("Conservación del momento angular total antes y después del choque")
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(angulos, lista_w1, '.-', c="r", label=r"$\omega_1$")
ax.plot(angulos, lista_w2, '.-', c="b", label=r"$\omega_2$")
ax.set_xlabel(r'$\theta$',fontsize=15)
ax.set_ylabel(r'$\omega$[rad/s]',fontsize=15)
ax.legend(loc='upper right')
ax.set_title("Velocidad angular de los dos discos después del choque")
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(angulos, lista_v1x, '.-', c="r", label=r"v$_{1x}$")
ax.plot(angulos, lista_v1y, '.-', c="b", label=r"v$_{1y}$")
ax.set_xlabel(r'$\theta$',fontsize=15)
ax.set_ylabel(r'$v_1$[m/s]',fontsize=15)
ax.legend(loc='upper left')
ax.set_title("Velocidad lineal del primer disco después del choque")
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(angulos, lista_v2x, '.-', c="r", label=r"v$_{2x}$")
ax.plot(angulos, lista_v2y, '.-', c="b", label=r"v$_{2y}$")
ax.set_xlabel(r'$\theta$',fontsize=15)
ax.set_ylabel(r'$v_2$[m/s]',fontsize=15)
ax.legend(loc='upper right')
ax.set_title("Velocidad lineal del segundo disco después del choque")
plt.show()
