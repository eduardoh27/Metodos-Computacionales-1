import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('dark_background')  


# Punto 1

url = 'https://raw.githubusercontent.com/diegour1/CompMetodosComputacionales/main/DataFiles/world_pop.csv'
df = pd.read_csv(url) 
seleccionados = df.loc[(df['Entity'] == 'Our World In Data') & (df['Year'] >= 700) & (df['Year'] <= 1900)] 

x = np.array(seleccionados['Year'])
y = np.array(seleccionados['World Population'])
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
pts = x.size


# Punto 2

P = np.array([np.ones([pts, 1]), x, x**2, x**3, x**4]).reshape(5, pts).T
v = (np.linalg.inv(P.T @ P) @ P.T) @ y
b, m , m2 , m3, m4 = v
coeffs = [m4[0], m3[0], m2[0], m[0], b[0]]

p = np.poly1d(np.array(coeffs))
x1 = np.linspace(670, 1910, 1000)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(x, y, s=15)    
ax.plot(x1, p(x1), c='r')
ax.set_xlabel(r'Year',fontsize=10)
ax.set_ylabel(r'World population',fontsize=10)
ax.set_title(r'Ajuste de grado 4',fontsize=13)
plt.savefig('HerreraEduardo_grafica.png')
plt.show()


# Punto 3

print(f"coeffs 4to grado = {coeffs}")
