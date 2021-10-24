import numpy as np
import matplotlib.pyplot as plt
import os.path as path
import wget
import random

file = 'data.dat'
url = 'https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/Matematicas.txt'
if not path.exists(file):
    Path_ = wget.download(url,file)
    print('--- File loaded ---')

data = np.loadtxt(file)
data = np.array(data)


max_ = np.max(data)
min_ = np.min(data)
print(f'max = {max_}, min = {min_}')

rango = max_ - min_
print(f'rango = {rango}')

data.sort()
print(f'sorted list = {data}')
print(f'list\'s length = {len(data)}')


# EJERCICIO
media = np.mean(data)
print(f'media = {media}')

mediana = np.percentile(data, 50)
print(f'mediana = {mediana}')

#elementoMitad = (data[39] + data[40]) / 2
#print(f'elMitad = {elementoMitad}')

def getFrecuenciaAcumulada(x: float):
    
    ii = np.where(data <= x)
    #len(data[ii])/len(data)
    cantidadMenores = len(data[ii])
    #data[cantidadMenores-1]
    return cantidadMenores

frec = getFrecuenciaAcumulada(59)
print(f'frecuencia = {frec}')
##

ii = np.where(data >= 75)
#print(f'ii = {ii}')
#print(data[ii])
#print(len(data[ii]))


ii = data > 65
#print(f'ii = {ii}')
jj = data <= 85

data1 = data[ii & jj]
#data1 = data[ii | jj]
print(f'{len(data1)/len(data)*100:.2f}')


todas  = np.linspace(0,100,101)

faltantes = []

for i in range(len(todas)):
    
    if todas[i] not in data:
        faltantes.append(todas[i])

#print(f'faltantes = {faltantes}')

help(random)