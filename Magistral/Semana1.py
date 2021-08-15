lista = ["*"*n for n in range(100)]
lista1 = lista[:10]
lista2 = lista[-10:]
lista3 = lista1 + lista2
for string in lista3:
    print(string)