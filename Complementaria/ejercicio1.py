def change_phrase(phrase: str)->str:
    palabras = phrase.split(" ")
    intercaladas = []
    for palabra in palabras:
        if palabra[0].isalpha():
            intercalada = intercalar(palabra, 0)
        else:
            intercalada = intercalar(palabra, 1)
        intercaladas.append(intercalada)

    frase_intercalada = " ".join(intercaladas)
    return frase_intercalada

def intercalar(palabra, bit):
    intercalada = ""
    if bit == 0:
        i = 0
        while i < len(palabra):
            if i%2 == 0:
                letra = palabra[i].upper()
            else:
                letra = palabra[i].lower()
            intercalada += letra
            i += 1
    else:
        i = 0
        while i < len(palabra):
            if i%2 != 0:
                letra = palabra[i].upper()
            else:
                letra = palabra[i].lower()
            intercalada += letra
            i += 1
    return intercalada

print(change_phrase("Hello World"))
print(change_phrase("After a black hole has formed, it can continue to grow"))
print(change_phrase("information is truly lost in black holes (the black hole information paradox)"))