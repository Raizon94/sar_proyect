######################################################################
#
# INTEGRANTES DEL EQUIPO/GRUPO:
#
# - Germán Soria Bustos
# - Jorge Rodríguez González
# - Julián Cussianovich Porto
#
######################################################################


import numpy as np

def levenshtein_matriz(x, y):
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
    return D[lenX, lenY]

def levenshtein_edicion(x, y):
    # a partir de la versión levenshtein_matriz
    return 0,[] # COMPLETAR Y REEMPLAZAR ESTA PARTE

def damerau_restricted_matriz(x, y):
    # completar versión Damerau-Levenstein restringida con matriz
    lenX, lenY = len(x), len(y)
    # COMPLETAR
    return 0 # COMPLETAR Y REEMPLAZAR ESTA PARTE

def damerau_restricted_edicion(x, y):
    # partiendo de damerau_restricted_matriz añadir recuperar
    # secuencia de operaciones de edición
    return 0,[] # COMPLETAR Y REEMPLAZAR ESTA PARTE

def damerau_intermediate_matriz(x, y):
    # completar versión Damerau-Levenstein intermedia con matriz
     return 0 # COMPLETAR Y REEMPLAZAR ESTA PARTE

def damerau_intermediate_edicion(x, y):
    # partiendo de matrix_intermediate_damerau añadir recuperar
    # secuencia de operaciones de edición
    # completar versión Damerau-Levenstein intermedia con matriz
    return 0,[] # COMPLETAR Y REEMPLAZAR ESTA PARTE
    
opcionesSpell = {
    'levenshtein_m': levenshtein_matriz,
    'damerau_rm':    damerau_restricted_matriz,
    'damerau_im':    damerau_intermediate_matriz,
}

opcionesEdicion = {
    'levenshtein': levenshtein_edicion,
    'damerau_r':   damerau_restricted_edicion,
    'damerau_i':   damerau_intermediate_edicion
}

if __name__ == "__main__":
    print(levenshtein_matriz("ejemplo", "campos"))
