######################################################################
#
# INTEGRANTES DEL EQUIPO/GRUPO:
#
# - COMPLETAR
# - COMPLETAR
#
######################################################################


import numpy as np

def levenshtein_matriz(x, y):
    """
    Calcula la distancia de Levenshtein entre dos cadenas usando una matriz completa.
    Esta es la implementación de referencia proporcionada.
    """
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i][0] = i
    for j in range(1, lenY + 1):
        D[0][j] = j
    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
    return D[lenX, lenY]

def levenshtein_edicion(x, y):
    """
    Calcula la distancia de Levenshtein y recupera la secuencia de operaciones de edición.
    Corresponde a la tarea 1 de la práctica.
    """
    # Paso 1: Construir la matriz de programación dinámica
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i, 0] = i
    for j in range(1, lenY + 1):
        D[0, j] = j

    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            cost = 1 if x[i - 1] != y[j - 1] else 0
            D[i, j] = min(
                D[i - 1, j] + 1,        # Borrado
                D[i, j - 1] + 1,        # Inserción
                D[i - 1, j - 1] + cost  # Sustitución/Coincidencia
            )

    # Paso 2: Retroceder (backtracking) para recuperar el camino de edición.
    edits = []
    i, j = lenX, lenY
    while i > 0 or j > 0:
        cost = 1 if (i > 0 and j > 0 and x[i-1] != y[j-1]) else 0
        
        # Prioridad a los movimientos diagonales (sustitución/coincidencia)
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + cost:
            edits.append((x[i - 1], y[j - 1]))
            i -= 1
            j -= 1
        # Borrado
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            edits.append((x[i - 1], ''))
            i -= 1
        # Inserción
        elif j > 0 and D[i, j] == D[i, j - 1] + 1:
            edits.append(('', y[j - 1]))
            j -= 1
        else: # Asegura que el bucle termine en los bordes
            break
            
    edits.reverse() # Invertir la lista para obtener la secuencia correcta.
    return D[lenX, lenY], edits

def damerau_restricted_matriz(x, y):
    """
    Calcula la distancia de Damerau-Levenshtein restringida usando una matriz completa.
    """
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i, 0] = i
    for j in range(1, lenY + 1):
        D[0, j] = j

    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            cost = 1 if x[i - 1] != y[j - 1] else 0
            d_val = min(
                D[i - 1, j] + 1,
                D[i, j - 1] + 1,
                D[i - 1, j - 1] + cost
            )
            # Añadir caso de transposición.
            if i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                d_val = min(d_val, D[i - 2, j - 2] + 1)
            D[i,j] = d_val
            
    return D[lenX, lenY]

def damerau_restricted_edicion(x, y):
    """
    Calcula la distancia de Damerau-Levenshtein restringida y recupera las ediciones.
    Corresponde a las tareas 2 y 3 de la práctica.
    """
    # Paso 1: Construir la matriz con la regla de transposición
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i, 0] = i
    for j in range(1, lenY + 1):
        D[0, j] = j

    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            cost = 1 if x[i-1] != y[j-1] else 0
            d_val = min(D[i-1, j] + 1, D[i, j-1] + 1, D[i-1, j-1] + cost)
            if i > 1 and j > 1 and x[i-1] == y[j-2] and x[i-2] == y[j-1]:
                d_val = min(d_val, D[i-2, j-2] + 1)
            D[i,j] = d_val

    # Paso 2: Backtracking, comprobando primero la transposición
    edits = []
    i, j = lenX, lenY
    while i > 0 or j > 0:
        # Transposición ('ab' -> 'ba')
        if i > 1 and j > 1 and x[i-1] == y[j-2] and x[i-2] == y[j-1] and D[i,j] == D[i-2,j-2] + 1:
            edits.append((x[i-2:i], y[j-2:j])) # El formato es ('ab', 'ba').
            i -= 2
            j -= 2
        else:
            cost = 1 if (i > 0 and j > 0 and x[i-1] != y[j-1]) else 0
            if i > 0 and j > 0 and D[i,j] == D[i-1,j-1] + cost:
                edits.append((x[i-1], y[j-1]))
                i -= 1
                j -= 1
            elif i > 0 and D[i,j] == D[i-1,j] + 1:
                edits.append((x[i-1], ''))
                i -= 1
            elif j > 0 and D[i,j] == D[i,j-1] + 1:
                edits.append(('', y[j-1]))
                j -= 1
            else:
                break
                
    edits.reverse()
    return D[lenX, lenY], edits

def damerau_intermediate_matriz(x, y):
    """
    Calcula la distancia de Damerau-Levenshtein intermedia usando una matriz completa.
    """
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i, 0] = i
    for j in range(1, lenY + 1):
        D[0, j] = j

    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            cost = 1 if x[i - 1] != y[j - 1] else 0
            d_val = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + cost)
            
            # Caso restringido (coste 1)
            if i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                d_val = min(d_val, D[i - 2, j - 2] + 1)
            
            # Casos intermedios (coste 2)
            # Caso 'acb' -> 'ba'.
            if i > 2 and j > 1 and x[i-3] == y[j-2] and x[i-1] == y[j-1]:
                d_val = min(d_val, D[i-3, j-2] + 2)
            # Caso 'ab' -> 'bca'.
            if i > 1 and j > 2 and x[i-2] == y[j-2] and x[i-1] == y[j-3]:
                d_val = min(d_val, D[i-2, j-3] + 2)
                
            D[i,j] = d_val
            
    return D[lenX, lenY]

def damerau_intermediate_edicion(x, y):
    """
    Calcula la distancia de Damerau-Levenshtein intermedia y recupera las ediciones.
    Corresponde a las tareas 4 y 5 de la práctica.
    """
    # Paso 1: Construir la matriz con todas las reglas de Damerau
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i, 0] = i
    for j in range(1, lenY + 1):
        D[0, j] = j

    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            cost = 1 if x[i-1] != y[j-1] else 0
            d_val = min(D[i-1,j] + 1, D[i,j-1] + 1, D[i-1,j-1] + cost)
            if i > 1 and j > 1 and x[i-1] == y[j-2] and x[i-2] == y[j-1]:
                d_val = min(d_val, D[i-2,j-2] + 1)
            if i > 2 and j > 1 and x[i-3] == y[j-2] and x[i-1] == y[j-1]:
                d_val = min(d_val, D[i-3, j-2] + 2)
            if i > 1 and j > 2 and x[i-2] == y[j-2] and x[i-1] == y[j-3]:
                d_val = min(d_val, D[i-2, j-3] + 2)
            D[i,j] = d_val

    # Paso 2: Backtracking, comprobando primero los casos más complejos
    edits = []
    i, j = lenX, lenY
    while i > 0 or j > 0:
        # Casos intermedios (coste 2)
        if i > 1 and j > 2 and x[i-2] == y[j-2] and x[i-1] == y[j-3] and D[i,j] == D[i-2, j-3] + 2:
            edits.append((x[i-2:i], y[j-3:j])) # 'ab' -> 'bca'
            i -= 2
            j -= 3
        elif i > 2 and j > 1 and x[i-3] == y[j-2] and x[i-1] == y[j-1] and D[i,j] == D[i-3, j-2] + 2:
            edits.append((x[i-3:i], y[j-2:j])) # 'acb' -> 'ba'
            i -= 3
            j -= 2
        # Caso restringido (coste 1)
        elif i > 1 and j > 1 and x[i-1] == y[j-2] and x[i-2] == y[j-1] and D[i,j] == D[i-2,j-2] + 1:
            edits.append((x[i-2:i], y[j-2:j])) # 'ab' -> 'ba'
            i -= 2
            j -= 2
        # Casos de Levenshtein
        else:
            cost = 1 if (i > 0 and j > 0 and x[i-1] != y[j-1]) else 0
            if i > 0 and j > 0 and D[i,j] == D[i-1,j-1] + cost:
                edits.append((x[i-1], y[j-1]))
                i -= 1
                j -= 1
            elif i > 0 and D[i,j] == D[i-1,j] + 1:
                edits.append((x[i-1], ''))
                i -= 1
            elif j > 0 and D[i,j] == D[i,j-1] + 1:
                edits.append(('', y[j-1]))
                j -= 1
            else:
                break
                
    edits.reverse()
    return D[lenX, lenY], edits

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
    print("--- Ejemplo Levenshtein ---")
    dist, edits = levenshtein_edicion("ejemplo", "campos")
    print(f"Distancia: {dist}")
    print(f"Ediciones: {edits}")
    print("\n--- Ejemplo Damerau Restringida ---")
    dist_dr, edits_dr = damerau_restricted_edicion("algoritmo", "algortimo")
    print(f"Distancia: {dist_dr}")
    print(f"Ediciones: {edits_dr}")
    print("\n--- Ejemplo Damerau Intermedia ---")
    dist_di, edits_di = damerau_intermediate_edicion("algoritmo", "algortximo")
    print(f"Distancia: {dist_di}")
    print(f"Ediciones: {edits_di}")