######################################################################
#
# INTEGRANTES DEL EQUIPO/GRUPO:
#
# - chati
# - lamine yamal
#
######################################################################

from collections import Counter # para la cota optimista
import numpy as np
#tercer argumento es necesario? Lo he añadido para que pase los test
def levenshtein_matriz(x, y, threshold=None):
    """
    Versión original con matriz completa.
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
            
            D[i, j] = min(
                D[i - 1, j] + 1,
                D[i, j - 1] + 1,
                D[i - 1, j - 1] + cost,
            )
            
    return D[lenX, lenY]

def levenshtein_reduccion(x, y, threshold=None):
    """
    TAREA 6: Implementación de Levenshtein con reducción de coste espacial.
    Usa solo dos filas para el cálculo, reduciendo el espacio de O(M*N) a O(N).
    """
    lenX, lenY = len(x), len(y)
    
    # Aseguramos que la cadena más corta (Y) defina el tamaño de las filas
    if lenX < lenY:
        return levenshtein_reduccion(y, x, threshold)

    prev_row = list(range(lenY + 1))
    
    for i in range(1, lenX + 1):
        curr_row = [i] + [0] * lenY # para cada fila, i = coste de eliminar i caracteres de y. 
        min_cost_in_row = i
        for j in range(1, lenY + 1):
            cost = 1 if x[i - 1] != y[j - 1] else 0
            curr_row[j] = min(
                prev_row[j] + 1,      # Borrado
                curr_row[j - 1] + 1,  # Inserción
                prev_row[j - 1] + cost # Sustitución
            )
            if curr_row[j] < min_cost_in_row:
                min_cost_in_row = curr_row[j]
            
        prev_row = curr_row

    return prev_row[lenY]

def levenshtein(x, y, threshold):
    """
    TAREA 7: Levenshtein con reducción de coste espacial y parada por umbral.
    """
    lenX, lenY = len(x), len(y)
    
    # Aseguramos que la cadena más corta (Y) defina el tamaño de las filas
    if lenX < lenY:
        return levenshtein(y, x, threshold)

    prev_row = list(range(lenY + 1))
    
    for i in range(1, lenX + 1):
        curr_row = [i] + [0] * lenY # para cada fila, i = coste de eliminar i caracteres de y. 
        min_cost_in_row = i
        for j in range(1, lenY + 1):
            cost = 1 if x[i - 1] != y[j - 1] else 0
            curr_row[j] = min(
                prev_row[j] + 1,      # Borrado
                curr_row[j - 1] + 1,  # Inserción
                prev_row[j - 1] + cost # Sustitución
            )
            if curr_row[j] < min_cost_in_row:
                min_cost_in_row = curr_row[j]
        
        # Si se proporciona un umbral y el coste mínimo de la fila lo supera, paramos
        if threshold is not None and min_cost_in_row > threshold:
            return threshold + 1
            
        prev_row = curr_row

    return prev_row[lenY]


def levenshtein_cota_optimista(x, y, threshold):
    """
    Implementa la distancia de Levenshtein con una cota optimista
    basada en el recuento de caracteres, como se especifica en la diapositiva.
    """
        
    # 1. Contar frecuencias de la primera cadena
    # Podriamos hacerlo de manera 'manual' recorriendo las cadenas...
    counts = Counter(x)
    #print(counts)

    # 2. Restar frecuencias de la segunda cadena
    for char in y:
        counts[char] -= 1
        
    # 3. Calcular sumas de positivos y negativos
    sum_pos = 0
    sum_neg = 0
    for value in counts.values():
        if value > 0:
            sum_pos += value
        elif value < 0:
            sum_neg += value
            
    # 4. Obtener la cota como el máximo de los valores absolutos
    cota_optimista = max(sum_pos, abs(sum_neg))
    
    # Si la cota ya es mayor que el threshold, no tiene sentido seguir
    if cota_optimista > threshold:
        return threshold + 1

    # CÁLCULO ESTÁNDAR DE LEVENSHTEIN  
    # si la cota es inferior al threshold
     
    return levenshtein(x,y,threshold)


def damerau_restricted(x, y, threshold=None):
    """
    Calcula la distancia de Damerau-Levenshtein restringida usando reducción
    de coste espacial (3 filas) y parada temprana por umbral.
    """

    lenX, lenY = len(x), len(y)

    # Cota optimista inicial
    if abs(lenX - lenY) > threshold:
        return threshold + 1

    # Aseguramos que la cadena más corta defina el tamaño de filas
    if lenX < lenY:
        return damerau_restricted(y, x, threshold)

    # Necesitamos 3 filas para la transposición
    prev_prev_row = list(range(lenY + 1))
    prev_row = list(range(lenY + 1))
    
    for i in range(1, lenX + 1):
        # La fila actual empieza con el coste de borrado
        curr_row = [i] + [0] * lenY
        min_cost_in_row = i

        for j in range(1, lenY + 1):
            cost = 0 if x[i - 1] == y[j - 1] else 1
            
            curr_row[j] = min(prev_row[j] + 1, # eliminar
                              curr_row[j - 1] + 1, # insertar
                              prev_row[j - 1] + cost) #sustituir

            # Comprobación de la TRANSPOSICIÓN
            if i > 1 and j > 1 and x[i-1] == y[j-2] and x[i-2] == y[j-1]:
                transposition_cost = prev_prev_row[j - 2] + 1 # El coste es 1, no 'cost'
                curr_row[j] = min(curr_row[j], transposition_cost)
            
            if curr_row[j] < min_cost_in_row:
                min_cost_in_row = curr_row[j]

        # Parada temprana si el mínimo de la fila ya supera el umbral
        if min_cost_in_row > threshold:
            return threshold + 1
        
        # Actualizamos las filas para la siguiente iteración
        prev_prev_row = prev_row
        prev_row = curr_row

    final_distance = prev_row[lenY]
    
    return final_distance if final_distance <= threshold else threshold + 1


def damerau_intermediate(x, y, threshold=None):
    """
    Calcula la distancia de Damerau-Levenshtein intermedia optimizada,
    con 4 filas en memoria.

    Operaciones consideradas:
    - Inserción, Borrado, Sustitución (coste 1)
    - Transposición 'ab' <-> 'ba' (coste 1)
    - Transposición 'acb' <-> 'ba' (coste 2)
    - Transposición 'ab' <-> 'bca' (coste 2)
    """
    lenX, lenY = len(x), len(y)


    # Cota optimista inicial para podar la búsqueda si es posible.
    if abs(lenX - lenY) > threshold:
        return threshold + 1

    # Aseguramos que la cadena más corta sea 'y' para optimizar el espacio.
    if lenX < lenY:
        return damerau_intermediate(y, x, threshold)

    # Inicializamos las filas necesarias
    row_i_minus_3 = list(range(lenY + 1))
    row_i_minus_2 = list(range(lenY + 1))
    prev_row = list(range(lenY + 1)) # Fila i-1
    
    for i in range(1, lenX + 1):
        curr_row = [i] + [0] * lenY # Fila i
        min_cost_in_row = i

        for j in range(1, lenY + 1):
            cost = 0 if x[i-1] == y[j-1] else 1
            
            # 1. Operaciones básicas de Levenshtein
            curr_row[j] = min(prev_row[j] + 1, # eliminar
                              curr_row[j-1] + 1, # insertar 
                              prev_row[j-1] + cost) # sustituir 

            # 2. Transposición restringida (ab <-> ba)
            if i > 1 and j > 1 and x[i-1] == y[j-2] and x[i-2] == y[j-1]:
                transposition_cost = row_i_minus_2[j-2] + 1
                curr_row[j] = min(curr_row[j], transposition_cost)

            # 3. Transposición intermedia (acb <-> ba)
            if i > 2 and j > 1 and x[i-1] == y[j-2] and x[i-3] == y[j-1]:
                transposition_cost = row_i_minus_3[j-2] + 2
                curr_row[j] = min(curr_row[j], transposition_cost)

            # 4. Transposición intermedia (ab <-> bca)
            if i > 1 and j > 2 and x[i-2] == y[j-1] and x[i-1] == y[j-3]:
                transposition_cost = row_i_minus_2[j-3] + 2
                curr_row[j] = min(curr_row[j], transposition_cost)

            # Actualizamos el mínimo coste de la fila para la parada temprana
            if curr_row[j] < min_cost_in_row:
                min_cost_in_row = curr_row[j]
        
        # Parada temprana
        if min_cost_in_row > threshold:
            return threshold + 1

        # Rotación de las filas para la siguiente iteración
        row_i_minus_3 = row_i_minus_2
        row_i_minus_2 = prev_row
        prev_row = curr_row

    final_distance = prev_row[lenY]

    return final_distance if final_distance <= threshold else threshold + 1




opcionesSpell = {
    'levenshtein_m': levenshtein_matriz,
    'levenshtein_r': levenshtein_reduccion,
    'levenshtein':   levenshtein,
    'levenshtein_o': levenshtein_cota_optimista,
    'damerau_r':     damerau_restricted,
    'damerau_i':     damerau_intermediate
}

if __name__ == "__main__":
    print(levenshtein_matriz("ejemplo", "campos"))
