######################################################################
#
# INTEGRANTES DEL EQUIPO/GRUPO:
#   Jorge Rodríguez González
#   Germán Soria Bustos
#   Julián Cussianovich Porto
#
######################################################################


import numpy as np

def levenshtein_matriz(x, y):
    
    lenX, lenY = len(x), len(y)
    
    # Se inicializa la matriz de programación dinámica con ceros.
    # El tamaño es +1 para incluir el caso de la cadena vacía.
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)

    # Inicialización de la primera fila y la primera columna.
    # D[i, 0] es el coste de convertir x[:i] a una cadena vacía, que requieren 'i' borrados.
    for i in range(1, lenX + 1):
        D[i, 0] = i
    # D[0, j] es el coste de convertir una cadena vacía a y[:j], que requiere 'j' inserciones.
    for j in range(1, lenY + 1):
        D[0, j] = j
        
    # Se rellena el resto de la matriz.
    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            # El coste de sustitución es 1 si los caracteres son diferentes, y 0 si son iguales.
            cost = 1 if x[i - 1] != y[j - 1] else 0
            
            # D[i, j] es el mínimo entre las tres operaciones de edición posibles:
            # 1. D[i - 1, j] + 1: Borrado del carácter x[i-1].
            # 2. D[i, j - 1] + 1: Inserción del carácter y[j-1].
            # 3. D[i - 1, j - 1] + cost: Sustitución de x[i-1] por y[j-1] (o coincidencia).
            D[i, j] = min(
                D[i - 1, j] + 1,
                D[i, j - 1] + 1,
                D[i - 1, j - 1] + cost,
            )
            
    # El resultado final es el valor en la esquina inferior derecha de la matriz.
    return D[lenX, lenY]

def levenshtein_edicion(x, y):
    
    # --- PASO 1: Construcción de la matriz de programación dinámica ---
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
                D[i - 1, j - 1] + cost
            )

    # --- PASO 2: Retroceso para recuperar el camino de edición ---
    edits = []
    i, j = lenX, lenY
    # Se parte de la esquina inferior derecha y se retrocede hacia el origen.
    while i > 0 or j > 0:
        # Se debe gestionar el caso de llegar a un borde ("pared") de la matriz.
        # Si estamos en la primera fila (i=0), solo podemos haber llegado desde la izquierda
        if i == 0:
            edits.append(('', y[j - 1]))
            j -= 1
            continue
        # Si estamos en la primera columna (j=0), solo podemos haber llegado desde arriba
        if j == 0:
            edits.append((x[i - 1], ''))
            i -= 1
            continue

        cost = 1 if x[i-1] != y[j-1] else 0
        
        # Se reconstruye el camino eligiendo el movimiento que generó el valor D[i,j].
        # Se prioriza la diagonal para agrupar sustituciones y aciertos. IMPORTANTE: influye en el orden de modificaciones. En clase se nos ha indicado que el orden de las modificaciones puede variar en función de la implementación
        # ya que varios conjuntos de modificaciones llevan al mismo coste
        if D[i, j] == D[i - 1, j - 1] + cost:
            # Movimiento diagonal: corresponde a una sustitución o un acierto.
            edits.append((x[i - 1], y[j - 1]))
            i -= 1
            j -= 1
        elif D[i, j] == D[i - 1, j] + 1:
            # Movimiento vertical: corresponde a un borrado de x[i-1].
            edits.append((x[i - 1], ''))
            i -= 1
        elif D[i, j] == D[i, j - 1] + 1:
            # Movimiento horizontal: corresponde a una inserción de y[j-1].
            edits.append(('', y[j - 1]))
            j -= 1
            
    # La lista de ediciones se construye en orden inverso, así que se revierte al final.
    edits.reverse()
    return D[lenX, lenY], edits

def damerau_restricted_matriz(x, y):
    
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i, 0] = i
    for j in range(1, lenY + 1):
        D[0, j] = j

    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            cost = 1 if x[i - 1] != y[j - 1] else 0
            # Se calculan primero los costes de las operaciones de Levenshtein.
            d_val = min(
                D[i - 1, j] + 1,
                D[i, j - 1] + 1,
                D[i - 1, j - 1] + cost
            )
            
            # Nuevo caso para la transposición
            # Se comprueba si los dos últimos caracteres están intercambiados
            if i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                # El coste es 1 + el coste de la subcadena anterior
                d_val = min(d_val, D[i - 2, j - 2] + 1)
                
            D[i,j] = d_val
            
    return D[lenX, lenY]

def damerau_restricted_edicion(x, y):
    
    # --- PASO 1: Construcción de la matriz ---
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

    # --- PASO 2: Backtracking con transposición ---
    edits = []
    i, j = lenX, lenY
    while i > 0 or j > 0:
        # Se comprueba primero el caso de la transposición, ya que implica un salto mayor.
        if i > 1 and j > 1 and x[i-1] == y[j-2] and x[i-2] == y[j-1] and D[i,j] == D[i-2,j-2] + 1:
            # Movimiento de transposición. Se anota como ('ab', 'ba').
            edits.append((x[i-2:i], y[j-2:j]))
            i -= 2
            j -= 2
        else:
            # Si no hay transposición, se procede con la lógica de Levenshtein normal.
            cost = 1 if (i > 0 and j > 0 and x[i-1] != y[j-1]) else 0
            
            # Movimiento diagonal (sustitución/acierto).
            if i > 0 and j > 0 and D[i,j] == D[i-1,j-1] + cost:
                edits.append((x[i-1], y[j-1]))
                i -= 1
                j -= 1
            # Movimiento vertical (borrado).
            elif i > 0 and D[i,j] == D[i-1,j] + 1:
                edits.append((x[i-1], ''))
                i -= 1
            # Movimiento horizontal (inserción).
            elif j > 0 and D[i,j] == D[i,j-1] + 1:
                edits.append(('', y[j-1]))
                j -= 1
            else:
                break
                
    edits.reverse()
    return D[lenX, lenY], edits

def damerau_intermediate_matriz(x, y):
    
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
            
            # Caso 1: Transposición restringida ('ab' -> 'ba', coste 1)
            if i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                d_val = min(d_val, D[i - 2, j - 2] + 1)
            
            # Caso 2: Nueva operación 'acb' -> 'ba' (coste 2).
            # Comprueba si x termina en 'acb' e y termina en 'ba' donde a y b coinciden.
            if i > 2 and j > 1 and x[i-3] == y[j-1] and x[i-1] == y[j-2]:
                d_val = min(d_val, D[i-3, j-2] + 2)

            # Caso 3: Nueva operación 'ab' -> 'bca' (coste 2).
            # Comprueba si x termina en 'ab' e y termina en 'bca' donde a y b coinciden.
            if i > 1 and j > 2 and x[i-2] == y[j-1] and x[i-1] == y[j-3]:
                d_val = min(d_val, D[i-2, j-3] + 2)
                
            D[i,j] = d_val
            
    return D[lenX, lenY]

def damerau_intermediate_edicion(x, y):
    
    # --- PASO 1: Construcción de la matriz ---
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
            if i > 2 and j > 1 and x[i-3] == y[j-1] and x[i-1] == y[j-2]:
                d_val = min(d_val, D[i-3, j-2] + 2)
            if i > 1 and j > 2 and x[i-2] == y[j-1] and x[i-1] == y[j-3]:
                d_val = min(d_val, D[i-2, j-3] + 2)
            D[i,j] = d_val

    # --- PASO 2: Backtracking con operaciones intermedias ---
    edits = []
    i, j = lenX, lenY
    while i > 0 or j > 0:
        # El orden de comprobación es crucial: de las operaciones más complejas a las más simples.
        # Esto asegura que se identifique el salto más grande posible que generó el coste mínimo.

        # 1. Comprobar 'acb' -> 'ba' (coste 2)
        if i > 2 and j > 1 and x[i-3] == y[j-1] and x[i-1] == y[j-2] and D[i,j] == D[i-3, j-2] + 2:
            edits.append((x[i-3:i], y[j-2:j]))
            i -= 3
            j -= 2
        # 2. Comprobar 'ab' -> 'bca' (coste 2)
        elif i > 1 and j > 2 and x[i-2] == y[j-1] and x[i-1] == y[j-3] and D[i,j] == D[i-2, j-3] + 2:
            edits.append((x[i-2:i], y[j-3:j]))
            i -= 2
            j -= 3
        # 3. Comprobar 'ab' -> 'ba' (coste 1, transposición restringida)
        elif i > 1 and j > 1 and x[i-1] == y[j-2] and x[i-2] == y[j-1] and D[i,j] == D[i-2,j-2] + 1:
            edits.append((x[i-2:i], y[j-2:j]))
            i -= 2
            j -= 2
        # 4. Si no es ninguna operación especial, usar la lógica de Levenshtein.
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

# Diccionario para acceder a las funciones de cálculo de distancia por su nombre.
# Lo usaremos para selección dinámica del algoritmo a utilizar.
opcionesSpell = {
    'levenshtein_m': levenshtein_matriz,
    'damerau_rm':    damerau_restricted_matriz,
    'damerau_im':    damerau_intermediate_matriz,
}

# Diccionario para acceder a las funciones que también recuperan la secuencia de edición.
opcionesEdicion = {
    'levenshtein': levenshtein_edicion,
    'damerau_r':   damerau_restricted_edicion,
    'damerau_i':   damerau_intermediate_edicion
}

if __name__ == "__main__":
    print("--- Ejemplo Levenshtein ('ejemplo' -> 'campos') ---")
    dist, edits = levenshtein_edicion("ejemplo", "campos")
    print(f"Distancia: {dist}")
    print(f"Ediciones: {edits}")
    
    print("\n--- Ejemplo Damerau Restringida ('algoritmo' -> 'algortimo') ---")
    
    dist_dr, edits_dr = damerau_restricted_edicion("algoritmo", "algortimo")
    print(f"Distancia: {dist_dr}")
    print(f"Ediciones: {edits_dr}")

    print("\n--- Ejemplo Damerau Intermedia ('acb' -> 'ba') ---")
    
    dist_di, edits_di = damerau_intermediate_edicion("acb", "ba")
    print(f"Distancia: {dist_di}")
    print(f"Ediciones: {edits_di}")