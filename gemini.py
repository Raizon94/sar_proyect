######################################################################
#
# INTEGRANTES DEL EQUIPO/GRUPO:
#
#
######################################################################


import numpy as np

def levenshtein_matriz(x, y):
    """
    Calcula la distancia de Levenshtein entre dos cadenas (x, y) utilizando el
    algoritmo de programación dinámica con una matriz completa.

    Este método implementa directamente la ecuación de recurrencia descrita en
    el boletín. Construye una matriz D de tamaño (len(x)+1) x (len(y)+1)
    donde D[i, j] almacena el coste mínimo para convertir el prefijo x[:i] en y[:j].

    Args:
        x (str): La primera cadena.
        y (str): La segunda cadena.

    Returns:
        int: La distancia de Levenshtein entre x e y.
    """
    lenX, lenY = len(x), len(y)
    
    # Se inicializa la matriz de programación dinámica con ceros.
    # El tamaño es +1 para incluir el caso de la cadena vacía.
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)


    # IMP: D[i,j] = D[filas,columnas]
    # Inicialización de la primera fila y la primera columna.
    # D[i, 0] es el coste de convertir x[:i] a una cadena vacía, que requiere 'i' borrados.
    for i in range(1, lenX + 1):
        D[i, 0] = i 
    # D[0, j] es el coste de convertir una cadena vacía a y[:j], que requiere 'j' inserciones.
    for j in range(1, lenY + 1):
        D[0, j] = j
        
    # Se rellena el resto de la matriz.
    # Bucles empiezan por 1 porque la filas y columna 0 ya han sido rellenadas.
    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            # El coste de sustitución es 1 si los caracteres son diferentes, y 0 si son iguales.
            cost = 1 if x[i - 1] != y[j - 1] else 0
            
            # D[i, j] es el mínimo entre las tres operaciones de edición posibles:
            # 1. D[i - 1, j] + 1: Borrado del carácter x[i-1]. (VERTICAL)
            # 2. D[i, j - 1] + 1: Inserción del carácter y[j-1]. (HORIZONTAL)
            # 3. D[i - 1, j - 1] + cost: Sustitución de x[i-1] por y[j-1] (o coincidencia). (DIAGONAL)
            D[i, j] = min(
                D[i - 1, j] + 1,
                D[i, j - 1] + 1,
                D[i - 1, j - 1] + cost,
            )
            
    # El resultado final es el valor en la esquina inferior derecha de la matriz.
    return D[lenX, lenY]

def levenshtein_edicion(x, y):
    """
    Calcula la distancia de Levenshtein y además recupera la secuencia de
    operaciones de edición para transformar x en y.
    
    Esta función corresponde a la Tarea 1 del boletín.
    El proceso consta de dos fases:
    1. Rellenar la matriz de programación dinámica D, igual que en levenshtein_matriz.
    2. Realizar un "backtracking" desde D[lenX, lenY] hasta D[0, 0] para reconstruir
       el camino de coste mínimo. Cada paso en el camino corresponde a una
       operación de edición.

    Args:
        x (str): La cadena de origen.
        y (str): La cadena de destino.

    Returns:
        tuple[int, list]: Una tupla conteniendo la distancia y la lista de
                          operaciones de edición. El formato de las operaciones es
                          una lista de tuplas (char_x, char_y).
    """
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
        # Si estamos en la primera fila (i=0), solo podemos haber llegado desde la izquierda (inserción).
        if i == 0:
            edits.append(('', y[j - 1]))
            j -= 1
            continue
        # Si estamos en la primera columna (j=0), solo podemos haber llegado desde arriba (borrado).
        if j == 0:
            edits.append((x[i - 1], ''))
            i -= 1
            continue

        cost = 1 if x[i-1] != y[j-1] else 0
        
        # Se reconstruye el camino eligiendo el movimiento que generó el valor D[i,j].
        # Se prioriza la diagonal para agrupar sustituciones y aciertos.
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
    """
    Calcula la distancia de Damerau-Levenshtein en su versión restringida.
    
    Esta función corresponde a la Tarea 2. Extiende la distancia de
    Levenshtein añadiendo la operación de transposición de dos caracteres
    adyacentes (e.g., 'ab' -> 'ba') con coste 1.
    
    La implementación modifica la recurrencia de Levenshtein para incluir un
    cuarto caso que comprueba esta posible transposición.

    Args:
        x (str): La primera cadena.
        y (str): La segunda cadena.

    Returns:
        int: La distancia de Damerau-Levenshtein restringida.
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
            # Se calculan primero los costes de las operaciones de Levenshtein.
            d_val = min(
                D[i - 1, j] + 1,
                D[i, j - 1] + 1,
                D[i - 1, j - 1] + cost
            )
            
            # Nuevo caso para la transposición (última línea de la ecuación del boletín).
            # Se comprueba si los dos últimos caracteres están intercambiados.
            if i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                # El coste es 1 (transposición) + el coste de la subcadena anterior (D[i-2, j-2]).
                d_val = min(d_val, D[i - 2, j - 2] + 1)
                
            D[i,j] = d_val
            
    return D[lenX, lenY]

def damerau_restricted_edicion(x, y):
    """
    Calcula la distancia de Damerau-Levenshtein restringida y recupera la
    secuencia de operaciones, incluyendo las transposiciones.

    Esta función corresponde a la Tarea 3. El backtracking se
    modifica para que, además de los movimientos de Levenshtein, pueda
    detectar un salto diagonal de 2x2 que corresponde a una transposición.
    Las transposiciones se representan con la tupla ('ab', 'ba').

    Args:
        x (str): La cadena de origen.
        y (str): La cadena de destino.

    Returns:
        tuple[int, list]: Distancia y lista de operaciones.
    """
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
            # Si no hay transposición, se procede con la lógica de Levenshtein.
            cost = 1 if (i > 0 and j > 0 and x[i-1] != y[j-1]) else 0
            # En el caso de que i o j sean 0, el coste de sustitución/acierto no se puede calcular.
            # En la práctica, el backtracking nunca evaluará esta condición si i o j es 0,
            # ya que los casos de borrado/inserción se evaluarán antes.
            
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
                # Si no se encuentra un camino válido, se detiene.
                # Esto no debería ocurrir en una implementación correcta.
                break
                
    edits.reverse()
    return D[lenX, lenY], edits

def damerau_intermediate_matriz(x, y):
    """
    Calcula la distancia de Damerau-Levenshtein en su versión "intermedia".

    Esta función corresponde a la Tarea 4. Se basa en la versión
    restringida y añade dos nuevas operaciones de edición:
    1. 'acb' -> 'ba' (coste 2)
    2. 'ab' -> 'bca' (coste 2)
    
    La implementación añade dos nuevos casos a la ecuación de recurrencia para
    contemplar estas transformaciones.

    Args:
        x (str): La primera cadena.
        y (str): La segunda cadena.

    Returns:
        int: La distancia de Damerau-Levenshtein intermedia.
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
            
            # Caso 1: Transposición restringida ('ab' -> 'ba', coste 1)
            if i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                d_val = min(d_val, D[i - 2, j - 2] + 1)
            
            # Caso 2: Nueva operación 'acb' -> 'ba' (coste 2).
            # Comprueba si x termina en 'acb' e y termina en 'ba' donde a y b coinciden.
            # x_prefijo_i termina en x[i-3]x[i-2]x[i-1]
            # y_prefijo_j termina en y[j-2]y[j-1]
            if i > 2 and j > 1 and x[i-3] == y[j-1] and x[i-1] == y[j-2]:
                d_val = min(d_val, D[i-3, j-2] + 2)

            # Caso 3: Nueva operación 'ab' -> 'bca' (coste 2).
            # Comprueba si x termina en 'ab' e y termina en 'bca' donde a y b coinciden.
            # x_prefijo_i termina en x[i-2]x[i-1]
            # y_prefijo_j termina en y[j-3]y[j-2]y[j-1]
            if i > 1 and j > 2 and x[i-2] == y[j-1] and x[i-1] == y[j-3]:
                d_val = min(d_val, D[i-2, j-3] + 2)
                
            D[i,j] = d_val
            
    return D[lenX, lenY]

def damerau_intermediate_edicion(x, y):
    """
    Calcula la distancia de Damerau-Levenshtein intermedia y recupera la
    secuencia de operaciones de edición.

    Esta función corresponde a la Tarea 5. El backtracking se
    amplía para reconocer los nuevos tipos de edición, que implican saltos
    de 3x2 y 2x3 en la matriz. Las nuevas operaciones se representan como
    tuplas ('acb', 'ba') o ('ab', 'bca').

    Args:
        x (str): La cadena de origen.
        y (str): La cadena de destino.

    Returns:
        tuple[int, list]: Distancia y lista de operaciones.
    """
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
# Útil para una selección dinámica del algoritmo a utilizar.
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

# Este bloque se ejecuta solo si el script es invocado directamente.
# Sirve como una pequeña prueba o demostración de las funciones implementadas.
if __name__ == "__main__":
    print("--- Ejemplo Levenshtein ('ejemplo' -> 'campos') ---")
    dist, edits = levenshtein_edicion("ejemplo", "campos")
    print(f"Distancia: {dist}") # Esperado: 5
    print(f"Ediciones: {edits}") # [('e', ''), ('j', 'c'), ('e', 'a'), ('m', 'm'), ('p', 'p'), ('l', 'o'), ('o', 's')]
    
    print("\n--- Ejemplo Damerau Restringida ('algoritmo' -> 'algortimo') ---")
    # Este es un caso clásico donde la transposición reduce la distancia.
    # Levenshtein daría 2 (sustituir 'i' por 't', 't' por 'i').
    # Damerau da 1 (transponer 'it' -> 'ti').
    dist_dr, edits_dr = damerau_restricted_edicion("algoritmo", "algortimo")
    print(f"Distancia: {dist_dr}") # Esperado: 1
    print(f"Ediciones: {edits_dr}") # [('a', 'a'), ('l', 'l'), ('g', 'g'), ('o', 'o'), ('r', 'r'), ('it', 'ti'), ('m', 'm'), ('o', 'o')]

    print("\n--- Ejemplo Damerau Intermedia ('acb' -> 'ba') ---")
    # En la versión restringida, la distancia sería 3 (borrar c, sustituir a->b, b->a).
    # En la intermedia, es una sola operación de coste 2.
    dist_di, edits_di = damerau_intermediate_edicion("acb", "ba")
    print(f"Distancia: {dist_di}") # Esperado: 2
    print(f"Ediciones: {edits_di}") # Esperado: [('acb', 'ba')]