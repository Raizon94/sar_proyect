# -*- coding: utf-8 -*-
import re

class SpellSuggester:

    """
    Clase que implementa el método suggest para la búsqueda de términos.
    """

    def __init__(self,
                dist_functions,
                vocab = [],
                default_distance = None,
                default_threshold = None):
        
        """Método constructor de la clase SpellSuggester

        Construye una lista de términos únicos (vocabulario),

        Args:
        dist_functions es un diccionario nombre->funcion_distancia
        vocab es una lista de palabras o la ruta de un fichero
        default_distance debe ser una clave de dist_functions
        default_threshold un entero positivo

        """
        self.distance_functions = dist_functions
        self.set_vocabulary(vocab)
        if default_distance is None:
            default_distance = 'levenshtein'
        if default_threshold is None:
            default_threshold = 3
        self.default_distance = default_distance
        self.default_threshold = default_threshold

    def build_vocabulary(self, vocab_file_path):
        """Método auxiliar para crear el vocabulario.

        Se tokeniza por palabras el fichero de texto,
        se eliminan palabras duplicadas y se ordena
        lexicográficamente.

        Args:
            vocab_file (str): ruta del fichero de texto para cargar el vocabulario.
            tokenizer (re.Pattern): expresión regular para la tokenización.
        """
        tokenizer=re.compile(r"\W+")
        with open(vocab_file_path, "r", encoding="utf-8") as fr:
            vocab = set(tokenizer.split(fr.read().lower()))
            vocab.discard("")  # por si acaso
            return sorted(vocab)

    def set_vocabulary(self, vocabulary):
        if isinstance(vocabulary,list):
            self.vocabulary = vocabulary # atención! nos quedamos una referencia, a tener en cuenta
        elif isinstance(vocabulary,str):
            self.vocabulary = self.build_vocabulary(vocabulary)
        else:
            raise Exception("SpellSuggester incorrect vocabulary value")

    def suggest(self, term, distance=None, threshold=None, flatten=True):
        """

        Args:
            term (str): término de búsqueda.
            distance (str): nombre del algoritmo de búsqueda a utilizar
            threshold (int): threshold para limitar la búsqueda
        """
        if distance is None:
            distance = self.default_distance
        if threshold is None:
            threshold = self.default_threshold

        previous_threshold_result = []
        resul = []

        ########################################
        # COMPLETAR
        ########################################

        # Buscamos en el diccionario 'self.distance_functions' la función que 
        # corresponde al nombre que nos han pasado en el parámetro 'distance'.
        if distance in self.distance_functions:
            dist_func = self.distance_functions[distance]
        else:
        # Si el nombre no existe en nuestro diccionario, lanzamos un error.
            raise ValueError(f"La función de distancia '{distance}' no es válida.")


        # Normalizamos el término de entrada a minúsculas
        term = term.lower()
        
        # Iteramos sobre cada palabra de nuestro vocabulario
        for vocab_word in self.vocabulary:

            # 3. Calcular la distancia usando la función seleccionada
            #    Le pasamos el 'threshold' para que el cálculo sea más rápido
            dist = dist_func(term, vocab_word, threshold)
            
            if dist <= threshold -1:
                previous_threshold_result.append(vocab_word)
            elif dist <= threshold:
                resul.append(vocab_word)


        if flatten:
            resul = [word for wlist in resul for word in wlist]

        previous_sorted = sorted(previous_threshold_result) 
        resul_sorted = sorted(resul)

        # no hay repetidos
        # de esta forma, el resultado siempre tiene la forma:
        # palabras aceptadas por el threshold anterior -> palabras aceptadas por el actual
        
        resul = previous_sorted + resul_sorted
        return resul

