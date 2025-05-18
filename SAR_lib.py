# versión 1.1

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle
import nltk
from SAR_semantics import SentenceBertEmbeddingModel, BetoEmbeddingCLSModel, BetoEmbeddingModel, SpacyStaticModel


# INICIO CAMBIO EN v1.1
## UTILIZAR PARA LA AMPLIACION
# Selecciona un modelo semántico
SEMANTIC_MODEL = "SBERT"
#SEMANTIC_MODEL = "BetoCLS"
#SEMANTIC_MODEL = "Beto"
#SEMANTIC_MODEL = "Spacy"
#SEMANTIC_MODEL = "Spacy_noSW_noA"

def create_semantic_model(modelname):
    assert modelname in ("SBERT", "BetoCLS", "Beto", "Spacy", "Spacy_noSW_noA")
    
    if modelname == "SBERT": return SentenceBertEmbeddingModel()    
    elif modelname == "BetoCLS": return BetoEmbeddingCLSModel()
    elif modelname == "Beto": return BetoEmbeddingModel()
    elif modelname == "Spacy": SpacyStaticModel(remove_stopwords=False, remove_noalpha=False)
    return SpacyStaticModel()
# FIN CAMBIO EN v1.1


class SAR_Indexer:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia
        
        Preparada para todas las ampliaciones:
          posicionales + busqueda semántica + ranking semántico

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # campo que se indexa
    DEFAULT_FIELD = 'all'
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10


    all_atribs = ['urls', 'index', 'docs', 'articles', 'tokenizer', 'show_all',
              'positional', 'semantic', 'chuncks', 'embeddings', 'chunck_index', 
              'kdtree', 'artid_to_emb', 'semantic_threshold', 'semantic_ranking', 
              'model', 'MAX_EMBEDDINGS']



    def __init__(self):
        """
        Constructor de la clase SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria pero
        	puedes añadir más variables si las necesitas. 

        """
        self.urls = set() # hash para las urls procesadas,
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list
        self.docs = {} # diccionario de terminos --> clave: entero(docid),  valor: ruta del fichero.
        self.articles = {} # hash de articulos --> clave entero (artid), valor: la info necesaria para diferencia los artículos dentro de su fichero
        self.tokenizer = re.compile(r"\W+") # expresion regular para hacer la tokenizacion
        self.show_all = False # valor por defecto, se cambia con self.set_showall()

        # PARA LA AMPLIACION
        self.positional = None 
        self.semantic = None
        self.chuncks = []
        self.embeddings = []
        self.chunck_index = []
        self.artid_to_emb = {}
        self.kdtree = None
        self.semantic_threshold = None
        self.semantic_ranking = None # ¿¿ ranking de consultas binarias ??
        self.model = None
        self.MAX_EMBEDDINGS = 200 # número máximo de embedding que se extraen del kdtree en una consulta
        
        
        
        

    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_showall(self, v:bool):
        """

        Cambia el modo de mostrar los resultados.

        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_semantic_threshold(self, v:float):
        """

        Cambia el umbral para la búsqueda semántica.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic es False el umbral no tendrá efecto.

        """
        self.semantic_threshold = v

    def set_semantic_ranking(self, v:bool):
        """

        Cambia el valor de semantic_ranking.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic_ranking es True se hará una consulta binaria y los resultados se rankearán por similitud semántica.

        """
        self.semantic_ranking = v


    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################


    def save_info(self, filename:str):
        """
        Guarda la información del índice en un fichero en formato binario

        """
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'wb') as fh:
            pickle.dump(info, fh)

    def load_info(self, filename:str):
        """
        Carga la información del índice desde un fichero en formato binario

        """
        #info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'rb') as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)


    ###############################
    ###                         ###
    ###   SIMILITUD SEMANTICA   ###
    ###                         ###
    ###############################

            
    def load_semantic_model(self, modelname:str=SEMANTIC_MODEL):
        """
    
        Carga el modelo de embeddings para la búsqueda semántica.
        Solo se debe cargar una vez
        
        """
        if self.model is None:
            # INICIO CAMBIO EN v1.1
            print(f"loading {modelname} model ... ",end="", file=sys.stderr)             
            self.model = create_semantic_model(modelname)
            print("done!", file=sys.stderr)
            # FIN CAMBIO EN v1.1

            
            

    # INICIO CAMBIO EN v1.2

    def update_chuncks(self, txt:str, artid:int):
        """
        Añade los chuncks (frases en nuestro caso) del texto "txt" correspondiente al articulo "artid" en la lista de chuncks
        Pasos:
            1 - extraer los chuncks de txt, en nuestro caso son las frases. Se debe utilizar "sent_tokenize" de la librería "nltk"
            2 - actualizar los atributos que consideres necesarios: self.chuncks, self.embeddings, self.chunck_index y self.artid_to_emb.
        """
        # 1 - Extraer frases usando sent_tokenize
        sentences = nltk.sent_tokenize(txt)
        
        # 2 - Actualizar atributos necesarios
        start_idx = len(self.chuncks)
        
        # Añadir las frases a self.chuncks
        self.chuncks.extend(sentences)
        
        # Actualizar self.chunck_index (indica qué articulo corresponde a cada frase)
        self.chunck_index.extend([artid] * len(sentences))
        
        # Actualizar self.artid_to_emb (mapeo de artículo a sus frases)
        if artid not in self.artid_to_emb:
            self.artid_to_emb[artid] = []
        
        # Añadir los índices de las frases para este artículo
        indices = list(range(start_idx, start_idx + len(sentences)))
        self.artid_to_emb[artid].extend(indices)
      
        

    def create_kdtree(self):
        """
        Crea el tktree utilizando un objeto de la librería SAR_semantics
        Solo se debe crear una vez despues de indexar todos los documentos
        
        # 1: Se debe llamar al método fit del modelo semántico
        # 2: Opcionalmente se puede guardar información del modelo semántico (kdtree y/o embeddings) en el SAR_Indexer
        """
        print(f"Creating kdtree ...", end="")
        
        if not self.chuncks:
            print("Error: No hay frases para crear el KDTree!")
            return
        
        # Asegurar que el modelo está cargado
        self.load_semantic_model()
        
        # 1. Llamar al método fit del modelo semántico
        self.model.fit(self.chuncks)
        
        # 2. Guardar información del modelo
        self.kdtree = self.model.kdtree
        self.embeddings = self.model.embeddings
        
        print("done!")



        
    def solve_semantic_query(self, query:str):
        """
        Resuelve una consulta utilizando el modelo semántico.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultados, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - si el último resultado tiene una distancia <= self.semantic_threshold 
                ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - también se puede salir si recuperamos todos los embeddings
            5 - tenemos una lista de chuncks que se debe pasar a artículos
        """
        self.load_semantic_model()
        
        # Verificar que existe el kdtree
        if self.kdtree is None:
            print("Error: KDTree no creado. Use create_kdtree() primero.")
            return []
        
        # 1. Utilizar el método query del modelo semántico
        top_k = self.MAX_EMBEDDINGS
        max_size = len(self.chuncks)
        
        # 2. Obtener resultados iniciales
        results = self.model.query(query, top_k=min(top_k, max_size))
        
        # 3. Si el último resultado tiene distancia <= threshold, aumentar top_k
        while (results and results[-1][0] <= self.semantic_threshold and top_k < max_size):
            top_k *= 2
            top_k = min(top_k, max_size)  # No exceder el número total de embeddings
            results = self.model.query(query, top_k=top_k)
        
        # 4. Filtrar resultados por threshold
        if self.semantic_threshold:
            results = [(dist, idx) for dist, idx in results if dist <= self.semantic_threshold]
        
        # 5. Convertir índices de frases a artículos
        article_set = set()
        for _, idx in results:
            if idx < len(self.chunck_index):
                article_set.add(self.chunck_index[idx])
        
        return list(article_set)



    def semantic_reranking(self, query:str, articles: List[int]):
        """
        Ordena los articulos en la lista 'article' por similitud a la consulta 'query'.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultado, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - a partir de los chuncks se deben obtener los artículos
            3 - si entre los artículos recuperados NO estan todos los obtenidos por la RI binaria
                ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - se utiliza la lista ordenada del kdtree para ordenar la lista "articles"
        """
        self.load_semantic_model()
        
        # Verificar que existe el kdtree
        if self.kdtree is None:
            print("Error: KDTree no creado. Use create_kdtree() primero.")
            return articles
        
        # 1. Utilizar el método query del modelo semántico
        top_k = self.MAX_EMBEDDINGS
        max_size = len(self.chuncks)
        
        # Conjunto de artículos que queremos encontrar
        articles_set = set(articles)
        found_articles = set()
        
        # 2. Obtener resultados iniciales
        results = self.model.query(query, top_k=min(top_k, max_size))
        
        # Mapear resultados a artículos
        ranked_articles = []
        for _, idx in results:
            if idx < len(self.chunck_index):
                art_id = self.chunck_index[idx]
                if art_id in articles_set and art_id not in found_articles:
                    ranked_articles.append(art_id)
                    found_articles.add(art_id)
        
        # 3. Si no encontramos todos los artículos, aumentar top_k
        while len(found_articles) < len(articles_set) and top_k < max_size:
            top_k *= 2
            top_k = min(top_k, max_size)
            
            results = self.model.query(query, top_k=top_k)
            
            for _, idx in results:
                if idx < len(self.chunck_index):
                    art_id = self.chunck_index[idx]
                    if art_id in articles_set and art_id not in found_articles:
                        ranked_articles.append(art_id)
                        found_articles.add(art_id)
        
        # 4. Añadir cualquier artículo faltante al final
        missing_articles = [art_id for art_id in articles if art_id not in found_articles]
        ranked_articles.extend(missing_articles)
        
        return ranked_articles

    
    # FIN CAMBIO EN v1.2

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def already_in_index(self, article:Dict) -> bool:
        """

        Args:
            article (Dict): diccionario con la información de un artículo

        Returns:
            bool: True si el artículo ya está indexado, False en caso contrario
        """
        return article['url'] in self.urls


    def index_dir(self, root:str, **args):
        """

        Recorre recursivamente el directorio o fichero "root"
        NECESARIO PARA TODAS LAS VERSIONES

        Recorre recursivamente el directorio "root"  y indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.positional = args['positional']
        self.semantic = args['semantic']
        if self.semantic is True:
            self.load_semantic_model()


        file_or_dir = Path(root)

        if file_or_dir.is_file():
            # is a file
            self.index_file(root)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in sorted(files):
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        #####################################################
        ## COMPLETAR SI ES NECESARIO FUNCIONALIDADES EXTRA ##
        #####################################################
        
        
    def parse_article(self, raw_line:str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una linea que representa un artículo del crawler

        Args:
            raw_line: una linea del fichero generado por el crawler

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'
        """
        
        article = json.loads(raw_line)
        sec_names = []
        txt_secs = ''
        for sec in article['sections']:
            txt_secs += sec['name'] + '\n' + sec['text'] + '\n'
            txt_secs += '\n'.join(subsec['name'] + '\n' + subsec['text'] + '\n' for subsec in sec['subsections']) + '\n\n'
            sec_names.append(sec['name'])
            sec_names.extend(subsec['name'] for subsec in sec['subsections'])
        article.pop('sections') # no la necesitamos
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs
        article['section-name'] = '\n'.join(sec_names)

        return article


    def index_file(self, filename:str):
        """

        Indexa el contenido de un fichero.

        input: "filename" es el nombre de un fichero generado por el Crawler cada línea es un objeto json
            con la información de un artículo de la Wikipedia

        NECESARIO PARA TODAS LAS VERSIONES

        dependiendo del valor de self.positional se debe ampliar el indexado

        """

        #docid unico
        docid = len(self.docs)
        self.docs[docid] = filename

        for i, line in enumerate(open(filename)):
            article = self.parse_article(line)
            # Comprobamos si el artículo ya está indexado
            if self.already_in_index(article):
                continue
            # Asignamos un identificador único al artículo
            artid = len(self.articles)
            # Guardamos la URL para evitar duplicados
            self.urls.add(article['url'])
            # Guardamos la información del artículo
            self.articles[artid] = {
                'docid': docid,
                'position': i,
                'title': article['title'],
                'url': article['url']
            }
            tokens = self.tokenize(article[self.DEFAULT_FIELD])
            if self.positional:
                for pos, token in enumerate(tokens):
                    # Alternativa más explícita para la lógica posicional
                    if token not in self.index:
                        self.index[token] = [(artid, [pos])]
                    else:
                        # El token ya existe
                        if self.index[token][-1][0] == artid:
                            # El token ya existe para este artículo, añadir posición
                            self.index[token][-1][1].append(pos)
                        else:
                            # El token existe, pero es la primera vez para este artículo
                            self.index[token].append((artid, [pos]))
            else: # self.positional is False
                for token in set(tokens):
                    if token not in self.index:
                        self.index[token] = []
                    # Corregido: Añadir si la lista está vacía O si el artid es diferente al último
                    if not self.index[token] or artid != self.index[token][-1]:
                        self.index[token].append(artid)

        #
        # 
        # Solo se debe indexar el contenido self.DEFAULT_FIELD
        #
        #
        #


    def tokenize(self, text:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()




    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Muestra estadisticas de los indices

        """
        pass
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        print("========================================")
        print("Estadísticas de indexación:")
        print("========================================")
        print(f"Número total de artículos indexados: {len(self.articles)}")
        print(f"Número total de documentos (ficheros) indexados: {len(self.docs)}")
        print(f"Tamaño del vocabulario (términos únicos): {len(self.index)}")
        print(f"Número total de URLs únicas: {len(self.urls)}")
        
        
        if self.positional:
            print("Índice posicional: Activado")
        else:
            print("Índice posicional: Desactivado")
            
        
        if self.semantic:
            print("Índice semántico: Activado")
        else:
            print("Índice semántico: Desactivado")
        
        print("========================================")



    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################


    def solve_query(self, query:str, prev:Dict={}):
        if not query:
            return [], {}

        # Extrae frases ("…") o tokens sueltos
        tokens = re.findall(r'"[^"]+"|\S+', query)
        result = None
        i = 0

        while i < len(tokens):
            token = tokens[i]

            # Operador NOT
            if token.upper() == 'NOT':
                # código del NOT...
                i += 2
                continue

            # Frase entre comillas
            if token.startswith('"') and token.endswith('"'):
                phrase = token[1:-1]
                terms = self.tokenize(phrase)
                p = self.get_positionals(terms, returning_phrase=True)
            else:
                # Token normal (ESTE ES EL CASO DE "The")
                p = self.get_posting(token.lower())

            result = p if result is None else self.and_posting(result, p)
            i += 1

        if result is None:
            result = []
        return result, {}




    def get_posting(self, term:str):
        """

        Devuelve la posting list asociada a un termino.
        Puede llamar self.get_positionals: para las búsquedas posicionales.


        param:  "term": termino del que se debe recuperar la posting list.

        return: posting list

        NECESARIO PARA TODAS LAS VERSIONES

        """
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        #Jorge
        
        if term in self.index:
            if self.positional:
                # Si es posicional
                return self.get_positionals(term.lower())
            else:
                # Si no es posicional, ya tenemos la lista de artids
                return self.index[term.lower()]
        return []



    def get_positionals(self, terms:str, returning_phrase:bool=False):
        """

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LAS BÚSQUESAS POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.

        return: posting list

        """

        #################################
        ## COMPLETAR PARA POSICIONALES ##
        #################################

        ## hecho en la versión 1 por X, pero creo que la implementación es incorrecta. "Jorge"
        if not self.positional:
            raise ValueError("Índice no posicional. No se puede buscar frases exactas.")
        if not terms:
            return []
        if len(terms) == 1:
            return self.index.get(terms[0], [])

        # Empezamos con la posting del primer término
        resultado = self.index.get(terms[0], [])

        for i in range(1, len(terms)):
            siguiente_posting = self.index.get(terms[i], [])
            if not siguiente_posting:
                return []  # Si un término no está, no puede estar la frase completa
            resultado = self.interseccion_posicional_con_punteros(resultado, siguiente_posting)
            if not resultado:
                return []  # Cortocircuito si ya no hay coincidencias

        # Al final de get_positionals, cuando es una frase:
        if returning_phrase:
            return [artid for artid, _ in resultado]  # Solo IDs para AND con otros términos
        else:
            return resultado  # Mantener posiciones para siguientes intersecciones posicionales
        
    # Función adicional, PREGUNTADO
    def interseccion_posicional_con_punteros(self, posting1:list, posting2:list):
        """
        Realiza intersección posicional de dos postings con punteros al estilo merge.
        Retorna los artid de posting2 donde alguna posición está justo después (p+1)
        de alguna posición de posting1.

        Parameters:
            posting1 (list): [(artid, [posiciones])]
            posting2 (list): [(artid, [posiciones])]

        Returns:
            list: [(artid, [posiciones de posting2 que cumplen la condición])]
        """
        resultado = []
        i = j = 0
        while i < len(posting1) and j < len(posting2):
            artid1, positions1 = posting1[i]
            artid2, positions2 = posting2[j]

            if artid1 == artid2:
                matches = []
                m = n = 0
                while m < len(positions1) and n < len(positions2):
                    if positions2[n] == positions1[m] + 1:
                        matches.append(positions2[n])
                        m += 1
                        n += 1
                    elif positions2[n] < positions1[m] + 1:
                        n += 1
                    else:
                        m += 1
                if matches:
                    resultado.append((artid1, matches))
                i += 1
                j += 1
            elif artid1 < artid2:
                i += 1
            else:
                j += 1
        return resultado






    def reverse_posting(self, p:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los artid exceptos los contenidos en p

        """
        
        pass
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        #Jorge:
        #implementación obtenida como una operación All AND NOT P, similar a la implementación de minus_posting
        all_docs = self.articles.keys()
        result = []
        i = j = 0
        while i < len(all_docs) and j < len(p):
            if all_docs[i] < p[j]:
                result.append(all_docs[i])
                i += 1
            elif all_docs[i] == p[j]:
                i += 1
                j += 1
            else:
                j += 1
        # Añadir los documentos restantes
        result.extend(all_docs[i:])
        return result



    def and_posting(self, p1:list, p2:list):
        # Convertir postings a formato compatible si es necesario
        if self.positional and p1 and isinstance(p1[0], tuple):
            p1_ids = [artid for artid, _ in p1]
        else:
            p1_ids = p1
            
        if self.positional and p2 and isinstance(p2[0], tuple):
            p2_ids = [artid for artid, _ in p2]
        else:
            p2_ids = p2
        
        # Realizar intersección de IDs
        result = []
        i = j = 0
        while i < len(p1_ids) and j < len(p2_ids):
            if p1_ids[i] == p2_ids[j]:
                result.append(p1_ids[i])
                i += 1
                j += 1
            elif p1_ids[i] < p2_ids[j]:
                i += 1
            else:
                j += 1
        return result







    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se incluye por si es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 y no en p2

        """

        
        pass
        ########################################################
        ## COMPLETAR PARA TODAS LAS VERSIONES SI ES NECESARIO ##
        ########################################################
        #Jorge: implementación obtenida de acuerdo con el ejercicio realizado al final del tema 1: a partir de las posting list de los términos A y B
        # proporciona el resultado de posting_list(A) AND NOT posting_list(B)
        result = []
        i = j = 0
        while i < len(p1) and j < len(p2):
            if p1[i] < p2[j]:
                result.append(p1[i])
                i += 1
            elif p1[i] == p2[j]:
                i += 1
                j += 1
            else:
                j += 1
        result.extend(p1[i:])
        return result





    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql:List[str], verbose:bool=True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r, _ = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results


    def solve_and_test(self, ql:List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                # INICIO CAMBIO EN v1.1
                result, _ = self.solve_query(query)
                result = len(result)
                # FIN CAMBIO EN v1.1
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True
            else:
                print(line)

        return not errors


    def solve_and_show(self, query:str):
        results, _ = self.solve_query(query)
        total = len(results)
        to_show = results if self.show_all else results[:self.SHOW_MAX]
        
        print(f"Recuperados {total} artículos para la consulta '{query}':")
        for idx, artid in enumerate(to_show, start=1):
            if isinstance(artid, (list, tuple)):
                artid = artid[0] if artid else None
            if artid is None:
                continue
            art = self.articles[artid]
            print(f"{idx}\t{artid}\t{art['title']}\t{art['url']}")
        
        return total




