import json
import os
import re
import sys
import math
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle
import numpy as np
import nltk


## UTILIZAR PARA LA AMPLIACION
if False:
    from nltk.tokenize import sent_tokenize
    import sentence_transformers
    from scipy.spatial import KDTree
    from scipy.spatial.distance import cosine
    nltk.download('punkt')

    def cosine_similarity(v1, v2):
        """
        
        Calcula la similitud coseno de dos vectores. La funcion 'cosine' devuelve la 'distancia coseno'
        
        similitud_coseno = 1 - distancia_coseno
        
        """
        return 1 - cosine(v1, v2)

    def euclidean_to_cosine(d:float):
        """
        
        Pasa de distancia euclidea DE VECTORES NORMALIZADOS a similitud coseno. 
        
        """
        return 1 - d**2/2
        
        

    SEMANTIC_MODEL = "jaimevera1107/all-MiniLM-L6-v2-similarity-es"

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
                  "semantic", "chuncks", "embeddings", "chunck_index", "kdtree", "artid_to_emb"]


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
            print(f"loading {modelname} model ... ",end="")
            self.model = sentence_transformers.SentenceTransformer(modelname)
            print("done!")
            

    def update_embeddings(self, txt:str, artid:int):
        """
        
        Añade los vectores (embeddings) de los chuncks del texto (txt) correspondiente al articulo artid a los indices.
        Pasos:
            1 - extraer los chuncks de txt
            2 - obtener con el LM los embeddings de cada chunck
            3 - normalizar los embeddings
            4 - actualizar: self.chuncks, self.embeddings, self.chunck_index y self.artid_to_emb
        
        """

        self.load_semantic_model()

	# COMPLETAR
        # 1
        # 2
        # 3
        # 4                
        

    def create_kdtree(self):
        """
        
        Crea el tktree utilizando la información de los embeddings
        Solo se debe crear una vez despues de indexar todos los documentos
        
        """
        print(f"Creating kdtree {len(self.embeddings)}...", end="")
	    
        self.kdtree = KDTree(self.embeddings)
        print("done!")


        
        
    def solve_semantic_query(self, query:str):
        """
        
        Resuelve una consulta utilizando el modelo de lenguaje.
        Pasos:
            1 - obtiene el embedding normalizado de la consulta
            2 - extrae los MAX_EMBEDDINGS embeddings más próximos
            3 - convertir distancias euclideas a similitud coseno
            4 - considerar solo las similitudes >= que self.semantic_threshold
            5 - obtener los artids y su máxima similitud
        
        """

        self.load_semantic_model()
        
        # COMPLETAR

        # 1
        # 2
        # 3
        # 4
        # 5


    def semantic_reranking(self, query:str, articles: List[int]):
        """

        Ordena los articulos en la lista 'article' por similitud a la consulta 'query'.
        Pasos:
            1 - obtener el vector normalizado de la consulta
            2 - calcular la similitud coseno de la consulta con todos los embeddings de cada artículo
            3 - ordenar los artículos en función de la mejor similitud.
            
        """
        
        print(self.artid_to_emb.keys())
        
        self.load_semantic_model()
        # COMPLETAR
        # 1
        # 2
        # 3
        
     

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
                    if token not in self.index:
                        self.index[token] = {}
                    if artid not in self.index[token]:
                        self.index[token][artid] = []
                    self.index[token][artid].append(pos)
            else:
                for token in set(tokens):  # Usamos set para eliminar duplicados, se puede??
                    if token not in self.index:
                        self.index[token] = []
                    if artid not in self.index[token]:
                        self.index[token].append(artid)
        # Julián:
        # Va linea por linea del "all" del articulo, y palabra por palabra, revisando si está en el diccionario para crear una entrada, y agregando el número del documento a su entrada
        # Solo se debe indexar el contenido self.DEFAULT_FIELD
        #
        # Jorge: asigna docid unico, procesa cada linea y la parsea, comprueba que no estuviera ya el artículo,
        # asigna artid unico, guarda info para poder localizar el artículo en un docid, indexa en función de self.positionals
        #
        # 
        #################
        ### COMPLETAR ###
        #################



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
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """
        
        if query is None or len(query) == 0:
            return []

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################




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
                # Si es posicional, devolvemos solo los artids (las claves)
                return self.get_positionals(term.lower())
            else:
                # Si no es posicional, ya tenemos la lista de artids
                return sorted(self.index[term.lower()])
        return []



    def get_positionals(self, terms:str):
        """

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LAS BÚSQUESAS POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.

        return: posting list

        """

        #################################
        ## COMPLETAR PARA POSICIONALES ##
        #################################
        pass



    def reverse_posting(self, p:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los artid exceptos los contenidos en p

        """
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        #Jorge:
        #implementación obtenida como una operación All AND NOT P, similar a la implementación de minus_posting
        all_docs = sorted(self.articles.keys())
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
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos en p1 y p2

        """
        
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

        #realizado de acuerdo con la descripción en pseudocódigo del tema 1 de la asignatura. Jorge.
        result = []
        i = j = 0
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                result.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
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
                result, _ = len(self.solve_query(query))
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True
            else:
                print(line)

        return not errors


    def solve_and_show(self, query:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados

        param:  "query": query que se debe resolver.

        return: el numero de artículo recuperadas, para la opcion -T

        """
        pass
        ################
        ## COMPLETAR  ##
        ################



