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
from distancias_parte2 import opcionesSpell # Opciones de modelos de cálculo de distancia para spellsuggester
from spellsuggester import SpellSuggester # Crear objeto SpellSuggester para correción ortográfica

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
              'model', 'MAX_EMBEDDINGS', 'use_spelling', 'speller']



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
        self.use_spelling = False   # Adiciones parte 2:
        self.speller = None         # Correción de ortografía con uso de suggest
        
        
        

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

    # Nuevo método Parte 2, llamado desde ALT_Searcher.py
    def set_spelling(self, use_spelling:bool, distance:str=None,threshold:int=None):
        """
        self.use_spelling a True activa la corrección ortográfica
        EN LAS PALABRAS NO ENCONTRADAS, en caso contrario NO utilizará
        corrección ortográfica
        input: "use_spell" booleano, determina el uso del corrector.
        "distance" cadena, nombre de la función de distancia.
        "threshold" entero, umbral del corrector
        """
        if use_spelling:
            if distance not in opcionesSpell:
                raise ValueError(f"La función de distancia '{distance}' no es válida.")
            elif threshold < 0:
                    raise ValueError(f"El valor de threshold '{threshold}' no es válido.")
            else: 
                self.use_spelling = True
                self.speller = SpellSuggester(opcionesSpell, list(self.index), distance, threshold)

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
            2 - actualizar los atributos que consideres necesarios: self.chuncks, self.chunck_index y self.artid_to_emb. Los embeddings correspondientes a estos chunks se calcularán y almacenarán cuando se llame a create_kdtree()
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
        Crea el kdtree utilizando un objeto de la librería SAR_semantics
        Solo se debe crear una vez despues de indexar todos los documentos
        
        # 1: Se debe llamar al método fit del modelo semántico
        # 2: Opcionalmente se puede guardar información del modelo semántico (kdtree y/o embeddings) en el SAR_Indexer
        """
        print(f"Creating kdtree ...", end="")
        
        if not self.chuncks:
            print("Error: No hay frases para crear el KDTree!")
            return
        
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



    def semantic_reranking(self, query: str, articles: List[int]):
        """
        Ordena los articulos en la lista 'articles' por similitud a la consulta 'query'.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultado, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - a partir los chuncks se deben obtener los artículos
            3 - si entre los artículos recuperados NO estan todos los obtenidos por la RI binaria
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - se utiliza la lista ordenada del kdtree para ordenar la lista "articles"
        """
        
        # Verificar que existe el kdtree
        if self.kdtree is None:
            print("Error: KDTree no creado. Use create_kdtree() primero.")
            return articles
        
        # Si no hay chunks o artículos, devolver la lista original
        if len(self.chuncks) == 0 or len(articles) == 0:
            return articles
        
        # Crear embedding de la consulta
        query_embedding = self.model.get_embeddings([query])[0]

        top_k=self.MAX_EMBEDDINGS
        max_k=len(self.chuncks)

        while True:
            #al principio k = MAX_EMBEDDINGS
            distances, indices = self.kdtree.query(
                query_embedding.reshape(1, -1), 
                k=top_k
            )
            
            # Mapear chunks a artículos manteniendo el orden de similitud
            articles_set = set(articles)
            ranked_articles = []
            found_articles = set()
            
            # Procesar resultados en orden de similitud (menor distancia = mayor similitud)
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.chunck_index):
                    art_id = self.chunck_index[idx]
                    # Solo añadir si está en la lista original y no lo hemos añadido ya
                    if art_id in articles_set and art_id not in found_articles:
                        ranked_articles.append(art_id)
                        found_articles.add(art_id)
                        # Si ya encontramos todos los artículos, podemos parar
                        if len(found_articles) == len(articles_set):
                            return ranked_articles
                        
            if top_k == max_k:
                break
            #aumentamos top_k
            top_k = min(top_k*2,max_k)

        #Añadir cualquier artículo faltante al final (por si acaso, dada la indexación empleada, no debería pasar nunca)
        missing_articles = [art_id for art_id in articles if art_id not in found_articles]
        ranked_articles.extend(missing_articles)
        
        


    
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


        # --- INICIO MODIFICACIÓN PARA SEMÁNTICA ---
        # Después de procesar todos los ficheros, si la indexación semántica está activa
        # y tenemos frases (chuncks), creamos el KDTree.
        if self.semantic:
            if self.chuncks: # Solo crear KDTree si hay frases
                self.create_kdtree()
            else:
                print("Warning: Semantic indexing enabled, but no chuncks found to create KDTree.", file=sys.stderr)
        # --- FIN MODIFICACIÓN PARA SEMÁNTICA ---
        
        
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
                    # Corregido: Añadir si la lista está vacía ó si el artid es diferente al último
                    if not self.index[token] or artid != self.index[token][-1]:
                        self.index[token].append(artid)

            # --- INICIO MODIFICACIÓN PARA SEMÁNTICA ---
            # Si la indexación semántica está activada, actualizamos los chuncks (frases)
            if self.semantic:
                # El contenido a pasar a update_chuncks es el texto completo del artículo del cual extraer frases
                self.update_chuncks(article[self.DEFAULT_FIELD], artid) 
            # --- FIN MODIFICACIÓN PARA SEMÁNTICA ---


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
        print(f"Número total de indexed files: {len(self.docs)}")
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


    def solve_query(self, query: str, prev: Dict = {}):
        """
        Resuelve una consulta booleana con evaluación ESTRICA DE IZQUIERDA A DERECHA.
        Maneja operadores AND, OR, NOT y frases.
        """

        # --- Lógica Semántica (la mantenemos como estaba) ---
        if self.semantic_threshold is not None and not self.semantic_ranking:
            semantic_results_artids = self.solve_semantic_query(query.lower())
            return semantic_results_artids, {}
        
        if not query:
            return [], {}

        # --- Lógica Booleana Izquierda-a-Derecha ---
        tokens = re.findall(r'"[^"]+"|\S+', query)
        if not tokens:
            return [], {}

        # 1. Inicializar final_result con el PRIMER término
        i = 0
        final_result = []
        
        if tokens[i].upper() == 'NOT':
            if len(tokens) < 2: return [], {} # Mal formada
            
            next_tok = tokens[i+1]
            if next_tok.startswith('"') and next_tok.endswith('"'): # Frase
                 terms = self.tokenize(next_tok[1:-1])
                 p = self.get_positionals(terms)
            else: # Término
                 p = self.get_posting(next_tok.lower())
            final_result = self.reverse_posting(p)
            i += 2
        
        elif tokens[i].startswith('"') and tokens[i].endswith('"'): # Frase
            terms = self.tokenize(tokens[i][1:-1])
            final_result = self.get_positionals(terms)
            i += 1
        
        elif tokens[i].upper() in ('AND', 'OR'): # Mal formada, no puede empezar con op
             return [], {}
        
        else: # Término normal
            final_result = self.get_posting(tokens[i].lower())
            i += 1
        
        # 2. Variable para guardar la operación (por defecto, AND implícito)
        current_op = self.and_posting 

        # 3. Procesar el resto de tokens
        while i < len(tokens):
            token = tokens[i]

            # Si es un operador, lo guardamos para la siguiente iteración
            if token.upper() == 'AND':
                current_op = self.and_posting
                i += 1
                continue
            elif token.upper() == 'OR':
                current_op = self.or_posting
                i += 1
                continue

            # Si no es un operador, es un término (o NOT + término)
            p = []
            if token.upper() == 'NOT':
                if i + 1 >= len(tokens): break # Mal formada
                
                next_tok = tokens[i+1]
                if next_tok.startswith('"') and next_tok.endswith('"'): # Frase
                    terms = self.tokenize(next_tok[1:-1])
                    p = self.get_positionals(terms)
                else: # Término
                    p = self.get_posting(next_tok.lower())
                
                p = self.reverse_posting(p)
                i += 2
            
            elif token.startswith('"') and token.endswith('"'): # Frase
                terms = self.tokenize(token[1:-1])
                p = self.get_positionals(terms)
                i += 1
            
            else: # Término normal
                p = self.get_posting(token.lower())
                i += 1

            # Aplicamos la última operación guardada
            final_result = current_op(final_result, p)
            
            # Reseteamos al AND implícito por defecto
            current_op = self.and_posting 


        # --- Lógica de Re-ranking (la mantenemos como estaba) ---
        if self.semantic_ranking:
            if not final_result: 
                return final_result, {}
            re_ranked_results = self.semantic_reranking(query, final_result)
            return re_ranked_results, {}
        else:
            return final_result, {}






    def get_posting(self, term:str):
        """
        Devuelve la posting list asociada a un termino.
        Puede llamar self.get_positionals: para las búsquedas posicionales.
        MODIFICADO: Utiliza self.or_posting para unir los resultados del speller.

        param:  "term": termino del que se debe recuperar la posting list.
        return: posting list
        """
        term = term.lower()
        if term in self.index:
            if self.positional:
                # Si es posicional
                return self.get_positionals(term)
            else:
                # Si no es posicional, ya tenemos la lista de artids
                return self.index[term]
        
        # --- Lógica de corrección (Parte 2) ---
        elif term not in self.index and self.use_spelling:
            terms = self.speller.suggest(term)
            if not terms: # No hay sugerencias
                return []
            
            # Inicializa la posting list de resultado
            resul = [] 
            
            if self.positional:
                for word in terms:
                    # Obtenemos la posting del término sugerido
                    word_posting = self.get_positionals(word)
                    # Hacemos la UNIÓN (OR) con el resultado acumulado
                    resul = self.or_posting(resul, word_posting)
            else: # No posicional
                for word in terms:
                    # Obtenemos la posting del término sugerido (asegurándonos de que existe)
                    word_posting = self.index.get(word, [])
                    # Hacemos la UNIÓN (OR) con el resultado acumulado
                    resul = self.or_posting(resul, word_posting)
            
            return resul # 'resul' ya está ordenada por la función or_posting
        
        # Si no está en el índice y no se usa el speller
        return []


    def get_positionals(self, terms):
        """

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LAS BÚSQUESAS POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.

        return: posting list

        """
        if not self.positional:
            raise ValueError("Índice no posicional. No se puede buscar frases exactas.")
        
        #Comprobamos formato de la entrada
        if isinstance(terms, str):
            terms = [terms]
        terms = [t.lower() for t in terms]
        if not terms:
            return []

        #Si es un solo término
        if len(terms) == 1:
            return [artid for artid, _ in self.index.get(terms[0], [])]

        resultado = self.index.get(terms[0])
        if not resultado:
            return []


        #si son n términos, n>1, hacemos n-1 llamadas a interseccion_posicional_con_punteros
        for i in range(1, len(terms)):
            siguiente = self.index.get(terms[i])
            if not siguiente:
                return []
            resultado = self.interseccion_posicional_con_punteros(resultado, siguiente)
            if not resultado:
                return []

        return [artid for artid, _ in resultado]

    # Función adicional
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
        Realizado de acuerdo con los algoritmos vistos en el tema 1.
        """
        all_docs = list(self.articles.keys())  # Obtengo todos los artid
            
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

        Calcula el AND de dos posting list de forma EFICIENTE. Realizado de acuerdo con algoritmo visto en tema 1.
        param:  "p1", "p2": posting lists sobre las que calcula
        return: posting list con los artid incluidos en p1 y p2

        """
        
        # Realizar intersección de IDs
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
    

    def or_posting(self, p1:list, p2:list):
        """
        Calcula el OR de dos posting list de forma EFICIENTE.
        param:  "p1", "p2": posting lists sobre las que calcula
        return: posting list con los artid incluidos en p1 o p2
        """
        result = []
        i = j = 0
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                result.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                result.append(p1[i])
                i += 1
            else: # p1[i] > p2[j]
                result.append(p2[j])
                j += 1
        
        # Añadir los elementos restantes de la lista que no se haya terminado
        result.extend(p1[i:])
        result.extend(p2[j:])
        
        return result







    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se incluye por si es util, no es necesario utilizarla.
        Realizado de acuerdo con explicación al final del tema 1.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 y no en p2

        """
        
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
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados

        param:  "query": query que se debe resolver.

        return: el numero de artículo recuperadas, para la opcion -T

        """
        results, _ = self.solve_query(query)
        total = len(results)
        #obtengo cantidad de resultados para mostrar
        to_show = results if self.show_all else results[:self.SHOW_MAX]
        
        print(f"Recuperados {total} artículos para la consulta '{query}':")
        for idx, artid in enumerate(to_show, start=1):
            if artid is None:
                continue
            art = self.articles[artid]
            print(f"{idx}\t{artid}\t{art['title']}\t{art['url']}")
        
        return total




