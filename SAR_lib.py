import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle
from distancias_parte2 import opcionesSpell # Opciones de modelos de cálculo de distancia para spellsuggester
from spellsuggester import SpellSuggester # Crear objeto SpellSuggester para correción ortográfica


##################################################
##                                              ##
##    LIBRERIA DE INDEXACION DE NOTICIAS        ##
##                                              ##
##              NO OPTIMIZADO !!                ##
## NO se puede utilizar PARA el proyecto de SAR ##
##                                              ##
##################################################


class SAR_Indexer:
    """
    Prototipo MINIMO del proyecto de SAR para ser utilizado en ALT
    """

    # campo que se indexa
    DEFAULT_FIELD = 'all'
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10

    all_atribs = ['urls', 'index', 'docs', 'articles', 'tokenizer', 'show_all',
                  "semantic", "chuncks", "embeddings", "chunck_index", "kdtree", "artid_to_emb"]


    def __init__(self, **args):
        """
        Constructor de la classe SAR_Indexer.

        Incluye todas las variables necesaria para todas las ampliaciones.
        Puedes añadir más variables si las necesitas 

        args permite acceder a los argumento de SAR_Indexer

        args['distance']
        args['threshold']

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
        
        # ALT - COMPLETAR
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
        
        Recorre recursivamente el directorio "root", también puede ser un único fichero.
        NECESARIO PARA TODAS LAS VERSIONES
        
        Recorre recursivamente el directorio "root" e indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.positional = args['positional']

        file_or_dir = Path(root)
        
        if file_or_dir.is_file():
            # is a file
            self.index_file(root)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in files:
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        ##########################################
        ## COMPLETAR PARA FUNCIONALIDADES EXTRA ##
        ##########################################
        
        
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

        dependiendo del valor de self.multifield y self.positional se debe ampliar el indexado
        """
        
        docid = len(self.docs)
        self.docs[docid] = filename
        for i, line in enumerate(open(filename, encoding='utf8')):
            j = self.parse_article(line)
            #print(j['url'])
            if self.already_in_index(j):
                continue
            artid = len(self.articles)
            self.articles[artid] = (docid, i)
            self.urls.add(j['url'])
            
            # ADD token in 'all' field to the index
            index = self.index.setdefault('all', {})
            #for token in set(self.tokenize(j['all'])):
            for token in self.tokenize(j['all']):
                if token not in index:
                    index[token] = [artid]
                elif index[token][-1] != artid:
                    index[token].append(artid)


    def set_spelling(self, use_spelling:bool, distance:str=None, threshold:int=None):
        """
        self.use_spelling a True activa la corrección ortográfica
        EN LAS PALABRAS NO ENCONTRADAS, en caso contrario NO utilizará
        corrección ortográfica
        
        input: "use_spell" booleano, determina el uso del corrector.
                "distance" cadena, nombre de la función de distancia.
                "threshold" entero, umbral del corrector
        """
        
        # ALT - COMPLETAR        
        if use_spelling: # Si está habilitado spelling
            if (distance is not None) and (distance not in opcionesSpell): # Si se utiliza una función válida
                raise ValueError(f"La función de distancia '{distance}' no es válida.")
            elif (threshold is not None) and (threshold < 0): # Si el threshold es mayor a 0
                    raise ValueError(f"El valor de threshold '{threshold}' no es válido.")
            else: # Se crea un objeto speller con los parámetros especificados que utilizará el Searcher
                self.speller = SpellSuggester(opcionesSpell, list(self.index), distance, threshold)

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
        print("=" * 40)
        print("Number of indexed days:", len(self.docs))
        print("-" * 40)
        print("Number of indexed articles:", len(self.articles))
        print("-" * 40)
        print('TOKENS:')
        print("\t# of tokens in 'all': %d" % (len(self.index['all'])))
        print("=" * 40)




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
        spt = query.split()
        i = 0
        if spt[i].lower() == 'not':
            neg = True
            i += 1
        else:
            neg = False
        # posting list 1
        term = spt[i]
        posting = self.get_posting(term)
        l1 = (neg, posting)
        i += 1
        while i < len(spt):
            conn = spt[i].lower()
            i += 1
            neg = False
            if spt[i].lower() == 'not':
                neg = True
                i += 1
            term = spt[i]
            posting = self.get_posting(term)
            l2 = (neg, posting)
            l1 = self.solve_conn(conn, l1, l2)
            i += 1
        if l1[0] is False:
            post = l1[1]
        else:
            post = self.reverse_posting(l1[1])
        return post

    def get_posting(self, term:str, field:str='all'):
        """
        Devuelve la posting list asociada a un termino. 
        Dependiendo de las ampliaciones implementadas "get_posting" puede llamar a:
            - self.get_positionals: para la ampliacion de posicionales


        param:  "term": termino del que se debe recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario si se hace la ampliacion de multiples indices

        return: posting list
        
        NECESARIO PARA TODAS LAS VERSIONES

        """

        # ALT - MODIFICAR
        term = term.lower()
        if term in self.index:
            if self.positional:
                # Si es posicional
                return self.get_positionals(term)
            else:
                # Si no es posicional, ya tenemos la lista de artids
                return self.index[term]
        elif term not in self.index and self.use_spelling:
            terms = self.speller.suggest(term)
            if terms is not None:
                    resul = set()
                    if self.positional:
                    # Si es posicional
                        for word in terms:
                            resul.update(self.get_positionals(word))
                    else:
                        for word in terms:
                    # Si no es posicional, ya tenemos la lista de artids
                            resul.update(self.index[word])
                    return sorted(list(resul))
        return []


    def reverse_posting(self, p:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los artid exceptos los contenidos en p

        """

        return [aid for aid in range(0, len(self.articles)) if aid not in p]




    def solve_conn(self, conn:str, r1:List, r2:List):
        """
        
        param:  "conn": conectiva, puede ser 'and' o 'or'.
            "r1": primer resultado formado por dos elementos, el primero es True o False y el segundo es la lista de postings; si el primer elemento es False se debe negar la lista
            "r2": segundo resultado
        """
        
        pl1 = r1[1] if r1[0] is False else self.reverse_posting(r1[1])
        pl2 = r2[1] if r2[0] is False else self.reverse_posting(r2[1])

        if conn == 'and':
            # AND
            r = self.and_posting(pl1, pl2)
        elif conn == 'or':
            # OR
            r = self.or_posting(pl1, pl2)
        else:
            # no deberia pasar
            r = []
        return False, r


    def and_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma NO EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular

        return: posting list con los artid incluidos en p1 y p2

        """
        
        return sorted(set(p1).intersection(p2))



    def or_posting(self, p1:list, p2:list):
        """
        Calcula el OR de dos posting list de forma NO EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 o p2

        """

        return sorted(set(p1).union(p2))



    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql:List[str], verbose:bool=True) -> List:
        """
        Resuelve una serie de consultas y devuelve el número de resultados, opcionalmente los muestra en el terminal. 

        param:  "ql": lista de queries que se debe resolver.
		"verbose": si True muestra el resultado por el termina, por defecto True

        return: la lista de resultados

        """
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results


    def solve_and_test(self, ql:List[str]) -> bool:
        """
        Resuelve una serie de consultas y las compara con un resultado de referencia. 

        param:  "ql": lista de queries que se debe resolver, cada query tiene un tabulador que separa la propia query del resultado esperado

        return: bool: True si todos los resultados coinciden

        """
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                result = len(self.solve_query(query))
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
        Resuelve una consulta y la muestra junto al numero de resultados 

        param:  "query": query que se debe resolver.

        return: el numero de artículo recuperadas, para la opcion -T

        """
        
        print('funcionalidad no implementada')        

        result = self.solve_query(query)
        rtotal = len(result)
        nmax = rtotal if self.show_all else min(self.SHOW_MAX, len(result))
        result = result[:nmax]
        
        print('=' * 40)
        for i, artid in enumerate(result):
            docid, artpos = self.articles[artid]



            #path = self.docs[docid]
            #print(path)
            #obj = self.get_article(path, artpos)
            #print(f'# {i+1:0{digs1}d} ({artid:{digs2}d}) {obj["title"]}:\t{obj["url"]} ')
            print(f'# {i+1:3d} ({docid}) ({artpos:d})')
        print('=' * 40)
        print('Number of results:', rtotal)





        

