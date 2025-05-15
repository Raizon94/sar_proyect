import unittest
import io
import sys
from pathlib import Path
from SAR_lib import SAR_Indexer

class TestSARIndexer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Determinar la ruta al directorio de datos JSON (1000 dentro de sar_proyect)
        project_root = Path(__file__).parent
        data_dir = project_root / "1000"
        # Inicializar y construir índice posicional básico
        cls.idx = SAR_Indexer()
        cls.idx.index_dir(str(data_dir), positional=True, semantic=False)
    def test_empty_query(self):
        # Consulta vacía debe devolver lista vacía
        r, _ = self.idx.solve_query("")
        self.assertEqual(r, [])

    def test_none_query(self):
        # Consulta None debe devolver lista vacía
        r, _ = self.idx.solve_query(None)
        self.assertEqual(r, [])

    def test_nonexistent_term(self):
        # Término que no existe en el índice -> sin resultados
        r, _ = self.idx.solve_query("zxqwy")
        self.assertEqual(r, [])

    def test_not_nonexistent(self):
        # NOT de un término inexistente -> todos los documentos
        r, _ = self.idx.solve_query("NOT zxqwy")
        all_ids = sorted(self.idx.articles.keys())
        self.assertEqual(r, all_ids)

    def test_solve_and_show_no_results(self):
        # Capturar salida de solve_and_show con cero resultados
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        total = self.idx.solve_and_show("zxqwy")
        sys.stdout = old_stdout
        # Debe devolver total=0 y no imprimir líneas
        self.assertEqual(total, 0)
        self.assertEqual(buf.getvalue().strip(), "")

if __name__ == '__main__':
    unittest.main()
