"""tools/cadastros.py: listar, substituir e apagar cadastros no banco."""
import contextlib
import io
import unittest
from types import SimpleNamespace

from tools.cadastros import CAMPOS, principal as cadastros


class Colecao:
    """Colecao do MongoDB simulada, com os documentos em memoria."""

    def __init__(self, documentos=()):
        self.documentos = list(documentos)
        self.projecoes = []

    def find(self, filtro=None, projecao=None):
        self.projecoes.append(projecao)

        async def gerar():
            for documento in list(self.documentos):
                if all(documento.get(campo) == valor for campo, valor in (filtro or {}).items()):
                    yield documento
        return gerar()

    async def delete_many(self, filtro):
        if "_id" in filtro:
            apagar = [d for d in self.documentos if d["_id"] in filtro["_id"]["$in"]]
        else:
            apagar = [d for d in self.documentos if d["nome"] == filtro["nome"]]
        self.documentos = [d for d in self.documentos if d not in apagar]
        return SimpleNamespace(deleted_count=len(apagar))


class RegistrationToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Ana se recadastrou com 5 fotos; o primeiro cadastro, de uma foto, e de
        # antes do campo fotos. O _id do MongoDB cresce com a hora de entrada.
        self.alunos = Colecao([
            {"_id": 1, "nome": "Ana", "embedding": [.1], "cadastrado_em": "01/10/2026 - 10:00:00"},
            {"_id": 3, "nome": "Ana", "embedding": [.2], "fotos": 5,
             "cadastrado_em": "04/10/2026 - 09:00:00"},
            {"_id": 2, "nome": "Bia", "embedding": [.3], "fotos": 3,
             "cadastrado_em": "02/10/2026 - 08:00:00"},
        ])
        self.aprendidos = Colecao([{"_id": f"a{i}", "nome": "Ana"} for i in range(3)]
                                  + [{"_id": "b0", "nome": "Bia"}])

    async def rodar(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            codigo = await cadastros(list(argv), self.alunos, self.aprendidos)
        return codigo, output.getvalue().splitlines()

    async def test_lists_each_person_without_reading_the_embedding(self):
        codigo, linhas = await self.rodar()
        self.assertEqual(codigo, 0)
        self.assertEqual(linhas, [
            "Ana: 2 cadastros, 3 referencias aprendidas",
            "  1 foto, 01/10/2026 - 10:00:00",
            "  5 fotos, 04/10/2026 - 09:00:00 (o mais recente)",
            "Bia: 1 cadastro, 1 referencia aprendida",
            "  3 fotos, 02/10/2026 - 08:00:00",
        ])
        self.assertEqual(self.alunos.projecoes, [CAMPOS])
        self.assertNotIn("embedding", CAMPOS)

    async def test_keeping_the_latest_drops_the_old_registration_only(self):
        codigo, linhas = await self.rodar("--manter-ultimo", "Ana")
        self.assertEqual(codigo, 0)
        self.assertIn("1 cadastro antigo. Fica o de 5 fotos", linhas[0])
        self.assertEqual([d["_id"] for d in self.alunos.documentos], [3, 2])
        # As referencias aprendidas ficam: a API confere na subida se ainda
        # passam perto do cadastro novo.
        self.assertEqual(len(self.aprendidos.documentos), 4)
        codigo, linhas = await self.rodar("--manter-ultimo", "Ana")
        self.assertEqual((codigo, len(self.alunos.documentos)), (0, 2))
        self.assertIn("nada a apagar", linhas[0])

    async def test_deleting_a_person_takes_the_learned_references_too(self):
        codigo, linhas = await self.rodar("--apagar", "Ana")
        self.assertEqual(codigo, 0)
        self.assertEqual(linhas, ["Apagados de 'Ana': 2 cadastros e 3 referencias aprendidas. "
                                  "Reinicie a API para ela esquecer tambem."])
        self.assertEqual([d["nome"] for d in self.alunos.documentos], ["Bia"])
        self.assertEqual([d["nome"] for d in self.aprendidos.documentos], ["Bia"])

    async def test_unknown_name_changes_nothing(self):
        for opcao in ("--apagar", "--manter-ultimo"):
            codigo, linhas = await self.rodar(opcao, "ana")
            self.assertEqual(codigo, 1)
            self.assertIn("Nenhum cadastro com o nome 'ana'", linhas[0])
        self.assertEqual((len(self.alunos.documentos), len(self.aprendidos.documentos)), (3, 4))


if __name__ == "__main__":
    unittest.main()
