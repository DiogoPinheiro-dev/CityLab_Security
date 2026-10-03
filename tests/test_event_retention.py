"""Banco: prazo dos eventos e referencias de rosto aprendidas no stream."""
import contextlib
import io
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

import Server.main as main
from Server.Db.database import (INDICE_RETENCAO_LOGS, INDICE_VALIDADE_APRENDIDOS,
                                configurar_retencao_logs, configurar_validade_aprendidos)
from tools.limpar_aprendidos import principal as limpar_aprendidos

TRINTA_DIAS = 30 * 24 * 60 * 60


class Colecao:
    def __init__(self, indices):
        self.indices = indices
        self.chamadas = []

    async def index_information(self):
        return dict(self.indices)

    async def create_index(self, key, **options):
        self.chamadas.append(("create", key, options))

    async def drop_index(self, name):
        self.chamadas.append(("drop", name))


class EventRetentionTests(unittest.IsolatedAsyncioTestCase):
    async def test_creates_the_ttl_index_when_missing(self):
        colecao = Colecao({"_id_": {"key": [("_id", 1)]}})
        resumo = await configurar_retencao_logs(30, colecao)
        self.assertEqual(colecao.chamadas, [("create", "data_hora_raw", {
            "name": INDICE_RETENCAO_LOGS, "expireAfterSeconds": TRINTA_DIAS})])
        self.assertIn("30 dias", resumo)

    async def test_changes_the_period_of_the_existing_index(self):
        colecao = Colecao({INDICE_RETENCAO_LOGS: {"expireAfterSeconds": 90 * 24 * 60 * 60}})
        await configurar_retencao_logs(30, colecao)
        # Recria em vez de collMod, que pede permissao de administrador.
        self.assertEqual(colecao.chamadas, [
            ("drop", INDICE_RETENCAO_LOGS),
            ("create", "data_hora_raw", {"name": INDICE_RETENCAO_LOGS,
                                         "expireAfterSeconds": TRINTA_DIAS})])

    async def test_same_period_changes_nothing(self):
        colecao = Colecao({INDICE_RETENCAO_LOGS: {"expireAfterSeconds": TRINTA_DIAS}})
        await configurar_retencao_logs(30, colecao)
        self.assertEqual(colecao.chamadas, [])

    async def test_zero_keeps_every_event(self):
        colecao = Colecao({INDICE_RETENCAO_LOGS: {"expireAfterSeconds": TRINTA_DIAS}})
        await configurar_retencao_logs(0, colecao)
        self.assertEqual(colecao.chamadas, [("drop", INDICE_RETENCAO_LOGS)])
        colecao = Colecao({})
        await configurar_retencao_logs(0, colecao)
        self.assertEqual(colecao.chamadas, [])

    async def test_api_starts_even_if_the_index_fails(self):
        class Vazio:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        recognizer = SimpleNamespace(face_service=None, close=Mock())
        saved = (main.recognizer, main.inference_executor)
        self.addCleanup(lambda: setattr(main, "recognizer", saved[0]))
        self.addCleanup(lambda: setattr(main, "inference_executor", saved[1]))
        output = io.StringIO()
        with patch.object(main, "validar_conexao_mongo", AsyncMock()), \
                patch.object(main, "configurar_retencao_logs",
                             AsyncMock(side_effect=RuntimeError("sem permissao"))), \
                patch.object(main, "UnifiedRecognitionService", Mock(return_value=recognizer)), \
                patch.object(main.colecao_alunos, "find", Mock(return_value=Vazio())), \
                contextlib.redirect_stdout(output):
            async with main.lifespan(main.app):
                self.assertIs(main.recognizer, recognizer)
        # O aviso sai no log e a API sobe e desce normalmente.
        self.assertIn("Retencao dos eventos nao configurada: sem permissao", output.getvalue())
        recognizer.close.assert_called_once()


class Aprendidos:
    """Colecao rostos_aprendidos simulada, com os documentos em memoria."""

    def __init__(self, documentos=(), falha=None):
        self.documentos = list(documentos)
        self.falha = falha
        self.indices = {}
        self.criados = []

    def find(self, filtro=None, projecao=None):
        async def gerar():
            for documento in list(self.documentos):
                yield documento
        return gerar()

    async def insert_many(self, documentos):
        if self.falha:
            raise self.falha
        self.documentos.extend(documentos)

    async def delete_many(self, filtro):
        if not filtro:
            apagar = list(self.documentos)
        elif "nome" in filtro:
            apagar = [d for d in self.documentos if d["nome"] == filtro["nome"]]
        else:
            apagar = [d for d in self.documentos if d["_id"] in filtro["_id"]["$in"]]
        self.documentos = [d for d in self.documentos if d not in apagar]
        return SimpleNamespace(deleted_count=len(apagar))

    async def index_information(self):
        return dict(self.indices)

    async def create_index(self, key, **options):
        self.criados.append((key, options))


class LearnedReferenceStorageTests(unittest.IsolatedAsyncioTestCase):
    async def test_learned_references_expire_by_their_own_date(self):
        colecao = Aprendidos()
        resumo = await configurar_validade_aprendidos(30, colecao)
        self.assertEqual(colecao.criados, [("aprendido_em", {
            "name": INDICE_VALIDADE_APRENDIDOS, "expireAfterSeconds": TRINTA_DIAS})])
        self.assertIn("30 dias", resumo)

    async def test_stream_saves_new_references_drops_swapped_ones_and_survives_a_failure(self):
        agora = datetime.now()
        item = {"id": "a1", "nome": "Aluno", "semelhanca": .7, "semelhanca_cadastro": .8,
                "aprendido_em": agora, "embedding": np.full(512, .5, dtype=np.float32)}
        colecao = Aprendidos([{"_id": "velha", "nome": "Aluno"}])
        with patch.object(main, "colecao_aprendidos", colecao):
            await main._guardar_aprendidos({"novos": [item], "removidos": ["velha"]})
        self.assertEqual([d["_id"] for d in colecao.documentos], ["a1"])
        documento = colecao.documentos[0]
        self.assertEqual((documento["nome"], documento["aprendido_em"],
                          documento["semelhanca_cadastro"]), ("Aluno", agora, .8))
        # O embedding vai como lista de float comum, que o MongoDB grava.
        self.assertEqual(len(documento["embedding"]), 512)
        self.assertIsInstance(documento["embedding"][0], float)
        output = io.StringIO()
        with patch.object(main, "colecao_aprendidos", Aprendidos(falha=RuntimeError("fora"))), \
                contextlib.redirect_stdout(output):
            await main._guardar_aprendidos({"novos": [item], "removidos": []})
        self.assertIn("Referencias aprendidas nao gravadas: fora", output.getvalue())

    async def test_startup_loads_references_and_drops_people_out_of_the_registry(self):
        agora = datetime.now()
        documentos = [{"_id": key, "nome": nome, "semelhanca": .7, "aprendido_em": agora,
                       "embedding": [.1] * 512} for key, nome in (("a", "Aluno"), ("b", "Saiu"))]
        colecao = Aprendidos(documentos)
        # O servico de rosto decide quem fica de fora; aqui, a de quem saiu.
        face_service = SimpleNamespace(replace_learned=Mock(return_value=["b"]))
        recognizer = SimpleNamespace(face_service=face_service)
        with patch.object(main, "colecao_aprendidos", colecao), \
                patch.object(main, "configurar_validade_aprendidos",
                             AsyncMock(side_effect=RuntimeError("sem permissao"))), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            await main._carregar_aprendidos(recognizer)
        # O prazo falhou, mas as referencias carregam, e as de fora saem do banco.
        self.assertIn("Prazo das referencias aprendidas nao configurado", output.getvalue())
        self.assertIn("1 referencias aprendidas carregadas", output.getvalue())
        self.assertEqual([d["_id"] for d in colecao.documentos], ["a"])
        carregadas = face_service.replace_learned.call_args.args[0]
        self.assertEqual([(item["id"], item["nome"]) for item in carregadas],
                         [("a", "Aluno"), ("b", "Saiu")])

    async def test_cleanup_tool_lists_and_deletes(self):
        colecao = Aprendidos([{"_id": str(i), "nome": nome}
                              for i, nome in enumerate(["Ana", "Ana", "Bia"])])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            await limpar_aprendidos([], colecao)
        self.assertEqual(output.getvalue().splitlines(), ["2  Ana", "1  Bia"])
        with contextlib.redirect_stdout(io.StringIO()):
            await limpar_aprendidos(["--nome", "Ana"], colecao)
        self.assertEqual([d["nome"] for d in colecao.documentos], ["Bia"])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            await limpar_aprendidos(["--tudo"], colecao)
        self.assertEqual(colecao.documentos, [])
        self.assertIn("1 referencias apagadas", output.getvalue())


if __name__ == "__main__":
    unittest.main()
