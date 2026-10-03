"""Prazo dos eventos no banco: o indice TTL e a API subindo mesmo se ele falhar."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import Server.main as main
from Server.Db.database import INDICE_RETENCAO_LOGS, configurar_retencao_logs

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


if __name__ == "__main__":
    unittest.main()
