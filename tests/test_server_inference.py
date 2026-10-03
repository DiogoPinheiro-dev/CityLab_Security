"""O servidor atende as outras rotas enquanto a thread de inferencia roda um frame."""
import contextlib
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

import Server.main as main


@contextlib.asynccontextmanager
async def sem_ciclo(_app):
    # Sem MongoDB nem modelos: o teste monta a pipeline falsa a mao.
    yield


class InferenceThreadTests(unittest.TestCase):
    FRAME_SECONDS = 1.0

    def setUp(self):
        self.calls = []
        self.started = threading.Event()

        def note(kind):
            self.calls.append((kind, threading.current_thread().name))

        def process_frame(frame):
            note("frame")
            self.started.set()
            time.sleep(self.FRAME_SECONDS)
            note("fim do frame")
            return {"faces": [], "gestures": [], "persons": [], "metrics": {}}

        def get(img):
            note("foto")
            return [SimpleNamespace(normed_embedding=np.ones(512, dtype=np.float32))]

        face_service = SimpleNamespace(
            app_insight=SimpleNamespace(get=get),
            average_embedding=lambda embeddings: (embeddings[0], None),
            replace_known_faces=lambda names, embeddings: note("memoria"))
        recognizer = SimpleNamespace(
            process_frame=process_frame, face_service=face_service,
            reset_gesture_history=lambda: note("gestos"),
            reset_face_identities=lambda: note("rostos"))

        saved = (main.recognizer, main.inference_executor, main.app.router.lifespan_context,
                 list(main.banco_rostos_memoria["nomes"]),
                 list(main.banco_rostos_memoria["embeddings"]))
        main.recognizer = recognizer
        main.inference_executor = ThreadPoolExecutor(max_workers=1,
                                                     thread_name_prefix="inferencia")
        main.app.router.lifespan_context = sem_ciclo

        def restore():
            main.inference_executor.shutdown(wait=True)
            (main.recognizer, main.inference_executor, main.app.router.lifespan_context,
             names, embeddings) = saved
            main.banco_rostos_memoria["nomes"][:] = names
            main.banco_rostos_memoria["embeddings"][:] = embeddings

        self.addCleanup(restore)
        self.jpeg = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))[1].tobytes()

    def test_other_routes_answer_while_a_frame_is_processed(self):
        # Com o cliente como contexto, stream e rotas dividem o mesmo laco, como
        # no uvicorn: um frame que travasse o laco atrasaria a rota junto.
        with TestClient(main.app) as client, client.websocket_connect("/stream") as ws:
            ws.send_bytes(self.jpeg)
            self.assertTrue(self.started.wait(5))
            started = time.perf_counter()
            response = client.get("/config/client")
            elapsed = time.perf_counter() - started
            self.assertEqual(response.status_code, 200)
            self.assertLess(elapsed, self.FRAME_SECONDS / 2)
            self.assertEqual(ws.receive_json()["frame"], 1)
        # O reinicio da conexao nova roda na mesma thread do frame, antes dele.
        self.assertEqual(self.calls, [("gestos", "inferencia_0"), ("rostos", "inferencia_0"),
                                      ("frame", "inferencia_0"),
                                      ("fim do frame", "inferencia_0")])

    def test_learned_references_go_to_the_database_and_not_to_the_client(self):
        item = {"id": "a1", "nome": "Aluno", "semelhanca": .7,
                "embedding": np.ones(512, dtype=np.float32)}
        mudancas = {"novos": [item], "removidos": ["velha"]}
        main.recognizer.process_frame = lambda frame: {
            "faces": [], "gestures": [], "persons": [], "metrics": {}, "aprendidos": mudancas}
        with TestClient(main.app) as client, client.websocket_connect("/stream") as ws, \
                patch.object(main, "_guardar_aprendidos", AsyncMock()) as guardar:
            ws.send_bytes(self.jpeg)
            resposta = ws.receive_json()
        guardar.assert_awaited_once_with(mudancas)
        self.assertNotIn("aprendidos", resposta)
        self.assertNotIn("embedding", str(resposta))

    def test_enrollment_waits_for_the_frame_and_swaps_names_between_frames(self):
        with TestClient(main.app) as client, client.websocket_connect("/stream") as ws, \
                patch.object(main.colecao_alunos, "insert_one", AsyncMock()):
            ws.send_bytes(self.jpeg)
            self.assertTrue(self.started.wait(5))
            response = client.post("/cadastro", data={"nome": "teste"},
                                   files=[("foto", ("1.jpg", self.jpeg, "image/jpeg"))])
            self.assertEqual(response.status_code, 200)
            ws.receive_json()
        # A foto e a troca da memoria entram na fila depois do frame, nunca no
        # meio dele, e na mesma thread dos modelos.
        kinds = [kind for kind, _ in self.calls]
        self.assertEqual(kinds[-3:], ["fim do frame", "foto", "memoria"])
        self.assertEqual({name for _, name in self.calls}, {"inferencia_0"})


if __name__ == "__main__":
    unittest.main()
