"""Regressoes de alertas, concorrencia e equivalencia do filtro facial."""
import asyncio
import os
import sys
import tempfile
import unittest
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from App.GestureRecon.detector import GestureAnalyzer
from App.inference_runtime import configure_insight_threads, configure_torch_threads
import test_rpi_optimization as fixtures
from test_rpi_optimization import load_class


class EventConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        fixture = fixtures.EventLoggerTests()
        fixture.setUp()
        self.fixture = fixture

    async def test_same_identity_is_reserved_during_insert(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def insert(payload):
            started.set()
            await release.wait()

        f = self.fixture
        f.collection.insert_one.side_effect = insert
        task = asyncio.create_task(f.logger.log_face_events(None, [f.face]))
        await started.wait()
        await f.logger.log_face_events(None, [f.face])
        self.assertEqual(f.collection.insert_one.await_count, 1)
        release.set()
        await task
        self.assertFalse(f.logger.in_flight)

    async def test_cancel_releases_identity_and_allows_retry(self):
        started = asyncio.Event()

        async def insert(payload):
            started.set()
            await asyncio.Event().wait()

        f = self.fixture
        f.collection.insert_one.side_effect = insert
        task = asyncio.create_task(f.logger.log_face_events(None, [f.face]))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(f.logger.in_flight)
        self.assertFalse(f.logger.last_logged)
        f.collection.insert_one.side_effect = None
        await f.logger.log_face_events(None, [f.face])
        self.assertEqual(f.collection.insert_one.await_count, 2)

    async def test_other_identity_does_not_wait_for_slow_insert(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def insert(payload):
            if payload["nome"] == "Teste":
                started.set()
                await release.wait()

        f = self.fixture
        f.collection.insert_one.side_effect = insert
        task = asyncio.create_task(f.logger.log_face_events(None, [f.face]))
        await started.wait()
        await f.logger.log_face_events(None, [{**f.face, "name": "Outro"}])
        self.assertIn("ALUNO:Outro", f.logger.last_logged)
        release.set()
        await task


class GestureSafetyTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = GestureAnalyzer()
        self.points = [[0, 0, 0] for _ in range(17)]
        # Braco esquerdo na horizontal: o punho so conta com o braco levantado.
        self.points[5], self.points[9] = [100, 100, 0.9], [140, 100, 0.9]

    def observe(self, now, closed=True):
        return self.analyzer.analyze(1, self.points, observed_at=now,
                                    hand_context={"left_closed": closed})["alerts"]

    def test_long_pause_does_not_confirm_gesture(self):
        self.observe(0)
        self.assertNotIn("Mao Fechada", self.observe(600))
        self.assertNotIn("Mao Fechada", self.observe(600.05, False))

    def test_normal_pi_interval_still_confirms(self):
        self.observe(0)
        self.assertIn("Mao Fechada", self.observe(17))
        self.assertLessEqual(self.analyzer.history[1]["fist_frames"], .2)
        self.assertNotIn("Mao Fechada", self.observe(17.05, False))

    def test_long_continuous_gesture_does_not_latch_alert(self):
        for now in range(100):
            self.observe(now)
        self.assertNotIn("Mao Fechada", self.observe(99.05, False))

    def test_session_reset_discards_previous_evidence(self):
        self.observe(0)
        self.observe(17)
        self.analyzer.clean_old_tracks([])
        self.assertNotIn("Mao Fechada", self.observe(17.1))

    def test_backward_timestamp_resets_evidence(self):
        self.observe(10)
        self.observe(11)
        self.assertNotIn("Mao Fechada", self.observe(2))


# O que App/FaceRecon/service.py importa no modulo, para carregar so a classe.
FACE_SERVICE_DEPENDENCIES = dict(
    DEBUG_PIPELINE=True, FACE_MIN_WIDTH=40, FACE_MIN_HEIGHT=40, FACE_MIN_CONFIDENCE=.45,
    FACE_MINIMAL_MODULES=True, FACE_PREFILTER=True, ONNX_INTRA_OP_THREADS=0,
    ONNX_ALLOW_SPINNING=True, FACE_REUSE_SECONDS=0.0, FACE_EMBED_FULL_FRAME=False,
    FACE_LEARN_FROM_STREAM=False, FACE_LEARNED_PER_PERSON=5, FACE_LEARN_INTERVAL_SECONDS=600.0,
    FACE_LEARNED_RETENTION_DAYS=30, FACE_DETECTOR_PATH="", FACE_ASYNC_RECOGNITION=False,
    Face=SimpleNamespace, uuid=uuid, datetime=datetime, timedelta=timedelta)


class Vector(list):
    def __getitem__(self, key):
        result = super().__getitem__(key)
        return Vector(result) if isinstance(key, slice) else result

    def astype(self, dtype):
        return Vector(dtype(value) for value in self)

    def tolist(self):
        return list(self)


class FaceOptimizationTests(unittest.TestCase):
    def make_service(self, prefilter, boxes, minimal=True, reuse_seconds=0.0):
        numpy_stub = SimpleNamespace(empty=lambda shape, dtype=None: [], float32=float,
                                     asarray=lambda value, dtype=None: value)
        cls = load_class("App/FaceRecon/service.py", "FaceRecognitionService",
                         **FACE_SERVICE_DEPENDENCIES, np=numpy_stub)
        service = cls.__new__(cls)
        service.prefilter = prefilter
        service.embed_full_frame = False
        service.learn_from_stream = False
        service.learned, service._learned_names, service._learned_embeddings = {}, [], []
        service.latest_learned, service.latest_forgotten = [], []
        service.debug_pipeline = True
        service.face_min_width = service.face_min_height = 40
        service.face_min_confidence = .45
        service.reuse_seconds = reuse_seconds
        service._identities = []
        service._identity_generation = 0
        service.async_recognition = False
        service._recognition_pool, service._job, service._job_count = None, None, 0
        service.latest_metrics = {}
        service._match_face = lambda embedding: ("Aluno", embedding[0])
        keys = [object() for _ in boxes]

        def recognize(frame, face):
            self.assertIn(face.kps, keys)
            face.normed_embedding = [.87]

        recognition = Mock(get=Mock(side_effect=recognize))
        addon = Mock(get=Mock())
        models = {"detection": Mock(), "recognition": recognition}
        if not minimal:
            models["landmark_3d_68"] = addon

        def get(frame):
            faces = [SimpleNamespace(bbox=box[:4], det_score=box[4], kps=keys[i])
                     for i, box in enumerate(boxes)]
            for face in faces:
                for name, model in models.items():
                    if name != "detection":
                        model.get(frame, face)
            return faces

        service.app_insight = SimpleNamespace(
            get=Mock(side_effect=get), models=models,
            det_model=SimpleNamespace(detect=Mock(return_value=(boxes, keys))))
        context = SimpleNamespace(processing_frame=object(),
                                  map_bbox_to_original=lambda box: [v * 2 for v in box],
                                  clip_original_bbox=lambda box: box)
        return service, context, recognition, addon

    def test_same_payload_with_less_embedding_work(self):
        boxes = [Vector([0, 0, 30, 30, .9]), Vector([0, 0, 10, 10, .9]),
                 Vector([0, 0, 30, 30, .4]), Vector([5, 5, 5, 5, .99])]
        old, frame, old_rec, _ = self.make_service(False, boxes)
        new, new_frame, new_rec, _ = self.make_service(True, boxes)
        self.assertEqual(old.recognize_faces(frame), new.recognize_faces(new_frame))
        self.assertEqual(old.latest_ignored_faces, new.latest_ignored_faces)
        self.assertEqual(old_rec.get.call_count, 4)
        self.assertEqual(new_rec.get.call_count, 1)
        new.app_insight.det_model.detect.assert_called_once_with(
            new_frame.processing_frame, max_num=0, metric="default")

    def test_face_time_is_split_into_detection_embedding_and_match(self):
        boxes = [Vector([0, 0, 30, 30, .9]), Vector([0, 0, 10, 10, .9]),
                 Vector([0, 0, 30, 30, .9])]
        new, frame, _, _ = self.make_service(True, boxes)
        new.recognize_faces(frame)
        metrics = new.latest_metrics
        # So os dois rostos aceitos pelo filtro de qualidade geram embedding.
        self.assertEqual(metrics["face_embeddings"], 2.0)
        for name in ("faces_ms", "face_detect_ms", "face_embed_ms", "face_match_ms"):
            self.assertGreaterEqual(metrics[name], 0.0)
        self.assertLessEqual(metrics["face_detect_ms"] + metrics["face_embed_ms"]
                             + metrics["face_match_ms"], metrics["faces_ms"])
        # Sem prefiltro o FaceAnalysis.get faz tudo junto: so o total e a contagem.
        old, old_frame, _, _ = self.make_service(False, boxes)
        old.recognize_faces(old_frame)
        self.assertEqual(old.latest_metrics["face_embeddings"], 3.0)
        self.assertEqual((old.latest_metrics["face_detect_ms"],
                          old.latest_metrics["face_embed_ms"]), (0.0, 0.0))

    def test_full_frame_embedding_uses_the_original_frame_and_scaled_points(self):
        box = Vector([10, 20, 40, 60, .9])
        kps = [[15, 30], [35, 30], [25, 40], [18, 50], [32, 50]]
        results = {}
        for full in (False, True):
            service, context, recognition, _ = self.make_service(True, [box])
            service.embed_full_frame = full
            service.app_insight.det_model.detect.return_value = ([box], [kps])
            seen = []

            def recognize(frame, face):
                seen.append((frame, face))
                face.normed_embedding = [.87]

            recognition.get.side_effect = recognize
            context = SimpleNamespace(**vars(context), original_frame=object(),
                                      scale_x=2.0, scale_y=2.0)
            results[full] = service.recognize_faces(context)
            # A deteccao continua na imagem reduzida; so o embedding muda de imagem.
            service.app_insight.det_model.detect.assert_called_once_with(
                context.processing_frame, max_num=0, metric="default")
            frame, face = seen[0]
            if full:
                self.assertIs(frame, context.original_frame)
                self.assertEqual(face.kps, [[30, 60], [70, 60], [50, 80], [36, 100], [64, 100]])
                self.assertEqual(face.bbox, [20, 40, 80, 120])
                self.assertEqual(face.det_score, .9)
            else:
                self.assertIs(frame, context.processing_frame)
                self.assertIs(face.kps, kps)
        # Mesmo payload: caixa no frame original, nome e semelhanca do embedding.
        self.assertEqual(results[False], results[True])
        self.assertEqual(results[True][0]["bbox"], [20, 40, 80, 120])

    def test_registration_averages_the_photos_and_flags_another_person(self):
        import numpy
        cls = load_class("App/FaceRecon/service.py", "FaceRecognitionService",
                         **FACE_SERVICE_DEPENDENCIES, np=numpy)

        def unit(*values):
            vector = numpy.array(values, dtype=numpy.float32)
            return vector / numpy.linalg.norm(vector)

        first, second, third = unit(1, .2, 0), unit(1, 0, .2), unit(1, .1, .1)
        # Uma foto so: o mesmo embedding do cadastro de antes.
        mean, outlier = cls.average_embedding([first])
        self.assertIsNone(outlier)
        numpy.testing.assert_allclose(mean, first, rtol=1e-6)
        # Varias da mesma pessoa: a media, normalizada como os embeddings do stream.
        mean, outlier = cls.average_embedding([first, second, third])
        self.assertIsNone(outlier)
        self.assertAlmostEqual(float(numpy.linalg.norm(mean)), 1.0, places=5)
        self.assertGreater(float(mean @ first), float(second @ first))
        # Uma foto de outra pessoa nao entra na media: o indice dela volta.
        other = unit(0, 0, 1)
        self.assertEqual(cls.average_embedding([first, other, second]), (None, 1))
        self.assertEqual(cls.average_embedding([first, other])[1], 0)

    def learning_service(self, per_person=5, interval=0):
        """Servico com o aprendizado ligado e numpy de verdade, cadastro de um aluno."""
        import numpy
        cls = load_class("App/FaceRecon/service.py", "FaceRecognitionService",
                         **FACE_SERVICE_DEPENDENCIES, np=numpy)
        service = cls.__new__(cls)
        service.similarity_threshold = .52
        service.learn_from_stream = True
        service.learned_per_person = per_person
        service.learn_interval = timedelta(seconds=interval)
        service.learned_retention = timedelta(days=30)
        service.learned, service._learned_names = {}, []
        service._learned_embeddings = numpy.empty((0, 512), dtype=numpy.float32)
        service.latest_learned, service.latest_forgotten = [], []
        service._identities, service._identity_generation = [], 0

        def vector(*values):
            result = numpy.zeros(512, dtype=numpy.float32)
            result[:len(values)] = values
            return result / numpy.linalg.norm(result)

        service.replace_known_faces(["Aluno"], [vector(1, 0)])
        return service, vector

    def test_stream_face_near_the_registration_becomes_a_reference(self):
        service, vector = self.learning_service()
        live = vector(1, .5)  # 0,894 do cadastro
        service._maybe_learn("Aluno", .894, live)
        self.assertEqual([entry["nome"] for entry in service.latest_learned], ["Aluno"])
        # Um rosto longe do cadastro (0,406) e perto do aprendido (0,772) e reconhecido.
        name, score = service._match_face(vector(.4, .9))
        self.assertEqual(name, "Aluno")
        self.assertAlmostEqual(score, .772, places=3)

    def test_learning_needs_margin_and_the_registration(self):
        service, vector = self.learning_service()
        service._maybe_learn("NAO ALUNO", .9, vector(1, .5))
        service._maybe_learn("Aluno", .55, vector(1, .5))  # abaixo de 0,60
        self.assertEqual(service.learned, {})
        service._maybe_learn("Aluno", .894, vector(1, .5))
        # Reconhecido so pelo aprendido, longe do cadastro: nao vira referencia,
        # para as referencias nao derivarem para outra pessoa.
        service._maybe_learn("Aluno", .772, vector(.4, .9))
        self.assertEqual(len(service.learned["Aluno"]), 1)

    def test_learning_respects_the_limit_and_the_interval(self):
        service, vector = self.learning_service(per_person=2)
        # Tres rostos a 0,8 do cadastro e a 0,64 entre si: nenhum e copia de outro.
        for values in ((.8, .6), (.8, 0, .6), (.8, 0, 0, .6)):
            service._maybe_learn("Aluno", .9, vector(*values))
        self.assertEqual(len(service.learned["Aluno"]), 2)
        service, vector = self.learning_service(interval=600)
        service._maybe_learn("Aluno", .9, vector(.8, .6))
        service._maybe_learn("Aluno", .9, vector(.8, 0, .6))
        self.assertEqual(len(service.learned["Aluno"]), 1)

    def test_full_slots_swap_the_reference_farthest_from_the_registration(self):
        service, vector = self.learning_service(per_person=2)
        service._maybe_learn("Aluno", .9, vector(.6, .8))  # 0,6 do cadastro
        service._maybe_learn("Aluno", .9, vector(.8, 0, .6))  # 0,8
        first = service.learned["Aluno"][0]["id"]
        service.latest_learned, service.latest_forgotten = [], []
        # Mais perto do cadastro (0,96): sai a mais longe, e o banco apaga.
        service._maybe_learn("Aluno", .9, vector(.96, 0, 0, .28))
        self.assertEqual(service.latest_forgotten, [first])
        self.assertEqual([round(ref["semelhanca_cadastro"], 3) for ref in service.learned["Aluno"]],
                         [.8, .96])
        # Mais longe que as duas (0,6): nao troca nada.
        service._maybe_learn("Aluno", .9, vector(.6, 0, 0, 0, .8))
        self.assertEqual(len(service.latest_forgotten), 1)
        self.assertEqual(len(service._learned_names), 2)

    def test_a_copy_only_competes_with_its_twin(self):
        service, vector = self.learning_service(per_person=2)
        service._maybe_learn("Aluno", .9, vector(.6, .8))  # 0,6 do cadastro
        service._maybe_learn("Aluno", .9, vector(.8, 0, .6))  # 0,8
        weakest, twin = (ref["id"] for ref in service.learned["Aluno"])
        service.latest_learned, service.latest_forgotten = [], []
        # Quase igual a de 0,8 e um pouco mais perto do cadastro: fica no lugar
        # dela, e a de 0,6, a menos parecida com o cadastro, continua.
        service._maybe_learn("Aluno", .9, vector(.82, 0, .57))
        self.assertEqual(service.latest_forgotten, [twin])
        self.assertEqual(service.learned["Aluno"][0]["id"], weakest)
        # A mesma foto de novo, ou quase igual e mais longe do cadastro, nao entra
        # nem com vaga sobrando: o video passado de novo nao enche as vagas.
        service, vector = self.learning_service()
        for values in ((.8, 0, .6), (.8, 0, .6), (.79, 0, .61)):
            service._maybe_learn("Aluno", .9, vector(*values))
        self.assertEqual(len(service.learned["Aluno"]), 1)

    def test_loading_drops_a_copy_of_a_reference_closer_to_the_registration(self):
        service, vector = self.learning_service(per_person=2)
        now = datetime.now()
        entries = [{"id": key, "nome": "Aluno", "semelhanca": .7, "embedding": vector(*values),
                    "aprendido_em": now} for key, values in (
                        ("perto", (.96, .28)), ("copia", (.95, .31)), ("meio", (.8, 0, .6)))]
        # A copia e mais parecida com o cadastro que a "meio", mas nao tira a vaga dela.
        self.assertEqual(service.replace_learned(entries), ["copia"])
        self.assertEqual([ref["id"] for ref in service.learned["Aluno"]], ["perto", "meio"])

    def test_detector_swap_keeps_the_recognizer(self):
        import numpy
        loaded = []

        def get_model(path, providers):
            loaded.append((Path(path).name, providers))
            return SimpleNamespace(taskname="recognition" if "w600k" in path else "detection")

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "det_500m.onnx").write_bytes(b"")
            (root / "w600k_mbf.onnx").write_bytes(b"")
            cls = load_class("App/FaceRecon/service.py", "FaceRecognitionService",
                             **FACE_SERVICE_DEPENDENCIES, np=numpy, Path=Path, PROJECT_ROOT=root,
                             insightface=SimpleNamespace(
                                 model_zoo=SimpleNamespace(get_model=get_model)))
            service = cls.__new__(cls)
            recognizer = object()
            service.app_insight = SimpleNamespace(
                models={"detection": "det_10g", "recognition": recognizer}, det_model="det_10g")
            # Caminho relativo vem da raiz do projeto; o reconhecedor fica.
            service._replace_detector("det_500m.onnx", None)
            self.assertEqual(loaded, [("det_500m.onnx", ["CPUExecutionProvider"])])
            self.assertIs(service.app_insight.det_model, service.app_insight.models["detection"])
            self.assertEqual(service.app_insight.det_model.taskname, "detection")
            self.assertIs(service.app_insight.models["recognition"], recognizer)
            # Outro reconhecedor no lugar do detector, ou arquivo que falta, param a subida.
            with self.assertRaisesRegex(ValueError, "nao e um detector"):
                service._replace_detector(str(root / "w600k_mbf.onnx"), None)
            with self.assertRaises(FileNotFoundError):
                service._replace_detector("falta.onnx", None)
        self.assertEqual(service.app_insight.det_model.taskname, "detection")

    def test_references_follow_the_registration_and_the_retention(self):
        service, vector = self.learning_service(per_person=2)
        now = datetime.now()
        entries = [{"id": key, "nome": name, "semelhanca": .7, "embedding": vector(*values),
                    "aprendido_em": now - timedelta(days=day)}
                   for key, name, values, day in (
                       ("perto", "Aluno", (.96, .28), 1), ("meio", "Aluno", (.8, 0, .6), 1),
                       ("sobra", "Aluno", (.6, 0, 0, .8), 1),
                       ("vencida", "Aluno", (1, 0, 0, 0, .1), 31),
                       ("longe", "Aluno", (0, 1), 1), ("saiu", "Saiu", (.96, .28), 1))]
        # Fora: quem saiu do cadastro, a que hoje nao entraria e a que passa do
        # limite. A vencida nem ocupa vaga; o banco a apaga pelo prazo.
        self.assertEqual(sorted(service.replace_learned(entries)), ["longe", "saiu", "sobra"])
        self.assertEqual([ref["id"] for ref in service.learned["Aluno"]], ["perto", "meio"])
        service.replace_known_faces(["Outro"], [vector(0, 1)])
        self.assertEqual((service.learned, service._learned_names), ({}, []))

    def test_only_a_fresh_embedding_can_teach(self):
        service, recognition, frame = self.reuse_frames()
        service.learn_from_stream = True
        service._maybe_learn = Mock()
        face = Vector([0, 0, 30, 30, .9])
        frame(100.0, face)
        frame(104.0, face)
        # O frame com o nome herdado pelo reuso nao passa pelo aprendizado.
        service._maybe_learn.assert_called_once_with("Aluno", .87, [.87])

    def reuse_frames(self, unknown_score=None):
        """Servico com reuso de 15 s e uma funcao que roda um frame no instante dado."""
        service, context, recognition, _ = self.make_service(True, [], reuse_seconds=15.0)
        if unknown_score is not None:
            service._match_face = lambda embedding: ("NAO ALUNO", unknown_score)
        keys = []

        def recognize(frame, face):
            face.normed_embedding = [.87]

        recognition.get.side_effect = recognize

        def frame(at, *boxes):
            kps = [object() for _ in boxes]
            keys.extend(kps)
            service.app_insight.det_model.detect.return_value = (list(boxes), kps)
            return service.recognize_faces(SimpleNamespace(**vars(context), observed_at=at))
        return service, recognition, frame

    def test_face_in_place_keeps_its_name_until_the_reuse_window_ends(self):
        service, recognition, frame = self.reuse_frames()
        face = Vector([0, 0, 30, 30, .9])
        first = frame(100.0, face)
        self.assertEqual(recognition.get.call_count, 1)
        # 4 s depois, no mesmo lugar: mesmo nome e semelhanca, sem embedding.
        self.assertEqual(frame(104.0, face), first)
        self.assertEqual(recognition.get.call_count, 1)
        self.assertEqual((service.latest_metrics["face_reused"],
                          service.latest_metrics["face_embeddings"]), (1.0, 0.0))
        # A validade conta do reconhecimento, nao do ultimo reuso: aos 15 s reconfere.
        frame(112.0, face)
        self.assertEqual(recognition.get.call_count, 1)
        frame(115.0, face)
        self.assertEqual(recognition.get.call_count, 2)

    def test_moved_or_new_face_is_recognized_and_the_others_keep_their_name(self):
        service, recognition, frame = self.reuse_frames()
        first, moved, new = (Vector([0, 0, 30, 30, .9]), Vector([40, 40, 70, 70, .9]),
                             Vector([100, 100, 130, 130, .9]))
        frame(100.0, first)
        frame(104.0, moved)
        self.assertEqual(recognition.get.call_count, 2)
        # Alguem entra: so o rosto novo gera embedding; quem ficou mantem o nome.
        self.assertEqual(len(frame(108.0, moved, new)), 2)
        self.assertEqual(recognition.get.call_count, 3)
        self.assertEqual(service.latest_metrics["face_reused"], 1.0)

    def test_unknown_face_is_reused_too(self):
        # Longe do cadastro, como um estranho: herda, sem embedding por frame.
        _, recognition, frame = self.reuse_frames(unknown_score=.1)
        face = Vector([0, 0, 30, 30, .9])
        self.assertEqual(frame(100.0, face)[0]["name"], "NAO ALUNO")
        self.assertEqual(frame(104.0, face)[0]["name"], "NAO ALUNO")
        self.assertEqual(recognition.get.call_count, 1)

    def test_near_miss_unknown_is_recognized_again_next_frame(self):
        # De 0,30 ate o limite costuma ser aluno de lado: reconhece de novo, e
        # o nome nao fica preso por 15 s quando ele volta a olhar para a camera.
        _, recognition, frame = self.reuse_frames(unknown_score=.45)
        face = Vector([0, 0, 30, 30, .9])
        frame(100.0, face)
        frame(104.0, face)
        self.assertEqual(recognition.get.call_count, 2)

    def async_frames(self):
        """Reconhecimento em segundo plano que so termina quando o teste libera."""
        service, context, recognition, _ = self.make_service(True, [], reuse_seconds=15.0)
        service.async_recognition = True
        service._recognition_pool = ThreadPoolExecutor(max_workers=1)
        gate = threading.Event()
        self.addCleanup(service.close)
        self.addCleanup(gate.set)

        def recognize(frame, face):
            gate.wait(5)
            face.normed_embedding = [.87]

        recognition.get.side_effect = recognize

        def frame(at, *boxes):
            service.app_insight.det_model.detect.return_value = (
                list(boxes), [object() for _ in boxes])
            return service.recognize_faces(SimpleNamespace(**vars(context), observed_at=at))

        def finish():
            gate.set()
            service._job["future"].result(timeout=5)
            gate.clear()

        return service, recognition, frame, finish

    def test_background_recognition_never_holds_the_frame(self):
        service, recognition, frame, finish = self.async_frames()
        service.learn_from_stream = True
        service._maybe_learn = Mock()
        face = Vector([0, 0, 30, 30, .9])
        self.assertEqual(frame(100.0, face)[0]["name"], "VERIFICANDO")
        self.assertEqual(service.latest_metrics["face_pending"], 1.0)
        # O mesmo rosto espera o mesmo reconhecimento, sem pedir outro.
        self.assertEqual(frame(101.0, face)[0]["name"], "VERIFICANDO")
        finish()
        result = frame(102.0, face)
        self.assertEqual((result[0]["name"], result[0]["confidence"]), ("Aluno", .87))
        self.assertEqual(service.latest_metrics["face_embeddings"], 1.0)
        # Aprende com o embedding novo, uma vez, quando o resultado chega.
        service._maybe_learn.assert_called_once_with("Aluno", .87, [.87])
        # Dai em diante herda, como no reuso.
        frame(104.0, face)
        self.assertEqual(recognition.get.call_count, 1)
        self.assertEqual(service.latest_metrics["face_reused"], 1.0)

    def test_one_background_recognition_at_a_time(self):
        service, _, frame, finish = self.async_frames()
        first, second = Vector([0, 0, 30, 30, .9]), Vector([100, 100, 130, 130, .9])
        frame(100.0, first)
        # Outro rosto chega com a thread ocupada: espera a vez, sem pedir.
        self.assertEqual([face["name"] for face in frame(101.0, first, second)],
                         ["VERIFICANDO", "VERIFICANDO"])
        self.assertEqual(service._job_count, 1)
        finish()
        self.assertEqual([face["name"] for face in frame(102.0, first, second)],
                         ["Aluno", "VERIFICANDO"])
        finish()
        self.assertEqual([face["name"] for face in frame(103.0, first, second)],
                         ["Aluno", "Aluno"])

    def test_known_face_keeps_its_name_while_it_is_recognized_again(self):
        service, _, frame, finish = self.async_frames()
        face = Vector([0, 0, 30, 30, .9])
        frame(100.0, face)
        finish()
        frame(101.0, face)
        # Passados os 15 s, reconhece de novo, e o nome fica enquanto isso.
        self.assertEqual(frame(116.0, face)[0]["name"], "Aluno")
        self.assertEqual(service._job_count, 2)

    def test_moved_face_shows_the_nearest_name_until_it_is_confirmed(self):
        service, _, frame, finish = self.async_frames()
        here, moved = Vector([0, 0, 30, 30, .9]), Vector([40, 0, 70, 30, .9])
        frame(100.0, here)
        finish()
        self.assertFalse(frame(101.0, here)[0]["pending"])
        # Andou mais que a sobreposicao, 80 px no frame original: o nome vem
        # junto so para a tela, e o rosto e reconhecido de novo onde esta.
        result = frame(102.0, moved)[0]
        self.assertEqual((result["name"], result["pending"]), ("Aluno", True))
        self.assertEqual(service._job_count, 2)
        finish()
        result = frame(103.0, moved)[0]
        self.assertEqual((result["name"], result["pending"]), ("Aluno", False))
        # Longe demais (320 px), pode ser outra pessoa: espera o proprio nome.
        self.assertEqual(frame(104.0, Vector([200, 0, 230, 30, .9]))[0]["name"], "VERIFICANDO")

    def test_moved_pending_face_keeps_its_background_job(self):
        service, _, frame, finish = self.async_frames()
        here, moved = Vector([0, 0, 30, 30, .9]), Vector([40, 0, 70, 30, .9])
        frame(100.0, here)
        job = service._job["id"]
        # O rosto anda antes do primeiro embedding terminar. O job continua
        # ligado a ele, mas o resultado levado pela distancia nao gera evento.
        result = frame(101.0, moved)[0]
        self.assertEqual((result["name"], result["pending"]), ("VERIFICANDO", True))
        self.assertEqual((service._job_count, service._identities[0]["job"]), (1, job))
        finish()
        result = frame(102.0, moved)[0]
        self.assertEqual((result["name"], result["confidence"], result["pending"]),
                         ("Aluno", .87, True))
        # Confirma de novo na posicao atual; so entao deixa de ser pendente.
        self.assertEqual(service._job_count, 2)
        finish()
        self.assertFalse(frame(103.0, moved)[0]["pending"])

    def test_moved_known_face_keeps_its_recheck_job(self):
        service, _, frame, finish = self.async_frames()
        here, moved = Vector([0, 0, 30, 30, .9]), Vector([40, 0, 70, 30, .9])
        frame(100.0, here)
        finish()
        frame(101.0, here)
        # A identidade vence, inicia a rechecagem e se move antes do resultado.
        frame(116.0, here)
        job = service._job["id"]
        result = frame(117.0, moved)[0]
        self.assertEqual((result["name"], result["pending"]), ("Aluno", True))
        self.assertEqual((service._job_count, service._identities[0]["job"]), (2, job))
        finish()
        self.assertTrue(frame(118.0, moved)[0]["pending"])
        self.assertEqual(service._job_count, 3)
        finish()
        self.assertFalse(frame(119.0, moved)[0]["pending"])

    def test_background_result_from_before_a_reset_is_dropped(self):
        service, _, frame, finish = self.async_frames()
        face = Vector([0, 0, 30, 30, .9])
        frame(100.0, face)
        service.reset_identities()
        finish()
        # O nome sairia da base anterior: o rosto pede um reconhecimento novo.
        self.assertEqual(frame(101.0, face)[0]["name"], "VERIFICANDO")
        self.assertEqual(service._job_count, 2)

    def test_registration_or_new_connection_forget_the_names(self):
        service, recognition, frame = self.reuse_frames()
        face = Vector([0, 0, 30, 30, .9])
        frame(100.0, face)
        service.replace_known_faces(["Outro"], [[.1]])
        frame(101.0, face)
        self.assertEqual(recognition.get.call_count, 2)
        service.reset_identities()
        frame(102.0, face)
        self.assertEqual(recognition.get.call_count, 3)

    def test_reuse_off_or_without_prefilter_recognizes_every_frame(self):
        face = Vector([0, 0, 30, 30, .9])
        for prefilter, reuse in ((True, 0.0), (False, 15.0)):
            service, context, recognition, _ = self.make_service(prefilter, [face],
                                                                 reuse_seconds=reuse)
            for _ in range(3):
                service.recognize_faces(context)
            calls = recognition.get.call_count
            self.assertEqual(calls, 3)
            self.assertEqual(service.latest_metrics["face_reused"], 0.0)

    def test_empty_scene_and_threshold_boundaries(self):
        for boxes, count in (([], 0), ([Vector([0, 0, 20, 20, .45])], 1)):
            service, frame, recognition, _ = self.make_service(True, boxes)
            self.assertEqual(len(service.recognize_faces(frame)), count)
            self.assertEqual(recognition.get.call_count, count)

    def test_full_modules_fallback_runs_addons_for_accepted_face(self):
        service, frame, _, addon = self.make_service(True, [Vector([0, 0, 30, 30, .9])], False)
        service.recognize_faces(frame)
        addon.get.assert_called_once()

    def test_constructor_preserves_registration_api_and_supports_rollback(self):
        for minimal in (True, False):
            service, frame, _, _ = self.make_service(True, [Vector([0, 0, 30, 30, .9])])
            app = service.app_insight
            app.prepare = Mock()
            factory = Mock(return_value=app)
            configure = Mock()
            namespace = type(service).__init__.__globals__
            namespace.update(os=os, np=SimpleNamespace(empty=lambda *a, **kw: [], float32=None),
                             insightface=SimpleNamespace(app=SimpleNamespace(FaceAnalysis=factory)),
                             configure_insight_threads=configure)
            with patch.object(type(service), "_load_database"):
                service.__init__(base_dir=".", lazy_person_model=True,
                                 minimal_modules=minimal, onnx_threads=2)
            self.assertEqual(factory.call_args.kwargs["allowed_modules"],
                             ["detection", "recognition"] if minimal else None)
            configure.assert_called_once_with(app, 2, allow_spinning=True)
            app.prepare.assert_called_once_with(ctx_id=0, det_size=(320, 320))
            # O cadastro continua usando get na imagem completa, sem o filtro do stream.
            self.assertEqual(len(service.app_insight.get(frame.processing_frame)), 1)


class ThreadConfigurationTests(unittest.TestCase):
    def test_parallel_failure_waits_for_other_model(self):
        fixture = fixtures.SharedPipelineTests()
        pipeline, faces, gestures = fixture.make_pipeline(True, True)
        started, release, completed = threading.Event(), threading.Event(), threading.Event()

        def gesture(*args):
            started.set()
            release.wait(2)
            completed.set()
            return []

        def face(*args):
            started.wait(2)
            raise RuntimeError("falha simulada")

        gestures.detect_gestures.side_effect = gesture
        faces.recognize_faces.side_effect = face
        try:
            with ThreadPoolExecutor(max_workers=1) as caller:
                future = caller.submit(pipeline.process_frame, None)
                try:
                    with self.assertRaises(TimeoutError):
                        future.result(timeout=.05)
                finally:
                    release.set()
                with self.assertRaises(RuntimeError):
                    future.result(timeout=2)
                self.assertTrue(completed.is_set())
        finally:
            fixture.doCleanups()

    def test_zero_does_not_require_native_libraries(self):
        configure_torch_threads(0)
        configure_insight_threads(None, 0)

    def test_onnx_options_are_applied_to_real_session_constructor(self):
        old = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"],
                              get_provider_options=lambda: {"CPUExecutionProvider": {}})
        model = SimpleNamespace(session=old, model_file="same.onnx")
        ort = SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=Mock())
        with patch.dict(sys.modules, {"onnxruntime": ort}):
            configure_insight_threads(SimpleNamespace(models={"recognition": model}), 2)
        args, kwargs = ort.InferenceSession.call_args
        self.assertEqual(args, ("same.onnx",))
        self.assertEqual(kwargs["sess_options"].intra_op_num_threads, 2)
        self.assertEqual(kwargs["providers"], ["CPUExecutionProvider"])
        self.assertEqual(kwargs["provider_options"], [{}])
        self.assertIs(model.session, ort.InferenceSession.return_value)

    def test_spinning_is_turned_off_only_when_asked(self):
        class Options:
            def __init__(self):
                self.entries = {}

            def add_session_config_entry(self, key, value):
                self.entries[key] = value

        for allow, expected in ((True, {}), (False, {"session.intra_op.allow_spinning": "0"})):
            old = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"],
                                  get_provider_options=lambda: {})
            model = SimpleNamespace(session=old, model_file="same.onnx")
            ort = SimpleNamespace(SessionOptions=Options, InferenceSession=Mock())
            with patch.dict(sys.modules, {"onnxruntime": ort}):
                configure_insight_threads(SimpleNamespace(models={"recognition": model}), 2,
                                          allow_spinning=allow)
            self.assertEqual(ort.InferenceSession.call_args.kwargs["sess_options"].entries,
                             expected)


if __name__ == "__main__":
    unittest.main()
