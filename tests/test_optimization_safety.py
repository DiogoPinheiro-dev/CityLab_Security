"""Regressoes de alertas, concorrencia e equivalencia do filtro facial."""
import asyncio
import os
import sys
import unittest
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
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


class Vector(list):
    def __getitem__(self, key):
        result = super().__getitem__(key)
        return Vector(result) if isinstance(key, slice) else result

    def astype(self, dtype):
        return Vector(dtype(value) for value in self)

    def tolist(self):
        return list(self)


class FaceOptimizationTests(unittest.TestCase):
    def make_service(self, prefilter, boxes, minimal=True):
        cls = load_class("App/FaceRecon/service.py", "FaceRecognitionService",
                         DEBUG_PIPELINE=True, FACE_MIN_WIDTH=40, FACE_MIN_HEIGHT=40,
                         FACE_MIN_CONFIDENCE=.45, FACE_MINIMAL_MODULES=True,
                         FACE_PREFILTER=True, ONNX_INTRA_OP_THREADS=0, Face=SimpleNamespace)
        service = cls.__new__(cls)
        service.prefilter = prefilter
        service.debug_pipeline = True
        service.face_min_width = service.face_min_height = 40
        service.face_min_confidence = .45
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
            configure.assert_called_once_with(app, 2)
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


if __name__ == "__main__":
    unittest.main()
