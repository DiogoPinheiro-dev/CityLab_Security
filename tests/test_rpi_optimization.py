"""Contratos sem modelos, OpenCV, MongoDB ou Raspberry Pi."""
import ast
import asyncio
import logging
import unittest
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from App.GestureRecon.detector import GestureAnalyzer


def load_class(path, name, **dependencies):
    # Isola dependencias nativas, executando a classe real por inteiro.
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, path, "exec"), dependencies)
    return dependencies[name]


class EventLoggerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock = [0.0]
        cls = load_class("Server/event_logger.py", "EventLogger",
                         COOLDOWN_ALUNO_SECONDS=5, COOLDOWN_NAO_ALUNO_SECONDS=5,
                         COOLDOWN_ALERTA_GESTO_SECONDS=5, datetime=datetime,
                         time=SimpleNamespace(monotonic=lambda: self.clock[0]),
                         logger=logging.getLogger("event-test"))
        self.collection = SimpleNamespace(insert_one=AsyncMock())
        self.logger = cls(self.collection)
        self.logger._crop_to_base64 = Mock(return_value=None)
        self.face = {"name": "Teste", "bbox": [0, 0, 10, 10]}

    async def test_failure_retries_without_consuming_cooldown(self):
        self.collection.insert_one.side_effect = [RuntimeError("offline"), None, None]
        with self.assertLogs("event-test", level="ERROR"):
            await self.logger.log_face_events(None, [self.face])
        self.assertEqual(self.logger.last_logged, {})
        await self.logger.log_face_events(None, [self.face])
        await self.logger.log_face_events(None, [self.face])
        self.assertEqual(self.collection.insert_one.await_count, 2)
        self.clock[0] = 6
        await self.logger.log_face_events(None, [self.face])
        self.assertEqual(self.collection.insert_one.await_count, 3)
        self.assertEqual(self.logger.last_logged["ALUNO:Teste"], 6)

    async def test_failed_gesture_does_not_block_following_event(self):
        self.collection.insert_one.side_effect = [RuntimeError("offline"), None]
        gestures = [{"track_id": i, "bbox": [0, 0, 10, 10], "alerts": ["Rendicao"]}
                    for i in (1, 2)]
        with self.assertLogs("event-test", level="ERROR"):
            await self.logger.log_gesture_events(None, gestures)
        self.assertEqual(self.collection.insert_one.await_count, 2)
        self.assertEqual(list(self.logger.last_logged), ["ALERTA_GESTO:2:Rendicao"])
        self.clock[0] = 6
        self.logger._should_log("ALUNO", "outro")
        self.assertEqual(self.logger.last_logged, {})

    async def test_cancel_is_not_swallowed(self):
        self.collection.insert_one.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.logger.log_face_events(None, [self.face])


class GestureTimeTests(unittest.TestCase):
    def run_sequence(self, timestamps, active=True):
        analyzer = GestureAnalyzer()
        keypoints = [[0, 0, 0] for _ in range(17)]
        alerts = []
        for now in timestamps:
            alerts = analyzer.analyze(1, keypoints, hand_context={"left_closed": active},
                                      observed_at=now)["alerts"]
        return analyzer, alerts

    def test_same_duration_at_different_rates(self):
        for timestamps in ([0, .05, .10, .15, .21], [0, .21]):
            analyzer, alerts = self.run_sequence(timestamps)
            self.assertIn("Mao Fechada", alerts)
            self.assertAlmostEqual(analyzer.history[1]["fist_frames"], .20)
        self.assertNotIn("Mao Fechada", self.run_sequence([0, .19])[1])

    def test_new_gesture_does_not_inherit_idle_interval(self):
        analyzer, _ = self.run_sequence([0], active=False)
        points = [[0, 0, 0] for _ in range(17)]
        result = analyzer.analyze(1, points, hand_context={"left_closed": True}, observed_at=17)
        self.assertNotIn("Mao Fechada", result["alerts"])
        result = analyzer.analyze(1, points, hand_context={"left_closed": True}, observed_at=34)
        self.assertIn("Mao Fechada", result["alerts"])

    def test_missing_track_resets_time_and_state(self):
        analyzer, _ = self.run_sequence([0, .21])
        analyzer.clean_old_tracks([])
        self.assertFalse(analyzer.history)
        self.assertFalse(analyzer.last_observed)
        self.assertFalse(analyzer.active_states)
        self.assertFalse(analyzer.elapsed)


class SharedPipelineTests(unittest.TestCase):
    def make_pipeline(self, parallel, shared, persons=None):
        cls = load_class("App/recognition_pipeline.py", "UnifiedRecognitionService",
                         PIPELINE_RUN_IN_PARALLEL=True, PIPELINE_SHARED_PERSON_POSE=False,
                         PIPELINE_MAX_WORKERS=2, PROCESS_SCALE=.5, EXPERIMENTAL_GRAYSCALE=False,
                         DEBUG_PIPELINE=False, ENABLE_PERFORMANCE_METRICS=True,
                         TORCH_NUM_THREADS=0, configure_torch_threads=lambda count: None,
                         OPENCV_NUM_THREADS=0, configure_opencv_threads=lambda count: None,
                         ThreadPoolExecutor=ThreadPoolExecutor, wait=wait, _env_bool=lambda name, default: default,
                         build_frame_context=lambda *args, **kwargs: object())
        faces = SimpleNamespace(detect_persons=Mock(return_value=[{"bbox": [1, 2, 3, 4]}]),
                                recognize_faces=Mock(return_value=[{"name": "Teste"}]),
                                latest_metrics={"faces_ms": 2, "persons_ms": 3})
        gestures = SimpleNamespace(detect_gestures=Mock(return_value=[]),
                                   latest_persons=persons or [],
                                   note_external_presence=Mock(),
                                   latest_metrics={"pose_ms": 5, "hands_ms": 0, "gestures_ms": 5})
        pipeline = cls(face_service=faces, gesture_service=gestures,
                       run_in_parallel=parallel, shared_person_pose=shared)
        self.addCleanup(pipeline.close)
        return pipeline, faces, gestures

    def test_pipeline_reports_face_presence_to_the_gesture_gate(self):
        for parallel in (False, True):
            pipeline, face, gesture = self.make_pipeline(parallel, True)
            pipeline.process_frame(None)
            gesture.note_external_presence.assert_called_once_with(True)
            # Sem rosto o sinal vai como falso e o gate volta a poder pular.
            face.recognize_faces.return_value = []
            gesture.note_external_presence.reset_mock()
            pipeline.process_frame(None)
            gesture.note_external_presence.assert_called_once_with(False)

    def test_single_pass_in_both_execution_modes(self):
        for parallel in (False, True):
            persons = [{"bbox": [2, 4, 6, 8], "confidence": .82}]
            pipeline, face, gesture = self.make_pipeline(parallel, True, persons)
            result = pipeline.process_frame(None)
            face.detect_persons.assert_not_called()
            face.recognize_faces.assert_called_once()
            gesture.detect_gestures.assert_called_once()
            self.assertEqual(result["persons"], persons)
            self.assertEqual(result["metrics"]["persons_ms"], 0)
            self.assertEqual(result["metrics"]["pose_ms"], 5)

    def test_empty_scene_still_runs_pose_and_face(self):
        pipeline, face, gesture = self.make_pipeline(False, True)
        self.assertEqual(pipeline.process_frame(None)["persons"], [])
        gesture.detect_gestures.assert_called_once()
        face.recognize_faces.assert_called_once()

    def test_fallback_when_gestures_disabled_or_unavailable(self):
        for unavailable in (False, True):
            pipeline, face, gesture = self.make_pipeline(False, True)
            if unavailable:
                pipeline.gesture_service = None
            pipeline.process_frame(None, detect_gestures=unavailable)
            face.detect_persons.assert_called_once()
            gesture.detect_gestures.assert_not_called()

    def test_person_output_flag_does_not_disable_gestures(self):
        pipeline, face, gesture = self.make_pipeline(False, True, [{"bbox": [1, 2, 3, 4]}])
        self.assertEqual(pipeline.process_frame(None, detect_persons=False)["persons"], [])
        gesture.detect_gestures.assert_called_once()

    def test_legacy_detector_remains_available(self):
        pipeline, face, gesture = self.make_pipeline(False, False)
        result = pipeline.process_frame(None)
        face.detect_persons.assert_called_once()
        self.assertEqual(result["persons"], [{"bbox": [1, 2, 3, 4]}])


class PoseOutputTests(unittest.TestCase):
    def make_service(self, results):
        cls = load_class("App/GestureRecon/service.py", "GestureRecognitionService",
                         GESTURE_ANALYZER_FPS=12, GESTURE_PUBLISH_MIN_CONFIDENCE=.25,
                         GESTURE_MOTION_GATE=False, GESTURE_MOTION_MIN_RATIO=.002,
                         GESTURE_MOTION_PIXEL_DELTA=25, GESTURE_MOTION_MAX_SKIP_SECONDS=30.)
        service = cls.__new__(cls)
        service.pose_model = SimpleNamespace(track=Mock(return_value=results))
        service.tracker = "bytetrack.yaml"
        service.analyzer = Mock()
        service.last_track_centers = {1: (10, 20)}
        service._to_numpy = lambda value: value
        # __new__ pula o __init__: o estado novo precisa ser posto a mao.
        service.publish_min_confidence = .25
        service.motion_gate = False
        service.motion_min_ratio = .002
        service.motion_pixel_delta = 25
        service.motion_max_skip_seconds = 30.
        service.motion_reference = None
        service.last_pose_at = None
        service.scene_occupied = False
        service.latest_metrics = {}
        context = SimpleNamespace(processing_frame=object(), observed_at=0.0,
                                  map_bbox_to_original=lambda box: [v * 2 for v in box],
                                  clip_original_bbox=lambda box: box)
        return service, context

    def test_boxes_preserved_even_without_keypoints(self):
        result = SimpleNamespace(boxes=SimpleNamespace(xyxy=[[1, 2, 3, 4]], conf=[.8]),
                                 keypoints=None)
        service, context = self.make_service([result])
        self.assertEqual(service.detect_gestures(context), [])
        self.assertEqual(service.latest_persons, [{"bbox": [2, 4, 6, 8], "confidence": .8}])
        service.pose_model.track.assert_called_once()

    def test_empty_result_removes_previous_boxes(self):
        service, context = self.make_service([])
        service.latest_persons = [{"bbox": [1, 2, 3, 4]}]
        service.detect_gestures(context)
        self.assertEqual(service.latest_persons, [])
        self.assertFalse(service.last_track_centers)

    def analisavel(self, service):
        """Deixa o laco de people rodar sem modelo de maos nem analisador real."""
        service._resolve_track_ids = lambda boxes, ids: list(range(1, len(boxes) + 1))
        service._detect_hands_in_body_roi = Mock(return_value=[])
        service._associate_hands = Mock(return_value={"matched_hands": []})
        service.analyzer.analyze = Mock(return_value={
            "alerts": [], "hand_context": {"matched_hands": []}, "hidden_debug": {}})

    def test_weak_boxes_feed_the_tracker_without_being_published(self):
        result = SimpleNamespace(
            boxes=SimpleNamespace(xyxy=[[1, 2, 3, 4], [5, 6, 7, 8]], conf=[.8, .1], id=None),
            keypoints=SimpleNamespace(data=[[[0, 0, 0]], [[0, 0, 0]]]))
        service, context = self.make_service([result])
        self.analisavel(service)
        people = service.detect_gestures(context)
        # A caixa de 0,1 existe so para o ByteTrack: nao vira pessoa nem gesto.
        self.assertEqual([person["confidence"] for person in people], [.8])
        self.assertEqual([person["confidence"] for person in service.latest_persons], [.8])
        service.analyzer.analyze.assert_called_once()
        # O track segue conhecido, senao quem oscila abaixo do limiar perde historico.
        service.analyzer.clean_old_tracks.assert_called_once_with([1, 2])

    def test_motion_gate_skips_pose_while_the_scene_is_still(self):
        service, context = self.make_service([])
        service.motion_gate = True
        service.last_pose_at = 0.
        service._motion_ratio = lambda frame: 0.
        context.observed_at = 10.
        self.assertEqual(service.detect_gestures(context), [])
        service.pose_model.track.assert_not_called()
        service.analyzer.clean_old_tracks.assert_called_once_with([])
        self.assertEqual(service.latest_metrics["pose_skipped"], 1.)

    def test_motion_gate_runs_pose_on_movement_and_after_the_time_ceiling(self):
        service, context = self.make_service([])
        service.motion_gate = True
        service.last_pose_at = 0.
        service._motion_ratio = lambda frame: .5
        context.observed_at = 1.
        service.detect_gestures(context)
        self.assertEqual(service.pose_model.track.call_count, 1)
        # Cena parada, mas o teto de tempo estourou: roda para nao perder quem
        # entrou em cena e ficou imovel.
        service._motion_ratio = lambda frame: 0.
        context.observed_at = 100.
        service.detect_gestures(context)
        self.assertEqual(service.pose_model.track.call_count, 2)
        self.assertEqual(service.latest_metrics["pose_skipped"], 0.)

    def test_motion_gate_never_skips_while_the_scene_is_occupied(self):
        """Pessoa parada quase nao gera movimento: pular a cegaria para gesto."""
        service, context = self.make_service([])
        service.motion_gate = True
        service.last_pose_at = 0.
        service._motion_ratio = lambda frame: 0.
        context.observed_at = 10.
        service.scene_occupied = True
        service.detect_gestures(context)
        service.pose_model.track.assert_called_once()
        self.assertEqual(service.latest_metrics["pose_skipped"], 0.)

    def test_face_presence_protects_the_next_frame_from_the_gate(self):
        service, context = self.make_service([])
        service.motion_gate = True
        service.last_pose_at = 0.
        service._motion_ratio = lambda frame: 0.
        context.observed_at = 10.
        # Sem sinal de rosto a cena parada e pulada...
        service.detect_gestures(context)
        self.assertEqual(service.latest_metrics["pose_skipped"], 1.)
        # ...mas um rosto visto pelo estagio paralelo protege o frame seguinte.
        service.note_external_presence(True)
        service.detect_gestures(context)
        self.assertEqual(service.latest_metrics["pose_skipped"], 0.)
        service.pose_model.track.assert_called_once()

    def test_pose_without_people_releases_the_gate_again(self):
        service, context = self.make_service([])
        service.motion_gate = True
        service.scene_occupied = True
        service.last_pose_at = 0.
        service._motion_ratio = lambda frame: 0.
        context.observed_at = 10.
        # A passada de pose e a leitura confiavel: sem ninguem, o gate volta.
        service.detect_gestures(context)
        self.assertFalse(service.scene_occupied)
        service.detect_gestures(context)
        self.assertEqual(service.latest_metrics["pose_skipped"], 1.)

    def test_motion_gate_disabled_never_skips(self):
        service, context = self.make_service([])
        service._motion_ratio = Mock(return_value=0.)
        service.last_pose_at = 0.
        context.observed_at = 1.
        service.detect_gestures(context)
        service.pose_model.track.assert_called_once()
        service._motion_ratio.assert_not_called()

    def test_legacy_empty_gate_skips_model_and_cleans_history(self):
        service, context = self.make_service([])
        service.detect_gestures(context, person_bboxes=[])
        service.pose_model.track.assert_not_called()
        service.analyzer.clean_old_tracks.assert_called_once_with([])
        self.assertEqual(service.latest_metrics["pose_ms"], 0)


if __name__ == "__main__":
    unittest.main()
