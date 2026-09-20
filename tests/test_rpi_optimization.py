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
                                   latest_metrics={"pose_ms": 5, "hands_ms": 0, "gestures_ms": 5})
        pipeline = cls(face_service=faces, gesture_service=gestures,
                       run_in_parallel=parallel, shared_person_pose=shared)
        self.addCleanup(pipeline.close)
        return pipeline, faces, gestures

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
                         GESTURE_ANALYZER_FPS=12)
        service = cls.__new__(cls)
        service.pose_model = SimpleNamespace(track=Mock(return_value=results))
        service.tracker = "bytetrack.yaml"
        service.analyzer = Mock()
        service.last_track_centers = {1: (10, 20)}
        service._to_numpy = lambda value: value
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

    def test_legacy_empty_gate_skips_model_and_cleans_history(self):
        service, context = self.make_service([])
        service.detect_gestures(context, person_bboxes=[])
        service.pose_model.track.assert_not_called()
        service.analyzer.clean_old_tracks.assert_called_once_with([])
        self.assertEqual(service.latest_metrics["pose_ms"], 0)


if __name__ == "__main__":
    unittest.main()
