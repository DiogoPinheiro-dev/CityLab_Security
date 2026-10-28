"""Passo 1 do TCC de PDI: instante do frame e ponto do PDI na cadeia."""
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, wait
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from App.frame_context import build_frame_context
from test_rpi_optimization import load_class


class Frame(SimpleNamespace):
    """Frame falso: so o formato e o tipo, que a cadeia confere depois do PDI."""


def video_frame():
    return Frame(shape=(480, 640, 3), dtype="uint8")


class PdiPipelineTests(unittest.TestCase):
    def make_pipeline(self, parallel, preprocessor=None):
        self.built = []

        def build_frame_context(frame, **options):
            self.built.append((frame, options))
            return SimpleNamespace(processing_frame=frame, original_frame=frame,
                                   observed_at=options.get("observed_at"))

        cls = load_class("App/recognition_pipeline.py", "UnifiedRecognitionService",
                         PIPELINE_RUN_IN_PARALLEL=True, PIPELINE_SHARED_PERSON_POSE=False,
                         PIPELINE_MAX_WORKERS=2, PROCESS_SCALE=.5, EXPERIMENTAL_GRAYSCALE=False,
                         DEBUG_PIPELINE=False, ENABLE_PERFORMANCE_METRICS=True,
                         TORCH_NUM_THREADS=0, configure_torch_threads=lambda count: None,
                         ensure_torch_threads=lambda count: count, threading=threading,
                         OPENCV_NUM_THREADS=0, configure_opencv_threads=lambda count: None,
                         ThreadPoolExecutor=ThreadPoolExecutor, wait=wait,
                         CITYLAB_ALLOW_PARTIAL_PIPELINE=False, CITYLAB_ENABLE_FACE_SERVICE=True,
                         CITYLAB_ENABLE_GESTURE_SERVICE=True,
                         build_frame_context=build_frame_context)
        faces = SimpleNamespace(detect_persons=Mock(return_value=[]),
                                recognize_faces=Mock(return_value=[]), latest_metrics={})
        gestures = SimpleNamespace(detect_gestures=Mock(return_value=[]), latest_persons=[],
                                   note_external_presence=Mock(), latest_metrics={})
        pipeline = cls(face_service=faces, gesture_service=gestures, run_in_parallel=parallel,
                       shared_person_pose=True, preprocessor=preprocessor)
        self.addCleanup(pipeline.close)
        return pipeline, faces, gestures

    def test_without_pdi_nor_instant_the_frame_goes_as_before(self):
        for parallel in (False, True):
            pipeline, _, _ = self.make_pipeline(parallel)
            frame = video_frame()
            metrics = pipeline.process_frame(frame)["metrics"]
            self.assertIs(self.built[0][0], frame)
            self.assertIsNone(self.built[0][1]["observed_at"])
            self.assertNotIn("pdi_ms", metrics)

    def test_the_video_instant_reaches_face_and_gesture(self):
        for parallel in (False, True):
            # Zero tambem: o primeiro frame de um video comeca em 0 s.
            for instant in (0.0, 3.4):
                pipeline, faces, gestures = self.make_pipeline(parallel)
                pipeline.process_frame(video_frame(), observed_at=instant)
                self.assertEqual(faces.recognize_faces.call_args.args[0].observed_at, instant)
                self.assertEqual(gestures.detect_gestures.call_args.args[0].observed_at, instant)

    def test_pdi_runs_once_before_the_reduction_and_is_timed(self):
        for parallel in (False, True):
            received, treated = video_frame(), video_frame()
            preprocessor = Mock(return_value=treated)
            pipeline, _, _ = self.make_pipeline(parallel, preprocessor)
            metrics = pipeline.process_frame(received)["metrics"]
            preprocessor.assert_called_once_with(received)
            # A reducao, e com ela todos os modelos, recebe o frame tratado.
            self.assertIs(self.built[0][0], treated)
            self.assertGreaterEqual(metrics["pdi_ms"], 0.0)
            self.assertLessEqual(metrics["pdi_ms"], metrics["total_ms"])

    def test_pdi_must_keep_the_frame_format(self):
        for treated in (Frame(shape=(240, 320, 3), dtype="uint8"),
                        Frame(shape=(480, 640, 3), dtype="float32")):
            pipeline, faces, _ = self.make_pipeline(False, Mock(return_value=treated))
            with self.assertRaisesRegex(ValueError, "mesmo formato e tipo"):
                pipeline.process_frame(video_frame())
            faces.recognize_faces.assert_not_called()


class FrameInstantTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)

    def test_informed_instant_is_kept_even_at_zero(self):
        for instant in (0.0, 3.4):
            context = build_frame_context(self.frame, process_scale=.5, observed_at=instant)
            self.assertEqual(context.observed_at, instant)
            self.assertEqual(context.processing_frame.shape, (24, 32, 3))

    def test_without_instant_the_clock_is_used(self):
        before = time.monotonic()
        context = build_frame_context(self.frame, process_scale=.5)
        self.assertTrue(before <= context.observed_at <= time.monotonic())


if __name__ == "__main__":
    unittest.main()
