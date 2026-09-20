"""Valida o perfil sem importar bibliotecas de inferencia."""
import os
import runpy
import sys
import unittest
import weakref
import io
import contextlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

from App.inference_runtime import configure_insight_threads, configure_opencv_threads, prepare_native_environment
from tools.run_rpi import main as run_rpi
from tools.benchmark_stream import detection_confidences


class RpiProfileTests(unittest.TestCase):
    def settings(self, **environment):
        with patch.dict(os.environ, environment, clear=True), patch.dict(
            sys.modules, {"dotenv": SimpleNamespace(load_dotenv=lambda *a, **kw: None)}
        ):
            return runpy.run_path("App/settings.py")

    def test_pi_defaults_preserve_image_quality_and_detectors(self):
        normal = self.settings()
        pi = self.settings(CITYLAB_PROFILE="rpi3")
        self.assertEqual([pi[name] for name in ("ONNX_INTRA_OP_THREADS", "TORCH_NUM_THREADS",
                                              "OPENCV_NUM_THREADS", "MAX_IN_FLIGHT_FRAMES")],
                         [1, 2, 1, 1])
        for name in ("PROCESS_SCALE", "STREAM_WIDTH", "STREAM_HEIGHT", "JPEG_QUALITY",
                     "FACE_MIN_CONFIDENCE", "FACE_MIN_WIDTH", "FACE_MIN_HEIGHT",
                     "PIPELINE_SHARED_PERSON_POSE", "POSE_MODEL_PATH"):
            self.assertEqual(pi[name], normal[name])

    def test_explicit_overrides_and_typo(self):
        values = self.settings(CITYLAB_PROFILE="rpi3", ONNX_INTRA_OP_THREADS="0",
                               TORCH_NUM_THREADS="1", MAX_IN_FLIGHT_FRAMES="2")
        self.assertEqual(values["ONNX_INTRA_OP_THREADS"], 0)
        self.assertEqual(values["TORCH_NUM_THREADS"], 1)
        self.assertEqual(values["MAX_IN_FLIGHT_FRAMES"], 2)
        with self.assertRaises(ValueError):
            self.settings(CITYLAB_PROFILE="rp13")

    def test_opencv_zero_and_configured_limit(self):
        cv2 = SimpleNamespace(setNumThreads=Mock(), getNumThreads=Mock(return_value=1))
        with patch.dict(sys.modules, {"cv2": cv2}):
            configure_opencv_threads(0)
            cv2.setNumThreads.assert_not_called()
            configure_opencv_threads(1)
            cv2.setNumThreads.assert_called_once_with(1)

    def test_old_onnx_session_released_before_replacement(self):
        class Session:
            def get_providers(self):
                return ["CPUExecutionProvider"]

            def get_provider_options(self):
                return {}

        model = SimpleNamespace(session=Session(), model_file="same.onnx")
        previous = weakref.ref(model.session)

        def create(*args, **kwargs):
            self.assertIsNone(previous())
            self.assertFalse(hasattr(model, "session"))
            return Session()

        ort = SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=create)
        with patch.dict(sys.modules, {"onnxruntime": ort}):
            configure_insight_threads(SimpleNamespace(models={"recognition": model}), 1)
        self.assertIsInstance(model.session, Session)

    def test_native_limits_preserve_explicit_user_overrides(self):
        with patch.dict(os.environ, {"OPENBLAS_NUM_THREADS": "2"}, clear=True):
            values = prepare_native_environment(1)
            self.assertEqual(values["OPENBLAS_NUM_THREADS"], "2")
            self.assertEqual(values["OMP_NUM_THREADS"], "1")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(prepare_native_environment(0), {})

    def test_launcher_configures_libraries_before_single_worker_start(self):
        settings = SimpleNamespace(**self.settings(CITYLAB_PROFILE="rpi3"))

        def start(*args, **kwargs):
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(kwargs["workers"], 1)
            self.assertFalse(kwargs["reload"])
            self.assertEqual(args, ("Server.main:app",))

        uvicorn = SimpleNamespace(run=Mock(side_effect=start))
        import App
        with patch.object(App, "settings", settings, create=True), patch.dict(
            sys.modules, {"App.settings": settings, "uvicorn": uvicorn,
                          "dotenv": SimpleNamespace(load_dotenv=lambda *a, **kw: None)}
        ), patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            run_rpi(["--show-config"])
            uvicorn.run.assert_not_called()
            run_rpi(["--host", "127.0.0.1"])
            uvicorn.run.assert_called_once()

    def test_benchmark_records_confidence_without_identity(self):
        data = detection_confidences({"pessoas": [{"confidence": .8, "nome": "privado"},
                                    {"confidence": None}, {"confidence": float("nan")}],
                                     "gestos": [{"confidence": .7}]})
        self.assertEqual(data, {"persons_confidence": [.8], "gestures_confidence": [.7]})


if __name__ == "__main__":
    unittest.main()
