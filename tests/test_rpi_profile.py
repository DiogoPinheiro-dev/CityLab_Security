"""Valida o perfil sem importar bibliotecas de inferencia."""
import json
import os
import runpy
import sys
import tempfile
import unittest
import weakref
import io
import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from App.inference_runtime import (configure_insight_threads, configure_opencv_threads,
                                   ensure_torch_threads, prepare_native_environment)
from tools.run_rpi import main as run_rpi
from tools.benchmark_stream import detection_confidences


class RpiProfileTests(unittest.TestCase):
    def settings(self, **environment):
        with patch.dict(os.environ, environment, clear=True), patch.dict(
            sys.modules, {"dotenv": SimpleNamespace(load_dotenv=lambda *a, **kw: None)}
        ):
            return runpy.run_path("App/settings.py")

    def test_pi_defaults_preserve_image_quality(self):
        normal = self.settings()
        pi = self.settings(CITYLAB_PROFILE="rpi3")
        self.assertEqual([pi[name] for name in ("ONNX_INTRA_OP_THREADS", "TORCH_NUM_THREADS",
                                              "OPENCV_NUM_THREADS", "MAX_IN_FLIGHT_FRAMES")],
                         [1, 3, 1, 1])
        for name in ("PROCESS_SCALE", "STREAM_WIDTH", "STREAM_HEIGHT", "JPEG_QUALITY",
                     "FACE_MIN_CONFIDENCE", "FACE_MIN_WIDTH", "FACE_MIN_HEIGHT",
                     "GESTURE_PUBLISH_MIN_CONFIDENCE", "POSE_MODEL_PATH"):
            self.assertEqual(pi[name], normal[name])

    def test_pi_profile_enables_the_measured_pose_path(self):
        names = ("PIPELINE_SHARED_PERSON_POSE", "GESTURE_MOTION_GATE")
        # Medido nos tres cenarios so no rpi3; o perfil default segue opt-in.
        self.assertEqual([self.settings()[name] for name in names], [False, False])
        self.assertEqual([self.settings(CITYLAB_PROFILE="rpi3")[name] for name in names],
                         [True, True])
        # Valor explicito prevalece, para comparar cada ajuste no Pi.
        off = self.settings(CITYLAB_PROFILE="rpi3", PIPELINE_SHARED_PERSON_POSE="0",
                            GESTURE_MOTION_GATE="0")
        self.assertEqual([off[name] for name in names], [False, False])

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

    def test_torch_threads_are_reapplied_only_when_they_drift(self):
        state = {"threads": 3}
        torch = SimpleNamespace(get_num_threads=lambda: state["threads"],
                                set_num_threads=Mock(side_effect=lambda n: state.update(threads=n)))
        with patch.dict(sys.modules, {"torch": torch}):
            # Thread que herdou o limite do Ultralytics volta ao configurado.
            self.assertEqual(ensure_torch_threads(2), 2)
            torch.set_num_threads.assert_called_once_with(2)
            # Sem desvio nao chama de novo: set_num_threads limpa o cache do oneDNN.
            self.assertEqual(ensure_torch_threads(2), 2)
            # Zero preserva a escolha da biblioteca.
            state["threads"] = 8
            self.assertEqual(ensure_torch_threads(0), 8)
            torch.set_num_threads.assert_called_once()

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

    def test_show_config_reports_the_pose_path_without_credentials(self):
        settings = SimpleNamespace(**self.settings(CITYLAB_PROFILE="rpi3"))
        output = io.StringIO()
        import App
        with patch.object(App, "settings", settings, create=True), patch.dict(
            sys.modules, {"App.settings": settings,
                          "dotenv": SimpleNamespace(load_dotenv=lambda *a, **kw: None)}
        ), patch.dict(os.environ, {"MONGO_DETAILS": "mongodb://usuario:segredo@host"},
                      clear=True), contextlib.redirect_stdout(output):
            run_rpi(["--show-config"])
        # O gate decide o ganho da cena vazia e nao aparecia nesta saida.
        configured = json.loads(output.getvalue())["configured"]
        self.assertIs(configured["PIPELINE_SHARED_PERSON_POSE"], True)
        self.assertIs(configured["GESTURE_MOTION_GATE"], True)
        self.assertEqual(configured["GESTURE_PUBLISH_MIN_CONFIDENCE"], 0.25)
        self.assertNotIn("segredo", output.getvalue())

    def test_launcher_validates_and_forwards_the_tls_pair(self):
        settings = SimpleNamespace(**self.settings(CITYLAB_PROFILE="rpi3"))
        uvicorn = SimpleNamespace(run=Mock())
        import App
        with patch.object(App, "settings", settings, create=True), patch.dict(
            sys.modules, {"App.settings": settings, "uvicorn": uvicorn,
                          "dotenv": SimpleNamespace(load_dotenv=lambda *a, **kw: None)}
        ), patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            with tempfile.TemporaryDirectory() as pasta:
                cert = Path(pasta) / "cert.pem"
                key = Path(pasta) / "key.pem"
                cert.write_text("cert", encoding="utf-8")
                key.write_text("key", encoding="utf-8")
                # Par incompleto ou caminho ausente falha antes de carregar modelos.
                with contextlib.redirect_stderr(io.StringIO()):
                    for argv in (["--ssl-certfile", str(cert)],
                                 ["--ssl-keyfile", str(key)],
                                 ["--ssl-certfile", str(Path(pasta) / "ausente.pem"),
                                  "--ssl-keyfile", str(key)]):
                        with self.assertRaises(SystemExit):
                            run_rpi(argv)
                uvicorn.run.assert_not_called()
                run_rpi(["--ssl-certfile", str(cert), "--ssl-keyfile", str(key)])
                forwarded = uvicorn.run.call_args.kwargs
                self.assertEqual(forwarded["ssl_certfile"], str(cert))
                self.assertEqual(forwarded["ssl_keyfile"], str(key))
                # Sem TLS o launcher segue valido, com o par vazio.
                run_rpi([])
                self.assertIsNone(uvicorn.run.call_args.kwargs["ssl_certfile"])
                self.assertIsNone(uvicorn.run.call_args.kwargs["ssl_keyfile"])

    def test_benchmark_records_confidence_without_identity(self):
        data = detection_confidences({"pessoas": [{"confidence": .8, "nome": "privado"},
                                    {"confidence": None}, {"confidence": float("nan")}],
                                     "gestos": [{"confidence": .7}]})
        self.assertEqual(data, {"persons_confidence": [.8], "gestures_confidence": [.7]})


if __name__ == "__main__":
    unittest.main()
