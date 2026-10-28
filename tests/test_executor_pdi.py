"""Passo 3 do TCC de PDI: partes do executor offline que rodam sem modelos nem banco."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import executor_pdi  # noqa: E402


class SamplingTests(unittest.TestCase):
    def test_one_frame_every_interval_of_video_with_its_instant(self):
        # 30 FPS, 10 s: quadros 0, 51, 102, ... nos instantes do video.
        samples = executor_pdi.sample_frames(30.0, 300, 1.7, 0.0)
        self.assertEqual([index for index, _ in samples], [0, 51, 102, 153, 204, 255])
        self.assertEqual([instant for _, instant in samples], [0.0, 1.7, 3.4, 5.1, 6.8, 8.5])

    def test_start_shifts_the_phase(self):
        samples = executor_pdi.sample_frames(30.0, 300, 1.7, 0.5)
        self.assertEqual(samples[0], (15, 0.5))
        self.assertEqual(len(samples), 6)

    def test_invalid_sampling_stops(self):
        for fps, interval, start in ((0, 1.7, 0), (30, 0, 0), (30, 1.7, -1)):
            with self.assertRaises(ValueError):
                executor_pdi.sample_frames(fps, 300, interval, start)


class RecordTests(unittest.TestCase):
    def test_gallery_hash_ignores_the_order_of_the_documents(self):
        documents = [{"nome": "B", "embedding": [0.0, 1.0]}, {"nome": "A", "embedding": [1.0, 0.0]}]
        names, vectors, digest = executor_pdi.gallery_from_documents(documents)
        self.assertEqual(names, ["A", "B"])
        self.assertEqual(vectors[0].tolist(), [1.0, 0.0])
        self.assertEqual(digest, executor_pdi.gallery_from_documents(documents[::-1])[2])
        documents[0]["embedding"] = [0.0, 0.9]
        self.assertNotEqual(digest, executor_pdi.gallery_from_documents(documents)[2])

    def test_face_and_gesture_records_keep_what_the_metrics_need(self):
        faces = executor_pdi.face_records([{"name": "Aluno", "bbox": [1, 2, 3, 4],
                                            "confidence": .612345, "det_confidence": .9,
                                            "pending": False}])
        self.assertEqual(faces, [{"nome": "Aluno", "bbox": [1, 2, 3, 4], "semelhanca": .6123,
                                  "deteccao": .9, "pendente": False}])
        gestures = executor_pdi.gesture_records([{"track_id": 3, "bbox": [0, 0, 5, 5],
                                                  "alerts": ["Rendicao"], "confidence": .8}])
        self.assertEqual(gestures, [{"track_id": 3, "bbox": [0, 0, 5, 5], "alertas": ["Rendicao"]}])

    def test_experiment_overrides_the_env_file(self):
        environ = {"FACE_LEARN_FROM_STREAM": "1", "CITYLAB_PROFILE": "default"}
        executor_pdi.apply_experiment_env("pose", "det.onnx", environ)
        self.assertEqual(environ["FACE_LEARN_FROM_STREAM"], "0")
        self.assertEqual(environ["CITYLAB_PROFILE"], "rpi3")
        self.assertEqual((environ["POSE_MODEL_PATH"], environ["FACE_DETECTOR_PATH"]),
                         ("pose", "det.onnx"))

    def test_summary_has_mean_and_dispersion(self):
        summary = executor_pdi.summarize([10.0, 20.0, 30.0])
        self.assertEqual((summary["media"], summary["desvio"], summary["mediana"]), (20.0, 10.0, 20.0))
        self.assertIsNone(executor_pdi.summarize([]))


if __name__ == "__main__":
    unittest.main()
