"""Passo 4 do TCC de PDI: metricas de rosto e de gesto contra a anotacao."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import metricas_pdi  # noqa: E402


def frame(t, faces=(), alerts=()):
    return {"t": t, "rostos": [{"nome": name, "pendente": pending} for name, pending in faces],
            "gestos": [{"track_id": 1, "alertas": list(alerts)}] if alerts else []}


ANNOTATION = {
    "pessoas": [{"id": "p1", "nome": "Aluno", "rosto": [[0.0, 10.0]]},
                {"id": "p2", "nome": None, "rosto": [[0.0, 3.4]]}],
    "gestos": [{"pessoa": "p1", "classe": "Mão Oculta", "inicio": 1.7, "fim": 6.8},
               {"pessoa": "p1", "classe": "Mao Fechada + Braco Estendido",
                "inicio": 8.5, "fim": 10.2}],
}


class FaceMetricsTests(unittest.TestCase):
    def test_counts_hits_misses_false_acceptances_and_rejections(self):
        frames = [
            frame(0.0, [("Aluno", False), ("NAO ALUNO", False)]),   # TP e TN
            frame(1.7, [("NAO ALUNO", False), ("NAO ALUNO", False)]),  # FN e TN
            frame(3.4, [("Aluno", False), ("Outro", False)]),        # TP; nao cadastrado aceito: FP
            frame(5.1, [("Aluno", True)]),                            # pendente nao confirma: FN
            frame(12.0, [("Aluno", False)]),                          # fora do trecho: FP
        ]
        result = metricas_pdi.face_metrics(frames, ANNOTATION)
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (2, 2, 2, 2))
        self.assertEqual(result["acuracia"], .5)
        self.assertEqual((result["precisao"], result["revocacao"], result["f1"]), (.5, .5, .5))


class GestureMetricsTests(unittest.TestCase):
    def test_interval_hit_miss_and_false_episodes_by_class(self):
        frames = [frame(0.0), frame(1.7), frame(3.4), frame(5.1, alerts=["Mao Oculta"]),
                  frame(6.8, alerts=["Mao Oculta"]),
                  # Duas sequencias de rendicao sem trecho: dois FP.
                  frame(8.5, alerts=["Rendicao"]), frame(10.2),
                  frame(11.9, alerts=["Rendicao", "Braco Estendido"])]
        result = metricas_pdi.gesture_metrics(frames, ANNOTATION, tolerance=1.7)["por_classe"]
        self.assertEqual((result["Mao Oculta"]["tp"], result["Mao Oculta"]["fn"]), (1, 0))
        self.assertEqual((result["Rendicao"]["fp"], result["Rendicao"]["trechos"]), (2, 0))
        # O trecho da ameaca vale para os componentes; o braco a 11,9 s ainda cai
        # na tolerancia. A ameaca e a mao fechada nao apareceram: FN.
        self.assertEqual((result["Braco Estendido"]["tp"], result["Braco Estendido"]["fp"]), (1, 0))
        self.assertEqual(result["Mao Fechada + Braco Estendido"]["fn"], 1)
        self.assertEqual(result["Mao Fechada"]["fn"], 1)

    def test_average_uses_annotated_classes_and_counts_other_alarms_apart(self):
        frames = [frame(5.1, alerts=["Mao Oculta", "Mao Fechada"])]
        annotation = {"gestos": [{"classe": "Mao Oculta", "inicio": 1.7, "fim": 6.8}]}
        result = metricas_pdi.gesture_metrics(frames, annotation, 0.0)
        self.assertEqual((result["media"]["classes"], result["media"]["f1"],
                          result["media"]["precisao"]), (1, 1.0, 1.0))
        self.assertEqual(result["alarmes_sem_trecho"], 1)

    def test_unknown_class_stops(self):
        with self.assertRaisesRegex(ValueError, "desconhecida"):
            metricas_pdi.gesture_spans({"gestos": [{"classe": "Aceno", "inicio": 0, "fim": 1}]})


class PrfTests(unittest.TestCase):
    def test_empty_counts_have_no_value_and_zero_hits_give_zero(self):
        self.assertEqual(metricas_pdi.prf(0, 0, 0), {"precisao": None, "revocacao": None, "f1": None})
        self.assertEqual(metricas_pdi.prf(0, 2, 1)["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
