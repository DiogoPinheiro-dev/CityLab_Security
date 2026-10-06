"""Banco de replay das regras de gesto: acoes 1 e 2 de docs/PLANO_GESTOS.md."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
# A ferramenta fica em tools/, que nao e pacote.
sys.path.insert(0, str(ROOT / "tools"))
from gesture_replay import GESTURES, PREVIOUS_MIN_OBSERVATIONS, replay, synthetic_sequences

FPS30 = 1 / 30
PI_INTERVAL = 6.2  # frame de duas pessoas no perfil rpi3 em 26/09
PI_INTERVAL_NOW = 1.7  # frame de uma pessoa no Pi desde 04/10, com NCNN e det_500m
BASELINE_INTERVAL = 13.2  # frame da linha de base de 20 e 21/09
RECORDINGS = ROOT / "resultados" / "gestos-reais"


class GestureReplayTests(unittest.TestCase):
    def setUp(self):
        self.sequences = synthetic_sequences()

    def first(self, name, interval, min_observations=None):
        return replay(self.sequences[name]["frames"], interval, min_observations)

    def test_each_sustained_posture_fires_only_its_alerts(self):
        for name in [name for name in self.sequences if not name.endswith("_breve")]:
            for interval in (FPS30, 1.0, PI_INTERVAL_NOW, PI_INTERVAL, BASELINE_INTERVAL):
                with self.subTest(name=name, interval=interval):
                    self.assertEqual(set(self.first(name, interval)),
                                     self.sequences[name]["esperados"])

    def test_relaxed_arms_do_not_fire_from_the_front_or_the_back(self):
        # Video neutro de 29/09: braco solto afastado contava como estendido, e a
        # mao perdida pelo detector, ou vista de costas, contava como oculta.
        for name in ("neutro_bracos_afastados", "neutro_mao_perdida", "neutro_de_costas"):
            for interval in (FPS30, PI_INTERVAL):
                self.assertEqual(self.first(name, interval), {}, (name, interval))

    def test_fist_with_the_arm_down_does_not_fire(self):
        # Decisao do responsavel em 30/09: com a mao solta ao lado do corpo o
        # detector nao separa o punho da mao relaxada, entao ele so conta levantado.
        for interval in (FPS30, PI_INTERVAL):
            self.assertEqual(self.first("punho_braco_solto", interval), {}, interval)

    def test_arm_the_camera_cannot_see_from_the_side_counts_as_hidden(self):
        # Decisao do responsavel em 29/09: de lado, o braco fora de vista e oculto.
        self.assertEqual(self.first("de_lado_mao_do_outro_lado", PI_INTERVAL), {"Mao Oculta": 3})

    def test_duration_alone_collapsed_every_rule_on_the_pi(self):
        # O defeito medido em 20 e 21/09, com o frame da linha de base: tudo
        # dispara na segunda observacao.
        for gesture in GESTURES:
            first = self.first(f"{gesture}_mantido", BASELINE_INTERVAL, PREVIOUS_MIN_OBSERVATIONS)
            self.assertEqual(set(first.values()), {2}, gesture)

    def test_rules_require_different_persistence_on_the_pi(self):
        # Criterio de aceite 1: na taxa do Pi as regras voltam a se diferenciar.
        first = {gesture: self.first(f"{gesture}_mantido", PI_INTERVAL) for gesture in GESTURES}
        self.assertEqual(first["mao_fechada"], {"Mao Fechada": 2})
        self.assertEqual(first["ameaca"], {"Mao Fechada": 2, "Mao Fechada + Braco Estendido": 2,
                                           "Braco Estendido": 4})
        self.assertEqual(first["rendicao"], {"Rendicao": 3})
        self.assertEqual(first["mao_oculta"], {"Mao Oculta": 3})
        self.assertEqual(first["braco_estendido"], {"Braco Estendido": 4})

    def test_brief_gesture_on_the_pi_only_confirms_two_observation_rules(self):
        fired = {gesture: set(self.first(f"{gesture}_breve", PI_INTERVAL)) for gesture in GESTURES}
        self.assertEqual(fired["mao_fechada"], {"Mao Fechada"})
        self.assertEqual(fired["ameaca"], {"Mao Fechada", "Mao Fechada + Braco Estendido"})
        for gesture in ("rendicao", "mao_oculta", "braco_estendido"):
            self.assertEqual(fired[gesture], set(), gesture)

    def test_current_pi_rhythm_uses_the_minimum_durations(self):
        # Decisao de 06/10: com o frame em 1,7 s, mao fechada pede 4 s e mao
        # oculta 8 s; as outras regras seguem decididas pelas observacoes.
        first = {gesture: self.first(f"{gesture}_mantido", PI_INTERVAL_NOW)
                 for gesture in GESTURES}
        self.assertEqual(first["mao_fechada"], {"Mao Fechada": 4})
        self.assertEqual(first["ameaca"], {"Mao Fechada": 4, "Mao Fechada + Braco Estendido": 2,
                                           "Braco Estendido": 4})
        self.assertEqual(first["rendicao"], {"Rendicao": 3})
        self.assertEqual(first["mao_oculta"], {"Mao Oculta": 6})
        self.assertEqual(first["braco_estendido"], {"Braco Estendido": 4})
        # Gesto de duas observacoes, 1,7 s: so a ameaca confirma.
        fired = {gesture: set(self.first(f"{gesture}_breve", PI_INTERVAL_NOW))
                 for gesture in GESTURES}
        self.assertEqual(fired["ameaca"], {"Mao Fechada + Braco Estendido"})
        for gesture in ("mao_fechada", "rendicao", "mao_oculta", "braco_estendido"):
            self.assertEqual(fired[gesture], set(), gesture)

    def test_thirty_fps_keeps_the_previous_behavior(self):
        # Criterio de aceite 3: a 30 FPS a duracao domina e nenhuma regra muda.
        for name in self.sequences:
            with self.subTest(name=name):
                self.assertEqual(self.first(name, FPS30),
                                 self.first(name, FPS30, PREVIOUS_MIN_OBSERVATIONS))

    def test_one_missed_observation_restarts_the_count(self):
        neutral = self.sequences["neutro"]["frames"][0]
        gesture = self.sequences["braco_estendido_mantido"]["frames"][0]
        # Tres observacoes, uma falha e quatro de novo: so a quarta seguida confirma.
        frames = [gesture] * 3 + [neutral] + [gesture] * 4
        self.assertEqual(replay(frames, PI_INTERVAL), {"Braco Estendido": 8})


class RecordedGestureTests(unittest.TestCase):
    """Gravacoes de 29/09/2026: uma pessoa de frente, de lado e de costas."""

    def load(self, name):
        path = RECORDINGS / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8"))["frames"]

    def test_neutral_front_and_back_no_longer_fire_arm_or_hidden_hand(self):
        frames = self.load("neutro")
        for start, end in ((2.0, 11.0), (38.5, 48.0)):
            for interval in (PI_INTERVAL_NOW, PI_INTERVAL):
                window = [frame for frame in frames if start <= frame["t"] < end]
                fired = replay(window, interval)
                self.assertNotIn("Braco Estendido", fired, (start, end, interval))
                self.assertNotIn("Mao Oculta", fired, (start, end, interval))

    def test_each_recorded_gesture_still_fires(self):
        expected = {"rendicao": "Rendicao", "mao_oculta": "Mao Oculta",
                    "braco_estendido": "Braco Estendido",
                    "ameaca": "Mao Fechada + Braco Estendido"}
        for name, alert in expected.items():
            for interval in (PI_INTERVAL_NOW, PI_INTERVAL):
                with self.subTest(name=name, interval=interval):
                    self.assertIn(alert, replay(self.load(name), interval))

    def test_recorded_fist_with_the_arm_down_no_longer_fires(self):
        for interval in (PI_INTERVAL_NOW, PI_INTERVAL):
            # mao_fechada.json e o punho com o braco solto, que parou de alertar em 30/09.
            self.assertNotIn("Mao Fechada", replay(self.load("mao_fechada"), interval))
            # Na ameaca, com o braco levantado, o punho continua contando.
            self.assertIn("Mao Fechada", replay(self.load("ameaca"), interval))
