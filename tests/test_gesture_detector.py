import unittest

import numpy as np

from App.GestureRecon.detector import GestureAnalyzer


def build_keypoints(
    left_elbow=(50.0, 150.0, 0.95),
    right_elbow=(250.0, 150.0, 0.95),
    left_wrist=(30.0, 200.0, 0.95),
    right_wrist=(270.0, 200.0, 0.95),
):
    keypoints = np.zeros((17, 3), dtype=float)
    keypoints[5] = [100.0, 100.0, 0.95]
    keypoints[6] = [200.0, 100.0, 0.95]
    keypoints[7] = left_elbow
    keypoints[8] = right_elbow
    keypoints[9] = left_wrist
    keypoints[10] = right_wrist
    keypoints[11] = [110.0, 220.0, 0.95]
    keypoints[12] = [190.0, 220.0, 0.95]
    return keypoints


class GestureAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = GestureAnalyzer(fps=12)
        self.track_id = 7

    def analyze_many(self, keypoints, hand_context, frames):
        result = None
        for _ in range(frames):
            result = self.analyzer.analyze(
                self.track_id,
                keypoints,
                hand_context=hand_context,
            )
        return result

    def test_detects_closed_left_hand_without_arm_extension(self):
        result = self.analyze_many(
            build_keypoints(),
            {
                "left_visible": True,
                "left_closed": True,
                "left_in_torso": False,
            },
            frames=self.analyzer.thresh_fist,
        )

        self.assertIn("Mao Fechada", result["alerts"])

    def test_detects_closed_right_hand_without_arm_extension(self):
        result = self.analyze_many(
            build_keypoints(),
            {
                "right_visible": True,
                "right_closed": True,
                "right_in_torso": False,
            },
            frames=self.analyzer.thresh_fist,
        )

        self.assertIn("Mao Fechada", result["alerts"])

    def test_detects_closed_hand_even_when_arm_is_not_extended(self):
        keypoints = build_keypoints(
            left_elbow=(85.0, 150.0, 0.95),
            left_wrist=(95.0, 185.0, 0.95),
        )

        result = self.analyze_many(
            keypoints,
            {
                "left_visible": True,
                "left_closed": True,
                "left_in_torso": True,
            },
            frames=self.analyzer.thresh_fist,
        )

        self.assertIn("Mao Fechada", result["alerts"])
        self.assertNotIn("Braco Estendido", result["alerts"])

    def test_hidden_hand_requires_arm_towards_torso(self):
        keypoints = build_keypoints(
            left_elbow=(135.0, 150.0, 0.95),
            left_wrist=(0.0, 0.0, 0.0),
        )

        result = self.analyze_many(
            keypoints,
            {
                "left_visible": False,
                "left_closed": False,
                "left_in_torso": False,
            },
            frames=self.analyzer.thresh_hidden,
        )

        self.assertIn("Mao Oculta", result["alerts"])
        self.assertTrue(result["hidden_debug"]["left"]["arm_towards_torso"])

    def test_hidden_hand_does_not_trigger_when_hand_is_detected(self):
        keypoints = build_keypoints(
            left_elbow=(135.0, 150.0, 0.95),
            left_wrist=(145.0, 185.0, 0.95),
        )

        result = self.analyze_many(
            keypoints,
            {
                "left_visible": True,
                "left_closed": False,
                "left_in_torso": True,
            },
            frames=self.analyzer.thresh_hidden,
        )

        self.assertNotIn("Mao Oculta", result["alerts"])

    def test_hidden_hand_clears_as_soon_as_hand_reappears(self):
        hidden_keypoints = build_keypoints(
            left_elbow=(135.0, 150.0, 0.95),
            left_wrist=(0.0, 0.0, 0.0),
        )
        visible_keypoints = build_keypoints(
            left_elbow=(135.0, 150.0, 0.95),
            left_wrist=(145.0, 185.0, 0.95),
        )

        hidden_result = self.analyze_many(
            hidden_keypoints,
            {
                "left_visible": False,
                "left_closed": False,
                "left_in_torso": False,
            },
            frames=self.analyzer.thresh_hidden,
        )
        visible_result = self.analyzer.analyze(
            self.track_id,
            visible_keypoints,
            hand_context={
                "left_visible": True,
                "left_closed": False,
                "left_in_torso": True,
            },
        )

        self.assertIn("Mao Oculta", hidden_result["alerts"])
        self.assertNotIn("Mao Oculta", visible_result["alerts"])

    def test_missing_hand_outside_torso_does_not_trigger_hidden(self):
        keypoints = build_keypoints(
            left_elbow=(45.0, 150.0, 0.95),
            left_wrist=(0.0, 0.0, 0.0),
        )

        result = self.analyze_many(
            keypoints,
            {
                "left_visible": False,
                "left_closed": False,
                "left_in_torso": False,
            },
            frames=self.analyzer.thresh_hidden,
        )

        self.assertNotIn("Mao Oculta", result["alerts"])
        self.assertFalse(result["hidden_debug"]["left"]["arm_towards_torso"])


if __name__ == "__main__":
    unittest.main()
