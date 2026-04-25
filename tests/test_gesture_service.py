import unittest

import numpy as np

from App.GestureRecon.service import GestureRecognitionService


def build_keypoints():
    keypoints = np.zeros((17, 3), dtype=float)
    keypoints[5] = [100.0, 100.0, 0.95]
    keypoints[6] = [200.0, 100.0, 0.95]
    keypoints[9] = [148.0, 165.0, 0.95]
    keypoints[10] = [240.0, 190.0, 0.95]
    keypoints[11] = [110.0, 220.0, 0.95]
    keypoints[12] = [190.0, 220.0, 0.95]
    return keypoints


class GestureRecognitionServiceTests(unittest.TestCase):
    def test_associate_hands_marks_visibility_and_torso_position(self):
        service = GestureRecognitionService.__new__(GestureRecognitionService)
        box = np.array([80.0, 80.0, 260.0, 260.0], dtype=float)
        keypoints = build_keypoints()
        hand_detections = [
            {
                "bbox": [138, 150, 160, 178],
                "center": [149, 164],
                "closed": True,
            },
            {
                "bbox": [228, 176, 252, 204],
                "center": [240, 190],
                "closed": False,
            },
        ]

        context = service._associate_hands(box, keypoints, hand_detections)

        self.assertTrue(context["left_visible"])
        self.assertTrue(context["right_visible"])
        self.assertTrue(context["left_closed"])
        self.assertFalse(context["right_closed"])
        self.assertTrue(context["left_in_torso"])
        self.assertFalse(context["right_in_torso"])
        self.assertEqual(len(context["matched_hands"]), 2)


if __name__ == "__main__":
    unittest.main()
