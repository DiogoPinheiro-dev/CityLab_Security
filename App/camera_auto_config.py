import cv2
import numpy as np


def apply_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / max(gamma, 0.01)
    table = np.array(
        [((value / 255.0) ** inv_gamma) * 255 for value in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def configure_camera_capture(cap, preferred_resolutions=None):
    if preferred_resolutions is None:
        preferred_resolutions = [
            (1280, 720),
            (960, 540),
            (640, 480),
        ]

    selected_resolution = None

    for width, height in preferred_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_width > 0 and actual_height > 0:
            selected_resolution = (actual_width, actual_height)

        if actual_width >= width and actual_height >= height:
            break

    return selected_resolution


class AutoImageOptimizer:
    def __init__(self):
        self.frame_counter = 0
        self.current_gamma = 1.0
        self.use_clahe = False

    def _analyze_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return {
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
        }

    def _estimate_gamma(self, brightness):
        if brightness < 70:
            return 1.45
        if brightness < 95:
            return 1.25
        if brightness > 185:
            return 0.85
        if brightness > 160:
            return 0.92
        return 1.0

    def optimize(self, frame):
        self.frame_counter += 1

        if self.frame_counter == 1 or self.frame_counter % 10 == 0:
            metrics = self._analyze_frame(frame)
            target_gamma = self._estimate_gamma(metrics["brightness"])

            self.current_gamma = (self.current_gamma * 0.7) + (target_gamma * 0.3)
            self.use_clahe = metrics["contrast"] < 45 and metrics["brightness"] < 150

        optimized = apply_gamma(frame, gamma=self.current_gamma)

        if self.use_clahe:
            optimized = apply_clahe(optimized)

        return optimized
