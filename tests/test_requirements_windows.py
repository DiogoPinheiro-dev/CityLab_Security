"""Passo 5 do TCC de PDI: dependencias do PC Windows fixadas e com um so OpenCV."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pinned():
    lines = (ROOT / "requirements-windows.txt").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


class WindowsRequirementsTests(unittest.TestCase):
    def test_every_package_has_an_exact_version(self):
        for line in pinned():
            self.assertRegex(line, r"^[A-Za-z0-9_.\-\[\]]+==[0-9][^\s]*$")

    def test_only_the_contrib_opencv_is_listed(self):
        names = [re.split(r"[=\[]", line)[0].lower() for line in pinned()]
        self.assertEqual([name for name in names if name.startswith("opencv")],
                         ["opencv-contrib-python"])

    def test_the_pi_file_is_left_alone(self):
        # O deploy do Pi so le o arquivo dele; os pinos do PC nao podem ir para la.
        pi = (ROOT / "requirements-rpi-bookworm.txt").read_text(encoding="utf-8")
        self.assertIn("mediapipe==0.10.18", pi)
        self.assertIn("numpy==1.26.4", pi)


if __name__ == "__main__":
    unittest.main()
