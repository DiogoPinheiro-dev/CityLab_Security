"""tools/atualizar_dependencias_rpi.sh com um python falso: quando instala e quando volta."""
import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")

# Anota cada chamada e responde como o teste manda, pelos arquivos da pasta.
FAKE_PYTHON = """#!/usr/bin/env bash
echo "$*" >> "$RAIZ/chamadas.log"
case "$*" in
  *"pip freeze --local"*) cat "$RAIZ/instalados.txt" ;;
  *"pip install --no-deps -r requirements-rpi-bookworm.txt"*)
    cp "$RAIZ/depois.txt" "$RAIZ/instalados.txt"
    cp "$RAIZ/importa_depois.txt" "$RAIZ/importa.txt"
    exit "$(cat "$RAIZ/pip.txt")" ;;
  *"pip install --no-deps -r "*)
    cp "$6" "$RAIZ/instalados.txt"
    echo 0 > "$RAIZ/importa.txt" ;;
  *"pip uninstall"*) ;;
  *"sys.version_info"*) ;;
  *"Server.main"*) exit "$(cat "$RAIZ/importa.txt")" ;;
  *) exit 9 ;;
esac
"""


@unittest.skipIf(BASH is None, "sem bash")
class DeployDependenciesTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        (self.root / "tools").mkdir()
        script = (ROOT / "tools" / "atualizar_dependencias_rpi.sh").read_text(encoding="utf-8")
        (self.root / "tools" / "atualizar_dependencias_rpi.sh").write_bytes(
            script.replace("\r\n", "\n").encode("utf-8"))
        (self.root / ".venv" / "bin").mkdir(parents=True)
        (self.root / ".venv" / "bin" / "python").write_bytes(FAKE_PYTHON.encode("utf-8"))
        self.write("requirements-rpi-bookworm.txt", "numpy==1.26.4\n")
        self.write("instalados.txt", "mediapipe==0.10.18\nnumpy==1.26.4\n")
        self.write("depois.txt", "mediapipe==0.10.18\nnumpy==1.26.4\n")
        self.write("pip.txt", "0\n")
        self.write("importa_depois.txt", "0\n")
        self.write("importa.txt", "0\n")

    def write(self, name, text):
        (self.root / name).write_bytes(text.encode("utf-8"))

    def read(self, name):
        path = self.root / name
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def run_script(self):
        return subprocess.run(
            [BASH, (self.root / "tools" / "atualizar_dependencias_rpi.sh").as_posix()],
            env={**os.environ, "RAIZ": self.root.as_posix()},
            capture_output=True, text=True, timeout=60)

    def installs(self):
        return self.read("chamadas.log").count("pip install --no-deps -r requirements")

    def test_installs_once_and_again_only_when_the_requirements_change(self):
        self.assertEqual(self.run_script().returncode, 0)
        requirements = (self.root / "requirements-rpi-bookworm.txt").read_bytes()
        self.assertEqual(self.read(".venv/.requirements-rpi.sha256").strip(),
                         hashlib.sha256(requirements).hexdigest())
        # Mesmo arquivo: o deploy seguinte nao chama o pip.
        result = self.run_script()
        self.assertEqual((result.returncode, self.installs()), (0, 1))
        self.assertIn("nada a instalar", result.stdout)
        self.write("requirements-rpi-bookworm.txt", "numpy==1.26.4\nncnn==1.0.20260526\n")
        self.assertEqual(self.run_script().returncode, 0)
        self.assertEqual(self.installs(), 2)

    def test_install_that_breaks_the_api_goes_back_to_the_previous_versions(self):
        # Como em 04/10/2026: o numpy trocado e um OpenCV novo por cima.
        self.write("depois.txt", "mediapipe==0.10.18\nnumpy==2.4.6\nopencv-python==5.0.0.93\n")
        self.write("importa_depois.txt", "1\n")
        result = self.run_script()
        self.assertEqual(result.returncode, 1)
        self.assertIn("pip uninstall -y opencv-python", self.read("chamadas.log"))
        self.assertEqual(self.read("instalados.txt"), "mediapipe==0.10.18\nnumpy==1.26.4\n")
        self.assertIn("voltou a importar", result.stdout)
        # Sem a marca, o proximo deploy tenta de novo.
        self.assertFalse((self.root / ".venv" / ".requirements-rpi.sha256").exists())

    def test_pip_error_also_goes_back_and_fails_the_step(self):
        self.write("pip.txt", "1\n")
        result = self.run_script()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.read("chamadas.log").count("pip install --no-deps -r "), 2)
        self.assertFalse((self.root / ".venv" / ".requirements-rpi.sha256").exists())


if __name__ == "__main__":
    unittest.main()
