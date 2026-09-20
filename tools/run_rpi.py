"""Inicia o perfil do Pi com limites nativos aplicados antes das bibliotecas."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--show-config", action="store_true", help="Exibe configuracao sem carregar modelos")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Porta invalida")
    from dotenv import load_dotenv

    # Respeitar o perfil ja escolhido nos arquivos antes de aplicar o default.
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / "Server" / ".env", override=True)
    os.environ.setdefault("CITYLAB_PROFILE", "rpi3")

    from App import settings
    from App.inference_runtime import prepare_native_environment

    native = prepare_native_environment(settings.NATIVE_NUM_THREADS)
    # Lista explicita: nunca imprimir credenciais ou todo o ambiente.
    config = {name: getattr(settings, name) for name in (
        "CITYLAB_PROFILE", "ONNX_INTRA_OP_THREADS", "TORCH_NUM_THREADS",
        "OPENCV_NUM_THREADS", "NATIVE_NUM_THREADS", "PIPELINE_RUN_IN_PARALLEL",
        "PIPELINE_MAX_WORKERS", "PIPELINE_SHARED_PERSON_POSE", "FACE_MINIMAL_MODULES",
        "FACE_PREFILTER", "MAX_IN_FLIGHT_FRAMES", "PROCESS_SCALE",
    )}
    print(json.dumps({"configured": config, "native_environment": native, "uvicorn_workers": 1}, indent=2))
    if args.show_config:
        return

    import uvicorn

    # Um processo evita duplicar os modelos na memoria de 1 GB.
    uvicorn.run("Server.main:app", host=args.host, port=args.port, workers=1, reload=False)


if __name__ == "__main__":
    main()
