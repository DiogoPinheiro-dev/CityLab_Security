"""Inicia o perfil do Pi com limites nativos aplicados antes das bibliotecas."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _tls_paths(parser, certfile, keyfile):
    """Exige o par completo e arquivos existentes antes de carregar modelos."""
    if (certfile is None) != (keyfile is None):
        parser.error("Informe --ssl-certfile e --ssl-keyfile juntos")
    if certfile is None:
        return None, None
    resolved = []
    for option, path in (("--ssl-certfile", certfile), ("--ssl-keyfile", keyfile)):
        candidate = Path(path).expanduser()
        if not candidate.is_file():
            parser.error(f"{option} nao encontrado: {candidate}")
        resolved.append(str(candidate))
    return resolved[0], resolved[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-certfile", type=Path, help="Certificado TLS; exige --ssl-keyfile")
    parser.add_argument("--ssl-keyfile", type=Path, help="Chave TLS; exige --ssl-certfile")
    parser.add_argument("--show-config", action="store_true", help="Exibe configuracao sem carregar modelos")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Porta invalida")
    # Validar o par antes do dotenv: erro de caminho nao deve custar carga de modelo.
    certfile, keyfile = _tls_paths(parser, args.ssl_certfile, args.ssl_keyfile)
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
    # Somente o estado do TLS: o par aponta para a chave privada do servidor.
    print(json.dumps({"configured": config, "native_environment": native,
                      "uvicorn_workers": 1, "tls": certfile is not None}, indent=2))
    if args.show_config:
        return

    import uvicorn

    # Um processo evita duplicar os modelos na memoria de 1 GB.
    uvicorn.run("Server.main:app", host=args.host, port=args.port, workers=1,
                reload=False, ssl_certfile=certfile, ssl_keyfile=keyfile)


if __name__ == "__main__":
    main()
