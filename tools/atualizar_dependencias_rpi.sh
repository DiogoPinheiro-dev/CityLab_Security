#!/usr/bin/env bash
# Instala o requirements-rpi-bookworm.txt no .venv da API, no deploy do Pi.
#
# Roda o pip so quando o arquivo mudou desde a ultima instalacao certa, e com
# --no-deps: so os pacotes listados, nas versoes do arquivo. Em 04/10/2026 uma
# dependencia de carona trocou o numpy e o OpenCV do ambiente da API. Se depois
# da instalacao a API nao importar, volta as versoes de antes e sai com erro.
#
# Uso, no Pi, na raiz do projeto: bash tools/atualizar_dependencias_rpi.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
REQ=requirements-rpi-bookworm.txt
MARCA=.venv/.requirements-rpi.sha256
# O que a API carrega na subida; o ncnn so quando instalado, para a pose NCNN.
IMPORTA='import importlib.util, Server.main, mediapipe, insightface, onnxruntime, ultralytics
if importlib.util.find_spec("ncnn"): import ncnn'

if [ ! -x "$PY" ]; then
    echo "Sem $PY: crie o ambiente da API, como em docs/RASPBERRY_PI.md."
    exit 1
fi
"$PY" -c "import sys; assert sys.version_info[:2] == (3, 11), 'O .venv da API precisa de Python 3.11'"

atual=$(sha256sum "$REQ" | cut -d' ' -f1)
if [ -f "$MARCA" ] && [ "$(cat "$MARCA")" = "$atual" ]; then
    echo "$REQ igual ao da ultima instalacao: nada a instalar."
    exit 0
fi

antes=$(mktemp)
"$PY" -m pip freeze --local > "$antes"
if "$PY" -m pip install --no-deps -r "$REQ" && "$PY" -c "$IMPORTA"; then
    echo "$atual" > "$MARCA"
    rm -f "$antes"
    echo "Dependencias instaladas no .venv da API."
    exit 0
fi

echo "A instalacao deixou a API sem importar: voltando as versoes de antes ($antes)."
depois=$(mktemp)
"$PY" -m pip freeze --local > "$depois"
novos=$(comm -13 <(cut -d= -f1 "$antes" | sort) <(cut -d= -f1 "$depois" | sort))
if [ -n "$novos" ]; then
    # shellcheck disable=SC2086
    "$PY" -m pip uninstall -y $novos || true
fi
"$PY" -m pip install --no-deps -r "$antes" || true
if "$PY" -c "$IMPORTA"; then
    echo "A API voltou a importar com as versoes de antes."
else
    echo "AVISO: a API continua sem importar; veja o log deste passo."
fi
exit 1
