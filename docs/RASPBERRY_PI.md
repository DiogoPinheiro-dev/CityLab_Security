# Raspberry Pi 3 B+ / Raspberry Pi OS Legacy 64-bit Bookworm

Este perfil mira o Raspberry Pi 3 B+ com Raspberry Pi OS Legacy 64-bit Bookworm
e Python 3.11. Ele evita Python 3.13 porque o MediaPipe usado pelo projeto nao
publica wheel Linux ARM64/aarch64 para essa versao.

## Base recomendada

- Placa: Raspberry Pi 3 B+
- SO: Raspberry Pi OS Legacy 64-bit Bookworm
- Python: 3.11
- MediaPipe: 0.10.18

O `requirements.txt` principal continua voltado ao ambiente de desenvolvimento
mais novo. Para o Pi, use `requirements-rpi-bookworm.txt`.

## Instalacao

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv libatlas-base-dev

python3 -m venv --system-site-packages .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-rpi-bookworm.txt
```

Copie o exemplo de variaveis:

```bash
cp .env.rpi.example .env
```

Suba a API:

```bash
python -m uvicorn Server.main:app --host 0.0.0.0 --port 8000
```

## Pipeline parcial

O Raspberry Pi 3 B+ tem pouca memoria e as dependencias de visao sao sensiveis a
arquitetura. Por isso, o backend aceita subir em modo parcial:

```env
CITYLAB_ALLOW_PARTIAL_PIPELINE=1
CITYLAB_ENABLE_FACE_SERVICE=0
CITYLAB_ENABLE_GESTURE_SERVICE=0
```

Com `CITYLAB_ALLOW_PARTIAL_PIPELINE=1`, a API nao cai se um servico pesado falhar
ao iniciar. O payload desse servico volta vazio.

Servicos:

- `CITYLAB_ENABLE_FACE_SERVICE`: reconhecimento facial, InsightFace, YOLO de pessoas.
- `CITYLAB_ENABLE_GESTURE_SERVICE`: YOLO pose e detector de maos MediaPipe.

Comece com ambos desligados para validar API, MongoDB e cliente. Depois ligue cada
servico conforme as dependencias de visao forem instaladas e testadas no hardware.

## Observacoes importantes

- `mediapipe==0.10.33` nao tem wheel Linux ARM64/aarch64 para o Raspberry Pi.
- `mediapipe==0.10.18` tem wheel Linux ARM64 para CPython 3.11.
- OpenCV deve vir do `apt` (`python3-opencv`) para evitar builds longos no Pi.
- `ultralytics`, `torch`, `insightface` e `onnxruntime` devem ser validados em
  separado. Para YOLO no Pi, prefira exportar modelos para NCNN quando possivel.
