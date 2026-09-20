# Raspberry Pi 3 B+ / Raspberry Pi OS Legacy 64-bit Bookworm

Este perfil mira o Raspberry Pi 3 B+ com Raspberry Pi OS Legacy 64-bit Bookworm
e Python 3.11. Ele evita Python 3.13 porque o MediaPipe usado pelo projeto nao
publica wheel Linux ARM64/aarch64 para essa versao.

## Base recomendada

- Placa: Raspberry Pi 3 B+
- SO: Raspberry Pi OS Legacy 64-bit Bookworm
- Python: 3.11
- MediaPipe: 0.10.18

O arquivo de dependencias desta branch e `requirements-rpi-bookworm.txt`.
Ele instala a base da API; os modelos pesados continuam opcionais.

## Instalacao

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv libatlas-base-dev

python3.11 -m venv --system-site-packages .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-rpi-bookworm.txt
```

Crie `.env` na raiz com `MONGO_DETAILS` e `MONGO_DB_NAME`, conforme o README.
A inicializacao da API valida a conexao com MongoDB antes de carregar os modelos.

Se a rede do Raspberry estiver usando IPv6 de forma instavel para o Atlas, tente
forcar IPv4 no `.env`:

```env
MONGO_FORCE_IPV4=1
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

## Medicao reproduzivel

Consulte [BENCHMARK.md](BENCHMARK.md) para protocolo, metricas e limites da medicao.

## Experimento de passada unica de pessoas e pose

Na branch `codex/raspberry-stream-optimization`, configure no `.env`:

```env
CITYLAB_ALLOW_PARTIAL_PIPELINE=0
CITYLAB_ENABLE_FACE_SERVICE=1
CITYLAB_ENABLE_GESTURE_SERVICE=1
PIPELINE_SHARED_PERSON_POSE=1
ENABLE_PERFORMANCE_METRICS=1
ENABLE_SYSTEM_MONITOR=1
```

Reinicie a API com um unico worker. O valor `0` restaura a deteccao separada
de pessoas e e o padrao enquanto o experimento nao for validado. Evite definir
a mesma chave em `Server/.env`, pois esse arquivo tem precedencia sobre o raiz.
Mantenha `PIPELINE_RUN_IN_PARALLEL` e `PIPELINE_MAX_WORKERS` iguais nas duas
configuracoes; use `DEBUG_PIPELINE=1` para conferir o modo efetivo no payload.

Com a opcao ativa, `persons_ms` e zero: o custo compartilhado esta em `pose_ms`,
dentro de `gestures_ms`. A cena vazia tambem executa pose e pode ficar mais lenta.
Rostos continuam sendo analisados em todos os frames. O detector separado de
pessoas e carregado sob demanda quando gestos estiverem desativados/indisponiveis.

Compare primeiro `0` e depois `1`, tres rodadas por cenario (vazio, uma e duas
pessoas), reiniciando a API a cada rodada. Exemplo no computador com webcam:

```bash
python tools/benchmark_stream.py --camera-index 0 --scenario many-persons --run-label pi3-shared-1-r1 --warmup 5 --frames 30 --output resultados/pi3-shared-1/duas-pessoas-r1.json
```

Anote o commit e a configuracao com cada rodada. Verifique pessoas distantes,
de lado e rostos pequenos, alem dos gestos. So considerar ganho acima de 5%
repetido nas tres rodadas e sem perda de deteccoes. As regras temporais de gesto
mudaram nesta branch; mantenha essa versao nos dois lados da comparacao.
