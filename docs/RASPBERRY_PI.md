# Raspberry Pi 3 B+ / Raspberry Pi OS Legacy 64-bit Bookworm

Este perfil mira o Raspberry Pi 3 B+ com Raspberry Pi OS Legacy 64-bit Bookworm
e Python 3.11. Ele evita Python 3.13 porque o MediaPipe usado pelo projeto nao
publica wheel Linux ARM64/aarch64 para essa versao.

## Base recomendada

- Placa: Raspberry Pi 3 B+
- SO: Raspberry Pi OS Legacy 64-bit Bookworm
- Python: 3.11
- MediaPipe: 0.10.18

## Perfil especifico do Pi 3 B+

O alvo tem quatro nucleos Cortex-A53 a 1,4 GHz e 1 GB de RAM
([especificacao oficial](https://www.raspberrypi.com/products/raspberry-pi-3-model-b-plus/)).
O perfil foi preparado para limitar a concorrencia interna das bibliotecas:

```env
CITYLAB_PROFILE=rpi3
```

Mescle `.env.rpi.example` no `.env` existente, preservando as credenciais.
Nao sobrescreva seu arquivo de ambiente inteiro. O perfil escolhe ONNX=1,
PyTorch=2 e OpenCV=1 thread, um frame pendente no cliente, a passada unica de
pessoas e pose (`PIPELINE_SHARED_PERSON_POSE`) e o gate de movimento antes da
pose (`GESTURE_MOTION_GATE`). Nao desativa reconhecimento nem reduz a
qualidade da imagem. Qualquer valor explicito dessas variaveis no ambiente tem
precedencia; confira tambem `Server/.env`.

Use um unico processo para nao duplicar modelos no 1 GB de RAM:

```bash
python tools/run_rpi.py --host 0.0.0.0 --port 8000
```

Este inicializador escolhe o perfil rpi3 se nenhum perfil estiver configurado,
aplica `NATIVE_NUM_THREADS=1` a BLAS/OpenMP antes de importar a API e fixa um
worker, sem reload. Variaveis nativas ja definidas sao preservadas. Para conferir
as configuracoes sem carregar modelos nem conectar ao banco:

```bash
python tools/run_rpi.py --show-config
```

O comando exige o ambiente do projeto instalado, incluindo python-dotenv.
Executar Uvicorn diretamente continua possivel, mas nao aplica o preparo nativo
desse inicializador. Os limites PyTorch tambem sao aplicados nos workers da
pipeline. Eles nao garantem um total fixo de threads de todas as bibliotecas.

O perfil completo foi medido no Pi em 21 e 26/09/2026: -45,8% com uma pessoa,
-53,4% a -54,4% com duas e -69,0% na cena vazia, sem perda de deteccao (ver
`docs/PLANO_OTIMIZACAO.md`). Os ajustes individuais nao foram medidos
isoladamente. Para comparar uma mudanca por vez, use `CITYLAB_PROFILE=default`
e ajuste individualmente `ONNX_INTRA_OP_THREADS`, `TORCH_NUM_THREADS`,
`OPENCV_NUM_THREADS`, `MAX_IN_FLIGHT_FRAMES`, `PIPELINE_SHARED_PERSON_POSE` e
`GESTURE_MOTION_GATE`. Reinicie o servidor e recarregue
o cliente quando mudar o limite de frames. Para voltar ao automatico, use
perfil default e remova os overrides, ou defina as tres opcoes de threads em 0.
O carregamento ajustado de ONNX libera a referencia da sessao antiga antes de
recria-la; ainda e necessario medir RAM nativa no hardware.

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

## Passada unica de pessoas e pose, e gate de movimento

No perfil rpi3 as caixas de pessoa saem da mesma passada do modelo de pose
(`PIPELINE_SHARED_PERSON_POSE=1`), e um gate por diferenca de frames pula a
pose quando a cena esta parada e desocupada (`GESTURE_MOTION_GATE=1`). As duas
opcoes comecaram como experimento e viraram padrao do perfil em 26/09/2026,
depois de tres rodadas em cada cenario. No perfil default continuam desligadas.

Com a passada unica, `persons_ms` e zero: o custo compartilhado esta em
`pose_ms`, dentro de `gestures_ms`. Caixas abaixo de
`GESTURE_PUBLISH_MIN_CONFIDENCE` alimentam o rastreador, mas nao saem
publicadas como pessoa. Sem o gate, a cena vazia pagaria a pose em todo frame;
com ele, a pose roda quando ha movimento, quando a cena esta ocupada e, no
maximo, a cada `GESTURE_MOTION_MAX_SKIP_SECONDS`. Rostos continuam sendo
analisados em todos os frames. O detector separado de pessoas e carregado sob
demanda quando gestos estiverem desativados/indisponiveis.

Para comparar com o caminho anterior, defina `PIPELINE_SHARED_PERSON_POSE=0`
ou `GESTURE_MOTION_GATE=0`, uma de cada vez, e reinicie a API com um unico
worker. Evite definir a mesma chave em `Server/.env`, pois esse arquivo tem
precedencia sobre o raiz. Mantenha `PIPELINE_RUN_IN_PARALLEL` e
`PIPELINE_MAX_WORKERS` iguais nas duas configuracoes; use `DEBUG_PIPELINE=1`
para conferir o modo efetivo no payload.

Sao tres rodadas por cenario (vazio, uma e duas pessoas), reiniciando a API a
cada rodada. Exemplo no computador com webcam:

```bash
python tools/benchmark_stream.py --camera-index 0 --scenario many-persons --run-label pi3-gate-r1 --url wss://<ip-do-pi>:8000/stream --warmup 5 --frames 30 --output resultados/pi3-gate/duas-pessoas-r1.json
```

Anote o commit e a configuracao com cada rodada. Verifique pessoas distantes,
de lado e rostos pequenos, alem dos gestos. So considerar ganho acima de 5%
repetido nas tres rodadas e sem perda de deteccoes. As regras temporais de gesto
mudaram nesta branch; mantenha essa versao nos dois lados da comparacao.

## Otimizacoes faciais e controles de CPU

Os padroes atuais sao:

```env
FACE_MINIMAL_MODULES=1
FACE_PREFILTER=1
ONNX_INTRA_OP_THREADS=0
TORCH_NUM_THREADS=0
GESTURE_IDLE_RESET_SECONDS=5
GESTURE_MAX_OBSERVATION_GAP_SECONDS=60
```

Os modulos faciais retidos sao deteccao e reconhecimento. O filtro antecipado
usa exatamente os limites atuais de qualidade, evitando embeddings de rostos
que seriam descartados. Cadastro, pesos, resolucao e similaridade permanecem
iguais. Para voltar ao caminho facial anterior, defina ambas as flags FACE
acima como `0` e reinicie. Para A/B, altere somente uma flag por rodada.

A equivalencia dos embeddings com essas flags ainda nao foi conferida com os
modelos reais. Compare nas mesmas fotos locais (rosto frontal, lateral,
pequeno, varias pessoas e cena vazia):

```bash
python tools/check_face_optimization.py frontal.jpg lateral.jpg varias.jpg vazia.jpg
```

A ferramenta exige as dependencias reais do pipeline e o buffalo_l. Compara
caixas, resultado do reconhecimento e embeddings, sem salvar fotos ou vetores.
Usa a base de rostos local; valide tambem o cadastro e o stream com a base Mongo
de teste. Nao e benchmark nem prova de recall fora das imagens fornecidas.

Os limites de threads sao experimentais. Comece comparando o valor automatico
`0` com `ONNX_INTRA_OP_THREADS=1` ou `2`, depois ajuste `TORCH_NUM_THREADS`
separadamente. Mantenha a opcao de paralelismo registrada e constante. O ajuste
ONNX recria sessoes no startup: monitore tambem o pico de RAM durante a carga.
Use um worker Uvicorn e uma camera por processo.

O limite de inatividade mede somente a espera por novos frames, nao o tempo de
inferencia. O limite de 60 s entre observacoes e uma protecao adicional; ajuste
se o pior tempo normal por frame ultrapassar esse valor. Gestos nao amostrados
continuam impossiveis de recuperar por software.

## Backend NCNN opcional

Em um ambiente com Ultralytics e ferramentas de exportacao, gere uma copia do
mesmo peso (pode exigir dependencias adicionais; preferir exportar fora do Pi):

```bash
python tools/export_pose_ncnn.py App/GestureRecon/yolov8n-pose.pt --imgsz 640
```

Copie o diretorio gerado para o Pi e configure:

```env
POSE_MODEL_PATH=App/GestureRecon/yolov8n-pose_ncnn_model
```

Caminhos relativos sao resolvidos a partir da raiz do projeto. Deixe a variavel
vazia para voltar ao .pt. O export usa FP32 e nao altera o peso original. Ainda
e necessario comparar caixas, keypoints, rastreamento e alertas no Pi: formato
e padding podem alterar resultados. Nao habilitar junto com outra mudanca na
mesma rodada, nem assumir ganho antes da medicao.
