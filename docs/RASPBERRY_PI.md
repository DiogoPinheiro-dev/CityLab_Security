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
Nao sobrescreva seu arquivo de ambiente inteiro. O perfil escolhe ONNX=2,
PyTorch=3 e OpenCV=1 thread, um frame pendente no cliente, a passada unica de
pessoas e pose (`PIPELINE_SHARED_PERSON_POSE`), o gate de movimento antes da
pose (`GESTURE_MOTION_GATE`), a pose em 416 px (`POSE_IMGSZ`), em vez dos 640
do Ultralytics, o reuso por 15 s do nome de um rosto que continua no mesmo
lugar (`FACE_REUSE_SECONDS`), o embedding do rosto no frame original
(`FACE_EMBED_FULL_FRAME`), o aprendizado de referencias pelo stream
(`FACE_LEARN_FROM_STREAM`) e 2 threads do ncnn quando a pose e NCNN
(`NCNN_NUM_THREADS`). A pose em NCNN em si depende da pasta exportada, fora do
Git, e liga pelo `POSE_MODEL_PATH` no `.env`; ver "Backend NCNN opcional".
Nao desativa reconhecimento nem reduz a qualidade da imagem. Qualquer valor
explicito dessas variaveis no ambiente tem precedencia; confira tambem
`Server/.env`.

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
pipeline e reaplicados no inicio de cada tarefa de gesto: o `select_device` do
Ultralytics chama `torch.set_num_threads` na thread que roda o primeiro
`track()`, e sem isso cada worker rodaria a pose com um limite diferente. Eles
nao garantem um total fixo de threads de todas as bibliotecas.

O perfil completo foi medido no Pi entre 21 e 27/09/2026: -56% com uma pessoa,
-63% com duas e -69% na cena vazia, sem perda de deteccao (ver
`docs/PLANO_OTIMIZACAO.md`). A pose em 416 px entrou no perfil em 01/10/2026,
depois de deixar o frame com uma pessoa 6% mais rapido, e no mesmo dia as 2
threads do ONNX Runtime, com mais 20% num video fixo de uma pessoa, e o reuso
do nome do rosto, com mais 13% na media. Com tudo isso, em 02/10/2026, a cena
vazia ficou em 0,93 s e duas pessoas em 3,41 s, -83% e -80% contra 19/09; uma
pessoa nao foi medida com eles na webcam. Em 03/10/2026 entraram o embedding
do rosto no frame original e o aprendizado pelo stream, que levou o video de
carga de 26 a 28 de 30 frames reconhecidos; os dois sem custo de tempo. Os
ajustes individuais nao foram medidos isoladamente. Para
comparar uma mudanca por vez, use `CITYLAB_PROFILE=default` e ajuste
individualmente `ONNX_INTRA_OP_THREADS`, `TORCH_NUM_THREADS`,
`OPENCV_NUM_THREADS`, `MAX_IN_FLIGHT_FRAMES`, `PIPELINE_SHARED_PERSON_POSE`,
`GESTURE_MOTION_GATE`, `POSE_IMGSZ`, `FACE_REUSE_SECONDS`,
`FACE_EMBED_FULL_FRAME` e `FACE_LEARN_FROM_STREAM`. Reinicie o servidor
e recarregue o cliente quando mudar o limite de frames. Para voltar ao automatico, use
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
Em seguida cria o indice que faz o MongoDB apagar os eventos com mais de
`EVENT_RETENTION_DAYS` dias (30 por padrao; 0 guarda tudo). Se o usuario do
banco nao puder criar indices, a API sobe assim mesmo e avisa no log.

Suba a API com o inicializador do perfil, descrito acima:

```bash
python tools/run_rpi.py --host 0.0.0.0 --port 8000
```

## Rodar como servico (uso continuo)

Para a API ficar no ar sem parar, instale-a como servico do usuario `citylab`:
ela sobe quando o Pi liga, volta sozinha 10 s depois de cair, por erro ou por
falta de memoria, e nao depende de sessao SSH. Uma vez so, na raiz do projeto:

```bash
bash tools/instalar_servico_rpi.sh CAMINHO/certificado.pem CAMINHO/chave.pem
```

O script recusa instalar se houver uma API iniciada a mao rodando, porque as
duas disputariam a porta 8000. Pede `sudo` uma vez, para o
`loginctl enable-linger`, que deixa os servicos do usuario rodarem sem ele
estar logado. Depois:

```bash
systemctl --user status citylab-api     # estado
systemctl --user stop citylab-api       # parar
systemctl --user start citylab-api      # subir de novo
journalctl --user-unit citylab-api -f   # log; se vier vazio, use sudo
```

O deploy para o servico antes de sincronizar e instalar, e sobe de novo no fim,
mesmo se a instalacao falhar; funciona sem `sudo` porque o runner roda com o
mesmo usuario `citylab`. O `.env` continua valendo; so e preciso reinstalar se
o certificado mudar de lugar.

As dependencias entram no `.venv` da API por
`tools/atualizar_dependencias_rpi.sh`, so quando o
`requirements-rpi-bookworm.txt` muda, e com `--no-deps`, para nenhum pacote
trocar outro de carona. Antes de instalar, o script guarda as versoes do
ambiente; se a API deixar de importar, ele volta a elas, e o passo do deploy
falha. A pasta `citylab_venv`, que o deploy criava ate 04/10/2026, nao e mais
usada.

Medido em 02/10/2026: depois de um `kill -9` a API voltou a responder em 1 min
51 s, e depois de `sudo reboot` ficou pronta em cerca de 2,5 min, sem ninguem
logar. Quase todo esse tempo e a carga dos modelos.

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

## Temperatura e limite de clock

No Pi 3 B+ o firmware baixa o clock de 1,4 para 1,2 GHz quando o chip atinge o
`temp_soft_limit`, 60 C por padrao. Com o perfil rpi3 sob carga os quatro
nucleos ficam ocupados, o limite chega em cerca de 75 s e dali em diante o frame
fica de 3% a 7% mais lento (medido em 26 e 27/09/2026, ver `docs/PLANO_OTIMIZACAO.md`).

Para conferir, rode em um segundo terminal durante a medicao, nao depois dela:

```bash
while true; do echo "$(date +%T) $(vcgencmd get_throttled) $(vcgencmd measure_clock arm) $(vcgencmd measure_temp)"; sleep 5; done
```

Em `get_throttled`, o bit `0x8` indica o limite ativo naquele instante e o bit
`0x80000` indica que ele ja foi atingido desde o boot. Terminada a rodada, o
clock volta ao repouso de 600 MHz, entao uma leitura isolada nao mostra a queda.

Um dissipador com ventoinha mantem o chip abaixo do limite. A alternativa e
subir `temp_soft_limit` em `/boot/firmware/config.txt`, ate 70 no 3 B+, com a
placa trabalhando mais quente.

Ar-condicionado na sala nao substitui o dissipador. Em 27/09/2026, com o ar em
21 C e sem vento direto no Pi, o limite chegou em 146 s em vez de 70 s e ficou
ativo em cerca de metade das leituras, contra tres quartos com a sala sem ar.
Abaixo do limite a velocidade e a mesma: o ar so adia o limite e reduz a perda.

No Pi do projeto o limite esta em 70 C desde 27/09/2026. Em 01/10/2026, em
seis rodadas de 3,5 min com uma pessoa, o Pi chegou a 63,9 C, com
`throttled=0x0` em todas as leituras e o clock nunca em 1,2 GHz. De 01 para
02/10/2026, em 10,5 h seguidas com o perfil atual, a temperatura ficou entre 56
e 63,4 C, com `throttled=0x0` em todas as 1318 leituras; uso por dias nao foi
medido. A copia do `config.txt` original ficou ao
lado; para voltar aos 60 C:

```bash
sudo cp /boot/firmware/config.txt.bak /boot/firmware/config.txt && sudo reboot
```

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

`FACE_EMBED_FULL_FRAME`, ligado no perfil rpi3 desde 03/10/2026, gera o
embedding no frame original em vez da imagem reduzida pela `PROCESS_SCALE`; a
deteccao continua na imagem reduzida. Vale so com `FACE_PREFILTER=1`. No Pi,
com o video de carga, a semelhanca com o cadastro subiu 0,008 na mediana e ate
0,089 no rosto mais dificil, sem mudar o tempo do frame; no PC, os rostos de
41 a 47 px subiram de 0,05 a 0,10. `FACE_EMBED_FULL_FRAME=0` volta a imagem
reduzida.

`FACE_DETECTOR_PATH` troca so o detector de rosto do `buffalo_l` (`det_10g`)
por outro `.onnx` do InsightFace, mantendo o reconhecedor `w600k_r50`: os
embeddings continuam compativeis com o cadastro, e ninguem precisa se
recadastrar. Vazio usa o detector do pacote. O candidato e o `det_500m`, do
pacote `buffalo_sc` dos releases v0.7 do InsightFace: no PC, em 04/10/2026,
detectou 6,7 vezes mais rapido, com os mesmos rostos no video de carga. Copie o
arquivo do PC, onde ele fica em `.insightface\models\buffalo_sc\` na pasta do
usuario:

```bash
scp $HOME/.insightface/models/buffalo_sc/det_500m.onnx citylab@<ip-do-pi>:.insightface/models/
```

E configure no `.env` do Pi:

```env
FACE_DETECTOR_PATH=~/.insightface/models/det_500m.onnx
```

A API para na subida se o arquivo faltar ou se nao for um detector.

`FACE_LEARN_FROM_STREAM`, ligado no perfil rpi3 desde 03/10/2026 e desligado no
default, faz o sistema aprender com o stream: um rosto reconhecido de um
embedding novo, com semelhanca de 0,60 ou mais e acima do limite contra o
proprio cadastro, vira referencia da pessoa. As referencias ficam na colecao `rostos_aprendidos` e na memoria da
API, ate `FACE_LEARNED_PER_PERSON` por pessoa (5), uma a cada
`FACE_LEARN_INTERVAL_SECONDS` (600), e o MongoDB apaga as com mais de
`FACE_LEARNED_RETENTION_DAYS` dias (30). Com as vagas cheias, uma nova troca a
referencia menos parecida com o cadastro, se for mais parecida que ela; a
quase igual a uma guardada, com 0,90 ou mais, so disputa a vaga dessa. Na
subida, a API carrega as referencias e apaga as de quem saiu do cadastro, as
que hoje nao entrariam e as copias. Para ver ou apagar:

```bash
cd ~/CityLab_Security && .venv/bin/python tools/limpar_aprendidos.py
cd ~/CityLab_Security && .venv/bin/python tools/limpar_aprendidos.py --tudo
```

Depois de apagar, reinicie a API, que guarda as referencias tambem na memoria.
O banco passa a guardar rostos tirados do stream sem a pessoa fazer nada, dado
biometrico na LGPD; `FACE_LEARN_FROM_STREAM=0` desliga. Com o aprendizado
ligado, as rodadas de medicao deixam de ser independentes: apague as
referencias antes de cada serie, ou desligue o aprendizado para medir outra
coisa.

Os cadastros ficam na colecao `alunos`, um documento por envio da pagina:
recadastrar um nome cria outro, e o antigo continua valendo. Para listar os
cadastros de cada pessoa, com fotos, data e referencias aprendidas, sem tirar o
embedding do banco; para apagar os antigos depois de um recadastro; e para
tirar uma pessoa, com as referencias aprendidas dela:

```bash
cd ~/CityLab_Security && .venv/bin/python tools/cadastros.py
cd ~/CityLab_Security && .venv/bin/python tools/cadastros.py --manter-ultimo "Nome"
cd ~/CityLab_Security && .venv/bin/python tools/cadastros.py --apagar "Nome"
```

O nome tem de ser igual ao da listagem, com maiusculas e acentos. Os eventos da
pessoa ficam e saem pelo prazo de `EVENT_RETENTION_DAYS`. Depois de apagar,
reinicie a API, que guarda os cadastros tambem na memoria. A pagina de cadastro
nao apaga nada: ela e aberta na rede, sem senha.

A equivalencia foi conferida em 27/09/2026 com os modelos reais, no PC: em 3
fotos de webcam com duas pessoas, o rosto aceito de cada foto teve a mesma
caixa, identidade, embedding e similaridade com e sem as flags. Para repetir,
compare nas mesmas fotos locais (rosto frontal, lateral, pequeno, varias
pessoas e cena vazia):

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

Exporte fora do Pi, num ambiente com Ultralytics; a exportacao instala `ncnn`
e `pnnx` se faltarem. Use uma copia do peso, porque a pasta sai ao lado dele:

```bash
python tools/export_pose_ncnn.py copia/yolov8n-pose.pt --imgsz 320 416
```

O NCNN roda na forma fixa do export, altura e largura. No perfil rpi3 o frame
de 640x480 e reduzido a 320x240 e o `.pt` o roda em 416 como 320x416; exportado
assim, o NCNN deu, nos 18 videos de validacao, os mesmos keypoints ate a quinta
casa decimal e os mesmos alertas. Exportado quadrado, em 416x416, a mao oculta
caiu de 182 para 152 fases, por causa das faixas de preenchimento. O servico
le a forma do `metadata.yaml` da pasta e ignora o `POSE_IMGSZ` nesse caso.

No Pi, instale o pacote no ambiente da API antes de ligar. Sem ele, o
Ultralytics tentaria compilar o ncnn a partir do Git na subida da API; por isso
o servico de gestos para antes, dizendo o que falta. Instale sem as
dependencias: o pacote pede `numpy` e `opencv-python` sem versao, e o pip troca
o numpy 1.26.4 pelo 2.x, que o mediapipe nao aceita, e poe o OpenCV 5 por cima
do que o projeto usa (aconteceu em 04/10/2026). O `portalocker`, que ele tambem
pede, e pequeno e entra a parte:

```bash
cd ~/CityLab_Security && .venv/bin/pip install --no-deps ncnn==1.0.20260526 portalocker==4.4.0
```

Os dois ficam comentados no `requirements-rpi-bookworm.txt`: quem monta o
ambiente do zero instala esse arquivo com dependencias, e o `ncnn` traria o
`opencv-python` junto. Por isso entram a mao, com o comando acima.

A pasta exportada fica fora do Git (`*_ncnn_model/` no `.gitignore`). Copie
para o Pi; o deploy nao a apaga, porque o `rsync` dele nao remove arquivos:

```bash
scp -r App/GestureRecon/yolov8n-pose_ncnn_model citylab@<ip-do-pi>:CityLab_Security/App/GestureRecon/
```

E configure no `.env` do Pi:

```env
POSE_MODEL_PATH=App/GestureRecon/yolov8n-pose_ncnn_model
```

Caminhos relativos sao resolvidos a partir da raiz do projeto. Deixe a variavel
vazia para voltar ao .pt. O export usa FP32 e nao altera o peso original. O
NCNN usa por padrao uma thread por nucleo fisico, 4 no Pi, sem o limite do
`TORCH_NUM_THREADS`, e disputa a CPU com o rosto. Medir no Pi antes de assumir
ganho, sem outra mudanca na mesma rodada.

Medido no Pi em 04/10/2026, com o video de carga e a fonte do Pi:

- Com as 4 threads do padrao do ncnn, o frame ficou 22% mais rapido, com os
  mesmos alertas frame a frame, mas a tensao caiu durante as rodadas:
  `vcgencmd get_throttled` deu `0x50005`, e o processador desceu para 600 MHz,
  o que nunca aconteceu com o `.pt`. Tensao baixa repetida arrisca o cartao SD.
- Com 2 threads, o frame ficou 40% mais rapido que com o `.pt`, com a mesma
  deteccao frame a frame e `throttled=0x0` em todas as leituras. O NCNN e o
  rosto somam 4 threads, uma por nucleo.

`NCNN_NUM_THREADS` e 2 no perfil rpi3 desde entao; zero volta ao padrao do
ncnn, uma thread por nucleo fisico. As convolucoes do ncnn fixam as threads ao
carregar a rede, entao o servico a recarrega uma vez, no primeiro frame, ja
com o valor. A metrica `ncnn_threads` mostra o valor usado em cada frame. Com o
NCNN ligado, confira `vcgencmd get_throttled` de vez em quando: qualquer valor
diferente de `0x0` e tensao baixa desde o ultimo boot, e o caminho e voltar ao
`.pt`, tirando o `POSE_MODEL_PATH` do `.env`.
