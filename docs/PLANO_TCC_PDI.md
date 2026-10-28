# Plano do TCC: pre-processamento de imagens no PC Windows

Branch `tcc-pdi-windows`, criada em 28/10/2026 a partir da `main` em `7ccf63c`.
Nela o CityLab Security roda num PC Windows local, e nao no Raspberry Pi 3 B+,
para os experimentos do TCC (TC II): o impacto de tecnicas de pre-processamento
digital de imagens (PDI) aplicadas antes da cadeia unificada sobre o
reconhecimento facial, o reconhecimento de gestos e o custo computacional.

A referencia do texto e a versao 1 do TC II, capitulos 1 a 4
(`tcc2_v1_cap3_cap4.docx`, 15/08/2026). Como nos outros planos, nada aqui esta
autorizado a ser implementado sem combinar com o responsavel, uma acao de cada
vez. O que e proposta esta escrito como proposta.

## Estado verificado em 28/10/2026

Lido no codigo da `main` em `7ccf63c`, sem rodar a API. Hardware, versoes e
testes conferidos no PC no mesmo dia.

### O texto descreve a versao de agosto

A secao 3.4 (cliente em 960 x 540 com JPEG 0,9; OpenCV do `python3-opencv`;
MediaPipe 0.10.18) corresponde a branch `raspberry-pi-version` (`be336ef`,
22/08/2026; `Client/index.js`, linhas 15 a 17). A `main` tem 76 commits que nao
estao nela. Entre as duas, `App/`, `Server/`, `Client/`, `tools/`, o
requirements e o `.env.rpi.example` mudaram em 31 arquivos, com 4.084 linhas
inseridas e 780 removidas.

### O que o texto diz e o que a main faz

| Secao | O TCC diz | Na main | Onde |
|---|---|---|---|
| 3.4 | o cliente envia 960 x 540 em JPEG 0,9 | 640 x 480, JPEG 0,65 e ate 10 FPS nos dois perfis, publicados pelo servidor em `/config/client` | `App/settings.py:146-150`, `Client/stream.js:25-31` |
| 3.4 | "fator de escala interno de 0,5" no rosto | `PROCESS_SCALE` 0,5 reduz o frame uma vez para a cadeia inteira: rosto, pessoas, pose e maos | `App/frame_context.py:44-66` |
| 3.4 | deteccao em 320 x 320 no InsightFace | confere: `buffalo_l` (det_10g e w600k_r50) em 320 x 320; o nome sai se a semelhanca de cosseno passar de 0,52, senao `NAO ALUNO` | `App/FaceRecon/service.py:61-64`, `:144`, `:686-690` |
| 1.3, 3.3 | pessoas detectadas por YOLO | no perfil `default`, o do PC, o `yolov8n.pt` roda em serie e decide se a pose roda: sem pessoa detectada nao ha gesto. No `rpi3` as pessoas saem da propria pose | `App/recognition_pipeline.py:143-176`, `App/GestureRecon/service.py:172-177` |
| 2.4, 3.3 | maos detectadas pelo MediaPipe | MediaPipe Tasks `GestureRecognizer` em modo IMAGE, no recorte da caixa de cada pessoa; para as regras so importa se a mao esta fechada | `App/GestureRecon/hand_detector.py:147-155` |
| 2.4, 3.7 | cinco classes de alerta | confere, com rotulos sem acento: `Braco Estendido`, `Rendicao`, `Mao Fechada`, `Mao Fechada + Braco Estendido` e `Mao Oculta`. Podem sair juntos no mesmo frame | `App/GestureRecon/detector.py:292-475` |
| 2.4 | regras temporais e geometricas | criterio duplo desde 29/09: duracao minima (8 s na mao oculta, 4 s na mao fechada, de 0,22 a 0,40 s nas outras) e de 2 a 4 observacoes seguidas. Braco estendido so de 45 a 135 graus da vertical; o punho so conta com o braco levantado | `App/GestureRecon/detector.py:57-65`, `:166-169` |
| 2.7, 3.8 | CPU, memoria e temperatura | por frame, so a memoria do processo e a temperatura. CPU e RAM em % vao so para o log, do sistema inteiro, a cada 5 s. No Windows a temperatura sai nula | `Server/system_monitor.py:43-92` |
| 3.4 | Python 3.11, NumPy 1.26.4, MediaPipe 0.10.18, OpenCV do sistema | e o ambiente do Pi; o do PC esta abaixo | `requirements-rpi-bookworm.txt` |

O texto tambem nao descreve recursos do rosto que vieram depois de agosto:
aprendizado com o stream (ligado nos dois perfis desde 05/10/2026), reuso da
identidade por 15 s (`rpi3`), reconhecimento em segundo plano com `VERIFICANDO`
(ligado pelo `.env` do Pi), aluno de perfil pendente, cadastro de 1 a 5 fotos
com a media, detector `det_500m` opcional (`FACE_DETECTOR_PATH`) e embedding
no frame original (`rpi3`).

### O que muda o resultado sem mudar a imagem

A secao 3.2 fixa modelos, pesos, limiares, resolucao, escala, paralelismo e
versoes. Quatro comportamentos da main ficam fora dessa lista e misturariam o
efeito da tecnica com outros efeitos:

1. **Aprendizado com o stream.** `FACE_LEARN_FROM_STREAM` vale verdadeiro nos
   dois perfis (`App/settings.py:92`). Um rosto reconhecido com semelhanca de
   0,60 ou mais vira referencia, entra no match no frame seguinte e e gravado
   em `rostos_aprendidos`, que a API recarrega na subida
   (`Server/main.py:214-215`). A galeria muda dentro da rodada e entre rodadas,
   e a ordem das tecnicas passa a mudar o resultado.
2. **Relogio das regras de gesto.** A duracao e medida com `time.monotonic()`,
   lido ao montar o frame (`App/frame_context.py:49`), e o contador soma o tempo
   real entre observacoes (`App/GestureRecon/detector.py:485-497`). No mesmo
   video, uma tecnica mais cara muda o frame em que o alerta confirma. No PC,
   com frames em decimos de segundo, a duracao decide todas as regras, que
   foram calibradas a 1,7 s por frame no Pi. `GESTURE_ANALYZER_FPS` e guardado
   e nunca lido (`App/GestureRecon/detector.py:46-47`).
3. **Descarte e recompressao na entrada.** Pelo navegador, o cliente so envia
   com vaga (no maximo 2 frames em voo no `default`) e a ate 10 FPS
   (`Client/stream.js:572`): tecnicas com custo diferente processam frames
   diferentes. O `tools/benchmark_stream.py` manda todos os frames, mas
   redimensiona para 640 x 480 sem manter a proporcao e recomprime em JPEG 65
   antes do servidor (`tools/benchmark_stream.py:130-131`).
4. **Estado do perfil `rpi3` e do `.env` do Pi.** Reuso da identidade,
   reconhecimento em segundo plano, gate de movimento e passada unica fazem a
   saida depender de frames anteriores ou da latencia. No `default` estao
   desligados. O `tools/run_rpi.py` liga o `rpi3` quando o `.env` nao define o
   perfil (`tools/run_rpi.py:45`).

No `default`, as threads do ONNX Runtime, do PyTorch e do OpenCV ficam no
automatico (`App/settings.py:99-108`), com rosto e gesto rodando em paralelo.

### O que muda com o PC Windows

A configuracao do PC esta em "Ambiente de teste", abaixo. A inferencia roda so
na CPU: `torch.cuda.is_available()` e falso, e o ONNX Runtime so tem os
provedores Azure e CPU.

- Nao ha arquivo de dependencias do PC; o repositorio so fixa o Pi.
- Tres distribuicoes de OpenCV estao instaladas juntas: `opencv-python` e
  `opencv-python-headless` 4.12.0.88 e `opencv-contrib-python` 4.13.0.92. O
  `cv2` carregado e o 4.13.0. As cinco tecnicas usam OpenCV.
- O monitor le a temperatura de `/sys/class/thermal` ou de
  `psutil.sensors_temperatures`, e nenhum dos dois existe no Windows:
  `temperature_c` sai nulo. O criterio de validade do Pi (`throttled=0x0`) nao
  tem equivalente no PC.
- O `.env` do PC desliga rosto e gesto (`CITYLAB_ENABLE_FACE_SERVICE=0` e
  `CITYLAB_ENABLE_GESTURE_SERVICE=0`), aceita cadeia parcial
  (`CITYLAB_ALLOW_PARTIAL_PIPELINE=1`) e nao liga as metricas. No `default`,
  sem o servico de rosto nao ha lista de pessoas, e o gesto nunca roda.
- A subida da API faz `ping` no MongoDB sem tratar erro
  (`Server/main.py:185-186`) e carrega os cadastros da colecao `alunos`. O
  `.env` do PC aponta para um MongoDB remoto (`mongodb+srv`) com o banco
  `recon-db`. Nao foi conferido se e o mesmo banco do Pi; se for, uma rodada no
  PC grava nele eventos com recorte do rosto e referencias aprendidas.
- Testes no PC: 152 de 155 testes Python e os 12 de Node passam. As tres falhas
  sao de `tests/test_deploy_dependencies.py`, que roda o script de deploy do Pi
  com o `bash` do PATH; aqui o primeiro e o do WSL
  (`C:\WINDOWS\system32\bash.exe`). A causa nao foi isolada.
- O deploy so dispara com push na `main` (`.github/workflows/deploy.yml:4-6`).
  Um merge desta branch na `main` levaria tudo ao Pi.

### Ambiente de teste

Conferido no PC em 28/10/2026, para a secao 3.4.

| Item | Configuracao |
|---|---|
| Processador | Intel Core i5-10400F, 2,90 GHz base, 6 nucleos e 12 threads, L2 de 1,5 MB e L3 de 12 MB |
| Memoria | 16 GB (2 x 8 GB TeamGroup UD4-3200), operando a 2400 MT/s |
| GPU | NVIDIA GeForce GTX 1650, driver 32.0.16.2101; a inferencia nao a usa |
| Armazenamento | SSD NVMe Kingston SNV3S500G de 500 GB (C:, sistema) e disco externo USB de 1 TB (D:, repositorio e `.venv`) |
| Sistema | Windows 11 Pro 64 bits, versao 10.0.26300; plano de energia Equilibrado |
| Python | 3.13.15 |
| Bibliotecas | NumPy 2.2.6; OpenCV 4.13.0 (carregado o `opencv-contrib-python` 4.13.0.92; `opencv-python` e `opencv-python-headless` 4.12.0.88 tambem instalados); MediaPipe 0.10.33; InsightFace 0.7.3; ONNX Runtime 1.23.2 (CPU); Ultralytics 8.3.226; PyTorch 2.9.0 (CPU); ncnn 1.0.20260526; lapx 0.9.4; psutil 7.1.3 |
| Modelos | configuracao final do Pi: det_500m (buffalo_sc) e w600k_r50 (buffalo_l) no rosto, YOLOv8n-pose em NCNN 320 x 416 na pose, `gesture_recognizer.task` do MediaPipe nas maos |

O que da para medir de recursos no PC, conferido no `.venv`:

- CPU e memoria do processo, pelo psutil (`cpu_percent` e `memory_info`). Hoje
  a memoria do processo ja sai por frame (`process_rss_mb`, com
  `ENABLE_PERFORMANCE_METRICS=1`), e a CPU so vai para o log, medida no
  sistema inteiro (`Server/system_monitor.py:52-74`).
- Leitura e escrita em disco do processo (`io_counters`).
- GPU: o `nvidia-smi` esta instalado, mas a inferencia roda so na CPU, entao o
  uso da GPU nao reflete a tecnica.
- Temperatura: o psutil nao a le no Windows.

### O que ainda nao existe para o protocolo do TCC

- **Modulo de PDI.** Nenhuma das cinco tecnicas esta no codigo. O unico gancho,
  `EXPERIMENTAL_GRAYSCALE`, age depois da reducao e so no frame reduzido
  (`App/frame_context.py:54-57`). Com `FACE_EMBED_FULL_FRAME` (ligado no
  `rpi3`) o embedding sai do frame original e escaparia de um PDI posto ali.
- **Executor offline.** Nada roda a cadeia unificada sobre um video sem
  servidor, WebSocket e MongoDB. O `tools/record_gesture_sequence.py` cobre so
  o gesto, e o `tools/check_face_optimization.py`, so o rosto em imagens.
- **Saida por frame com identidade.** O `tools/benchmark_stream.py` grava
  contagens e nomes de alerta, sem nome nem caixa de rosto, sem a pessoa de cada
  alerta e sem o tempo do video. Acuracia, precisao e F1 do rosto exigem
  identidade e caixa por rosto, pseudonimizadas.
- **Anotacao e metricas.** Nao ha formato de anotacao (identidade por rosto,
  intervalo por gesto) nem calculo de precisao, revocacao e F1. Os planos de
  gesto medem "fases", nao F1.
- **Metadados por rodada.** O coletor nao grava commit, configuracao efetiva,
  tecnica, parametros, CPU do processo nem identificador da rodada. Nao ha tag
  no repositorio.
- **Base de videos.** Os conjuntos existentes sao de gesto, so com o
  responsavel, e o video de carga tem 36 frames. Nao ha cenarios de
  iluminacao, distancia e ruido nem pessoas nao cadastradas, como pede a secao
  3.6. O `.gitignore` nao cobre arquivos de video.

## Decisoes do responsavel em 28/10/2026

- **Plataforma.** O Raspberry Pi nao e usado neste TCC, que apresenta os
  testes e o resultado feitos no PC. As partes do texto escritas para o Pi
  precisam ser reescritas: titulo ("sistema embarcado"), resumo, problema,
  objetivo geral e as secoes 1.3, 2.5, 3.1, 3.4, 3.8, 3.9 e 4.5. A secao 1.4.3
  das diretrizes do TC pede que a mudanca em relacao ao tema do TC I
  ("sistemas embarcados") seja justificada.
- **Perfil.** O do Pi: `CITYLAB_PROFILE=rpi3`, com a configuracao final do Pi
  (det_500m, pose NCNN 320 x 416, passada unica). Com ele vem ligado o
  embedding no frame original, o reuso do nome do rosto por 15 s e o gate de
  movimento.
- **Onde aplicar o PDI.** No frame recebido, antes da reducao pela
  `PROCESS_SCALE`: o frame original e o reduzido saem tratados, e o embedding
  no frame original tambem ve a tecnica.
- **PC no texto.** O PC e o ambiente de teste e aparece com a sua
  configuracao ("Ambiente de teste", acima). O uso de recursos (CPU, memoria,
  GPU e disco) entra como metrica, junto com o tempo por quadro e o tempo do
  PDI. A temperatura fica fora do texto.
- **Galeria.** Os cadastros que ja estao no MongoDB que o Pi usava, sem PDI,
  iguais para todas as tecnicas e congelados durante a coleta: aprendizado
  desligado e nenhum cadastro novo. Na ultima listagem registrada, em
  04/10/2026, eram quatro pessoas, cada uma com um cadastro de uma foto
  (`docs/PLANO_OTIMIZACAO.md:695-698`). Com uma foto so, o rosto cadastrado
  ficou perto do limite de 0,52 em 02/10/2026 (README, "Reconhecimento perto
  do limite"). A classe de rejeicao da secao 3.7 depende de aparecerem nos
  videos rostos que nao estao na galeria.
- **Banco.** O mesmo MongoDB que o Pi usava. Falta conferir se o `MONGO_DETAILS`
  do `.env` do PC aponta para ele. Com o aprendizado desligado, a API nao
  carrega as referencias aprendidas (`Server/main.py:214-215`).
- **Caminho dos testes.** Executor offline: um script que carrega a mesma
  cadeia e o mesmo perfil, le a galeria do banco do Pi uma vez, no comeco da
  rodada, e nao grava nada nele. Ele le o video quadro a quadro, aplica o PDI,
  usa a base de tempo escolhida e grava um registro por quadro em arquivo
  local, sem servidor, WebSocket nem recompressao JPEG. Pela API, o servidor
  gravaria no banco eventos com recorte do rosto (`Server/event_logger.py:162`).
- **Imagens de rosto no texto.** So as do responsavel.
- **Base de tempo.** Tempo do video no ritmo do Pi: o executor pega um quadro
  a cada 1,7 s de video e passa a cadeia o instante dele no video. As regras
  de gesto, o reuso do nome do rosto (15 s no `rpi3`), o aluno de perfil e o
  teto do gate de movimento contam o tempo por esse instante, que hoje e o
  relogio do PC (`App/frame_context.py:49`, `App/FaceRecon/service.py:253`,
  `App/GestureRecon/service.py:182-183`). Assim o custo de cada tecnica nao
  muda os alertas. As duracoes atuais (8 s na mao oculta, 4 s na mao fechada)
  foram validadas de 1,0 a 3,4 s por quadro (`docs/PLANO_GESTOS.md`, "Decisao:
  mao oculta 8 s, mao fechada 4 s"). Nesse ritmo a mao oculta confirma uns
  8,5 s depois de comecar, entao os gestos gravados precisam durar mais que
  isso.

## Proposta de ordem de trabalho

Cada passo e combinado antes, com testes e um commit proprio:

1. **Tempo do quadro e ponto do PDI na cadeia.** `process_frame` passa a
   aceitar o instante do quadro e uma etapa de PDI aplicada ao frame recebido,
   antes de `build_frame_context`, com o tempo em `pdi_ms`. Sem os dois, nada
   muda na API nem no Pi.
2. **Modulo de PDI.** As cinco tecnicas e as combinacoes, com parametros
   configuraveis e valores iniciais da literatura ate o ensaio piloto (secao
   3.5): equalizacao, CLAHE e gama no canal de luminancia; gaussiano e mascara
   de nitidez no quadro BGR.
3. **Executor offline.** Galeria lida do banco; perfil do Pi com aprendizado e
   reconhecimento em segundo plano desligados; um quadro a cada 1,7 s de
   video, com o instante dele no video; registro por quadro com CPU, memoria
   e disco do processo; cabecalho da rodada com commit, configuracao efetiva,
   versoes, hardware e hash do video e dos modelos. A combinar neste passo: o
   quadro inicial muda quais quadros entram, e os testes de gesto de outubro
   rodaram todas as fases (17 por video a 1,7 s, `docs/PLANO_GESTOS.md:108-113`).
4. **Anotacao e metricas.** Formato da anotacao (identidade por rosto,
   intervalo por gesto) e o calculo de acuracia, precisao, revocacao e F1.
5. **Ambiente do Windows.** Requirements do PC com versoes fixas e uma unica
   distribuicao de OpenCV, e o `.env` do experimento.

### Passo 1 em 28/10/2026

- `build_frame_context` aceita `observed_at`; sem ele, segue o relogio
  (`App/frame_context.py`).
- `UnifiedRecognitionService` aceita um `preprocessor`, aplicado uma vez ao
  frame recebido, antes da reducao, com o tempo em `pdi_ms`. Ele precisa
  devolver o frame no mesmo formato e tipo; senao a cadeia para com
  `ValueError`. `process_frame` aceita `observed_at`
  (`App/recognition_pipeline.py`). Sem os dois, a API e o Pi seguem iguais:
  o servidor chama `process_frame(frame)` (`Server/main.py:94`).
- Correcao no rosto: o instante 0 virava o relogio do PC
  (`App/FaceRecon/service.py`). Com o tempo do video, a janela do reuso ficava
  negativa a partir do primeiro frame, e o nome nunca vencia. O teste novo
  falha sem a correcao.
- Testes: `tests/test_pdi_pipeline.py` (6) e um em
  `tests/test_optimization_safety.py`. No PC, 159 de 162 testes Python e os 12
  de Node passam; as 3 falhas sao as de antes, do script de deploy do Pi
  (`tests/test_deploy_dependencies.py`).

### Passo 2 em 28/10/2026

- `App/preprocessing.py`: as tecnicas `nenhuma`, `equalizacao`, `clahe`,
  `gama`, `gaussiano` e `nitidez`, e a classe `Preprocessor`, que aplica uma
  sequencia na ordem dada e entra na cadeia como o `preprocessor` do passo 1.
- Equalizacao, CLAHE e gama mexem so no Y do YCrCb; gaussiano e mascara de
  nitidez agem no frame BGR. Cada tecnica devolve o frame no mesmo formato e
  tipo.
- Valores iniciais, a fixar no ensaio piloto (secao 3.5): CLAHE com limite 2,0
  e grade 8 x 8; gama 0,8 (saida = 255 x (entrada/255)^gama, abaixo de 1
  clareia); gaussiano 5 x 5 com sigma 1,0; mascara de nitidez com sigma 1,0 e
  quantidade 0,5.
- Uma configuracao se escreve como `clahe+gaussiano` ou
  `gama:gamma=0.6+nitidez:amount=1.0`; `Preprocessor.config` devolve tecnicas,
  parametros efetivos e ordem para o registro da rodada. Tecnica ou parametro
  desconhecido para com `ValueError`.
- Testes: `tests/test_preprocessing.py` (8). No PC, 167 de 170 testes Python e
  os 12 de Node passam; as 3 falhas sao as do script de deploy do Pi.

### Passo 3 em 28/10/2026

- `tools/executor_pdi.py VIDEO --pdi CONFIG --saida ARQUIVO.jsonl`: forca o
  perfil `rpi3`, os dois servicos, `FACE_LEARN_FROM_STREAM=0`,
  `FACE_ASYNC_RECOGNITION=0` e `YOLO_AUTOINSTALL=False`, com a pose NCNN e o
  `det_500m` por padrao. Le os cadastros da colecao `alunos` uma vez, so
  leitura, e nao grava nada no banco.
- Pega um quadro a cada `--intervalo` segundos de video (1,7 por padrao), a
  partir de `--inicio` (a fase), reduz a 640 px de largura mantendo a
  proporcao, sem JPEG, e passa a cadeia o instante do quadro no video. Antes
  da medicao, `--aquecimento` passadas no primeiro quadro, e depois historico
  de gesto, nomes e rastreador zerados.
- Saida em JSONL: cabecalho (rodada, commit, configuracao de PDI e efetiva,
  video com sha256, galeria com contagem e sha256, hash dos modelos, versoes e
  maquina), um registro por quadro (rostos com nome, caixa, semelhanca e
  pendente; gestos com track e alertas; metricas da cadeia com `pdi_ms`; CPU,
  memoria e disco do processo) e um resumo com media e desvio. A pasta
  `resultados/tcc-pdi/` e os videos ficam fora do Git, porque tem nomes e
  rostos.
- Rodada de teste no PC com `mao_oculta_r1.avi` (30 s, 30 FPS, 640 x 480) e
  CLAHE: 18 quadros, 4 cadastros na galeria (o mesmo numero da listagem do Pi
  em 04/10), rosto conhecido em 18/18, mao oculta em 14 quadros, `pdi_ms` medio
  2,4 ms e quadro medio de 80 ms. O reuso do nome venceu no quadro de 15,3 s,
  pelo tempo do video. Quatro rodadas deram rostos e alertas identicos quadro
  a quadro, com quadro medio de 79 a 81 ms.
- Achado: no primeiro quadro, o InsightFace (pelo `skimage`) e a pose importam
  o `scipy` cada um na sua thread, e o import circular derrubou uma rodada. O
  executor importa os dois antes, na thread principal. A API tem o mesmo
  arranjo de threads; nao foi visto la nem corrigido.
- A combinar: a fase da amostragem (`--inicio`).
- Testes: `tests/test_executor_pdi.py` (7), sem modelos nem banco. No PC, 174
  de 177 testes Python passam; as 3 falhas sao as do script de deploy do Pi.

### Passo 4 em 28/10/2026

- `tools/metricas_pdi.py RODADA.jsonl ANOTACAO.json`: compara uma rodada do
  executor com a anotacao do video. O formato da anotacao esta no topo do
  arquivo: pessoas com o nome do cadastro (ou `null`, nao cadastrada) e os
  trechos, em segundos de video, em que o rosto aparece; gestos com classe,
  inicio e fim.
- Rosto, por quadro amostrado: nome esperado presente e TP, ausente e FN; nome
  reconhecido sem par esperado e FP (aceitacao falsa, nome trocado ou deteccao
  espuria); nao cadastrado com um `NAO ALUNO` para ele e TN. Rosto pendente
  conta como `NAO ALUNO`. Saem acuracia, precisao, revocacao, F1 e as
  contagens. O casamento e por nome, sem a caixa: a anotacao nao precisa de
  caixas, mas duas pessoas trocadas no mesmo quadro nao se distinguem.
- Gesto, por classe: trecho anotado com o alerta em algum quadro dentro dele,
  ou ate `--tolerancia` s depois do fim (1,7 por padrao), e TP; sem o alerta, FN.
  Cada sequencia de quadros seguidos com o alerta fora de todo trecho e um FP.
  O trecho da ameaca vale tambem para braco estendido e mao fechada. A media
  usa so as classes com trecho anotado; os alarmes das outras saem em
  `alarmes_sem_trecho`. Classes comparadas sem acento e sem caixa.
- Teste de ponta a ponta com a rodada do passo 3 e uma anotacao provisoria
  (mao oculta no video inteiro), so para exercitar o caminho: rosto 18 de 18,
  mao oculta 1 de 1, e a mao fechada de um quadro saiu como alarme sem trecho.
  Nao e resultado: o video nao foi anotado.
- Testes: `tests/test_metricas_pdi.py` (5). No PC, 179 de 182 testes Python
  passam; as 3 falhas sao as do script de deploy do Pi.
