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

PC conferido em 28/10/2026: Intel Core i5-10400F (6 nucleos, 12 threads),
15,9 GB de RAM, NVIDIA GeForce GTX 1650 e Windows 11 Pro build 26300. O `.venv`
tem Python 3.13.15, NumPy 2.2.6, MediaPipe 0.10.33, InsightFace 0.7.3, ONNX
Runtime 1.23.2, Ultralytics 8.3.226, PyTorch 2.9.0, lapx 0.9.4 e psutil 7.1.3.
A inferencia roda so na CPU: `torch.cuda.is_available()` e falso, e o ONNX
Runtime so tem os provedores Azure e CPU.

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

## Decisoes em aberto

Para o responsavel e, quando mexer no texto, para o orientador:

1. **Tema.** As diretrizes do TC (secao 1.4.3) mantem no TC II o tema do TC I,
   salvo excecao justificada e autorizada. O TC I trata de "sistemas
   embarcados", e o titulo do TC II v1 ainda diz "sistema embarcado". Rodar no
   PC muda isso e as partes escritas para o Pi: titulo, resumo, problema,
   objetivo geral e as secoes 1.3, 2.5, 3.1, 3.4, 3.8, 3.9 e 4.5.
2. **Perfil e modelos no PC.** O `default` (det_10g, pose `.pt` em 640, pessoas
   pelo `yolov8n.pt`) ou a configuracao final do Pi (det_500m, pose NCNN
   320 x 416, passada unica).
3. **Onde aplicar o PDI.** No frame recebido, antes da reducao: cobre todos os
   modulos em qualquer perfil e custa 4 vezes os pixels. Ou no frame reduzido:
   mais barato, mas so vale com `FACE_EMBED_FULL_FRAME=0`. A grade do CLAHE e
   os nucleos do gaussiano e da mascara dependem da resolucao escolhida.
4. **Base de tempo dos gestos.** Proposta: tempo do video, numa cadencia fixa
   igual para todas as tecnicas, com o custo medido a parte. Hoje
   `build_frame_context` sempre usa o relogio, entao isso pede um parametro
   novo.
5. **Galeria.** Fixa, cadastrada sem PDI, como no uso real, ou refeita por
   tecnica.
6. **Temperatura no PC.** Outra fonte ou tirar a metrica do texto.
7. **Banco.** Um banco so do TCC ou o executor offline, sem banco.
8. **Participantes.** Os videos sao dado biometrico, e o TCC nao fala de
   consentimento.
