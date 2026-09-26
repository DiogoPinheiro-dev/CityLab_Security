# Plano de otimizacao do stream (Raspberry Pi 3 B+)

Documento compartilhado entre os agentes que trabalham neste repositorio (Codex e
Claude) e o responsavel pelo projeto. Quem for mexer em performance le este
arquivo antes e atualiza o estado aqui depois. Nao duplicar o plano em outro
lugar: este e a fonte unica **de latencia e vazao**.

As regras de gesto, ou seja, quando um alerta dispara e o que ele significa,
ficam em `docs/PLANO_GESTOS.md`. A medicao de 21/09/2026 mostrou que aquilo e
problema de comportamento do produto, nao de desempenho, e que ficar mais
rapido nao resolve.

## Objetivo

Melhorar latencia e vazao do stream sem perder deteccoes de rostos, gestos ou
alertas. Trabalhar uma acao por vez, medindo no Pi antes e depois.

Ponto de partida medido em 19/09/2026, commit `b05058f`, pipeline completo:
5,41 s por frame na cena vazia, 13,41 s com uma pessoa e 16,62 s com duas,
entre 0,06 e 0,18 FPS. A decomposicao por estagio esta no "Balanco da fase 1" e
a fila de trabalho no "Backlog de otimizacao". Se as acoes 3 e 4 do backlog
entregarem o que a decomposicao sugere, o frame de duas pessoas cai para algo
entre 6 e 11 s: estimativa derivada da medicao, nao meta acordada.

## Regras de trabalho

1. Medicao antes de otimizacao. A linha de base da fase 1 foi fechada em
   19/09/2026. Daqui em diante cada acao do backlog e combinada com o
   responsavel antes de ser implementada, uma de cada vez.
2. Planejar nao autoriza implementar. O backlog e uma fila de propostas.
3. Toda mudanca e comparada com a mesma configuracao e com a cena ao vivo
   mais estavel possivel:
   mediana e p95 de `rtt_ms`, `completed_fps`, tempos por estagio,
   `process_rss_mb`, `temperature_c` e comportamento dos alertas.
4. Preservar recall. Ganho de tempo que perde deteccao nao e ganho.
5. Atualizar a secao "Estado verificado" deste arquivo ao concluir uma etapa.
6. Criterio de significancia: so tratar como ganho real uma variacao de
   latencia acima de 5% que se repita nas tres rodadas do mesmo cenario. Com 30
   amostras o p95 serve para observar caudas, nao para decidir.
7. Nao comparar contagem de alertas entre versoes com desempenho diferente. A
   conversao para tempo decorrido nao resolveu isso: com o frame custando muito
   mais que os limiares, as cinco regras colapsaram em "gesto presente em duas
   observacoes". Ver `docs/PLANO_GESTOS.md`.


## Estado verificado em 26/09/2026

### Duas pessoas no perfil rpi3 com gate - ganho confirmado

Commit medido: `3e67f56`. O Pi recebeu o deploy de `ea97065`, que so muda
documentacao e resultados; os hashes de `App/settings.py`,
`App/recognition_pipeline.py` e `App/GestureRecon/service.py` foram conferidos
no dispositivo. Mesma configuracao da serie de 21/09, conferida com
`tools/run_rpi.py --show-config` e com `GESTURE_MOTION_GATE=1` no `.env`, que o
`--show-config` nao exibe. Resultados em
`resultados/pi3-3e67f56-gate/duas-pessoas-r1.json` a `r3.json`. Cena igual a da
linha de base: duas pessoas reais, uma de frente e outra de lado. API
reiniciada antes de cada rodada.

| Cenario | Linha de base | Agora | Ganho |
|---|---|---|---|
| Duas pessoas | 16615,63 ms | 7597,0 / 7568,6 / 7740,2 | **-53,4% a -54,4%** |

- Amplitude entre rodadas: 2,3%, contra 2,7% da linha de base.
- Vazao: 0,059 a 0,062 FPS para 0,135 a 0,144 FPS.
- **Recall preservado nas 90 amostras**: 2 pessoas em 90/90, 2 tracks de gesto
  em 90/90, 1 rosto em 89/90 e 2 rostos em 1/90. A linha de base tambem tinha
  mediana de 1 rosto, porque a pessoa de lado nao gera rosto aceito. Nenhuma
  terceira caixa, contra a pessoa a mais sistematica da linha de base.
  Confianca da segunda pessoa entre 0,430 e 0,894.
- Duas pessoas custam 4,4% a mais que uma: 7597,0 contra 7276,8 ms, mediana das
  medianas. O rosto, entre 5,3 e 5,5 s, roda inteiro em paralelo com a pose, com
  residuo de 3,1 ms nas tres rodadas. O caminho critico agora e a pose.
- `process_rss_mb` entre 628,2 e 692,3 MB, abaixo do teto de 719 MB da fase 1.
  **Mas o sistema ja usa swap**: `vmstat 5` durante a r2 mostrou cerca de
  236 MB em swap e 59 MB livres, com rajadas de ate 2,4 MB/s de saida para o
  swap. Na maior parte dos intervalos a troca ficou em zero, entao nao e
  thrashing continuo. O runner do GitHub Actions estava ligado e ocupa parte
  dessa memoria. O RSS do processo sozinho subestima o risco de OOM.
- `temperature_c` entre 52,6 e 59,1 C. `vcgencmd get_throttled` devolveu `0x0`:
  nenhuma reducao de frequencia por temperatura ou alimentacao desde o boot.
- Alertas: 56, 52 e 42 por rodada, presentes em 85 de 90 frames. E o colapso
  descrito em `docs/PLANO_GESTOS.md` e nao serve para comparar versoes.

### Pose bimodal - a latencia depende de qual worker roda o gesto

- A pose tem duas velocidades, sem meio-termo: entre 5,0 e 5,3 s ou entre 6,94
  e 6,99 s. Com a pose rapida o frame fica em 6,0 a 6,3 s; com a lenta, em 7,6
  a 7,8 s. Foram 30 de 90 frames na fase rapida, 5, 14 e 11 por rodada.
- Nao e processador mais lento: quando a pose fica lenta, o rosto fica mais
  rapido, de cerca de 5,7 para 5,3 s. `get_throttled` descarta temperatura e
  alimentacao, e no `vmstat` a rajada de swap veio antes da troca de fase da
  r2, com a pose ainda rapida.
- Mecanismo observado: `_process_parallel` submete primeiro o rosto e depois o
  gesto ao `ThreadPoolExecutor` de 2 workers. O worker que terminou por ultimo
  no frame anterior recebe o gesto no frame seguinte, entao os papeis ficam
  presos e so trocam quando o rosto termina depois do gesto. Essa regra previu
  85 de 87 transicoes nas tres rodadas; as duas falhas sao os primeiros frames
  da r2. A troca da r1 (frame 10) e a da r3 (frame 24, logo apos o unico frame
  com 2 rostos, com 8806 ms de rosto) vem exatamente depois de um frame em que o
  rosto terminou por ultimo.
- Leitura: um dos dois workers roda a pose cerca de 1,4 vez mais rapido que o
  outro, embora os dois executem `configure_torch_threads(2)` no inicializador.
  A causa nao foi isolada; a hipotese principal e o numero efetivo de threads
  do PyTorch em cada worker. A serie de uma pessoa de 21/09 confirma a regra:
  pose entre 6934 e 7138 ms em todos os 90 frames, e em nenhum deles o rosto
  terminou depois do gesto, entao os papeis nunca trocaram.
- Oportunidade, nao ganho medido: com o gesto sempre na configuracao rapida, o
  frame de duas pessoas cairia de cerca de 7,6 s para 6,0 a 6,3 s, perto de 20%.
  Exige registrar worker e threads por frame antes de mexer em qualquer coisa.

### Promocao a padrao do perfil rpi3

Combinada com o responsavel apos a serie de duas pessoas. No perfil rpi3,
`PIPELINE_SHARED_PERSON_POSE` e `GESTURE_MOTION_GATE` passam a vir ligados por
padrao em `App/settings.py`, e `.env.rpi.example` deixa de desliga-los. O
perfil default nao muda, porque nao foi medido. Valor explicito no ambiente
continua prevalecendo, entao `=0` volta ao caminho anterior para comparar.
`tools/run_rpi.py --show-config` passou a exibir `GESTURE_MOTION_GATE` e
`GESTURE_PUBLISH_MIN_CONFIDENCE`; antes, o gate que decide o ganho da cena
vazia so aparecia com `grep` no `.env`. No Pi o comportamento nao muda, porque
o `.env` de la ja define as duas chaves como `1`.

### O que fica aberto

1. **Pose bimodal**, acima. Proxima acao candidata de desempenho, a combinar.
2. **Memoria do sistema.** Medir com o runner parado para separar o que e dele,
   e decidir se o runner fica ligado como servico.
3. Limiares de gesto, em `docs/PLANO_GESTOS.md`.
4. Acoes 6, 7, 8 e 9 do backlog seguem sem medicao isolada. A equivalencia dos
   embeddings da acao 6, ligada por padrao desde 20/09, nunca foi conferida com
   `tools/check_face_optimization.py` nos modelos reais.

## Estado verificado em 21/09/2026

### Gate de movimento e limiar de publicacao - ganho confirmado nos dois cenarios

Commit medido: `3e67f56`, perfil rpi3 completo, `GESTURE_MOTION_GATE=1`,
`GESTURE_PUBLISH_MIN_CONFIDENCE=0.25`. Resultados em
`resultados/pi3-3e67f56-gate/`. API reiniciada antes de cada rodada, iniciada
por `tools/run_rpi.py` com TLS.

| Cenario | Linha de base | Agora | Ganho |
|---|---|---|---|
| Uma pessoa | 13413,5 ms | 7276,8 / 7271,7 / 7287,7 | **-45,7% a -45,8%** |
| Cena vazia | 5409,98 ms | 1676,3 / 1679,1 / 1677,2 | **-69,0% nas tres** |

- Amplitude entre rodadas: 0,22% com uma pessoa e 0,17% na cena vazia. A linha
  de base tinha 1,7% e 2,1%. A configuracao e mais reprodutivel que ela.
- Vazao: 0,075 para 0,137 FPS com uma pessoa; 0,184 para 0,491 FPS na cena vazia.
- **Recall preservado nas 90 amostras com uma pessoa**: rosto 30/30, pessoa
  30/30 e gesto 30/30 nas tres rodadas, caixa entre 0,844 e 0,917, zero alertas.
- **Caixa fantasma eliminada**: zero pessoa, zero gesto e zero alerta nas 90
  amostras de cena vazia, contra 74 de 90 antes das correcoes.
- `process_rss_mb` entre 641,4 e 694,0 MB, abaixo do teto de 719 MB da fase 1 e
  bem abaixo dos 748,5 MB medidos sem o gate. Nao rodar a pose libera os buffers
  do modelo, o que tambem reduz o risco de OOM na placa de 906 MB.
- `temperature_c` entre 47,2 e 55,8 C, contra 59 a 60 C da linha de base.

### Defeito do gate, medido e corrigido no mesmo dia

- A primeira versao decidia so por movimento. Com uma pessoa sentada parada, o
  `motion_ratio` ficou entre 0,00000 e 0,0017, abaixo do limiar de 0,002, e a
  pose foi pulada em 22 de 30 frames. O rosto apareceu em 30 de 30, entao a
  pessoa estava presente o tempo todo: em 73% dos frames ela nao existia para o
  pipeline de pose, sem caixa, sem track e sem analise de gesto. A evidencia
  esta em `resultados/pi3-5325cff-gate/evidencia/`.
- Correcao: o gate passou a exigir tambem que a cena nao esteja ocupada. A
  passada de pose e a leitura confiavel de ocupacao, e o pipeline repassa o
  rosto do frame via `note_external_presence`, porque rosto e gesto correm em
  paralelo e so o resultado do frame anterior chega a tempo da decisao.
- Verificacao no hardware: nas tres rodadas com uma pessoa, `pose_skipped` ficou
  em 0 de 30 mesmo com 26, 20 e 3 frames abaixo do limiar de movimento. A r3,
  com a pessoa se movendo mais, deu a mesma latencia das outras duas: o
  resultado nao depende de quanto a pessoa fica parada.
- Na cena vazia o gate segue pulando 28 de 30 frames. Os outros 2 sao a passada
  forcada por `GESTURE_MOTION_MAX_SKIP_SECONDS`, e explicam o p95 alto: nao e
  cauda anomala, e a protecao funcionando.

### O que fica aberto

1. **Cenario de duas pessoas nunca foi medido com este perfil.** E onde o
   reconhecimento facial escala e onde a RAM aperta.
2. **Limiares de gesto continuam colapsados.** A 7,4 s por frame o intervalo
   entre analises ainda e 18 vezes maior que o maior limiar, de 0,40 s. Punho,
   rendicao, mira e ameaca permanecem indistinguiveis, e o ganho de desempenho
   nao resolve: as regras so voltam a discriminar perto de 3 a 5 FPS. E defeito
   de comportamento do produto, nao de latencia.
3. Acoes 6, 7, 8 e 9 do backlog seguem sem medicao. A acao 5 ficou obsoleta:
   com a passada unica as caixas vem do modelo de pose, e o limiar de publicacao
   ja cobre o caso.

## Estado verificado em 20/09/2026

### Cena vazia no perfil rpi3 - latencia neutra, deteccao reprovada

- Tres rodadas em `resultados/pi3-970a384-rpi3full/vazia-r1.json` a `r3.json`.
  Medianas 5179,4; 5200,8; 5175,1 ms contra 5409,98 da linha de base: -3,9% a
  -4,3%, com 0,5% de amplitude. **Nao atinge os 5%: a cena vazia nao melhorou
  nem piorou.** A previsao de que a passada unica a deixaria mais lenta nao se
  confirmou, porque o paralelismo absorveu a troca do detector pela pose.
- **Caixa fantasma reprova o cenario.** 74 de 90 frames reportaram uma pessoa
  numa cena declarada vazia pelo responsavel, contra 0 de 90 na linha de base.
  Zero rostos nos 90 frames confirma que nao havia ninguem. Foram criados 74
  tracks de gesto numa sala vazia. Os alertas ficaram em zero, mas por
  coincidencia dos keypoints, nao por protecao.
- Causa identificada: `GestureRecognitionService` chama `pose_model.track()` sem
  `conf=`, e o Ultralytics 8.3.226 forca `conf=0.1` nesse modo, porque o
  ByteTrack precisa de deteccoes fracas na associacao de segundo estagio. O
  `detect_persons` da passada dupla usava o default de predict, 0,25. **Trocar a
  passada dupla pela unica baixou o limiar de deteccao de 0,25 para 0,10 como
  efeito colateral da API, sem decisao de projeto.**
- Agravante: `latest_persons` e o laco de `people` em `App/GestureRecon/service.py`
  sao montados direto de `result.boxes`, sem filtro. As caixas que existem so
  para alimentar o tracker saem publicadas como pessoas detectadas, e a analise
  de maos e as regras de gesto rodam sobre elas.
- **Um limiar de publicacao nao resolve sozinho.** Simulacao sobre os dados:
  a 0,25 sobrariam 17/30 frames na r1, 3/30 na r2 e 1/30 na r3; a 0,50 ainda
  sobrariam 14/30 na r1, com caixas chegando a 0,804. O modelo de pose produz
  falso positivo de alta confianca onde o `yolov8n.pt` nao produz na mesma cena.
- `process_rss_mb` ficou entre 721,8 e 743,9 MB mesmo sem ninguem em cena,
  acima do teto de 719 MB da fase 1.
- Combinado com o cenario de uma pessoa: a passada unica entrega -45% com gente
  e neutralidade com a cena vazia, ao custo de um track fantasma em 82% dos
  frames ociosos. **Nao promover a padrao sem tratar o fantasma.**

### Acoes combinadas com o responsavel em 20/09/2026

1. Gate barato antes da pose, por diferenca de frames. Camera ociosa nao tem
   movimento, entao a pose nao roda e o fantasma nao aparece; o ganho com gente
   e preservado. Risco a cobrir: pessoa parada nao pode ficar invisivel.
2. Limiar de publicacao separado da entrada do tracker: manter `conf=0.1`
   alimentando o ByteTrack e filtrar o que e publicado como pessoa e o que entra
   na analise de gesto. Corrige um limiar que mudou sem decisao.

Ambas combinadas apos a medicao da cena vazia, para serem medidas em seguida.

### Perfil rpi3 completo - ganho confirmado com uma pessoa

- **As medicoes anteriores deste dia rodaram com as otimizacoes desligadas.** O
  `.env` do Pi, que nao e versionado, tinha `PIPELINE_RUN_IN_PARALLEL=0`,
  `PIPELINE_MAX_WORKERS=1`, nenhum `CITYLAB_PROFILE` e a passada unica desligada.
  A API tambem subia por uvicorn direto, sem os limites BLAS/OpenMP que o
  `tools/run_rpi.py` aplica antes dos imports nativos. Das acoes implementadas,
  so as de rosto e as de gesto estavam ativas.
- Correcao do registro anterior: o paralelismo nao estava falhando por
  contencao, estava desligado na configuracao. Com a flag em 0 o
  `ThreadPoolExecutor` nem chega a ser criado.
- Configuracao medida agora, conferida com `tools/run_rpi.py --show-config`:
  `CITYLAB_PROFILE=rpi3`, ONNX 1, PyTorch 2, OpenCV 1, nativas 1,
  `PIPELINE_RUN_IN_PARALLEL=1`, `PIPELINE_MAX_WORKERS=2`,
  `PIPELINE_SHARED_PERSON_POSE=1`, `MAX_IN_FLIGHT_FRAMES=1`.
- Tres rodadas com uma pessoa, protocolo da fase 1, resultados em
  `resultados/pi3-970a384-rpi3full/`. Medianas de `rtt_ms`: 7380,0; 7359,1;
  7527,3 ms contra 13413,5 da linha de base. **Ganho de 43,9% a 45,1% repetido
  nas tres rodadas**, com 2,3% de amplitude. Vazao de 0,075 para 0,132 a 0,135 FPS.
- **Recall preservado nas tres**: rosto em 30/30 frames, gesto em 30/30, zero
  alertas, caixa de pessoa entre 0,877 e 0,914.
- `persons_ms` foi a zero: a passada unica eliminou o detector separado. O
  paralelismo passou a sobrepor de fato, com `pipeline_ms` menos
  `persons_ms + max(faces_ms, gestures_ms)` em 3,2 ms nas tres rodadas. No
  perfil default o mesmo calculo dava 1986 ms e batia com a soma dos estagios.
- p95 entre 7554 e 7708 ms contra mediana entre 7359 e 7527: dispersao de 2,4%.
  No perfil default o p95 chegava a 15188 contra mediana de 13328, 14%.
- **Risco aberto: RAM.** `process_rss_mb` chegou a 748,5 MB numa placa de 906 MB,
  acima do teto de 719 MB da fase 1. Rosto e pose sobrepostos mantem buffers dos
  dois modelos vivos ao mesmo tempo.
- Falta medir com este perfil: cena vazia e duas pessoas. A cena vazia pode
  piorar por construcao, porque passa a pagar a pose no lugar do detector de
  pessoas, e e a condicao mais comum de uma camera ociosa. **Nao promover este
  perfil a padrao antes dessas duas medicoes.**

### Cenario de uma pessoa no perfil default - sem ganho

- Tres rodadas em `resultados/pi3-970a384-webcam/uma-pessoa-r1.json` a `r3.json`.
  Medianas 12927,4; 13328,3; 13360,6 ms, entre -3,6% e -0,4% contra a linha de
  base. Nao atinge os 5% exigidos.
- Ganho estavel por estagio: `pose_ms` em 4852,6; 4867,9; 4860,7 ms contra
  5204,5 a 5299,0, ou -7,0% a -8,3% com 0,3% de amplitude. `logs_ms` caiu de
  cerca de 52,8 para 35 ms. Somados, cerca de 0,5 s, ou 3,8% do frame.
- `faces_ms` nao e estavel entre rodadas: 1983,6; 2710,5; 1952,6 ms com o mesmo
  codigo e enquadramento equivalente. Depende do tamanho e do angulo do rosto.
  Nao atribuir ganho a esse estagio com webcam ao vivo.
- Duas rodadas foram descartadas como medicao de latencia por cena
  inconsistente e ficaram em `resultados/pi3-970a384-webcam/descartadas/`.
- Protocolo acrescentado: antes de cada rodada cheia, uma sondagem de 3 frames
  sem aquecimento confirma o enquadramento pela confianca da caixa. Custa cerca
  de 40 s e evita perder uma rodada inteira por cena fora do padrao.

### Regras de gesto por tempo decorrido - efeito medido (acao 2)

- A conversao esta implementada, mas **nao tornou os alertas comparaveis**. Os
  limiares sao de 0,20 a 0,40 s e o intervalo real entre analises e de 13,2 s no
  perfil default e 7,4 s no perfil rpi3. Em `_update_counter`,
  `history = min(limit, history + elapsed)` satura no primeiro incremento.
- Efeito pratico: as cinco regras colapsam em "gesto presente em dois frames
  consecutivos". A distincao entre punho, rendicao, mira e ameaca deixa de
  existir neste hardware, e o ganho de 45% nao resolve: as regras so voltam a
  discriminar perto de 3 a 5 FPS.
- Na linha de base, contando frames, o limiar equivalia a 10 frames, ou 132 s de
  gesto continuo, e por isso ela nunca alertava. A rodada `uma-pessoa-r3.json`
  do perfil default registrou 7 alertas em 7 frames consecutivos, confirmados
  pelo responsavel como gesto real. O cooldown de 5 s nao deduplica quando o
  frame custa 13 s: cada frame gravou um evento com recorte de imagem.
- A contagem de alertas continua sem servir para comparar versoes, agora por
  excesso de sensibilidade em vez de falta.

### Caixa fantasma - evidencia medida (acao 5)

- `detect_persons` chama o YOLO sem `conf=`, entao vale o default do Ultralytics,
  0,25. As caixas falsas medidas ficaram entre 0,252 e 0,692.
- Custo com uma pessoa em cena: desprezivel. Frames com duas caixas contra
  frames com uma diferiram em +97,9 ms de `faces_ms`, +40,6 ms de `pose_ms` e
  +1,0 ms de `hands_ms`, tudo dentro do ruido.
- Custo em cena vazia: alto. Os 5 frames com caixa falsa da rodada vazia r1
  pularam de 5,3 s para 10,3 a 14,8 s, porque a caixa liga a pose num frame que
  senao pularia o gesto inteiro. Penalidade de 2x a 3x.
- Nenhum limiar unico se sustentou: 0,50 nao perdeu frame na cena bem enquadrada
  e perderia 4 de 30 na cena fraca. Com enquadramento bom a separacao e limpa
  abaixo de 0,40.
- A acao 5 se reposiciona: nao e otimizacao de latencia com gente em cena, e
  protecao de cena ociosa e de alerta falso. Fica obsoleta se a passada unica for
  promovida, porque as caixas passam a vir do modelo de pose.

### Ferramenta

- `tools/run_rpi.py` aceita `--ssl-certfile` e `--ssl-keyfile`. Sem isso o
  launcher oficial do perfil do Pi nao subia HTTPS, e a alternativa era exportar
  `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` e
  `NUMEXPR_NUM_THREADS` no shell antes do uvicorn. Esquecer o export faz o
  servidor subir sem erro e sem os limites, que foi o que mascarou as medicoes
  deste dia. O par e validado antes do dotenv e o JSON reporta so o booleano
  `tls`, nunca os caminhos.

### Medicao apos deploy - cena vazia

- O workflow da `main` (`970a384`) concluiu com sucesso. A API foi iniciada
  manualmente no Pi com HTTPS e `.venv`; o workflow nao reinicia a API. O startup
  completo confirmou rosto e gesto ativos, MongoDB de teste `recon-db` e 4
  cadastros. `CITYLAB_ALLOW_PARTIAL_PIPELINE=0`, `ENABLE_PERFORMANCE_METRICS=1`
  e `ENABLE_SYSTEM_MONITOR=1`. O hash do arquivo de codigo no Pi ainda nao foi
  conferido; o SHA nos rotulos identifica o deploy esperado.
- Tres rodadas com webcam local (indice 0), cena vazia declarada pelo responsavel,
  640x480, JPEG 65, 5 frames de aquecimento e 30 medidos, uma pendencia por vez.
  A API foi reiniciada antes de cada rodada. Resultados em
  `resultados/pi3-970a384-webcam/vazia-r1.json` a `vazia-r3.json`.
- Medianas de `rtt_ms`: 5995,50; 5296,94; 5259,38 ms. Vazao: 0,145; 0,175;
  0,186 FPS. A linha de base `b05058f` teve 5300,14; 5409,98; 5412,11 ms.
  Nao houve ganho acima de 5% repetido nas tres rodadas.
- Na primeira rodada, 5 de 30 frames retornaram uma pessoa e um track de gesto
  na cena vazia, com confianca de pessoa entre 0,253 e 0,382; nenhum alerta.
  Esses frames executaram pose e elevaram o p95 a 12010,40 ms. As rodadas 2 e 3
  nao tiveram deteccoes. A linha de base teve zero nas tres rodadas vazias.
  A webcam nao repete frames; investigar a caixa falsa sem elevar o limiar antes
  de testar pessoas reais e recall. P95 nao decide ganho com 30 amostras.
- A API permaneceu online ao fim das tres rodadas. Ainda faltam os cenarios com
  uma e duas pessoas, a comparacao da passada unica e a verificacao de recall.

### Integracao na main

- `otimizations-tests` (`5618a6e`) foi integrada a `main` a partir de
  `343d8da`, com preferencia pela branch de otimizacao em conflitos de conteudo.
  O merge automatico terminou sem conflitos pendentes. As mudancas exclusivas
  de `main` no aviso de camera em contexto seguro e no README foram preservadas.
- O workflow passou a sincronizar em `/home/citylab/CityLab_Security`, caminho
  verificado no Pi em 19/09/2026, e verifica a existencia da pasta antes do
  `rsync`. O workflow instala dependencias, mas nao reinicia o processo da API.
- Validacao local apos o merge: 39 testes Python, 3 testes do cliente e
  compilacao dos arquivos Python passaram. Modelos reais, MongoDB, runner e
  camera no Pi nao foram exercitados nesta etapa. O push na `main` exige conferir
  a execucao do workflow e o processo ativo antes de afirmar que houve deploy.

### Integracao na branch otimizations-tests

- `codex/raspberry-stream-optimization` (`e1585e2`) foi integrada em
  `otimizations-tests` por fast-forward a partir de `53de153`, sem conflitos.
- Apos a integracao, passaram 39 testes Python e 3 testes do cliente;
  `git diff --check 53de153..e1585e2` nao apontou erros.
- Esta integracao nao alterou `main` nem executou deploy ou nova medicao no Pi.
  Ganho de desempenho e recall ainda exigem validacao no dispositivo.

### Fechamento da implementacao local para Pi 3 B+

- `tools/run_rpi.py` inicia o perfil com um processo e sem reload. Prepara
  limites de BLAS/OpenMP antes dos imports nativos, mantendo overrides.
  `--show-config` imprime somente opcoes de desempenho, sem credenciais.
- `NATIVE_NUM_THREADS` centralizado em settings: 1 no perfil rpi3, 0 no default.
  PyTorch tambem configurado no inicializador dos workers de inferencia.
- Falha em um modelo paralelo agora aguarda o outro terminar antes de liberar
  o estado compartilhado, evitando inferencia pendente na proxima chamada.
- Coletor inclui confiancas de caixas de pessoas/gestos por amostra, sem nomes
  ou imagens; os limiares nao foram elevados. Isso prepara a avaliacao da caixa
  fantasma com evidencia, em vez de supor um valor melhor.
- Validacao final local: 39 testes Python e 3 testes do cliente, sintaxe 3.11
  e diff-check OK. Modelos reais, MongoDB e camera nao foram exercitados aqui.
- Responsavel nao conhece o endereco SSH do Pi nesta sessao. O pacote local
  esta pronto para teste; concluir ganho de desempenho e recall exige medir no
  dispositivo. Nao houve deploy, exportacao NCNN nem nova medicao real.
  Inicializacao: `python tools/run_rpi.py`. Nao promover os experimentos de
  passada unica/NCNN sem a comparacao de deteccoes descrita neste plano.

### Ajuste especifico para Raspberry Pi 3 B+

- Hardware confirmado pelos registros do projeto: Pi 3 B+, Cortex-A53 com
  quatro nucleos a 1,4 GHz e 1 GB de RAM; Bookworm 64-bit, Python 3.11.
- Adicionado perfil opt-in `CITYLAB_PROFILE=rpi3` e `.env.rpi.example`.
  Defaults do perfil: ONNX intra-op 1, PyTorch intra-op 2, OpenCV 1 thread,
  um frame pendente no cliente. Variaveis explicitas prevalecem sobre o perfil.
  Sao candidatos para medicao, nao uma distribuicao garantida de todos os
  threads do processo nem a configuracao vencedora comprovada no hardware.
- Preservados pesos, resolucao, qualidade JPEG, filtros faciais e o detector
  separado de pessoas como padrao. A passada unica permanece opt-in.
- Corrigida a sobreposicao de sessoes ONNX durante o ajuste de threads: remover
  a referencia antiga e coletar antes de construir a substituta. Teste com
  weakref verifica a ordem; nao mede pico de RAM nativa ou devolucao ao SO.
  Falha de recriacao interrompe o carregamento conforme a politica de pipeline.
- Validacao atual: 35 testes Python e 3 do cliente passaram. Nao houve deploy
  nem medicao nova no Raspberry. Comparar os ajustes individualmente antes de
  adotar o perfil completo. O pipeline requer um worker e uma camera por processo.

### Implementacao inicial

- Pedido atual: implementar otimizacoes com base no PDF em branch nova.
  Branch local `codex/raspberry-stream-optimization`, criada a partir de
  `origin/otimizations-tests` (`53de153`), que contem a linha de base citada.
- Acao 1 implementada: falha de insert e registrada sem sair do stream;
  cooldown so comeca apos sucesso, usa relogio monotono e remove entradas
  expiradas na proxima consulta. Nao ha fila nem repeticao automatica do mesmo
  payload; a proxima deteccao pode tentar novamente. Cancelamento continua
  propagando. Eventos durante indisponibilidade do banco podem nao ser salvos.
- Acao 2 implementada: acumuladores usam segundos entre observacoes do track,
  com um timestamp por frame de gesto. Limiares: mao oculta 0,35 s, rendicao
  0,30 s, braco estendido 0,40 s, mao fechada 0,20 s e ameaca 0,22 s.
  `GESTURE_ANALYZER_FPS` permanece aceito por compatibilidade, mas nao determina
  mais os limiares. Os nomes internos `*_frames` agora guardam segundos.
  A primeira observacao ativa nao herda tempo anterior; a seguinte acumula o
  intervalo se o gesto continuar ativo. Decaimentos tambem usam segundos, com
  as mesmas taxas relativas anteriores. Tracks ausentes perdem o historico.
  Isso supoe continuidade entre amostras ativas: nao recupera gestos que a
  camera/servidor nao amostraram. Pausas longas com o mesmo track tambem entram
  no intervalo. Os limiares exigem validacao funcional no Pi.
- Acao 3 implementada como experimento reversivel:
  `PIPELINE_SHARED_PERSON_POSE=1` usa caixas/confiancas da pose para pessoas,
  mantendo coordenadas originais e a analise de maos. O YOLO separado de pessoas
  so e carregado se um caminho de fallback precisar dele. A configuracao
  funciona em modo sequencial e paralelo, inclusive em cena vazia. Sem gesto
  ativo/disponivel, usa o detector de pessoas existente.
  Padrao `0`, ate confirmar qualidade e desempenho no hardware.
- Validacao local sem modelos: testes de falha/repeticao/cooldown/cancelamento,
  tempo de gesto, passada unica, fallback, coordenadas e cena vazia.
  Resultado: 16 testes Python e 3 testes do cliente passaram; sintaxe dos
  arquivos Python validada para 3.11. Os testes Python rodam no host 3.13,
  com dependencias nativas substituidas por doubles nos contratos de servico.
  A execucao real dos modelos, o recall e o ganho no Raspberry ainda nao foram
  medidos nesta branch; nao ha resultado de performance novo nem deploy.
- Acoes 4 a 9 continuam pendentes. Nao foram alterados limiares dos detectores,
  modulos do InsightFace, resolucoes, threads internas ou transporte.
  Proximo passo de medicao: comparar `PIPELINE_SHARED_PERSON_POSE=0` e `1`
  nesta mesma branch, com as regras temporais iguais nos dois casos, seguindo
  `docs/BENCHMARK.md`. Nao comparar contagens de alertas com a regra antiga.

### Complemento implementado apos a revisao (20/09/2026)

O responsavel autorizou as otimizacoes propostas, condicionadas a preservar o
proposito do software. Este registro atualiza o estado acima; os registros da
primeira implementacao e da revisao abaixo permanecem como historico.

- P1 corrigido: acumuladores limitados ao limiar de cada gesto; intervalo
  superior a 60 s ou timestamp regressivo limpa o track. O primeiro frame
  valido de cada conexao limpa o historico de gestos (o cliente fecha o socket
  ao pausar). Espera por frame superior a 5 s tambem limpa o historico, sem
  contar tempo de inferencia nesse limite. Configuracao em
  `GESTURE_MAX_OBSERVATION_GAP_SECONDS` e `GESTURE_IDLE_RESET_SECONDS`.
  O timestamp e obtido ao preparar o frame, antes dos modelos, evitando incluir
  variacoes do tempo de reconhecimento facial na observacao de gesto.
- P2 corrigido: reserva por identidade enquanto o insert esta pendente, sem
  bloquear outras identidades. Sucesso marca cooldown; falha/cancelamento
  liberam a reserva. Continua sem garantia de entrega durante falha do banco.
- InsightFace: `FACE_MINIMAL_MODULES=1` (novo padrao) mantem detection e
  recognition. `FACE_PREFILTER=1` (novo padrao) executa os mesmos filtros de
  tamanho/confianca antes dos embeddings. Nao muda pesos, escala, det_size,
  filtros nem limiar de similaridade. O cadastro continua usando get na imagem
  completa. As duas flags podem ser desligadas independentemente para A/B.
- CPU: `ONNX_INTRA_OP_THREADS` e `TORCH_NUM_THREADS` configuraveis, padrao 0
  preserva o comportamento da biblioteca. InsightFace 0.7 nao encaminha
  sess_options; por isso o ajuste ONNX recria as sessoes explicitamente no
  startup, antes de prepare, preservando pesos e providers. Ha custo transitorio
  de RAM nesse startup; medir antes de adotar no Pi.
- NCNN: `POSE_MODEL_PATH` aceita o diretorio exportado do mesmo modelo de pose.
  `tools/export_pose_ncnn.py` prepara exportacao CPU/FP32 opcional. O padrao
  continua sendo o arquivo .pt versionado; nao foi exportado ou ativado NCNN.
- `tools/check_face_optimization.py` compara respostas, identidades, caixas e
  embeddings nas mesmas imagens usando uma copia dos modelos. Nao salva dados
  biometricos; uma rodada sem rostos aceitos e inconclusiva. Precisa ser rodado
  no ambiente com modelos reais antes de validar equivalencia em producao.
- Validacao local: 31 testes Python e 3 do cliente passaram; sintaxe Python
  3.11 verificada em 18 arquivos. Inclui concorrencia, cancelamento, pausa,
  inferencia de 17 s, retirada de alerta, resposta facial equivalente com
  menos embeddings, limites dos filtros, construtor/cadastro e rollback.
  Dependencias nativas continuam substituidas por doubles nesses testes.
- Pendentes no hardware: equivalencia real dos embeddings, todos os gestos,
  recall de pessoas/rostos, consumo de memoria e tres rodadas por cenario.
  Nenhuma melhora de latencia foi medida nesta etapa. A passada unica continua
  opt-in. Nao reduzir resolucao nem elevar confianca sem outra comparacao.
- O pipeline ainda possui rastreador global e se destina a uma camera por
  processo; isolamento completo de multiplas cameras nao faz parte desta etapa.

### Revisao do commit 01153c0 (20/09/2026)

Revisao solicitada apos o push. Nenhuma nova otimizacao implementada nesta
revisao. Os 16 testes Python e 3 testes do cliente continuam passando, mas
reproducoes adicionais encontraram dois defeitos nao cobertos:

- P1, `App/GestureRecon/detector.py`: duas observacoes ativas do mesmo track
  em t=0 e t=600 acumulam 600 segundos, mesmo se o stream ficou pausado nesse
  intervalo. Em t=600,05, com mao nao detectada como fechada e nao visivel, o
  acumulador ainda vale 599,7 e emite `Mao Fechada`. O estado nao expira por
  inatividade e os acumuladores nao possuem teto. Corrigir descontinuidade de
  sessao/pausa e limitar a evidencia acumulada; incluir testes de pausa,
  retorno, perda de visibilidade e retirada de alerta. Nao escolher um timeout
  menor que o intervalo normal de inferencia do Pi sem medir.
- P2, `Server/event_logger.py`: duas chamadas concorrentes para a mesma
  identidade passam por `_should_log` antes que o primeiro insert termine.
  Reproduzido com `asyncio.gather` e insert que cede o event loop: duas
  gravacoes dentro do cooldown. Proteger a identidade enquanto o insert esta
  em andamento, liberando em sucesso, falha e cancelamento; manter cooldown
  definitivo somente apos sucesso. O logger e global e aceita varios sockets.

Os testes atuais extraem classes por AST e usam doubles; nao validam imports,
construtores reais, qualidade de caixas da pose nem integracao dos modelos.
Ainda falta a comparacao A/B no Pi. A passada unica segue experimental.

Oportunidades para combinar e medir separadamente apos as correcoes:

1. `allowed_modules=['detection', 'recognition']` no InsightFace (acao 6).
   O codigo oficial v0.7 executa todos os modulos habilitados por rosto; o
   ArcFace alinha usando `face.kps` fornecidos pelo detector. Conferir os
   embeddings e o cadastro na versao instalada antes de promover a mudanca.
2. Antecipar o filtro de qualidade facial existente: hoje `FaceAnalysis.get`
   calcula embeddings antes de `_validate_face` descartar rostos. Separar
   deteccao, filtro atual e reconhecimento evitaria trabalho em rostos que ja
   seriam descartados, sem elevar limiares. Ganho depende da cena; nenhum
   ganho esperado se todos os rostos passarem no filtro.
3. Comparar paralelismo e limites de threads ONNX/PyTorch (acao 4), registrando
   configuracao efetiva. O compartilhamento de pose ja permite submeter rosto
   e pose sem esperar pelo detector separado; concorrencia nao garante ganho
   nos quatro nucleos do Pi.
4. Avaliar exportacao do mesmo YOLOv8-pose para NCNN em experimento posterior,
   mantendo entrada e thresholds para comparar caixas, keypoints e tracking.
   A documentacao atual lista suporte a pose YOLOv8, mas a compatibilidade com
   as versoes instaladas no Pi ainda precisa ser validada.

Fontes tecnicas consultadas:
- https://raw.githubusercontent.com/deepinsight/insightface/v0.7/python-package/insightface/app/face_analysis.py
- https://raw.githubusercontent.com/deepinsight/insightface/v0.7/python-package/insightface/model_zoo/arcface_onnx.py
- https://onnxruntime.ai/docs/performance/tune-performance/threading.html
- https://docs.ultralytics.com/integrations/ncnn/

### Registro historico de 19/09/2026

- No inicio da validacao, branch `otimizations-tests`, commit `b05058f`,
  working tree limpo. As adaptacoes locais para webcam e este plano ainda
  estao em trabalho; o Pi continua no commit `b05058f`.
- `CityLab_Security/otimizations-tests` aponta para o mesmo `b05058f`.
- `main` esta em `343d8da` e **nao contem** `b05058f` (merge-base `08aae9b`).
  O deploy (`.github/workflows/deploy.yml`) so dispara em push na `main`.
  A instrumentacao foi colocada no Pi manualmente em 19/09/2026; isso nao
  valida o caminho de deploy do workflow.
- No Pi, o checkout real fica em `/home/citylab/CityLab_Security` (o caminho
  `/home/citylab/Desktop/Projects/CityLab_Security` nao existe nessa maquina).
  A branch local `otimizations-tests` foi avancada por fast-forward de
  `26a7e0b` para `b05058f`, sem commits locais exclusivos; o working tree
  terminou limpo. O ambiente `.venv` usa Python 3.11.2, OpenCV 4.6.0 do sistema
  e `psutil` 5.9.4. InsightFace, Ultralytics e MediaPipe importaram no Pi.
- A API foi iniciada manualmente a partir de `b05058f` e respondeu HTTP 200
  usando o banco de teste `recon-db`. No teste ao vivo com camera no computador
  via tunel SSH, a tela mostrou 2 pessoas, 1 rosto e um Round Trip pontual de
  7337 ms. A configuracao usada tinha gestos desativados e pipeline parcial
  permitido. Pausar e retomar o stream funcionou. Cinco linhas consecutivas do
  monitor registraram CPU entre 80,9% e 86,1%, RAM em 72,2%, temperatura entre
  59,61 e 60,15 C, `avg_fps` de 0,12 e `avg_frame_ms` entre 7786,72 e 7832,20.
  Essas leituras do teste ao vivo nao sao a linha de base reproduzivel.
- O startup do pipeline completo tambem foi confirmado no Pi: processo Uvicorn
  `1017`, MongoDB de teste conectado, modelos InsightFace e gesto carregados,
  4 cadastros em memoria e `Application startup complete`. A inicializacao foi
  feita com `CITYLAB_ALLOW_PARTIAL_PIPELINE=0` e
  `CITYLAB_ENABLE_GESTURE_SERVICE=1`; os dois indicadores de desempenho ficaram
  habilitados no `.env` raiz. Isso valida o carregamento dos servicos, ainda
  nao a deteccao nem o desempenho do gesto em frames reais.
- Testes locais rodados em 19/09/2026 neste commit: `2 passed` em
  `python -m unittest discover -s tests -p 'test_*.py'` e `3 passed` em
  `node --test tests/client_stream.test.cjs`.
- O responsavel optou por medir com a webcam, sem gravar videos. Um piloto de
  1 frame de aquecimento e 3 medidos gerou `.tmp/webcam-pilot.json` no cliente
  Windows, fora do Git. Todos os frames medidos retornaram 1 pessoa, 1 rosto,
  1 rastreamento de gesto e 0 alertas. Medianas: `rtt_ms` 12832,57,
  `pipeline_ms` 12740,71, `persons_ms` 4533,48, `faces_ms` 2764,74,
  `gestures_ms` 5297,67, `process_rss_mb` 694,29 e temperatura 54,23 C;
  vazao 0,078 FPS. Essas tres amostras apenas validam o coletor e o pipeline
  completo; nao fecham a linha de base da fase 1.
- Primeira rodada valida da linha de base, em 19/09/2026 as 17:14 UTC:
  `resultados/pi3-b05058f-webcam/duas-pessoas-r1.json`, rotulo
  `pi3-b05058f-full-webcam-2p-r1`, cena de varias pessoas com 2 pessoas reais
  (uma de frente, outra de lado), 5 frames de aquecimento e 30 medidos em
  485,51 s. Vazao 0,062 FPS. Medianas: `rtt_ms` 16351,30 (p95 17671,77),
  `pipeline_ms` 16235,68, `persons_ms` 4760,28, `faces_ms` 5328,19,
  `gestures_ms` 6156,84, `process_rss_mb` 706,09 e temperatura 59,07 C
  (p95 59,61). Deteccao por frame: mediana de 2 pessoas (p95 3), 1 rosto
  (p95 2), 2 rastreamentos de gesto (p95 3) e 1 alerta (p95 5).
  A rodada foi executada com o rotulo errado (`one-person`) e corrigida depois
  para `many-persons`; o JSON guarda `label_correction` com o motivo e o SHA-256
  do relatorio original. O arquivo original ficou em `.tmp/`, fora do Git.
  Uma rodada isolada nao fecha a linha de base.
- Rodada de cena vazia em 19/09/2026 as 17:41 UTC:
  `resultados/pi3-b05058f-webcam/vazia-r1.json`, rotulo
  `pi3-b05058f-full-webcam-empty-r1`, 30 frames medidos em 162,00 s e vazao
  0,185 FPS. Medianas: `rtt_ms` 5300,14 (p95 6069,57), `pipeline_ms` 5279,26,
  `persons_ms` 4569,45, `faces_ms` 705,05, `gestures_ms` 0,00, `decode_ms` 7,35,
  `logs_ms` 0,02, `receive_wait_ms` 17,72, `process_rss_mb` 561,60 e temperatura
  59,61 C. Nenhum dos 30 frames retornou pessoa, rosto, gesto ou alerta, o que
  confirma a cena vazia.
- Ambiente alvo: Raspberry Pi OS Legacy 64-bit Bookworm, Python 3.11,
  `requirements-rpi-bookworm.txt` (ver `docs/RASPBERRY_PI.md`).

## Fase 1 - instrumentacao e linha de base (FECHADA em 19/09/2026)

### Ja implementado

- Cliente: timestamp por frame pendente (`state.pendingSentAt`) em
  `Client/index.js`.
- Servidor: tempos separados em `Server/main.py` - `receive_wait_ms`,
  `decode_ms`, `pipeline_ms`, `logs_ms`, `response_ready_ms`, `send_ms`,
  `total_ms`, `effective_fps`.
- Recursos: `Server/system_monitor.py` fornece `process_rss_mb` e
  `temperature_c` por frame; CPU e RAM percentuais aparecem **apenas** no log do
  monitor, nao no JSON do benchmark.
- Utilitario `tools/benchmark_stream.py` aceita video fixo ou webcam local;
  procedimento em `docs/BENCHMARK.md`.

### Rodadas coletadas e protocolo usado

Progresso das rodadas (atualizar a cada execucao). Arquivos em
`resultados/pi3-b05058f-webcam/`, todos com 5 frames de aquecimento e 30
medidos, commit `b05058f`, pipeline completo:

| Cenario | Rodada | Arquivo | FPS | rtt mediana (ms) | Deteccao mediana por frame |
|---|---|---|---|---|---|
| varias pessoas (2 reais) | r1 | `duas-pessoas-r1.json` | 0,062 | 16351,30 | 2 pessoas, 1 rosto, 2 gestos, 1 alerta |
| varias pessoas (2 reais) | r2 | `duas-pessoas-r2.json` | 0,059 | 16794,78 | 3 pessoas, 1 rosto, 3 gestos, 2 alertas |
| varias pessoas (2 reais) | r3 | `duas-pessoas-r3.json` | 0,060 | 16615,63 | 3 pessoas, 1 rosto, 3 gestos, 2 alertas |
| cena vazia | r1 | `vazia-r1.json` | 0,185 | 5300,14 | 0 em todos os 30 frames |
| cena vazia | r2 | `vazia-r2.json` | 0,183 | 5409,98 | 0 em todos os 30 frames |
| cena vazia | r3 | `vazia-r3.json` | 0,184 | 5412,11 | 0 em todos os 30 frames |
| uma pessoa | r1 | `uma-pessoa-r1.json` | 0,075 | 13340,14 | 1 pessoa, 1 rosto, 1 gesto, 0 alertas nos 30 frames |
| uma pessoa | r2 | `uma-pessoa-r2.json` | 0,075 | 13413,53 | 1 pessoa, 1 rosto, 1 gesto, 0 alertas; 1 frame com 2 pessoas |
| uma pessoa | r3 | `uma-pessoa-r3.json` | 0,073 | 13567,14 | 1 rosto e 1 gesto nos 30 frames; 9 frames com 2 pessoas |

**Varias pessoas fechada** com as tres rodadas, todas com 2 pessoas reais.
Linha de base do cenario: `rtt_ms` mediana 16615,63 (rodadas entre 16351,30 e
16794,78, amplitude 2,7%), `pipeline_ms` 16498,96, `persons_ms` 4760,28,
`faces_ms` 5337,39, `gestures_ms` 6376,94 (`pose_ms` 5287,70 e `hands_ms`
1038,96), `logs_ms` 85,71, `process_rss_mb` 713,49 e vazao entre 0,059 e
0,062 FPS. A medicao foi interrompida a pedido do responsavel com a r3 em
andamento e retomada em seguida; o coletor nao publica execucao incompleta,
entao a rodada descartada nao deixou arquivo.

**Uma pessoa fechada** com as tres rodadas. Linha de base do cenario:
`rtt_ms` mediana 13413,53 (rodadas entre 13340,14 e 13567,14, amplitude 1,7%),
`pipeline_ms` 13340,26, `persons_ms` 4748,25, `faces_ms` 3044,71,
`gestures_ms` 5539,38 (`pose_ms` 5234,59 e `hands_ms` 286,16), `logs_ms` 52,74,
`process_rss_mb` 701,15 e vazao entre 0,073 e 0,075 FPS.

Falso positivo do detector de pessoas, confirmado em 19/09/2026: na r2 de
varias pessoas, com apenas 2 pessoas na sala segundo o responsavel, o detector
devolveu mediana de 3 pessoas (um frame com 4) e 3 rastreamentos de gesto,
contra mediana de 2 e 2 na r1. Nessa rodada a caixa extra custou trabalho:
`hands_ms` subiu 22,6% (963,42 para 1180,87 ms) e `gestures_ms` 5,8%, porque
cada caixa vira um track e uma regiao de maos a analisar. A mediana de alertas
dobrou, de 1 para 2.

Consequencia que vai alem de desempenho: alertas gerados por caixa fantasma sao
gravados no MongoDB como eventos `ALERTA_GESTO`, com recorte de imagem. O
sistema registra alerta de gente que nao existe. Tratar isso junto com a fase 4,
e medir o efeito de um limiar de confianca no detector tanto em tempo quanto em
alertas falsos.

Lacuna do coletor para essa investigacao: o relatorio guarda a contagem de
pessoas, nao a confianca de cada caixa. Sem isso nao da para escolher o limiar
pelos dados ja coletados. Incluir a confianca no coletor depois de fechar a
linha de base, para nao trocar a ferramenta no meio da medicao.

Impureza registrada: com uma pessoa so no enquadramento, o detector devolveu
duas pessoas em 1 frame da r2 e em 9 frames da r3. Nesses frames o rosto e o
rastreamento de gesto continuaram em 1, `hands_ms` nao mudou (286,48 contra
285,72 ms) e o `rtt` nao subiu: a caixa extra nao virou trabalho nem track. E
falso positivo do detector de pessoas, nao contaminacao dos tempos. Vale como
dado para a fase 4, ao mexer em confianca e tamanho de entrada do YOLO.

**Cena vazia fechada** com as tres rodadas, todas com a API reiniciada antes e
nenhuma deteccao em nenhum dos 90 frames. Linha de base do cenario:
`rtt_ms` mediana 5409,98 (rodadas entre 5300,14 e 5412,11, amplitude 2,1%),
`pipeline_ms` 5387,22, `persons_ms` 4651,92 (amplitude 1,9%), `faces_ms` 725,11,
vazao entre 0,183 e 0,185 FPS.

Repetibilidade observada entre `vazia-r1` e `vazia-r2`, ambas com a API
reiniciada antes: `rtt_ms` mediana variou +2,1%, `pipeline_ms` +2,0% e
`persons_ms` +1,8%. Duas ressalvas para as comparacoes futuras:

- Dentro de cada rodada o tempo sobe do primeiro ao ultimo frame (de 5,25 s
  para 5,60 s em `vazia-r2`), com a temperatura subindo junto. A deriva interna
  de uma rodada tem a mesma ordem de grandeza da diferenca entre rodadas.
- `process_rss_mb` ficou em 561,60 na r1 e 629,88 na r2, ambas com processo
  recem-iniciado e cena vazia. Dentro de cada rodada o valor e estavel; entre
  reinicios, nao. Diferencas de RAM abaixo de uns 15% nao significam nada.
- Nas tres rodadas de cena vazia, a mediana de `rtt_ms` variou 2,1% e o p95
  variou 9,4%. Com 30 amostras, o p95 serve para observar caudas, nao para
  decidir se uma mudanca melhorou o sistema.

Como referencia pratica: so tratar como ganho real uma variacao de latencia
acima de 5% que se repita nas tres rodadas do mesmo cenario.

1. Coletar a linha de base com webcam conforme `docs/BENCHMARK.md`: tentar
   cenas vazia, uma pessoa e varias pessoas, com 5 frames de aquecimento e 30
   medidos por rodada, tres rodadas por cenario disponivel e API reiniciada
   antes de cada rodada. Manter enquadramento e luz tao estaveis quanto possivel.
2. Usar `ENABLE_PERFORMANCE_METRICS=1`, `ENABLE_SYSTEM_MONITOR=1` e
   `CITYLAB_ALLOW_PARTIAL_PIPELINE=0` em cada rodada, mantendo rosto e gesto
   ativos; confirmar esse estado apos cada reinicio da API.
3. Registrar commit, versoes, configuracao, pesos, temperatura, contagem de
   pessoas e os JSON de resultado. A webcam nao permite repetir os mesmos
   frames entre versoes, entao os numeros nao provam diferencas pequenas de
   latencia nem preservacao exata de recall.

Limitacao conhecida do utilitario: ele mantem uma pendencia por vez
(`max_in_flight: 1`), enquanto o cliente real usa `MAX_IN_FLIGHT_FRAMES=2`. Os
numeros nao medem a renderizacao do navegador. Com webcam, cada rodada recebe
frames diferentes e o p95 de 30 amostras e apenas indicativo.

### Onde estao os resultados

Todos os arquivos da linha de base estao versionados em
`resultados/pi3-b05058f-webcam/`, um por rodada, no padrao
`<cenario>-r<numero>.json`:

| Arquivo | Cenario | Rodada |
|---|---|---|
| `vazia-r1.json`, `vazia-r2.json`, `vazia-r3.json` | cena vazia | 1 a 3 |
| `uma-pessoa-r1.json`, `uma-pessoa-r2.json`, `uma-pessoa-r3.json` | uma pessoa | 1 a 3 |
| `duas-pessoas-r1.json`, `duas-pessoas-r2.json`, `duas-pessoas-r3.json` | varias pessoas, 2 reais | 1 a 3 |

Cada arquivo tem o cabecalho da rodada (cenario, rotulo, camera, resolucao,
`started_at_utc`, versao do cliente, configuracao), os resumos de `rtt_ms`,
`detections` e `metrics` em mediana e p95, os bytes trafegados e, em `samples`,
as 30 amostras individuais com 13 metricas por frame. Sao 9 arquivos, 7306
linhas e 240 KB no total; o volume vem do `indent=2` e das amostras por frame,
cerca de 27 linhas por frame medido.

Os arquivos guardam apenas contagens e tempos. Nao ha imagem, base64,
embedding nem nome de pessoa. A conferencia foi feita antes do commit.

As amostras por frame ficaram versionadas de proposito: foram elas que
mostraram que os frames com a pessoa fantasma eram os primeiros da rodada, e
nao frames mais lentos, e que a caixa extra custava trabalho na cena de duas
pessoas mas nao na de uma. So com os resumos, as duas leituras teriam saido
erradas.

Fora do Git, em `.tmp/` no cliente Windows, ficaram o piloto de 3 frames
(`webcam-pilot.json`) e o relatorio original da rodada que foi rotulada errada
(`baseline-one-person-r1.json`). O `label_correction` dentro de
`duas-pessoas-r1.json` guarda o SHA-256 desse original, mas o arquivo em si nao
e recuperavel pelo repositorio.

## Balanco da fase 1

Linha de base completa: 9 rodadas, 270 frames medidos, commit `b05058f`,
pipeline completo com rosto e gesto, API reiniciada antes de cada rodada.

| Metrica (mediana) | Vazia | 1 pessoa | 2 pessoas |
|---|---|---|---|
| `rtt_ms` | 5409,98 | 13413,53 | 16615,63 |
| `completed_fps` | 0,184 | 0,075 | 0,060 |
| `persons_ms` | 4651,92 | 4748,25 | 4760,28 |
| `pose_ms` | 0,00 | 5234,59 | 5287,70 |
| `faces_ms` | 725,11 | 3044,71 | 5337,39 |
| `hands_ms` | 0,00 | 286,16 | 1038,96 |
| `decode_ms` | 8,11 | 7,58 | 7,68 |
| `logs_ms` | 0,02 | 52,74 | 85,71 |
| `process_rss_mb` | 629,88 | 701,15 | 713,49 |
| `temperature_c` | 59,88 | 59,61 | 59,07 |

Amplitude entre as tres rodadas de cada cenario: 2,1% na cena vazia, 1,7% com
uma pessoa e 2,7% com duas, sempre na mediana de `rtt_ms`. A temperatura ficou
entre 59 e 60 C em todas as rodadas, sem indicio de throttling.

O que a fase 1 estabeleceu:

1. O gargalo e inferencia, nao transporte nem I/O. `decode_ms` (7,6 ms),
   `logs_ms` (ate 88,97 ms) e `receive_wait_ms` (cerca de 18 ms) somados nao
   chegam a 1% do frame.
2. Existe um custo fixo de cerca de 10 s por frame sempre que ha alguem na
   cena: `persons_ms` mais `pose_ms`, duas passadas YOLO sobre o frame inteiro
   cujo tempo quase nao muda entre uma e duas pessoas.
3. O custo variavel e o reconhecimento facial, perto de 2,3 s por rosto.
4. A RAM residente do processo fica entre 630 e 719 MB numa placa de 906 MB.
   Nao sobra espaco para dobrar modelos ou manter filas grandes em memoria.
5. O detector de pessoas devolve uma caixa a mais do que existe de forma
   sistematica: 3 pessoas com 2 reais nas rodadas r2 e r3, 2 com 1 real em
   parte das rodadas de uma pessoa.

Ressalva metodologica que vale para todas as fases seguintes: as regras de
gesto contam analises, nao tempo. Nesta linha de base, cada analise cobre entre
5 e 17 s de mundo real. Se uma otimizacao dobrar a vazao, os limiares passam a
ser atingidos em menos tempo real e a contagem de alertas muda sozinha, sem que
o reconhecimento tenha melhorado nem piorado. **Enquanto as regras nao forem
convertidas para tempo decorrido, a contagem de alertas desta linha de base nao
serve para verificar preservacao de recall depois de uma mudanca de
desempenho.** Essa conversao deixa de ser item da fase 4 e passa a ser
pre-requisito de qualquer comparacao de alertas.

A ordem das fases originais nao sobreviveu a medicao. O backlog abaixo
substitui aquela sequencia e esta ordenado por retorno medido, nao por numero
de fase.

## Backlog de otimizacao

Nada aqui esta autorizado a ser implementado: a proxima acao e combinada com o
responsavel, uma por vez, com medicao antes e depois. Os ganhos sao estimativas
derivadas da linha de base, nao promessas; o frame de referencia e o de duas
pessoas, 16615,63 ms de `rtt` e 16498,96 ms de `pipeline`.

| # | Acao | Alvo medido | Ganho estimado | Risco principal |
|---|---|---|---|---|
| 1 | Corrigir `event_logger` | stream cai em falha de insert | nenhum em tempo | baixo |
| 2 | Regras de gesto por tempo decorrido | comparabilidade de alertas | nenhum em tempo | muda quando o alerta dispara |
| 3 | Passada unica de deteccao (pessoas + pose) | 4760 + 5288 ms | 4,8 a 5,3 s, 29 a 32% | qualidade das caixas do modelo de pose |
| 4 | Sobreposicao real entre rosto e gesto | 5337 ms de `faces_ms` | ate 5,4 s, 32,5% no teorico | 4 nucleos ja saturados; pode dar zero |
| 5 | Limiar de confianca no detector | caixa fantasma | 0,1 a 0,2 s e alertas falsos | perder pessoa distante ou de lado |
| 6 | `allowed_modules` no InsightFace | parte dos 5337 ms de rosto | a medir | modulo necessario ao alinhamento |
| 7 | `imgsz` e `det_size` explicitos | `persons_ms` e `faces_ms` | proporcional ao lado da entrada | recall de rosto pequeno |
| 8 | Fila/worker no WebSocket (fase 2) | comportamento da conexao | nenhum em latencia | RAM apertada; escolher o que descartar |
| 9 | Ajustes de cliente (fase 5) | correlacao, timeout, pausa | nenhum em latencia | baixo |

### 1. Corrigir o `event_logger`

Nao e otimizacao, e defeito: uma falha de `insert_one` encerra o loop do
WebSocket e derruba o stream, e o cooldown ja foi marcado antes da gravacao,
entao a nova tentativa fica bloqueada. Independe de medicao e pode ser feito a
qualquer momento. Detalhe na secao da fase 3.

### 2. Converter as regras de gesto para tempo decorrido

Pre-requisito de todo o resto que mexe em desempenho. Enquanto as regras
contarem analises, qualquer ganho de vazao muda a contagem de alertas por
efeito colateral, e a comparacao de recall fica sem base. Detalhe na secao da
fase 4.

### 3. Passada unica de deteccao para pessoas e pose

Hoje rodam dois modelos sobre o mesmo frame: `yolov8n.pt` em
`FaceRecognitionService.detect_persons` e `yolov8n-pose.pt` em
`GestureRecognitionService`. O segundo ja produz caixas de pessoa junto com os
keypoints. Usar uma unica passada elimina de 4,8 a 5,3 s do frame com gente.

Contrapartida a medir: hoje a cena vazia paga so o detector de pessoas
(4651,92 ms) e pula o gesto. Com pose primeiro, a cena vazia passaria a pagar
o modelo de pose (5287,70 ms), ficando cerca de 0,6 s mais lenta. O ganho
aparece quando ha gente; a cena vazia piora um pouco. Decidir se compensa manter
um gate barato antes da pose.

### 4. Fazer o paralelismo valer alguma coisa

A medicao mostrou soma, nao sobreposicao. Primeiro passo e barato: conferir
`PIPELINE_RUN_IN_PARALLEL` no ambiente do Pi. Se estiver ligada, o problema e
contencao: limitar as threads intra-op do ONNX Runtime para que dois modelos
caibam nos quatro nucleos sem se atropelar. Pode dar zero, e nesse caso a
conclusao tambem vale: desligar a flag e economizar o ThreadPoolExecutor.

### 5. Limiar de confianca no detector de pessoas

O detector devolve sistematicamente uma caixa a mais. Custa `hands_ms` extra e,
pior, gera alerta de gesto de gente que nao existe, gravado no MongoDB com
recorte de imagem. Antes de escolher o limiar, incluir a confianca de cada
caixa no coletor, que hoje guarda so a contagem.

### 6 e 7. Modulos do InsightFace, `imgsz` e `det_size`

O rosto custa cerca de 2,3 s por face. O `FaceAnalysis` carrega todos os
modulos do `buffalo_l`, incluindo landmarks 2D e 3D, que o reconhecimento por
embedding pode nao usar. Reduzir modulos e fixar tamanhos de entrada sao ajustes
independentes; medir um de cada vez, com os cenarios de uma e duas pessoas.

### 8 e 9. Fila no WebSocket e ajustes de cliente

Ficaram por ultimo por motivo medido, nao por desinteresse. Com 0,06 a 0,18
FPS, o trabalho e o mesmo com ou sem fila: o que muda e quais frames sao
descartados e como a conexao se comporta enquanto o servidor pensa. Tem valor,
mas nao reduz latencia. Vale mais depois que as acoes 3 a 7 mudarem a ordem de
grandeza do frame.

## Fase 2 - tirar decodificacao e inferencia do caminho sincrono

Backlog: acao 8. Adiada por medicao, nao por desinteresse.

Hoje `cv2.imdecode` e `recognizer.process_frame` rodam dentro do handler do
WebSocket em `Server/main.py`. Proposta: fila/worker de capacidade limitada.

Decidir antes de implementar: o estado do rastreador atende so uma camera ou
precisa ser isolado por camera? Situacao atual verificada: o estado ja e global
- `recognizer` e instancia unica de modulo, e `App/GestureRecon/service.py` usa
`pose_model.track(persist=True)` com `last_track_centers`, `next_track_id` e
`analyzer.history` indexados apenas por `track_id`. Duas cameras simultaneas ja
misturariam tracks hoje, antes de qualquer fila.

## Fase 3 - tirar a gravacao de eventos do caminho da resposta

Backlog: acao 1, a correcao do defeito. A parte de tirar a gravacao do
caminho da resposta nao tem ganho de tempo relevante.

`Server/event_logger.py` grava no MongoDB com `await` dentro do handler. Alem do
custo no caminho da resposta, ha dois defeitos confirmados:

- `_should_log` grava `last_logged[key] = now` **antes** do `insert_one`. Se o
  insert falhar, o evento se perde e o cooldown ainda bloqueia nova tentativa.
- A excecao do `insert_one` sobe ate o `except Exception` de
  `websocket_reconhecimento`, que esta **fora** do `while`: uma falha de gravacao
  encerra o loop e derruba a conexao do stream.
- `last_logged` nunca expira. A chave de `NAO_ALUNO` e a bbox quantizada em
  grade de 40 px, entao o dicionario cresce por posicao na tela.

Evidencia medida em 19/09/2026: com eventos acontecendo, `logs_ms` ficou em
84,52 ms de mediana (p95 135,51) dentro de um frame de 16,3 s, e em 0,02 ms na
cena vazia. Tirar a gravacao do caminho da resposta vale por robustez e pela
correcao do cooldown, nao por latencia: o ganho de tempo e inferior a 1%.

Proposta: fila limitada, tratamento de falhas sem derrubar o stream e correcao
do cooldown (marcar apos sucesso, com expiracao).

## Fase 4 - otimizacoes de modelo (medir cada uma)

Backlog: acoes 2 a 7. E aqui que esta a latencia.

Evidencia medida em 19/09/2026: na cena vazia, `persons_ms` ficou em 4569,45 ms
de mediana sem nenhuma pessoa no enquadramento, contra 4760,28 ms na cena com
duas pessoas. O detector de pessoas custa praticamente o mesmo com a cena cheia
ou vazia e define o piso de latencia; `faces_ms` foi de 705,05 ms na cena vazia
para 5328,19 ms com rostos presentes. `decode_ms` (7,35 ms) e `logs_ms`
(0,02 ms sem eventos) sao irrelevantes nessa escala.

Na cena com duas pessoas, `pose_ms` sozinho foi 5202,80 ms de mediana, contra
963,42 ms de `hands_ms` dentro dos 6156,84 ms de `gestures_ms`. Somando com
`persons_ms`, os dois modelos YOLO consomem cerca de 10 s dos 16,3 s do frame.
Reaproveitar uma unica passada de deteccao para pessoas e pose e a maior
alavanca isolada identificada ate aqui.

Com as tres cenas medidas, o custo se separa em duas partes:

| Estagio (mediana das 3 rodadas, ms) | Vazia | 1 pessoa | 2 pessoas |
|---|---|---|---|
| `persons_ms` | 4651,92 | 4748,25 | 4760,28 |
| `pose_ms` | 0,00 | 5234,59 | 5287,70 |
| `faces_ms` | 725,11 | 3044,71 | 5337,39 |
| `hands_ms` | 0,00 | 286,16 | 1038,96 |
| `pipeline_ms` | 5387,22 | 13340,26 | 16498,96 |
| `rtt_ms` | 5409,98 | 13413,53 | 16615,63 |

`persons_ms` e `pose_ms` praticamente nao mudam entre uma e duas pessoas: sao
passadas de rede sobre o frame inteiro, com custo fixo de cerca de 9,9 s sempre
que ha alguem na cena. O que cresce com a cena e o reconhecimento facial
(725 ms sem ninguem, 3042 ms com uma pessoa, 5328 ms com duas, algo perto de
2,3 s por rosto processado) e, bem menos, `hands_ms`. Ou seja: reduzir o custo
fixo das duas passadas YOLO vale mais do que qualquer ajuste proporcional a
quantidade de pessoas.

Terceiro achado, sobre o paralelismo: `pipeline_ms` e igual a soma de
`persons_ms`, `faces_ms` e `gestures_ms` com diferenca de 0,2% nos tres
cenarios (5377 contra 5387 na cena vazia, 13332 contra 13340 com uma pessoa,
16475 contra 16499 com duas). Rosto e gesto **nao se sobrepoem na pratica**,
apesar de `PIPELINE_RUN_IN_PARALLEL` vir ligado por padrao e de
`_process_parallel` submeter os dois ao ThreadPoolExecutor. Se houvesse
sobreposicao real, o frame de duas pessoas cairia de 16499 ms para cerca de
11137 ms, 32,5% menos. Duas explicacoes possiveis, ainda nao separadas: a flag
pode estar desligada no ambiente do Pi, ou os quatro nucleos ja estao saturados
pelas threads internas de cada modelo, de modo que rodar dois modelos juntos so
divide o mesmo processador. O `.env` local nao define a flag; o do Pi nao foi
lido nesta sessao.

Pontos verificados em `App/FaceRecon/service.py` e `App/GestureRecon/service.py`:

- `insightface.app.FaceAnalysis` criado sem `allowed_modules` (carrega tudo).
- `det_size` fixo em `(320, 320)`.
- Chamadas YOLO sem `imgsz` explicito.
- Sem configuracao de threads do ONNX Runtime.
- `PIPELINE_RUN_IN_PARALLEL` ligado por padrao com 2 workers, mas medido sem
  nenhuma sobreposicao: confirmar a flag no Pi e testar limitar as threads do
  ONNX Runtime antes de concluir que paralelismo nao serve nesta placa.
- Dois modelos separados (`yolov8n.pt` para pessoas e `yolov8n-pose.pt` para
  pose): avaliar reaproveitar a deteccao de pose.
- Filtro de rostos ruins (`FACE_MIN_WIDTH`, `FACE_MIN_HEIGHT`,
  `FACE_MIN_CONFIDENCE`).

Pre-requisito: as regras de gesto em `App/GestureRecon/detector.py` sao baseadas
em contagem de analises (`_confirm_gesture` sobre `_update_counter`). Converter
para tempo decorrido **antes** de reduzir a frequencia de inferencia, senao o
significado dos alertas muda junto.

## Fase 5 - ajustes de cliente

Backlog: acao 9.

- Correlacao explicita frame/resposta: hoje e posicional (`pendingSentAt.shift()`).
- Timeout por frame: hoje nao existe watchdog de resposta.
- Renderizacao durante a pausa: o `cancelAnimationFrame` para o overlay.

## Protocolo de comparacao

Mesma camera, cenario e configuracao, mesmo estado inicial do banco e
temperatura comparavel. A webcam nao repete os mesmos frames. Registrar em cada
rodada: mediana e p95 de `rtt_ms`,
`completed_fps`, tempos por estagio, `process_rss_mb`, `temperature_c` e o
comportamento dos alertas. Detalhes e significado de cada metrica em
`docs/BENCHMARK.md`.
