# Plano de otimizacao do stream (Raspberry Pi 3 B+)

Documento compartilhado entre os agentes que trabalham neste repositorio (Codex e
Claude) e o responsavel pelo projeto. Quem for mexer em performance le este
arquivo antes e atualiza o estado aqui depois. Nao duplicar o plano em outro
lugar: este e a fonte unica.

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
7. Enquanto as regras de gesto contarem analises em vez de tempo, nao comparar
   contagem de alertas entre versoes com desempenho diferente. Ver o "Balanco
   da fase 1".

## Estado verificado em 20/09/2026

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
