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
entre 0,06 e 0,18 FPS. Essa referencia e o backlog original ficam preservados
como historico. A fila de propostas atual esta em "Pesquisa e planejamento em
01/10/2026"; os resultados efetivamente medidos, no "Estado verificado".

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

## Pesquisa e planejamento em 01/10/2026

O responsavel pediu uma revisao do repositorio, pesquisa na internet e um novo
planejamento de desempenho. Esta secao reabre o levantamento de propostas;
nao autoriza implementar, mudar o ambiente do Pi ou iniciar medicoes. Cada
acao continua dependendo de combinacao individual. O ultimo estado medido
permanece na secao seguinte.

### Base da revisao

Checkout conferido: `main`, commit `89c93b8`, sem alteracoes locais antes deste
planejamento. Foram revisados o historico Git, os dois planos, o protocolo de
benchmark, o guia do Pi, README, codigo da pipeline/servidor/cliente, ferramentas,
dependencias e registros de resultados. Nao houve acesso ao Pi, execucao de
modelos, leitura de credenciais ou nova medicao de desempenho nesta revisao.
As fontes externas abaixo foram consultadas em 01/10/2026.

O que ja foi entregue e nao deve voltar como proposta nova:

- Metricas por frame, coletor de webcam/video e tres rodadas por cenario.
- Uma passada para pessoas e pose, sobreposicao de rosto e gesto, limite de
  threads nativas e reaplicacao do limite PyTorch em cada tarefa.
- Gate de movimento que continua analisando cena ocupada e forca nova pose
  pelo teto de tempo, mais filtro de publicacao das caixas fracas do tracker.
- InsightFace apenas com deteccao/reconhecimento e filtro de qualidade antes
  do embedding. Equivalencia conferida com modelos reais em tres fotos no PC.
- Eventos por episodio, tratamento de falha de insert e controles de cooldown.
- Replay e evidencia temporal de gestos, criterio de tempo e observacoes,
  correlacao frame/resposta, timeout e limpeza do overlay na pausa.
- Pose em 416 no rpi3, com ganho de RTT acima de 5% nas tres rodadas de uma
  pessoa. Pose em 320 foi reprovada por perder deteccoes.
- Caminho opcional e exportador NCNN preparados; backend ainda nao medido.

As referencias de tempo disponiveis tem condicoes diferentes:

| Cenario | Ultima serie valida | RTT mediano por rodada | Limite da evidencia |
|---|---|---|---|
| Uma pessoa frontal | `c1f1ac0`, pose 416, limite termico 70 C | 5299,0 / 5317,4 / 5319,8 ms | pessoa, rosto e gesto em 90/90 frames, sem alertas |
| Vazia | `3e67f56`, pose 640, limite 60 C | 1676,3 a 1679,1 ms | nao mede o codigo/perfil completo atual |
| Duas pessoas | `bc0a441`, pose 640, limite 60 C | 6159,8 a 6182,2 ms | uma frontal e uma de lado; normalmente so um rosto aceito |

Nao usar essa tabela como tres cenarios de uma baseline uniforme. A ultima
serie de 416 esta em `resultados/pi3-c1f1ac0-pose416/`. Com uma pessoa,
`faces_ms` ficou em 5276 a 5294 ms, `pose_ms` em 2232 a 2241 ms e
`gestures_ms` em cerca de 2,7 s. O rosto terminou por ultimo em todos os frames.

### Diagnostico que orienta a ordem

Hoje a resposta espera rosto e gesto. O custo se aproxima do mais lento dos
dois, acrescido dos demais estagios; nao da soma dos tempos paralelos.
Otimizar apenas a pose pode liberar CPU/RAM e ajudar cenas sem rosto, mas nao
garante que o frame frontal fique muito mais rapido. A primeira alavanca a
investigar e o caminho facial, sem mudar pesos ou filtros.

`faces_ms` junta detector SCRFD, alinhamento/ArcFace e busca na base; ainda
nao ha decomposicao atual desses custos. O detector recebe 320x320, enquanto
o reconhecedor usa o recorte facial alinhado. Reduzir `det_size` nao reduz
automaticamente o custo de gerar embeddings. A comparacao com a base ja e
vetorizada; trocar o MongoDB ou criar indice vetorial nao e prioridade com os
poucos cadastros documentados.

A configuracao ORT=1/PyTorch=3 foi escolhida quando a pose ocupava os nucleos
por mais tempo. Agora ha oportunidade de rever a divisao, mas mais threads
tambem podem aumentar contencao e temperatura. O ganho precisa aparecer no
RTT completo, com os dois servicos ligados.

### Fila proposta e pontos de decisao

Esforco abaixo e relativo ao desenvolvimento/validacao, sem prometer prazo
de calendario nem FPS. A ordem e condicional: cada resultado decide a proxima
acao, e cada subexperimento altera uma variavel por vez.

| ID | Proposta | Motivo para priorizar | Esforco / risco | Dependencia |
|---|---|---|---|---|
| P0 | Fechar referencia em 416 e validar uso continuo | evita decidir com cenarios/temperaturas diferentes | baixo em codigo; exige camera/Pi | primeira acao a combinar |
| P1 | Decompor rosto e ajustar ONNX Runtime | atua no caminho critico mantendo pesos | baixo a medio / baixo | P0 |
| P2 | Medir a mesma pose em NCNN | caminho pronto, potencial de CPU/temperatura; RTT incerto | medio / medio | P0; preferir apos P1 |
| P3 | Avaliar modelo facial menor conforme o perfil medido | pode reduzir o maior custo e a RAM | medio a alto / alto | P1; conjunto facial de validacao |
| P4 | Provider ARM ou INT8 do mesmo modelo | alternativas se P1/P3 nao bastarem | alto / alto | perfil por operador e prova de compatibilidade |
| P5 | Separar cadencias de rosto/gesto ou reutilizar identidade | pode aumentar observacoes de gesto sem esperar todo rosto | alto / alto, muda contrato temporal | decisao de produto e replay |
| P6 | Worker limitado para responsividade | rotas/conexao continuam atendidas durante inferencia | medio / medio | contrato de uma camera e fila definidos |

#### P0 - referencia e qualidade antes de novo ajuste

Combinar tambem o objetivo operacional: quantidade de pessoas/rostos que deve
suportar, intervalo desejado entre observacoes de gesto e atraso maximo de
alerta. O criterio de >5% distingue ganho de variacao; nao define sozinho se
o software ja atende ao uso pretendido.

Primeiro repetir o perfil atual nos tres cenarios do protocolo: vazio, uma
pessoa frontal e duas pessoas (frontal + perfil). Acrescentar, como cenario
separado, duas pessoas frontais com dois rostos aceitos: o historico de duas
pessoas nao demonstra esse custo. Camera escolhida pelo responsavel, mesmos
cadastros/pesos, API reiniciada, 5 aquecimentos e 30 frames, tres rodadas por
cenario. Conferir commit, hashes e configuracao no Pi, nao apenas no PC.

Em uma acao separada, conferir perfil e punho levantado nos mesmos frames em
640/416. Os JSONs de keypoints ja extraidos nao permitem testar uma nova rede:
precisam dos videos locais ou de novos videos combinados com o responsavel.
Guardar apenas resultados numericos no Git. Essa verificacao complementa o
replay e nao presume que mais alertas signifiquem mais recall.

Depois, combinar ensaio continuo inicial de 30 a 60 min, com pessoa/rosto,
log de `vcgencmd`, RAM disponivel, RSS, swap e `vmstat` (`si`/`so`). Estender
para a duracao de uso esperada somente se a primeira janela ficar estavel.
O log de 01/10 cobre horas de relogio, mas tambem repouso e reinicios: nao e
prova de horas de carga continua. Swap alocado sozinho nao prova que ha I/O
de swap durante a inferencia.

Entrega: JSONs e LEIAME com ambiente, cenas, variacao entre rodadas e periodo
continuo, incluindo falhas. Se houver throttling atual, combinar refrigeracao
antes de comparar codigo. Dissipador/ventoinha e alternativa de estabilidade;
o limite de 70 C ja esta aplicado, e nao sera aumentado por este plano.
A documentacao oficial explica a reducao 1,4 -> 1,2 GHz e o limite do 3 B+
([Raspberry Pi](https://www.raspberrypi.com/documentation/computers/config_txt.html#overclocking)).

#### P1 - ONNX Runtime no caminho facial

1. Medir separadamente detector, alinhamento/embedding por rosto e matching
   em `App/FaceRecon/service.py`. Encaminhar tempos numericos pela pipeline e
   coletor, sem nomes/embeddings. Usar o profiler nativo do ORT para uma rodada
   diagnostica curta e desligar no benchmark de aceite: o tracing tem custo
   ([profiling oficial](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)).
2. Comparar `ONNX_INTRA_OP_THREADS=1`, depois 2 e depois 3 com pose 416 e
   PyTorch=3 fixos. A opcao ja existe; nao precisa trocar bibliotecas ou pesos.
   Medir primeiro o cenario frontal; so levar candidatos promissores aos demais.
3. Se a disputa inicial com a pose limitar o resultado, combinar uma proxima
   comparacao PyTorch=2 contra 3 mantendo o ORT vencedor. Nao tratar esse teste
   como repeticao dos resultados da pose em 640: a carga atual e outra.
4. Com threads escolhidas, avaliar `session.intra_op.allow_spinning=0` em
   experimento separado. Em ORT=1 nao ha workers intra-op adicionais; nao
   esperar ganho relevante desse controle sozinho. Se detector/embedding
   precisarem limites diferentes, propor controle por sessao no startup,
   sem reconstruir sessoes por frame.

A referencia de threads explica contencao e spinning
([ORT](https://onnxruntime.ai/docs/performance/tune-performance/threading.html));
as chaves basicas foram conferidas tambem na versao 1.23.2, citada no projeto
([codigo da versao](https://github.com/microsoft/onnxruntime/blob/v1.23.2/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)).
Confirmar a versao realmente instalada antes de usar opcoes novas da documentacao.

Entrega de codigo, se autorizada: instrumentacao e controles em
`App/settings.py`, `App/inference_runtime.py`, `tools/run_rpi.py`, pipeline,
coletor e testes pertinentes. Preservar flags de retorno e evitar manter duas
sessoes pesadas em RAM. Controle de arena so entra se houver pressao de memoria
medida; pode trocar RAM por tempo. `ORT_ENABLE_ALL` ja e o padrao da biblioteca:
"ligar otimizacoes de grafo" nao e uma nova alavanca. Salvar grafo otimizado
serve principalmente ao startup e exige hardware/provider compativeis
([grafos ORT](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)).

#### P2 - NCNN da mesma pose, sem trocar o modelo

Exportar fora do Pi uma copia do `yolov8n-pose.pt`, no ambiente compativel
com a versao Ultralytics medida. A chamada completa do exportador existente
e `python tools/export_pose_ncnn.py App/GestureRecon/yolov8n-pose.pt --imgsz 416`;
preferir substituir o caminho pela copia isolada, pois o diretorio exportado
e criado junto ao peso. Fixar versoes/hash de peso, exportador, pnnx e NCNN;
confirmar wheel Linux aarch64/Python 3.11 e executar smoke test no Pi.
Nao deixar o primeiro startup de producao instalar dependencias do Git.
Existe wheel CPython 3.11/aarch64 em
[NCNN 1.0.20260526](https://pypi.org/project/ncnn/1.0.20260526/), candidato a
comparacao; sua existencia nao comprova compatibilidade deste modelo/wrapper.

A integracao suporta pose, mas a documentacao atual pode usar argumentos
diferentes dos de 8.3.226: conferir o codigo da versao antes de adaptar
([NCNN/Ultralytics](https://docs.ultralytics.com/integrations/ncnn/)).

`imgsz=416` nao garante entrada equivalente: no caminho PyTorch com
`rect=True`, um frame 320x240 pode virar tensor 320x416; o export estatico NCNN
pode usar 416x416, 30% mais pixels. Registrar shape real e letterbox em ambos,
comparar primeiro os caminhos como realmente executam, e tratar eventual
export retangular como outro experimento. Esse comportamento foi conferido
no pacote local Ultralytics 8.3.226 (`engine/predictor.py`, `nn/autobackend.py`
e `engine/exporter.py`). Conferir os 17 keypoints, scores,
caixas, ByteTrack e alertas nos mesmos videos, incluindo perfil, punho,
oclusao, entrada/saida e duas pessoas.

Medir com pipeline completa e verificar, como diagnostico separado, cenas
com pessoa sem rosto. `half=False` preserva pesos exportados FP32, mas nao
prova a precisao interna de todo kernel NCNN. Registrar opcoes efetivas,
inclusive `net.opt.num_threads`: o limite PyTorch nao limita o pool NCNN.
Se precisar ajustar o pool, comparar separadamente e expor o controle em
`App/settings.py`. Nao combinar INT8/FP16/Vulkan nessa primeira comparacao.
O wrapper ainda importa Ultralytics/PyTorch e
reaplica threads; exportar nao garante retirar PyTorch da RAM.

Entrega: manifesto do artefato, relatorio de equivalencia e desempenho/RAM,
ajuste de compatibilidade somente se necessario. `POSE_MODEL_PATH` vazio
retorna ao .pt. Se a pose melhorar sem >5% de ganho no RTT alvo, registrar
como ganho de estagio; CPU/RAM/temperatura podem justificar outra decisao
explicita, sem chamar isso de ganho confirmado de latencia.

#### P3 - modelo facial menor, escolhido pela decomposicao

Antes de trocar pesos, se houver uma prova viavel de baixo custo, comparar os
mesmos ONNX em OpenCV DNN ou provider alternativo (P4). Isso preserva modelo
e banco, mas ainda exige reproduzir preprocessamento, alinhamento e outputs.
O esforco de adaptacao determina se essa prova vem antes da troca de modelo.

Se o detector pesar mais, comparar primeiro SCRFD-2.5GF (como o de `buffalo_m`)
mantendo o reconhecedor atual de `buffalo_l`, ResNet50/WebFace600K; conferir
hash do reconhecedor, nao so nome da arquitetura. Se o embedding pesar mais,
avaliar MobileFaceNet, como o de `buffalo_s`/`buffalo_sc`, em ambiente isolado.
Esses pacotes tambem mudam o detector para SCRFD-500MF. A tabela oficial mostra
perda de acuracia do reconhecedor leve em parte dos benchmarks; tamanho de
pacote no disco nao equivale a RSS do processo
([modelos oficiais](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md)).

YuNet + SFace e outra alternativa completa pelo OpenCV. Exige adaptador de
deteccao/alinhamento/reconhecimento, base de embeddings e limiares proprios,
alem de conferir a versao OpenCV do apt e o modelo compativel. Nao transferir
numeros de Pi 4 para o Pi 3 B+ nem tratar como troca equivalente do InsightFace
([tutorial oficial](https://docs.opencv.org/4.13.0/d0/dd4/tutorial_dnn_face.html),
[YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
[SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)).

Entrega de desenvolvimento: selecao explicita de modelos/provedores em
`App/settings.py`, manifesto com hashes e versao do modelo de embedding,
validacao de alinhamento, cadastro e inferencia com o mesmo contrato.
Primeiro provar compatibilidade nas mesmas imagens; se o reconhecedor mudar,
nao misturar embeddings de modelos diferentes, mesmo com 512 componentes.

A rota de cadastro hoje salva embedding/nome/data, sem a foto de origem.
Outro reconhecedor pode exigir recadastro; a migracao precisa de decisao
propria, base de teste, backup e retorno ao modelo/base anteriores. Mesmo
mantendo ArcFace, outro detector pode alterar os landmarks, alinhamento e
similaridade. Nao herdar cegamente o limiar 0,52.

Comparar rostos conhecidos e desconhecidos, frontal/perfil, pequeno/distante,
luz variavel, oclusao e duas pessoas frontais. Medir deteccoes, falso aceite,
falsa rejeicao e similaridades perto do limiar. Definir tolerancias com o
responsavel antes do teste; rejeitar candidato que so fica rapido porque
deixa de aceitar rosto. A avaliacao de licenca dos novos pesos e parte da
escolha do artefato, sem alterar o escopo de uso do projeto.

#### P4 - providers ARM e quantizacao, experimentos condicionais

- **XNNPACK no ORT:** permite kernels para ARM, mas requer build com o provider
  e cobertura dos operadores. Conferir `get_available_providers`, registrar
  fallback CPU e comparar somente um modelo por vez. O provider tem pool de
  threads proprio: nao somar seu pool ao do ORT sem medir contencao
  ([XNNPACK oficial](https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html)).
- **INT8 do mesmo SCRFD/ArcFace:** calibrar no PC com imagens representativas
  separadas do conjunto de validacao, verificar operadores e preservar uma
  copia FP32. Para essas CNNs, partir de quantizacao estatica com calibracao,
  testando primeiro o modelo que mais custa. O ORT alerta que
  quantizacao pode ficar mais lenta em hardware antigo; ARM64 nao garante
  aceleracao. Aceite inclui deteccao e falsos aceites/rejeicoes, alem de RTT e
  RAM ([quantizacao ORT](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)).
- **ACL/Arm NN, OpenCV DNN ou TFLite:** reservas se os anteriores falharem,
  condicionadas a conversao, operadores e runtime no Bookworm/Python 3.11.
  Implementar somente o backend escolhido pelo perfil, preservando
  preprocessamento, NMS, alinhamento e resultados. Compilar fora do Pi em
  ambiente ARM64 compativel, sem instrucoes que o Cortex-A53 nao suporta.
  ACL e Arm NN sao providers comunitarios, nao recursos automaticos do wheel
  CPU ([ACL](https://onnxruntime.ai/docs/execution-providers/community-maintained/ACL-ExecutionProvider.html),
  [Arm NN](https://onnxruntime.ai/docs/execution-providers/community-maintained/ArmNN-ExecutionProvider.html)).
  A disponibilidade de TFLite ARM64 nao comprova conversao direta dos ONNX
  InsightFace; exigir prova sem TensorFlow completo no Pi.

Essas alternativas nao entram todas no calendario. Fazer primeiro uma prova
curta com o modelo real; descartar as que exigem reescrita grande sem vantagem
medida. Se houver wheel/provider compativel pronto, priorizar a prova do mesmo
modelo antes de trocar pesos em P3. Atualizar bibliotecas por si so nao
comprova ganho.

#### P5 - taxa de gesto independente do reconhecimento facial

Se o rosto continuar dominando depois das acoes anteriores, avaliar dois
caminhos independentes: gestos sobre frames novos e reconhecimento facial
com cadencia propria. Outra proposta, separada, e reutilizar identidade por
track por tempo limitado. Nenhuma delas e apenas um ajuste de desempenho:
muda a idade da informacao e pode esconder troca de pessoa.

Antes de implementar, combinar atraso facial maximo e validade da identidade.
O desenho precisa de associacao rosto/pessoa, timestamps por resultado,
invalidacao em entrada/saida, oclusao e cruzamento, reconhecimento imediato
de nova pessoa e retorno ao processamento integral quando houver duvida.
Nao reutilizar keypoints/maos como novas observacoes do `GestureAnalyzer`.
O cadastro precisa invalidar caches; rotulos antigos nao podem abrir/fechar
episodios como se fossem deteccoes novas.

Entrega: contrato de payload/cliente/coletor e testes de expiracao, troca,
reconexao e eventos. Medir intervalo entre observacoes reais de gesto, idade
do frame/rosto, tempo ate alerta, deteccoes e RTT separadamente. O replay de
`docs/PLANO_GESTOS.md` deve usar os intervalos realmente observados. Reduzir
o RTT devolvendo resultado velho nao atende ao objetivo.

#### P6 - responsividade do servidor, sem promessa de inferencia mais rapida

Mover o trabalho sincrono do handler para um worker de inferencia com fila
limitada, mantendo uma camera e um processo. O executor interno de rosto/gesto
pode permanecer; chamadas completas de `process_frame` devem ser serializadas.
Cadastro que usa a mesma sessao facial tambem precisa dessa coordenacao.
`asyncio.to_thread` sem limite nao resolve o estado global do tracker
([executor no Python 3.11](https://docs.python.org/3.11/library/asyncio-eventloop.html#executing-code-in-thread-or-process-pools)).

Combinar antes a politica de rejeitar/descartar frames e o numero na resposta;
o cliente atual espera resposta correlacionada para cada envio. Validar
desconexao/cancelamento sem resetar o tracker enquanto a tarefa anterior roda.
Medir latencia de `/` e demais rotas durante inferencia e frescor do resultado,
alem do protocolo atual. Nao abrir varios processos com copias dos modelos.

### Alternativas fora da fila principal

| Opcao | Quando considerar | Restricao / retorno esperado |
|---|---|---|
| Refrigeracao ativa | se o ensaio continuo mostrar queda de clock | evita perda termica; nao acelera nucleo ja em 1,4 GHz |
| ZRAM | se houver I/O de swap relevante apos reduzir memoria dos modelos | comprime paginas na RAM e custa CPU; medir, nao assumir ganho de inferencia |
| Inferencia em computador da LAN | se a meta de observacao nao couber no Pi | Pi pode manter API/captura; precisa protocolo, timeout, disponibilidade e medicao de ponta a ponta |
| Placa com mais CPU/RAM | se o requisito exceder o Pi 3 B+ | novo alvo exige perfil e baseline proprios; numeros de Pi 4/5 nao predizem este Pi |
| Coral USB/Edge TPU | somente apos prova de modelo/runtime compativel | nao executa .pt/ONNX diretamente; exige TFLite INT8 compilado e operadores suportados |

ZRAM e uma mitigacao de memoria, nao substitui reduzir o conjunto ativo dos
modelos ([kernel Linux](https://docs.kernel.org/admin-guide/blockdev/zram.html)).
O guia Coral inclui o Pi 3 B+, mas lista PyCoral para Python 3.6 a 3.9; nao
valida instalacao direta no ambiente Python 3.11/Bookworm do projeto
([requisitos](https://coral.ai/docs/accelerator/get-started/)). O formato e a
cobertura do modelo precisam de prova antes de compra
([modelos Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)).

Nao priorizar: mais workers/processos, batching que acumula frames, somente
mudar FastAPI/Uvicorn, JPEG/WebSocket/logging ou desligar um reconhecimento.
Nao repetir como nova proposta pose 320, gate que pula pessoa parada, parar
runner ocioso para liberar RAM ou mudar recorte/veto de maos ja reprovados.
MediaPipe em modo VIDEO nao e troca direta: hoje usa IMAGE sobre recortes de
pessoas no mesmo reconhecedor; estado temporal compartilhado pode misturar
pessoas, e intervalos de segundos limitam o beneficio.

### Aceite, retorno e encerramento de cada acao

1. Baseline e candidato usam os mesmos pesos, cadastros, camera e condicoes,
   exceto a variavel explicitamente testada. Registrar versoes, shapes,
   hashes, configuracao, energia/refrigeracao e cena. Profiler desligado no
   aceite; microbenchmark de uma rede nao substitui pipeline completa.
2. Ganho de latencia confirmado somente acima de 5% nas tres rodadas do mesmo
   cenario. P95 com 30 frames e diagnostico. Nao somar tempos paralelos nem
   comparar com codigo/temperatura anteriores como se fossem iguais.
3. Preservar deteccoes nas imagens/videos comuns, rastreamento, cadastro,
   identidades, episodios e alertas esperados. No Pi, conferir comportamento
   ao vivo; nenhuma contagem de alerta substitui recall. As limitacoes
   conhecidas de perfil/punho continuam declaradas.
4. Sem OOM, crescimento progressivo de memoria ou nova regressao termica.
   Comparar pico de carga dos modelos e uso continuo, alem de RSS por frame.
   Economia de RAM/energia sem ganho de RTT e resultado separado, sujeito a
   decisao explicita sobre o beneficio operacional.
5. Para codigo, rodar os testes Python/Node aplicaveis e conferir com modelos
   reais antes do Pi. Manter flags/artefatos/base anterior para retorno. Testes
   simulados nao comprovam qualidade dos modelos nem desempenho ARM.
6. Deploy, somente quando combinado: parar API antes do push na `main`,
   confirmar workflow, arquivos/pesos/configuracao e processo ativo. `rsync`
   nao apaga artefatos antigos: remover no Pi apenas o que for explicitamente
   identificado. Reiniciar e conferir startup/HTTP 200 antes de medir.
7. Atualizar o "Estado verificado" com ganho, empate ou reprovacao, fontes dos
   resultados e motivo da proxima prioridade. Nenhum candidato vira padrao
   automaticamente. Se nao restar melhoria que preserve deteccao, encerrar
   a fila e decidir requisito/hardware, sem prometer tempo real no Pi 3 B+.

Proxima acao recomendada para combinar: **P0, referencia atual em 416**.
Depois, **P1, decomposicao facial e teste ORT=2**, pelo menor custo de mudanca
e por atacar o gargalo observado. Demais itens permanecem propostas.

### Decisao do responsavel sobre esta fila

Ainda em 01/10/2026 o responsavel escolheu outra ordem, uma acao por vez: P1
(rosto com 2 threads), depois as duas partes de P5, primeiro reaproveitar a
identidade do rosto e depois a cadencia propria dos gestos. P2 (NCNN) e P3
(modelo facial menor) ficam fora por enquanto. P0 fica para quando ele puder
estar na frente da camera; ate la, cada opcao e comparada na mesma sessao com
um video fixo. A decomposicao do rosto entrou no codigo em `9334b9e`, mas, a
pedido dele, o teste de threads foi medido antes do deploy dela. Resultados em
"Estado verificado em 01/10/2026".

## Estado verificado em 03/10/2026

O responsavel nao pode fazer cenas com duas pessoas por enquanto e pediu para
desenvolver o que falta, deixando um teste geral para depois. Ordem combinada:
o reconhecimento perto do limite, depois a thread separada para a inferencia
(P6) e a limpeza dos eventos antigos, por fim o NCNN na pose (P2).

### Reconhecimento perto do limite - a foto do cadastro pesa mais que a resolucao

`tools/benchmark_stream.py` passou a guardar `faces_width_px`, a largura de
cada rosto em pixels do frame enviado, e `alerts`, o nome de cada alerta ativo,
com resumo em `faces_width_px` e `alerts_by_name`; sem nome de pessoa nem
imagem.

Pelo codigo, o cadastro gera o embedding numa foto so, a da pagina de cadastro,
que no celular abre a camera frontal. O stream gera na imagem reduzida pela
`PROCESS_SCALE`: com 0,5, um rosto de 40 a 60 px no frame de 640x480 vira 20 a
30 px antes de ser ampliado para os 112x112 do ArcFace. Para separar os dois
efeitos, no PC, com os modelos reais: os 36 frames do video de carga contra uma
referencia tirada de 10 frames do video `neutro` do celular (1080x1920, rosto
de 121 a 185 px), da mesma pessoa. Rascunho em `.tmp/escala_embedding.py`, fora
do Git; so numeros.

| Referencia | Embedding | Mediana | Minimo | Acima de 0,52 |
|---|---|---|---|---|
| Media das 10 fotos | na imagem reduzida, como hoje | 0,677 | 0,477 | 35/36 |
| Media das 10 fotos | no frame inteiro | 0,693 | 0,563 | 36/36 |
| Uma foto so | na imagem reduzida, como hoje | 0,311 a 0,662 | 0,268 a 0,370 | 0 a 32/36 |

- **A resolucao pesa pouco**: o embedding no frame inteiro sobe a mediana em
  0,016 e ganha em 28 dos 36 frames, mas o ganho se concentra nos rostos
  menores, de 0,05 a 0,10 nos quatro frames com rosto de 41 a 47 px.
- **A referencia pesa muito**: com uma foto so, a mediana vai de 0,31 a 0,66
  conforme a foto, e tres das dez reconheceriam em no maximo 4 dos 36 frames.
  A media das dez fica acima do limite em 35 de 36. No Pi, com o cadastro real
  de uma foto, o mesmo video teve mediana de 0,60, na faixa das fotos isoladas.
- Os frames do video do celular nao sao fotos de cadastro: a pessoa se mexe e
  vira o rosto, entao a variacao entre fotos isoladas exagera a de uma foto
  tirada com cuidado. A comparacao mostra a direcao, nao o ganho exato no Pi.

O responsavel escolheu fazer as duas correcoes, uma de cada vez:

- **`FACE_EMBED_FULL_FRAME`**, desligado nos dois perfis: a deteccao continua
  na imagem reduzida e o embedding usa o frame original, com os pontos do rosto
  multiplicados pela escala. O ArcFace recebe 112x112 de qualquer jeito, entao
  o custo quase nao muda. No PC, com o servico real e a mesma referencia, o
  resultado da tabela se repete: mediana de 0,677 para 0,693 e pior frame de
  0,477 para 0,563, com o embedding de 113 para 117 ms.
- **Cadastro com ate 5 fotos** no mesmo campo `foto`. Guarda a media
  normalizada dos embeddings no campo `embedding` de sempre e o numero de fotos
  em `fotos`; com uma foto so, o resultado e o de antes. Uma foto com
  semelhanca abaixo de 0,3 com a media das outras recusa o cadastro, para nao
  misturar dois rostos: nas 10 fotos da mesma pessoa, a pior ficou em 0,50. A
  pagina junta as fotos uma de cada vez, porque no celular cada toque abre a
  camera e devolve uma foto, e sugere de 3 a 5. Conferido com o FastAPI e o
  banco simulado: uma, tres e seis fotos, pessoas diferentes e foto sem rosto.

O que fica aberto destas duas:

1. Medir no Pi com o video de carga e o cadastro real: o padrao contra
   `FACE_EMBED_FULL_FRAME=1` no `.env`. Depois, recadastrar com 3 a 5 fotos e
   medir de novo; os cadastros atuais continuam valendo, mas so quem for
   recadastrado ganha com a media.
2. A seguir na ordem combinada: P6 e a limpeza dos eventos antigos; depois o
   NCNN na pose.

## Estado verificado em 02/10/2026

### Teste de uma noite - 10,5 h sem degradar

O responsavel definiu em 01/10/2026 que o sistema deve ficar ligado sem parar,
com cuidado de memoria para nao cair, e escolheu medir antes de limpar. Codigo
`3ab0547`, so o perfil rpi3 no `.env`; API iniciada com `nohup`, fora da sessao
SSH. Uma conexao de 10,48 h com o video de carga em loop, com as opcoes novas
`--loop` e `--samples-jsonl` de `tools/benchmark_stream.py`, e no Pi um
registro a cada 30 s com `vcgencmd`, `free`, `vmstat` e a memoria da API em
`/proc`. Resultado em `resultados/pi3-3ab0547-video-noite/`.

| Hora, no PC | Frame, media | p95 | RSS da API | Temperatura |
|---|---|---|---|---|
| 23h | 3783 ms | 4400 ms | 645 a 662 MB | 49,9 a 60,1 C |
| 00h | 3765 ms | 4412 ms | 484 a 646 MB | 58,5 a 62,8 C |
| 01h a 05h | 3764 a 3770 ms | 4404 a 4418 ms | 485 a 508 MB | 56,4 a 63,4 C |
| 06h a 10h | 3764 a 3772 ms | 4400 a 4429 ms | 486 a 509 MB | 58,5 a 62,8 C |

- **Sem degradacao**: a media por hora ficou entre 3764 e 3783 ms nas 10,5 h, o
  pior frame levou 5,2 s e nenhum intervalo entre amostras passou de 15 s;
  0,265 FPS no total.
- **Sem vazamento de memoria**: na primeira meia hora o sistema mandou cerca de
  210 MB da API para o swap, e o RSS dela caiu de cerca de 650 para 500 MB.
  Depois tudo parou: nas leituras de hora em hora, RSS da API de 501 a 508 MB,
  API no swap de 208 a 210 MB, swap total de 281 a 298 MB e memoria disponivel
  de 316 a 344 MB. A troca com o swap quase nao aconteceu: `si` acima de zero
  em 18 e `so` em 4 das 1320 leituras, no comeco da noite.
- A queda de RSS do teste de 33,6 min tem a mesma explicacao: memoria da API
  mandada para o swap, nao devolvida pelo processo.
- **Temperatura** entre 56 e 63,4 C depois do aquecimento, `throttled=0x0` nas
  1318 leituras.
- **Deteccao estavel**: pessoa, rosto e gesto nos 10.000 frames; a cada hora,
  cerca de 50% dos rostos reaproveitados, 89% reconhecidos e 12 alertas a cada
  36 frames. A gravacao de eventos rodou junto a noite toda.
- Limites: um video de uma pessoa, sem cadastro mudando nem conexao caindo e
  voltando, e 10,5 h, nao dias.

### Leitura para o uso continuo

Com a memoria parada depois da primeira meia hora, uma limpeza periodica nao tem
crescimento para conter. O risco que sobra para ficar ligado sem parar e o
processo cair, por falta de memoria com mais pessoas, falha ou queda da sessao
SSH em que a API e iniciada, e ninguem subir de novo.

### API como servico - volta sozinha em menos de 2 min

O responsavel pediu o servico em seguida. `3caba9b` traz
`tools/citylab-api.service` e `tools/instalar_servico_rpi.sh`: servico do
usuario `citylab`, com `Restart=always`, 10 s de espera e sem limite de
tentativas, e linger ligado para rodar sem ninguem logado. Instalado no Pi em
02/10/2026 as 11:14, com os arquivos conferidos por hash e `Linger=yes`; a API
respondeu um minuto depois.

- Queda simulada com `pkill -9`, o sinal que o sistema usa quando mata por
  falta de memoria: o servico subiu outra API 10 s depois, e ela voltou a
  responder 1 min 51 s depois da queda, quase todo o tempo carregando modelos.
- Reinicio do Pi com `sudo reboot` as 11:43:12, sem ninguem logar depois: o
  servico foi iniciado as 11:43:25 e a API ficou pronta as 11:45:41
  ("Application startup complete" no `journalctl`), cerca de 2,5 min depois do
  comando.
- O deploy continua sem reiniciar a API: parar o servico antes do push e subir
  depois.

### Queda de rede - Wi-Fi reconectado, Pi fora de alcance por 9 min

Logo depois do teste acima, as 11:22:21 pelo relogio do Pi, o Wi-Fi se
reassociou sozinho ao ponto de acesso da mesma rede, agora na faixa de 5 GHz, e
renovou o mesmo IP em 3 s. Para o Pi a rede voltou ali; o PC, porem, ficou sem
alcancar o Pi, nem por ping, SSH ou API, por cerca de 9 min. O mais provavel e o
roteador, ou o PC, ter continuado mandando pacotes pelo caminho antigo ate a
tabela de enderecos expirar. O Pi nao reiniciou, a API seguiu rodando com o
mesmo processo e `throttled` ficou em `0x0`.

- Foi a unica reassociacao no registro desde 01/10 as 11:20, incluindo a noite
  de teste, sem nenhuma pausa no stream.
- A economia de energia do Wi-Fi esta ligada (`Power save: on`), causa comum de
  reconexoes no Raspberry Pi.
- Decisao do responsavel: deixar a rede como esta, como risco conhecido. As
  opcoes eram cabo de rede, a mais firme, ou desligar a economia de energia.

### Cena vazia com o perfil atual - 0,93 s, sem deteccao falsa

Primeira cena dos testes com camera. Codigo `af2872d` no Pi, so o perfil rpi3,
API como servico e reiniciada antes de cada rodada. Webcam do PC apontada para
um canto sem ninguem, 5 frames de aquecimento e 30 medidos, com sondagem de 3
frames que exige o quadro vazio. Resultado em `resultados/pi3-af2872d-vazia/`.

| Rodada | Mediana | p95 | Contra 27/09 |
|---|---|---|---|
| r1 | 930,6 ms | 1195,5 ms | -44,5% |
| r2 | 932,3 ms | 942,8 ms | -44,4% |
| r3 | 936,1 ms | 964,2 ms | -44,2% |

- **Ganho confirmado** contra os 1676,3 a 1679,1 ms de `3e67f56` em 27/09;
  contra a linha de base de 19/09, 5,41 s, -83%. Amplitude de 0,6%.
- O ganho vem das 2 threads no rosto: com a pose pulada pelo gate, o frame e a
  busca por rostos, que roda em todo frame e caiu para cerca de 0,91 s.
- **Sem deteccao falsa**: nenhuma pessoa, rosto, gesto ou alerta nos 90 frames.
- Pose pulada em 87 de 90 frames, sem movimento na imagem. Ela rodou uma vez
  por rodada, na passada obrigatoria de 30 em 30 s, com o frame em cerca de
  2,8 s; por isso a media fica perto de 1 s.
- Temperatura de 45,1 a 51,5 C, `throttled=0x0` em todo o log; RSS da API de
  655 a 709 MB.

### Deploy automatico - para e sobe a API sozinho

Pedido pelo responsavel em 02/10/2026 e entregue em `af2872d`: o workflow para o
servico antes de sincronizar e instalar e o sobe no fim, mesmo se a instalacao
falhar. O runner roda como `citylab`, o mesmo usuario do servico, entao nao
precisa de `sudo`. As dependencias seguem indo para `citylab_venv`, e nao para o
`.venv` da API.

- Primeiro deploy, de `af2872d`, com a API ja parada: o job terminou as
  12:19:51 e a API respondeu as 12:21:03, sem ninguem mexer no Pi.
- Deploy de `61d7961`, com a API rodando, pela vigia do PC: o job ja rodava as
  18:28:25, a API estava fora do ar as 18:28:35, o job tinha terminado as
  18:30:30 e a API voltou as 18:31:23. O passo de parar esta conferido.
- Antes, esse job ficou na fila das 18:17 as 18:28: o runner nao tinha
  conectado depois de o Pi ser ligado, e reiniciar o servico dele resolveu.
  Deploy parado na fila aponta primeiro para o runner.

### Duas pessoas com o perfil atual - 3,41 s, sem perder deteccao

Segunda cena com camera. Codigo `61d7961` no Pi, levado pelo deploy acima, so o
perfil rpi3, API como servico: a r1 pegou o processo subido pelo deploy, e
antes da r2 e da r3 o servico foi reiniciado. Webcam do PC ao vivo, com a cena
de setembro: duas pessoas reais, uma de frente, com cadastro, e outra de lado,
sem cadastro. 5 frames de aquecimento e 30 medidos, com sondagem de 3 frames
que exige duas pessoas e um rosto. Resultado em
`resultados/pi3-61d7961-duas-pessoas/`.

| Rodada | Mediana | p95 | Media | Contra 26/09, mediana | Contra 26/09, media |
|---|---|---|---|---|---|
| r1 | 3550,5 ms | 4663,8 ms | 3706,1 ms | -42,6% | -39,9% |
| r2 | 3405,1 ms | 4578,0 ms | 3633,7 ms | -44,8% | -40,8% |
| r3 | 3389,3 ms | 4469,1 ms | 3588,7 ms | -45,0% | -41,5% |

- **Ganho confirmado** contra `bc0a441`, de 26/09, rodada a rodada, pela
  mediana e pela media; o pior cruzamento da -42,4% na mediana. Contra a linha
  de base de 19/09, 16,62 s, -80% na mediana das medianas. Vazao de 0,162 a
  0,163 para 0,269 a 0,278 FPS. Amplitude entre rodadas de 4,8%: a r1, com a
  API subida pelo deploy, foi a mais lenta.
- O ganho soma tudo o que mudou desde 26/09: pose em 416, 2 threads e reuso no
  rosto e o limite de temperatura em 70 C, que sozinho deixou a serie de uma
  pessoa em 640 de 4,5% a 5,4% mais rapida.
- Com o reuso o frame tem dois ritmos, e a media e a medida justa, como no
  video de uma pessoa: 23 ou 24 dos 30 frames reaproveitam o nome e levam
  cerca de 3,4 s; os 6 ou 7 que geram embedding, cerca de 4,4 s.
- O gesto termina por ultimo nos 90 frames. Com o nome reaproveitado, o rosto
  fica so na deteccao, cerca de 1,2 s, e a pose roda em cerca de 2,7 s, contra
  5,3 a 5,4 s em 26/09. Nos frames com embedding o rosto leva cerca de 3,3 s, a
  pose sobe para cerca de 3,7 s, disputando os nucleos com ele, e o gesto para
  cerca de 4,4 s.
- **Deteccao preservada**: 2 pessoas, 1 rosto e 2 tracks de gesto em 90/90; a
  pessoa de lado continua sem rosto aceito, como em setembro. Confianca das
  caixas de pessoa entre 0,51 e 0,94.
- **Reconhecimento perto do limite**: a pessoa de frente saiu como cadastrada
  em 9, 21 e 25 dos 30 frames, e nos outros como desconhecida. Nas 20
  comparacoes com o cadastro dentro das rodadas, a semelhanca ficou entre 0,424
  e 0,589, mediana de 0,531, e 12 passaram do limite de 0,52. No video fixo,
  gravado a uns 2 ou 3 m, a mesma pessoa sozinha teve mediana de 0,60. O
  limite e o `det_size` de 320 sao os mesmos desde o primeiro commit, e threads
  e reuso nao mudam o embedding. O reuso so agrupa o resultado em blocos de
  15 s: 60% das comparacoes e 61% dos frames reconhecidos. A causa nao foi
  identificada, porque o benchmark nao guarda o tamanho do rosto; luz e rosto
  virado sao candidatos.
- Alertas: 16, 15 e 33 por rodada, em 16, 15 e 23 frames, contra 55 a 60 em
  26/09. Ninguem fez gesto de alerta. O benchmark conta os alertas sem guardar
  a regra que disparou; a mao oculta de quem esta de lado, limitacao conhecida,
  e candidata, mas nao da para afirmar. Pela regra 7, a contagem nao compara
  versoes.
- `process_rss_mb` de 650 a 660 MB, contra 666 a 727 MB em 26/09. Temperatura
  de 50,5 a 56,9 C pela API, contra ate 60,1 C. Sem log de `vcgencmd`, que tinha
  parado depois da cena vazia: `get_throttled` deu `0x0` as 19:37 com o Pi
  ligado desde 17:50 sem reiniciar, entao nao houve subtensao, corte de clock
  nem limite de temperatura desde o boot.

### O que fica aberto

1. Com camera: dois rostos de frente, com videos que o responsavel vai gravar,
   e por ultimo duas pessoas trocando de lugar, que mostra se o reuso troca
   nomes. Cena vazia e duas pessoas foram feitas. A linha de uma pessoa do
   README ainda e de 01/10, antes das 2 threads e do reuso, que so foram
   medidos com video.
2. Reconhecimento perto do limite com duas pessoas, acima. Para achar a causa,
   o benchmark pode passar a guardar a largura do rosto e o nome dos alertas,
   sem imagem nem nome de pessoa.
3. De lado e punho levantado em 416: o responsavel decidiu em 02/10/2026 nao
   gravar videos novos para comparar com 640. Fica como risco conhecido, ja
   anotado no README.
4. Testes de dias, agora com a API como servico. Reinicio programado ou limpeza
   de memoria so se aparecer crescimento.
5. Rede: Wi-Fi com economia de energia e uma queda de 9 min em 24 h, mantido
   por decisao do responsavel.
6. P4 e P6 da fila de 01/10, sem acao combinada.
7. O resto do "Encerramento em 27/09/2026".

## Estado verificado em 01/10/2026

O responsavel reabriu o plano para a acao 7, so na pose: o `det_size` do
InsightFace ja estava fixo em 320. A pose rodava sem `imgsz`, entao o
Ultralytics usava 640 e ampliava o frame de 320x240 que sai da `PROCESS_SCALE`.
Depois da medicao no Pi, no mesmo dia, o responsavel promoveu 416 a padrao do
perfil rpi3. Na sequencia vieram as 2 threads no rosto e o reuso da identidade
do rosto, os dois promovidos; 2 threads no PyTorch e o spinning desligado do
ONNX Runtime foram medidos e nao entraram.

### Medicao no PC

Medido no PC, sem o Pi, com os videos de validacao de `docs/PLANO_GESTOS.md`
(webcam, 18 videos de 30 s) e os seis do celular, extraidos com cada tamanho.
O tempo e a mediana de 60 frames de 320x240 num processo so; os alertas, as
fases em que o alerta esperado dispara, de 62 por video a cada 6,2 s:

| Pose | Tempo no PC | Mao oculta, webcam | Rendicao, webcam | Mao oculta, celular |
|---|---:|---:|---:|---:|
| 640 | 54,5 ms | 184/186 | 167/186 | 26/62 |
| 416 | 33,5 ms (-39%) | 182/186 | 168/186 | 23/62 |
| 320 | 25,4 ms (-53%) | 130/186 | 151/186 | 23/62 |

- A pessoa foi vista em todos os frames da webcam nos tres tamanhos.
- Em 320, mao oculta e rendicao perderam deteccao: reprovado pela regra 4.
- Em 416, os alertas esperados ficaram iguais na webcam, inclusive mao fechada
  e ameaca, com 185 e 48 de 186 contra 186 e 48. No celular, a mao oculta caiu
  de 26 para 23 fases, a mao fechada da ameaca de 48 para 44, e o braco
  estendido ficou em 42. Os alarmes falsos mudaram para os dois lados: a
  ameaca falsa no braco aberto subiu de 75 para 84 fases, e a mao oculta falsa
  na rendicao caiu de 20 para 4.

Implementado em `c1f1ac0`: `POSE_IMGSZ`, ainda desligado por padrao, com zero
mantendo os 640, e mostrado em `tools/run_rpi.py --show-config`, para comparar
640 e 416 no Pi com o mesmo codigo, trocando so o `.env`.

### Medicao no Pi - frame 6% mais rapido com uma pessoa

Codigo medido: `c1f1ac0`, conferido no Pi por hash e com `--show-config`; as
duas series so diferem no `POSE_IMGSZ=416` do `.env`. Tres rodadas de uma
pessoa em cada tamanho, conforme `docs/BENCHMARK.md`: API reiniciada antes de
cada uma, sondagem de 3 frames, runner do GitHub Actions parado e log de
`vcgencmd` rodando junto. Resultados em `resultados/pi3-c1f1ac0-pose640/` e
`resultados/pi3-c1f1ac0-pose416/`.

| Rodada | 640 | 416 | Diferenca |
|---|---|---|---|
| r1 | 5632,8 ms | 5299,0 ms | -5,9% |
| r2 | 5673,9 ms | 5317,4 ms | -6,3% |
| r3 | 5687,9 ms | 5319,8 ms | -6,5% |

- **Ganho confirmado**: acima de 5% nas tres rodadas. Mesmo a rodada mais lenta
  de 416 contra a mais rapida de 640 da -5,6%. Amplitude entre rodadas: 1,0% em
  640 e 0,4% em 416, sem tendencia dentro de nenhuma rodada.
- **Recall preservado**: pessoa, rosto e gesto em 30/30 frames nas seis rodadas
  e zero alertas. Confianca da caixa da pessoa entre 0,81 e 0,93 em 416, contra
  0,85 e 0,91 em 640.
- A pose caiu de 5012 a 5041 ms para 2232 a 2241 ms (-55%), mas o frame so 6%.
  Com uma pessoa o rosto ja terminava por ultimo (24, 24 e 29 de 30 frames em
  640) e agora termina por ultimo em 30/30. O ganho vem do rosto, de 5600 a
  5660 ms para 5276 a 5294 ms, que fica com a CPU livre depois da pose.
- `process_rss_mb` entre 661,0 e 673,6 MB, contra 677,5 e 712,8 MB em 640.
  Temperatura maxima de 54,8 C, contra 63,4 C.
- Temperatura: `throttled=0x0` nas 3967 leituras do log, das 12:14 as 17:47
  pelo relogio do Pi, com maxima de 63,9 C no fim da terceira rodada de 640 e o
  clock nunca em 1,2 GHz. Primeira medicao com o `temp_soft_limit` em 70 C, que
  nao chegou a atuar. Por isso a serie de 640 ficou 4,5% a 5,4% mais rapida que
  a de `bc0a441`, que em 26/09 rodou com o limite de 60 C ativo em parte do
  tempo: e efeito do limite, nao do codigo.

### Rodada com gesto em 416

Mesmo coletor e mesmas nove situacoes da "Rodada com gesto no Pi" de
`docs/PLANO_GESTOS.md`, com a API em 416. Resultado em
`resultados/pi3-c1f1ac0-pose416-gestos/rodada-gestos-r1.json`, comparado com as
rodadas de 640 de 30/09 em `resultados/pi3-338ac06-gestos/`: a r2 para as duas
poses de braco, que na r1 foram feitas com o braco apontado para a camera.

Primeiro frame em que cada alerta aparece, de 6 por situacao: F mao fechada, A
mao fechada com braco estendido, B braco estendido, R rendicao, O mao oculta.

| Situacao | Esperado | 640, 30/09 | 416 |
|---|---|---|---|
| Neutro, punho com braco solto, de costas | nenhum | nenhum | nenhum |
| Mao oculta | O3 | O3 | O3 |
| Rendicao | R3 | R3 e F a mais | R3 e F a mais |
| Ameaca | F2 A2 B4 | B4 | B4 |
| Braco estendido | B4 | B4, F e A a mais | B4, F e A a mais |
| Punho levantado | F2 | F2 | F3 |
| De lado | O3 | O4 | nenhum |

- Igual em sete situacoes, inclusive nas limitacoes ja conhecidas: o punho na
  ponta do braco estendido nao e lido, e a mao aberta com o braco levantado e
  lida fechada.
- De lado, a mao oculta nao disparou em 416. Nesta rodada o rosto foi
  reconhecido em 3 dos 6 frames, contra nenhum em 30/09: a pessoa estava menos
  de perfil.
- Punho levantado: o alerta veio um frame depois, com o punho lido em 2 frames,
  contra 3.
- Com 6 frames por situacao e a pose da pessoa diferente em cada dia, nao da
  para separar o efeito do tamanho do efeito da cena. Essas duas situacoes nao
  estao nos videos de validacao.
- Frame de 5,3 a 5,4 s, contra 5,6 a 5,9 s em 30/09. De costas, sem rosto, 2,7
  s contra 5,3 s: sem rosto, o caminho critico volta a ser a pose. Sao 6
  frames, nao medicao de cenario.

### Decisao

O responsavel promoveu 416 a padrao do perfil rpi3 em 01/10/2026, com as duas
situacoes acima sem conferencia. `POSE_IMGSZ=0` volta aos 640. No perfil
default o valor segue 640, porque nao foi medido.

### Rosto com 2 threads - frame 20% mais rapido com uma pessoa, em video fixo

Com a pose em 416, o rosto ficou sozinho no caminho critico, com 1 thread do
ONNX Runtime. Codigo medido: `89c93b8`; as duas series so diferem em
`ONNX_INTRA_OP_THREADS` no `.env`. O valor 2 foi conferido no Pi com
`--show-config`; o 1, pelo tempo do rosto, igual ao da serie de 416.

O responsavel nao podia ficar na frente da camera, entao a carga foi um video
fixo, no modo de video de `tools/benchmark_stream.py`: 36 frames de 640x480
tirados dos videos r1 das seis situacoes de validacao de `docs/PLANO_GESTOS.md`,
6 de cada, um a cada 5 s, como se o Pi processasse um frame a cada 5 s. Todas as
rodadas recebem os mesmos frames. O video tem o rosto de quem gravou e fica
fora do repositorio; os JSONs guardam so o SHA-256 dele. API reiniciada antes de
cada rodada, 5 frames de aquecimento e 30 medidos. Resultados em
`resultados/pi3-89c93b8-video-onnx1/` e `resultados/pi3-89c93b8-video-onnx2/`.

| Rodada | 1 thread | 2 threads | Diferenca |
|---|---|---|---|
| r1 | 5388,9 ms | 4291,4 ms | -20,4% |
| r2 | 5324,0 ms | 4295,3 ms | -19,3% |
| r3 | 5335,7 ms | 4279,0 ms | -19,8% |

- **Ganho confirmado**: mesmo a rodada mais lenta de 2 threads contra a mais
  rapida de 1 da -19,3%. Amplitude entre rodadas: 1,2% com 1 thread e 0,4% com
  2.
- **Deteccao identica frame a frame** nas seis rodadas: pessoa, rosto e gesto
  em 30/30, rosto reconhecido como cadastrado em 25/30 e 11 alertas; a
  semelhanca com o cadastro mudou no maximo 0,0001. Os 5 frames sem
  reconhecimento sao os de rosto menor ou mais virado do video, os mesmos nas
  duas series.
- O rosto caiu de 5292 a 5340 ms para 3346 a 3414 ms (-36%), mas a pose subiu
  de 2235 a 2244 ms para 3681 a 3691 ms: com 2 threads no rosto e 3 no PyTorch,
  os dois disputam os 4 nucleos. Agora o gesto termina por ultimo em 30/30.
- Temperatura de 52,1 a 59,6 C, contra 47,8 a 53,7 C com 1 thread, pelo
  `temperature_c` da API. O log de `vcgencmd` rodou junto, mas o resumo dele
  nao foi coletado. Rodadas de 2,5 a 3 min; uso continuo nao foi medido.
- No PC, com os modelos reais e o mesmo video, o embedding foi 81% do rosto com
  1 thread (271 de 334 ms), a deteccao 19% e a comparacao com o cadastro 0,05
  ms. No Pi, a decomposicao entra no proximo deploy.

### Promocao das 2 threads e reuso da identidade do rosto

O responsavel promoveu `ONNX_INTRA_OP_THREADS=2` a padrao do perfil rpi3
(`d98da86`). A divisao de threads entre rosto e pose fica para depois do reuso
abaixo, que muda a carga do rosto.

No mesmo commit entrou `FACE_REUSE_SECONDS`, desligado por padrao. Um rosto
aceito pelo filtro de qualidade cuja caixa cobre pelo menos metade da caixa de
um rosto do frame anterior (IoU de 0,5 ou mais) herda nome e semelhanca do
reconhecimento feito ha menos de N segundos, sem gerar embedding. Regras
combinadas com o responsavel: 15 s de validade, contada do reconhecimento e nao
do ultimo reuso; vale tambem para rosto desconhecido; quando alguem entra ou
sai, so os rostos novos sao reconhecidos. Conexao nova, pausa de mais de
`GESTURE_IDLE_RESET_SECONDS` e cadastro alterado esquecem os nomes. A deteccao
continua em todo frame, entao os episodios de evento seguem as deteccoes reais.
Consequencia aceita: enquanto a identidade vale, o nome aparece mesmo num frame
em que o reconhecimento falharia.

No PC, com o video de carga e um frame a cada 4,3 s, houve reuso em 16 de 30
frames. Nos outros, o rosto, com cerca de 27 px na imagem processada, andou o
bastante para a sobreposicao ficar abaixo de metade. A mediana do rosto caiu de
187 para 40 ms.

### Reuso no Pi - frame 13% mais rapido na media, com uma pessoa em video fixo

Codigo medido: `419ca7f`, que traz `9334b9e` e `d98da86`, conferido no Pi por
hash e com `--show-config`; as duas series so diferem em `FACE_REUSE_SECONDS`
no `.env`. Mesmo video e protocolo da serie de threads acima. Resultados em
`resultados/pi3-419ca7f-video-reuso0/` e `resultados/pi3-419ca7f-video-reuso15/`.

Primeiro, a decomposicao do rosto no Pi, com 2 threads e sem reuso: deteccao
1173 ms (34%), embedding 2250 ms (66%) e comparacao com o cadastro 0,25 ms, na
mediana da r1. O reuso corta o embedding.

| Rodada | Sem reuso, media | Com 15 s, media | Diferenca | Mediana com 15 s |
|---|---|---|---|---|
| r1 | 4284,3 ms | 3709,3 ms | -13,4% | 3324,7 ms (-22,8%) |
| r2 | 4270,7 ms | 3710,2 ms | -13,1% | 3359,5 ms (-22,3%) |
| r3 | 4266,7 ms | 3739,0 ms | -12,4% | 3491,5 ms (-18,6%) |

- **Ganho confirmado** pela media e pela mediana nas tres rodadas; o pior
  cruzamento da -12,4% na media. Vazao de 0,233 a 0,234 para 0,267 a 0,269
  FPS. Sem reuso, as medianas foram 4309,0, 4321,5 e 4290,8 ms.
- **A media e a medida justa**: o frame fica com dois ritmos. Com o nome
  reaproveitado, 16 dos 30 frames, os mesmos nas tres rodadas, levam cerca de
  3,2 s; os que reconhecem de novo continuam em cerca de 4,3 s. A mediana cai
  num ritmo ou no outro conforme a rodada e exagera o ganho.
- Nos frames com reuso o rosto fica so na deteccao, cerca de 1,2 s, e a pose
  cai de 3,7 para 2,6 s, porque para de disputar os nucleos. O gesto continua
  terminando por ultimo em 30/30 nas duas series.
- **Deteccao**: os mesmos 11 alertas, frame a frame. O rosto aparece como
  cadastrado em 26/30 contra 25/30: no frame 15 o nome foi herdado de um frame
  em que o reconhecimento passou, embora sozinho ele ficasse em 0,516, logo
  abaixo do limite de 0,52. E a consequencia aceita. Um frame que vinha como
  desconhecido continuou desconhecido.
- Temperatura maxima de 56,9 a 58,0 C, contra 59,1 a 60,1 C sem reuso. Log de
  `vcgencmd` da serie sem reuso: `throttled=0x0` nas 378 leituras.
- A troca de nome entre duas pessoas que trocam de lugar nao aparece num video
  de uma pessoa e nao foi testada.

### PyTorch com 2 threads - reprovado

No lugar da cadencia propria dos gestos, que pelos numeros acima nao ganharia
nada, o responsavel pediu medir `TORCH_NUM_THREADS=2` com o reuso ligado, para
tirar a disputa de nucleos. Base: a serie com reuso acima. Resultado em
`resultados/pi3-419ca7f-video-torch2/`; uma primeira tentativa rodou com 3
threads e ficou em `descartadas/`.

- Media de 3620,0 ms contra 3709,3 a 3739,0 ms: -2,4% a -3,2%, abaixo do
  criterio. Como a r1 ja nao passa, a serie parou nela.
- O frame fica mais regular, com p95 de 3820,5 ms contra 4343,1 a 4489,2 ms: a
  pose fica mais lenta nos frames com reuso (cerca de 2,9 s) e mais rapida nos
  que reconhecem (cerca de 3,2 s). A mediana piora, 3606,8 ms.
- Fica em 3 threads.

### Spinning do ONNX Runtime desligado - sem efeito

Passo 4 de P1: `ONNX_ALLOW_SPINNING=0` faz as threads do rosto dormirem entre
operadores em vez de girar a espera (`session.intra_op.allow_spinning`). Entrou
desligado por padrao em `3ab0547`, junto com a promocao do reuso. Medido com
esse commit no Pi, conferido por hash, ONNX Runtime 1.23.2. Base: a serie com
reuso acima. Resultado em `resultados/pi3-3ab0547-video-spin0/`.

- Media de 3733,8 ms contra 3709,3 a 3739,0 ms: igual. Os tempos de cada
  etapa tambem. A serie parou na r1, e a opcao fica no padrao da biblioteca.

### Decisoes

O responsavel promoveu `FACE_REUSE_SECONDS=15` a padrao do perfil rpi3
(`3ab0547`); no perfil default segue desligado. PyTorch fica em 3 threads e o
spinning no padrao. A cadencia propria dos gestos, segunda parte de P5, ficou
fora: com o gesto ja no caminho critico em todos os frames, separar o rosto nao
encurta o frame.

### Teste continuo de 33,6 min - estavel

Parte de P0 que nao precisa de camera. Codigo `3ab0547` no Pi, so o perfil
rpi3 no `.env` (pose em 416, 2 threads no rosto, 3 no PyTorch, reuso de 15 s),
API reiniciada antes. Uma conexao so, 5 frames de aquecimento e 535 medidos, com
os 36 frames do video de carga repetidos 15 vezes; no Pi, um registro a cada 5
a 6 s com `vcgencmd`, `free` e `vmstat`. Resultado em
`resultados/pi3-3ab0547-video-continuo/`.

- **Velocidade estavel**: media de 3766,8 ms e 0,265 FPS no total. Cada volta
  de 36 frames ficou entre 3738 e 3792 ms, 3775 ms na primeira completa e 3768
  ms na ultima, sem piora ao longo da rodada. Igual as rodadas curtas com reuso.
- **Deteccao estavel**: pessoa, rosto e gesto em 535/535; em toda volta
  completa, 18 rostos reaproveitados, 32 reconhecidos como cadastrados e 12
  alertas.
- **Temperatura**: sobe ate 59 a 61 C em uns 10 min e fica ali; maxima de 62,3
  C. `throttled=0x0` nas 532 leituras do registro.
- **Memoria**: o swap ja estava em 255 MB antes da rodada, chegou a 302 MB e
  terminou em 267 MB; a memoria disponivel minima foi 150 MB, no inicio. Houve
  troca com o swap em 34 de 532 leituras (`si` ate 8156 KiB/s) e saida para ele
  em 4 (`so` ate 10568 KiB/s): rajadas, nao troca continua, e sem efeito
  visivel no tempo dos frames. O `process_rss_mb` da API caiu de cerca de 686
  para 573 a 588 MB na 12a volta, quando o swap ja estava descendo; parece
  memoria devolvida pelo proprio processo, mas sem o swap por processo nao da
  para afirmar.
- Uma rodada de 33,6 min nao prova horas de uso; a primeira janela ficou
  estavel, como pede P0 para estender.

### O que fica aberto

1. Com camera: cena vazia, duas pessoas e dois rostos de frente, que nao foram
   medidos com a pose em 416, 2 threads e o reuso. Duas pessoas trocando de
   lugar mostram se o reuso troca nomes.
2. Uso continuo: o responsavel definiu em 01/10/2026 que o sistema deve ficar
   ligado sem parar. Falta um teste de horas, e a memoria e o ponto de atencao:
   o swap passa de 250 MB. Hoje a API e iniciada a mao num terminal SSH, entao
   cair a conexao derruba a API.
3. Conferir, nos mesmos frames em 640 e 416, as duas situacoes que mudaram na
   rodada com gesto: de lado e punho levantado. Pede gravar videos novos, como
   os de validacao, e comparar no PC.
4. P4 e P6 da fila de 01/10, sem acao combinada.
5. O resto do "Encerramento em 27/09/2026". A acao 7 ficou feita na pose, e o
   item 1 de la ganhou dados: em rodadas de 3,5 min com uma pessoa o Pi ficou em
   ate 63,9 C, e em 33,6 min seguidos com o perfil atual, em ate 62,3 C, sem
   atingir o limite de 70 C.

## Encerramento em 27/09/2026

O responsavel encerrou o trabalho de desempenho em 27/09/2026, por considerar
que o Pi 3 B+ chegou ao limite com este pipeline. O plano fica como registro;
um item novo o reabre, combinado com o responsavel pelas regras acima.

Resultado final, mediana de `rtt_ms` com webcam e pipeline completo:

| Cenario | Linha de base (`b05058f`) | Final | Ganho |
|---|---|---|---|
| Cena vazia | 5409,98 ms | 1676,3 a 1679,1 ms (`3e67f56`) | -69% |
| Uma pessoa | 13413,5 ms | 5930,5 a 5997,3 ms (`bc0a441`) | -55% a -56% |
| Duas pessoas | 16615,63 ms | 6139,1 a 6182,2 ms (`bc0a441` e `6f5756d`) | -63% |

Recall preservado em todos os cenarios, e a caixa de pessoa falsa da linha de
base sumiu. As alavancas foram a passada unica de pessoas e pose (acao 3), o
paralelismo com threads limitadas (acao 4), o gate de movimento, o limiar de
publicacao e o limite de threads do PyTorch reaplicado em cada tarefa.

Configuracao do Pi no encerramento: perfil rpi3 como em `.env.rpi.example` e
`temp_soft_limit=70` em `/boot/firmware/config.txt`, aplicado em 27/09/2026 e
conferido com `vcgencmd get_config`, sem rodada medida depois. A copia do
arquivo original ficou em `/boot/firmware/config.txt.bak`. A documentacao do
Raspberry Pi avisa que subir o limite acima de 60 C pode causar instabilidade.

Conferido no encerramento: a equivalencia do reconhecimento facial com
`FACE_MINIMAL_MODULES` e `FACE_PREFILTER`, com `tools/check_face_optimization.py`
e os modelos reais no PC (InsightFace 0.7.3, ONNX Runtime 1.23.2). Em 3 fotos
de webcam com duas pessoas, o rosto aceito de cada foto teve a mesma caixa,
identidade, embedding e similaridade nos dois caminhos. As fotos foram
apagadas depois da conferencia.

Fica para depois, sem data:

1. Rodada longa com o `temp_soft_limit` em 70 C, para ver se o Pi se estabiliza
   abaixo do limite com o stream ligado direto.
2. Cena vazia com o codigo final.
3. Acoes 7 e 8 do backlog e o NCNN na pose. A acao 9 foi feita em 29/09/2026.
4. Regras de gesto: acao 4 de `docs/PLANO_GESTOS.md`. As acoes 1, 2 e 5 foram
   feitas em 29/09/2026 e conferidas no Pi em 30/09/2026.

## Estado verificado em 27/09/2026

### Um evento por episodio e limpeza de codigo - latencia neutra

Codigo medido: `6f5756d`, que inclui `05bc7ce`, a acao 3 de
`docs/PLANO_GESTOS.md`. O Pi recebeu o deploy de `614316b`, que so muda
documentacao e `.gitignore`; os hashes dos sete arquivos de codigo alterados
desde `bc0a441` foram conferidos no dispositivo, e `tools/run_rpi.py
--show-config` mostrou a mesma configuracao da serie de `bc0a441`. Uma rodada
de duas pessoas, mesma cena (uma de frente, outra de lado), API reiniciada
antes, sondagem de 3 frames e log de `vcgencmd` rodando junto. Resultado em
`resultados/pi3-6f5756d-eventos/duas-pessoas-r1.json`.

| Cenario | `bc0a441` | Agora | Diferenca |
|---|---|---|---|
| Duas pessoas | 6182,2 / 6168,9 / 6159,8 ms | 6139,1 ms | -0,3% a -0,7% |

- Latencia neutra: a diferenca fica dentro da variacao entre rodadas, e parte
  dela e o `logs_ms`, que caiu cerca de 30 ms por frame com a acao 3.
- Recall preservado: 2 pessoas e 2 tracks de gesto em 30/30 frames, 1 rosto em
  29/30 e 2 rostos em 1/30. O frame com 2 rostos levou 9621 ms, o maior da
  rodada.
- `logs_ms` com mediana de 30,7 ms, contra 58,5 a 62,2 ms na serie de
  `bc0a441`. A contagem de eventos no MongoDB esta em `docs/PLANO_GESTOS.md`.
- `process_rss_mb` entre 709,3 e 734,2 MB, e entre 732,0 e 743,0 MB na rodada
  com ar-condicionado da secao seguinte. As duas passam do teto de 719 MB da
  fase 1; 743,0 MB e o maior valor desde os 748,5 MB de 20/09, sem o gate.
- Condicao registrada: segundo o responsavel, o ar-condicionado da sala do Pi
  nunca ficou ligado nas rodadas anteriores a esta secao. Rodada com o ar
  ligado e outra condicao termica e nao se compara com elas.

### Ar-condicionado na sala - adia o limite de temperatura, mas nao evita

Segunda rodada, mesmo codigo, mesma cena e mesmo protocolo, com o ar da sala em
21 C e ventilacao maxima, sem vento direto no Pi, ligado cerca de 20 min antes.
Resultado em `resultados/pi3-6f5756d-ar/duas-pessoas-r1.json`; os logs de
`vcgencmd` das duas rodadas estao nas pastas `evidencia/` de cada uma.

| | Sem ar | Com ar |
|---|---|---|
| Pi parado antes da rodada | 42 a 43 C | 39 C |
| Limite ativo pela primeira vez | 70 s apos o inicio | 146 s apos o inicio |
| Leituras com o limite ativo, dali ao fim | 22 de 29 (76%) | 6 de 13 (46%) |
| Frames terminados antes do limite (mediana) | 5795,2 ms | 5806,0 ms |
| Frames inteiros no periodo limitado (mediana) | 6169,3 ms (+6,5%) | 5983,7 ms (+3,1%) |
| Rodada (mediana de `rtt_ms`) | 6139,1 ms | 5929,9 ms (-3,4%) |

- Abaixo do limite as duas rodadas tem a mesma velocidade. O ar nao acelera o
  Pi: so adia o limite e reduz a parte do tempo em que o firmware baixa o clock.
- Sem ar, o limite custou 6,5% no periodo limitado, contra 3,2% a 4,2% nas
  rodadas de duas pessoas de 26/09, quando ficava ativo em cerca de metade das
  leituras. A perda acompanha a fracao do tempo a 1,2 GHz.
- Os -3,4% vem de uma rodada de cada lado e ficam abaixo do criterio de 5%. Nao
  sao ganho confirmado; medem o efeito da temperatura com o codigo igual.
- No fim da rodada com ar o Pi ja estava em 59 a 60 C, com o limite ativo em
  metade das leituras. Uma rodada dura cerca de 3,5 min e o stream roda por
  horas, entao em uso continuo o Pi chega ao limite com ou sem ar. Uma medicao
  mais longa nao foi feita.
- Recall igual: 2 pessoas, 1 rosto e 2 gestos em 30/30 frames.

### O que fica aberto

Ver "Encerramento em 27/09/2026". Com duas pessoas, no periodo limitado, o
limite de temperatura custou 6,5% por frame com a sala sem ar e 3,1% com o ar
ligado; com esses numeros o responsavel decidiu subir o `temp_soft_limit`.

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
  A causa esta na secao seguinte. A serie de uma pessoa de 21/09 confirma a regra:
  pose entre 6934 e 7138 ms em todos os 90 frames, e em nenhum deles o rosto
  terminou depois do gesto, entao os papeis nunca trocaram.
- Oportunidade, nao ganho medido: com o gesto sempre na configuracao rapida, o
  frame de duas pessoas cairia de cerca de 7,6 s para 6,0 a 6,3 s, perto de 20%.
  Exige registrar worker e threads por frame antes de mexer em qualquer coisa.

### Causa da pose bimodal - limite de threads trocado pelo Ultralytics

- Causa: na primeira chamada de `track()`, o `setup_model` do Ultralytics chama
  `select_device`, que executa `torch.set_num_threads(NUM_THREADS)`, com
  `NUM_THREADS = min(8, nucleos - 1)`: 3 no Pi. O limite vale por thread, entao
  so o worker que rodou o primeiro `track()` passa a 3; o outro fica nos 2
  configurados pelo inicializador. A fase rapida e a pose com 3 threads; a
  lenta e a pose com 2, que era o valor do perfil.
- Reproduzido no PC com as versoes do projeto (Ultralytics 8.3.226, torch
  2.9.0) e o mesmo inicializador: o worker do primeiro `track()` terminou com 8
  threads, o `NUM_THREADS` do PC, e o outro com 2. No pipeline real, com o
  servico de gesto real e o rosto simulado para forcar a troca de workers, a
  pose alternou entre cerca de 54 ms com 8 threads e 80 ms com 2, conforme o
  worker.
- Correcao: o pipeline reaplica `TORCH_NUM_THREADS` no inicio de cada tarefa de
  gesto (`ensure_torch_threads`), so quando o valor da thread diverge, porque
  `set_num_threads` limpa o cache do oneDNN. No mesmo teste do PC, os dois
  workers ficaram em 2 threads e a pose entre 76 e 83 ms em todos os frames.
- O perfil rpi3 passou de 2 para 3 threads do PyTorch, o valor da fase rapida.
  Nos frames dessa fase em 26/09, a pose ficou entre 5,0 e 5,3 s e o frame
  entre 6,0 e 6,3 s, contra 6,94 a 6,99 s e 7,6 a 7,8 s com 2 threads.
- O pipeline passou a publicar `gesture_worker`, `face_worker` e
  `gesture_torch_threads` por frame, para a medicao mostrar qual worker rodou a
  pose e com quantas threads.
- Medido no Pi no mesmo dia, secao seguinte.

### Pose sempre em 3 threads - ganho confirmado com uma e duas pessoas

Commit medido: `bc0a441`, perfil rpi3 com `TORCH_NUM_THREADS=3`, conferido no
Pi com `--show-config` e com os hashes dos arquivos alterados. Resultados em
`resultados/pi3-bc0a441-threads/`. API reiniciada antes de cada rodada. Uma
rodada de duas pessoas foi descartada por cena instavel e repetida; o motivo
esta em `descartadas/LEIAME.md`.

| Cenario | Linha de base | `3e67f56` | Agora | Ganho sobre `3e67f56` |
|---|---|---|---|---|
| Uma pessoa | 13413,5 ms | 7276,8 ms | 5930,5 / 5997,3 / 5954,4 | **-17,6% a -18,5%** |
| Duas pessoas | 16615,63 ms | 7597,0 ms | 6182,2 / 6168,9 / 6159,8 | **-18,6% a -18,9%** |

- Contra a linha de base: -55,3% a -55,8% com uma pessoa e -62,8% a -62,9% com
  duas. Amplitude entre rodadas: 1,1% e 0,4%.
- **As duas velocidades sumiram**: a pose rodou com 3 threads nos 180 frames, e
  os dois workers ficaram com a mesma pose, entre 5253 e 5451 ms de mediana por
  worker e rodada. Antes eram 5,0 contra 7,0 s.
- **Recall preservado**: com uma pessoa, pessoa, rosto e gesto em 90/90 frames
  e zero alertas; com duas, 2 pessoas, 1 rosto e 2 gestos em 90/90, como na
  serie de `3e67f56`.
- Com uma pessoa o rosto virou o caminho critico em 86 de 90 frames: a pose
  termina em cerca de 5,3 s e o rosto em 5,9 s. Com duas, rosto e gesto
  terminam quase juntos, com o rosto por ultimo em 18 de 90 frames.
- `process_rss_mb` entre 665,7 e 726,8 MB. A r3 de duas pessoas passou do teto
  de 719 MB da fase 1, com 723 a 727 MB; as outras cinco ficaram abaixo. A
  variacao entre reinicios ja era conhecida, mas com o swap visto no mesmo dia a
  folga real e menor do que o RSS sugere.
- Todas as rodadas chegaram a 59-60 C e o limite de clock atuou, secao
  seguinte. Os numeros acima ja incluem esse efeito.
- Cena vazia nao foi medida com `bc0a441`. Com o gate a pose roda em 2 de 30
  frames, entao a mediana nao deve mudar.

### Limite de temperatura - o Pi baixa o clock no meio da rodada

- Com a pose em 3 threads e o rosto em 1, os quatro nucleos ficam ocupados e o
  Pi atinge o `temp_soft_limit` padrao do 3 B+, 60 C, cerca de 75 s depois do
  inicio da rodada. Ao atingi-lo, o firmware baixa o clock de 1,4 para 1,2 GHz.
- Evidencia em `resultados/pi3-bc0a441-threads/evidencia/`: um log de
  `vcgencmd` a cada 5 s durante a rodada `duas-pessoas-r2`. A rodada comeca a
  1,4 GHz e 45,1 C; aos 59,1 C aparece `throttled=0x80008`, limite ativo
  naquele instante, com 1,2 GHz. Dali ao fim, o firmware alterna entre 1,2 e
  1,4 GHz para segurar 58,5 a 60,7 C: limite ativo em 15 de 28 leituras e
  1,2 GHz em 13 delas, uma media perto de 1,3 GHz.
- Efeito medido: frames a 59 C ou mais ficaram 6,0% a 7,0% mais lentos que os
  anteriores nas tres rodadas de uma pessoa, e 3,2% a 4,2% nas de duas. E a
  subida lenta de 5,7 para 6,1 s que aparece dentro de cada rodada.
- `vcgencmd get_throttled` ainda dava `0x0` depois da primeira rodada de duas
  pessoas de `3e67f56`, com a pose em 2 threads na maior parte dos frames. A
  primeira leitura com o bit `0x80000`, limite ja atingido desde o boot, veio
  depois da primeira rodada de uma pessoa de `bc0a441`.
- Leitura feita depois da rodada engana: o bit `0x80000` fica gravado, mas o
  clock ja voltou ao repouso de 600 MHz e a temperatura caiu. O log precisa
  rodar durante a medicao; o comando esta em `docs/RASPBERRY_PI.md`.
- Consequencia: parte do ganho de `bc0a441` fica com a temperatura, e a linha
  de base, que rodou entre 59 e 60 C, pode ter sido afetada sem registro. Um
  dissipador com ventoinha deve manter o frame no valor de antes do limite.
  Subir o `temp_soft_limit` (ate 70 C no 3 B+) e a outra opcao, com a placa
  mais quente. Decisao do responsavel.

### Memoria do sistema - o runner nao e a causa do swap

- Com a API desligada, o runner ocioso ocupava 7 MB de RAM e 40 MB de swap.
  Parar o runner liberou 72 MB de swap (104 para 32 de 511 MB) e nada de
  memoria disponivel (787 para 785 MB). Dados em
  `resultados/pi3-bc0a441-memoria/`.
- O swap das rodadas vem da propria API: cerca de 700 MB residentes numa placa
  de 906 MB, com o sistema usando uns 120 MB. O kernel manda para o swap o que
  esta frio. O arquivo de swap tem 511 MB e o maior uso visto foi 236 MB, entao
  OOM exigiria a API crescer bem alem do que se mediu.
- As rodadas nao mostram efeito do swap na latencia: amplitude de 0,4% entre as
  rodadas de duas pessoas de `bc0a441`, e a rodada extra com o runner ligado deu
  6117,0 ms, dentro da serie.
- Conclusao: o runner pode ficar ligado. Folga de memoria continua pequena, mas
  estavel; reduzir a RAM da API so vale se um cenario novo, como mais pessoas ou
  mais cameras, apertar.

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

1. Cena vazia com `bc0a441`, que nao deve mudar porque o gate pula a pose em 28
   de 30 frames.
2. **Temperatura.** Decidir entre dissipador com ventoinha e `temp_soft_limit`
   maior, e medir de novo com o `vcgencmd` rodando junto. Adiado pelo
   responsavel em 26/09/2026.
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

- Cliente: timestamp por frame pendente (`state.pendingSentAt`; desde a acao 9,
  `state.pendingFrames`, pelo numero do frame) em
  `Client/stream.js`.
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

Registro historico da fila derivada da linha de base de 19/09. Algumas acoes
abaixo ja foram feitas; os estados posteriores registram o resultado. Para
decidir trabalho novo, usar "Pesquisa e planejamento em 01/10/2026".

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

Feita em 29/09/2026, sem medicao, porque nao muda latencia:

- O servidor numera os frames recebidos em cada conexao e devolve `frame` em
  toda resposta, inclusive nas de erro; o cliente casa a resposta pelo numero.
  O envio nao muda, entao `tools/benchmark_stream.py` segue igual.
- Um frame sem resposta por 30 s fecha o socket e agenda a reconexao, que
  reinicia a numeracao dos dois lados. No Pi, um frame leva de 6 a 10 s.
- Ao pausar ou perder a conexao, o cliente apaga os ultimos resultados, que
  antes ficavam desenhados sobre o video ao vivo sem analise nenhuma.

## Protocolo de comparacao

Mesma camera, cenario e configuracao, mesmo estado inicial do banco e
temperatura comparavel. A webcam nao repete os mesmos frames. Registrar em cada
rodada: mediana e p95 de `rtt_ms`,
`completed_fps`, tempos por estagio, `process_rss_mb`, `temperature_c` e o
comportamento dos alertas. Detalhes e significado de cada metrica em
`docs/BENCHMARK.md`.
