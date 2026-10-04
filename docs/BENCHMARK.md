# Benchmark do stream

Na implementacao otimizada, cada amostra tambem inclui `persons_confidence` e
`gestures_confidence` (somente numeros finitos, sem identidade). Esses campos
nao existem nos JSON historicos. Use-os para investigar falsos positivos antes
de alterar limiares. Desde 01/10/2026 entram tambem `faces_confidence`, a
semelhanca de cada rosto com o cadastro, e `known_faces_count`, quantos rostos
foram reconhecidos como cadastrados; o nome nunca e gravado. Desde 03/10/2026,
`faces_width_px` traz a largura de cada caixa de rosto em pixels do frame
enviado, e `alerts`, o nome de cada alerta de gesto ativo, que diz a regra e
nao a pessoa. O relatorio resume as larguras em `faces_width_px` e conta os
alertas por nome em `alerts_by_name`. A API reduz o frame pela `PROCESS_SCALE`
(0,5 por padrao) antes de achar o rosto e gerar o embedding: o rosto que o
reconhecedor ve tem metade dessa largura.

Registre a saida de `python tools/run_rpi.py --show-config` junto da rodada; ela
informa a configuracao selecionada, nao comprova uso de CPU ou velocidade
efetiva.

O utilitario `tools/benchmark_stream.py` usa uma webcam local ou reproduz um
video fixo contra a API real. Ele envia os frames em ordem, com uma pendencia
por vez e sem pausa artificial. Mede a vazao desse perfil; nao simula o
navegador com duas pendencias ou captura a 10 FPS. Nao mede CPU/renderizacao
do navegador.

## Preparacao

Para a medicao atual, use a webcam do computador e uma instancia e banco de
teste: o stream grava eventos reais. Feche a pagina do navegador antes de abrir
a webcam pelo utilitario. O JSON registra as metricas, sem gravar imagens.
Mantenha camera, posicao, luz, resolucao e quantidade de pessoas tao estaveis
quanto possivel. O conteudo dos frames muda entre rodadas; os resultados da
webcam servem como linha de base operacional aproximada, nao como comparacao
exata de recall ou p95 entre versoes.

O modo de video fixo continua disponivel para quem tiver tres arquivos locais:
cena vazia, uma pessoa e varias pessoas. Cada um precisa ter pelo menos 330
frames para os valores padrao. O relatorio registra SHA-256; os videos nao estao
incluidos no repositorio.

No servidor, habilite `ENABLE_PERFORMANCE_METRICS=1` e
`ENABLE_SYSTEM_MONITOR=1` antes de iniciar. Instale o perfil Bookworm, incluindo
psutil, e as dependencias opcionais dos modelos que serao avaliados. Para a
referencia completa, use `CITYLAB_ALLOW_PARTIAL_PIPELINE=0` e ambos os servicos
habilitados. Confirme no startup que os dois foram carregados. Instale previamente
os pesos e mantenha os mesmos pesos e cadastros em todas as execucoes.

Guarde junto aos resultados, no Raspberry:

```bash
git rev-parse HEAD > benchmark-commit.txt
python --version > benchmark-python.txt
python -m pip freeze > benchmark-packages.txt
uname -a > benchmark-host.txt
```

Registre tambem resolucao, escala, workers, paralelismo, limites de threads,
servicos ativos, hashes dos pesos, quantidade de cadastros, alimentacao e
refrigeracao. Nao inclua credenciais do banco nos artefatos. `--run-label`
identifica esse registro; nao detecta automaticamente o ambiente remoto.

## Execucao com webcam

No computador com a camera, ative o ambiente virtual do projeto. Se a API roda
no Pi com HTTPS, aponte o coletor direto para ela com
`--url wss://IP_DO_PI:8000/stream`; o certificado precisa ser confiavel nesse
computador. Sem HTTPS, use um tunel SSH para `localhost:8000`. Com a API
iniciada e sem a pagina do navegador usando a camera, rode uma cena por vez:

```powershell
python tools/benchmark_stream.py --camera-index 0 --scenario empty --run-label pi3-webcam-01 --warmup 5 --frames 30 --output resultados/vazia-01.json
python tools/benchmark_stream.py --camera-index 0 --scenario one-person --run-label pi3-webcam-01 --warmup 5 --frames 30 --output resultados/uma-01.json
python tools/benchmark_stream.py --camera-index 0 --scenario many-persons --run-label pi3-webcam-01 --warmup 5 --frames 30 --output resultados/varias-01.json
```

Reinicie a API antes de cada execucao para zerar rastreador e caches de eventos.
Repita cada cenario tres vezes, se as cenas estiverem disponiveis. Use o mesmo
aquecimento, estado inicial do banco, enquadramento e temperatura comparavel.
Registre a contagem aproximada de pessoas e as condicoes de luz de cada rodada.
No Raspberry Pi, deixe o log de `vcgencmd` de `docs/RASPBERRY_PI.md` rodando
durante cada rodada: o limite de temperatura baixa o clock no meio da medicao e
muda os tempos sem nenhuma mudanca de codigo.
Com 30 frames, p95 e especialmente sensivel a variacao da cena: leia as amostras
individuais e evite atribuir ganhos pequenos a mudancas de codigo.

## Execucao com video fixo (opcional)

Na raiz do projeto, com a API iniciada e sem outras cameras conectadas:

```bash
python tools/benchmark_stream.py videos/vazia.mp4 --scenario empty --run-label pi3-baseline-01 --output resultados/vazia-01.json
python tools/benchmark_stream.py videos/uma-pessoa.mp4 --scenario one-person --run-label pi3-baseline-01 --output resultados/uma-01.json
python tools/benchmark_stream.py videos/varias-pessoas.mp4 --scenario many-persons --run-label pi3-baseline-01 --output resultados/varias-01.json
```

Reinicie a API antes de cada execucao para zerar rastreador e caches de eventos.
Repita cada cenario tres vezes com o mesmo aquecimento e estado inicial do banco.
Espere temperatura comparavel antes das repeticoes. Para uso continuo, por horas
numa conexao so, `--loop` volta ao inicio do video quando ele acaba e
`--samples-jsonl ARQUIVO` grava cada amostra, com o horario UTC, assim que ela
chega: uma queda no meio nao perde o que ja foi medido. O cliente pode rodar em outra
maquina usando `--url ws://ENDERECO:8000/stream`; registre a topologia de rede.
Nao misture resultados do cliente local e remoto.

## Significado das medidas

- `rtt_ms`: envio do JPEG ate recebimento da resposta, incluindo espera no
  servidor e transporte. No navegador, comeca depois da codificacao JPEG.
- `completed_fps`: respostas medidas / tempo real da carga, excluindo aquecimento
  e incluindo preparacao dos frames posteriores ao primeiro frame medido.
- `receive_wait_ms`: tempo aguardando `receive_bytes`; inclui ociosidade e nao
  deve ser interpretado como latencia de rede.
- `response_ready_ms`: decodificacao, pipeline, eventos e montagem da resposta,
  antes da coleta de recursos, serializacao e envio.
- `send_ms` e `total_ms`: medidos no servidor apos envio; total exclui espera
  pelo proximo frame e inclui envio ao ASGI, sem garantir recebimento no cliente.
  Nao aparecem na resposta do proprio frame; alimentam o monitor do servidor.
- `persons_ms`, `faces_ms`, `pose_ms`, `hands_ms`, `gestures_ms`: duracoes dos
  estagios existentes, nao tempo exclusivo de cada rede. Estagios paralelos e
  tempos de gestos/pose/maos se sobrepoem; nao some esses valores.
- `face_detect_ms`, `face_embed_ms`, `face_match_ms`: partes do `faces_ms`, so
  com `FACE_PREFILTER`: deteccao, embedding dos rostos aceitos e comparacao com
  o cadastro. `face_embeddings` conta os embeddings gerados no frame, e
  `face_reused` os rostos que herdaram o nome com `FACE_REUSE_SECONDS`.
- `ncnn_threads`: threads da rede da pose em NCNN, com `NCNN_NUM_THREADS`; zero
  com o peso `.pt`.
- `process_rss_mb`: RAM residente do processo servidor; `temperature_c`: leitura
  disponivel no host. Ausencia aparece como null e contagem zero no resumo.
- `detections`: contagem de pessoas, rostos, gestos e alertas por frame. O
  relatorio guarda as contagens, sem nomes, embeddings ou imagens. O campo
  `started_at_utc` identifica quando a rodada comecou, antes do aquecimento.
- Trafego: bytes dos payloads JPEG e JSON, sem cabecalhos WebSocket/TCP/TLS.

Os resumos usam mediana e p95 pelo metodo nearest-rank; as amostras individuais
tambem ficam no JSON. Em modo webcam, `video_sha256` e null e o relatorio inclui
indice, largura, altura e FPS informados pela camera. A coleta de recursos tem
custo e deve ficar igualmente habilitada nas comparacoes. O utilitario falha em
video curto, leitura da webcam, timeout, erro da API ou metricas desativadas,
em vez de publicar uma execucao incompleta.

## Validacao disponivel

```bash
node --test tests/client_stream.test.cjs tests/client_cadastro.test.cjs
python -m unittest discover -s tests -p 'test_*.py'
```

Esses testes validam a correlacao da resposta pelo numero do frame, o tempo
limite por frame, a pausa sem resultado antigo, callbacks de conexoes antigas,
falha de codificacao, limites dos tempos do handler com dependencias simuladas
e resumo estatistico. Nao executam modelos, MongoDB ou Raspberry. A fase de medicao so
fecha com as cargas combinadas no hardware e os resultados registrados.
