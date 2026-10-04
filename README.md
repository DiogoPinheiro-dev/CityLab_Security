# CityLab Security - Servidor

API FastAPI para reconhecimento em tempo real com:

- reconhecimento facial
- deteccao de pessoas
- analise de gestos suspeitos
- deteccao de maos para refinar gestos

O alvo de producao e um Raspberry Pi 3 B+ com Raspberry Pi OS Bookworm 64-bit e
Python 3.11. O servidor sobe usando a pipeline unificada em
[App/recognition_pipeline.py](App/recognition_pipeline.py), que roda rosto e
gesto em paralelo sobre o mesmo frame.

## Estrutura

- [Server/main.py](Server/main.py): servidor FastAPI, rotas e WebSocket do stream
- [Server/event_logger.py](Server/event_logger.py): gravacao de eventos no MongoDB
- [Server/system_monitor.py](Server/system_monitor.py): RAM, CPU e temperatura do host
- [Server/Db/database.py](Server/Db/database.py): conexao com MongoDB
- [App/settings.py](App/settings.py): configuracao por variavel de ambiente, incluindo o perfil do Pi
- [App/recognition_pipeline.py](App/recognition_pipeline.py): pipeline unificada
- [App/FaceRecon/service.py](App/FaceRecon/service.py): reconhecimento facial (InsightFace) e deteccao de pessoas
- [App/GestureRecon/service.py](App/GestureRecon/service.py): pose (YOLOv8-pose), maos (MediaPipe) e gate de movimento
- [App/GestureRecon/detector.py](App/GestureRecon/detector.py): regras de interpretacao de pose
- [App/inference_runtime.py](App/inference_runtime.py): limites de threads das bibliotecas nativas
- [Client/](Client/): paginas do stream (`stream.html`) e do cadastro (`cadastros.html`)
- [tools/run_rpi.py](tools/run_rpi.py): inicializador do servidor no Pi
- [tools/benchmark_stream.py](tools/benchmark_stream.py): medicao do stream com webcam ou video
- [tests/](tests/): testes que rodam sem modelos, MongoDB ou Raspberry
- [docs/](docs/): guia do Pi, protocolo de medicao e planos de trabalho
- [resultados/](resultados/): medicoes feitas no Pi, um JSON por rodada

## Branches e deploy

- `main`: branch estavel. Um push nela dispara
  [.github/workflows/deploy.yml](.github/workflows/deploy.yml), que roda no
  runner self-hosted do proprio Pi: para o servico da API, sincroniza os
  arquivos em `/home/citylab/CityLab_Security`, atualiza as dependencias no
  `.venv` da API e sobe o servico de novo, mesmo se a instalacao falhar.
- outras branches: features, correcoes e experimentos.

No Pi 3 B+ a API nao pode rodar durante o deploy: com 1 GB de RAM ele disputa
memoria com ela, e o runner ja caiu num deploy feito com a API rodando. Com a
API como servico, o workflow cuida disso; com ela iniciada a mao, pare antes do
push. As dependencias vem de `requirements-rpi-bookworm.txt`, por
[tools/atualizar_dependencias_rpi.sh](tools/atualizar_dependencias_rpi.sh): o
pip so roda quando o arquivo muda, com `--no-deps`, e, se a API deixar de
importar, o script volta as versoes de antes e o deploy marca falha. Um pacote
novo entra no arquivo junto com as dependencias que faltarem, todas com versao.

## Requisitos e instalacao

Siga [docs/RASPBERRY_PI.md](docs/RASPBERRY_PI.md) para instalar OpenCV do
sistema, criar o ambiente e instalar `requirements-rpi-bookworm.txt`. As
dependencias dos modelos (Ultralytics, InsightFace e ONNX Runtime) ficam
comentadas nesse arquivo e precisam ser instaladas e validadas para executar o
reconhecimento completo. MongoDB local ou Atlas e necessario.

O workflow de deploy exige que o OpenCV ja esteja instalado no host. Se o
ambiente existente usa outro Python, recrie-o com Python 3.11 antes do deploy.

## Variaveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
MONGO_DETAILS=mongodb://localhost:27017
MONGO_DB_NAME=recon-db
MONGO_SERVER_SELECTION_TIMEOUT_MS=10000
```

Opcoes mais usadas da pipeline:

| Variavel | Padrao | O que faz |
|---|---|---|
| `CITYLAB_PROFILE` | `default` | `rpi3` liga o perfil medido do Raspberry Pi 3 B+ |
| `CITYLAB_ENABLE_FACE_SERVICE` | `1` | liga o reconhecimento facial |
| `CITYLAB_ENABLE_GESTURE_SERVICE` | `1` | liga pose, maos e regras de gesto |
| `CITYLAB_ALLOW_PARTIAL_PIPELINE` | `0` | com `1`, a API sobe mesmo se um servico falhar ao carregar |
| `ENABLE_PERFORMANCE_METRICS` | `0` | publica os tempos de cada estagio na resposta do stream |
| `ENABLE_SYSTEM_MONITOR` | `0` | registra CPU, RAM e temperatura no log do servidor |

As demais ficam em [App/settings.py](App/settings.py). O perfil do Pi esta em
[.env.rpi.example](.env.rpi.example), para mesclar no `.env` sem apagar as
credenciais.

## Perfil do Raspberry Pi 3 B+

`CITYLAB_PROFILE=rpi3` escolhe 2 threads para o ONNX Runtime, 3 para o PyTorch e
1 para o OpenCV, e um frame pendente por vez no cliente. Tambem liga a passada
unica de pessoas e pose e o gate de movimento, que pula a pose quando a cena
esta parada e desocupada, roda a pose em 416 px em vez de 640, deixa um rosto
no mesmo lugar herdar por ate 15 s o nome ja reconhecido, gera o embedding do
rosto no frame original, e nao na imagem reduzida, e aprende referencias da
pessoa com o stream, guardadas no banco. Um valor explicito no ambiente sempre
prevalece sobre o perfil.

No Pi, a pose roda em NCNN desde 04/10/2026, pelo `POSE_MODEL_PATH` no `.env`,
com a pasta exportada copiada a mao (ver
[docs/RASPBERRY_PI.md](docs/RASPBERRY_PI.md)). Com o ncnn em 2 threads, padrao
do perfil, o frame com uma pessoa no video de carga ficou 40% mais rapido que
com o `.pt`, com a mesma deteccao.

Resultado medido no Pi com webcam, tres rodadas por cenario, mediana do tempo
por frame:

| Cenario | Linha de base (19/09/2026) | Perfil atual | Ganho |
|---|---|---|---|
| Cena vazia | 5,41 s | 0,93 s | -83% |
| Uma pessoa | 13,41 s | 5,32 s | -60% |
| Duas pessoas | 16,62 s | 3,41 s | -80% |

Nenhuma deteccao de rosto, pessoa ou gesto se perdeu em relacao a linha de
base, e a caixa de pessoa falsa que ela produzia sumiu. Cena vazia e duas
pessoas foram medidas em 02/10/2026 com o perfil atual completo. Na vazia,
nenhuma deteccao falsa. Com duas pessoas, uma de frente e outra de lado, as
duas, o rosto de frente e os gestos das duas foram detectados em todos os
frames, e a media, que conta os frames que reconhecem o rosto de novo, foi de
3,59 a 3,71 s. Uma pessoa foi medida em 01/10/2026, com a pose em 416 px, mas
antes das 2 threads do ONNX Runtime e do reuso da identidade do rosto; por isso
aparece mais lenta que duas. Num video fixo de uma pessoa, as 2 threads
deixaram o frame 20% mais rapido, de 5,33 para 4,29 s, e o reuso tirou mais
13% na media, para 3,72 s; com a webcam, nao foram medidos. Com 60 C, sob
carga continua o firmware baixa o clock para 1,2 GHz, o que custa de 3% a 7%;
um dissipador com ventoinha evita isso. A pose tambem pode rodar em NCNN, que
nos videos de validacao deu os mesmos keypoints e alertas do `.pt`; o tempo
ainda nao foi medido no Pi. Detalhes em
[docs/PLANO_OTIMIZACAO.md](docs/PLANO_OTIMIZACAO.md) e
[docs/RASPBERRY_PI.md](docs/RASPBERRY_PI.md).

## Como rodar o servidor

### No computador, para desenvolvimento

Com o ambiente virtual ativo:

```powershell
python -m uvicorn Server.main:app --reload --host 0.0.0.0 --port 8000
```

### No Raspberry Pi

```bash
.venv/bin/python tools/run_rpi.py --ssl-certfile CAMINHO/certificado.pem --ssl-keyfile CAMINHO/chave.pem
```

O inicializador sobe um unico processo, sem reload, aplica os limites de threads
antes de importar as bibliotecas nativas e usa o perfil `rpi3` se nenhum outro
estiver configurado. O par de certificados liga o HTTPS, necessario para a
camera do navegador em outra maquina. Para conferir a configuracao efetiva sem
carregar modelos nem conectar ao banco:

```bash
.venv/bin/python tools/run_rpi.py --show-config
```

A saida lista so opcoes de desempenho, nunca credenciais.

Para uso continuo, sem depender de um terminal aberto, instale a API como
servico, que sobe com o Pi e volta sozinha se cair: ver "Rodar como servico" em
[docs/RASPBERRY_PI.md](docs/RASPBERRY_PI.md).

### Enderecos

- Na propria maquina: `http://127.0.0.1:8000/`
- Em outra maquina da rede: `http://IP_DA_MAQUINA_DO_SERVIDOR:8000/`, ou
  `https://` quando a API sobe com certificado
- Swagger UI: `http://127.0.0.1:8000/docs`

`0.0.0.0` serve para o Uvicorn escutar em todas as interfaces, mas nao deve ser
usado como endereco no navegador.

## Fluxo de processamento

Quando o servidor sobe:

1. valida a conexao com o MongoDB
2. instancia a pipeline unificada
3. carrega embeddings dos alunos do MongoDB para memoria
4. sincroniza esses embeddings com o `FaceRecognitionService`

Durante o stream WebSocket:

1. o cliente envia um frame JPEG
2. o servidor decodifica o frame
3. o frame vai para a thread de inferencia, a unica que roda os modelos; ali
   rosto e gesto rodam em paralelo, e no perfil `rpi3` as pessoas saem da mesma
   passada do modelo de pose
4. os eventos vao para o MongoDB
5. o servidor responde com o payload consolidado

Enquanto um frame roda, as outras rotas continuam respondendo. O cadastro usa a
mesma thread: as fotos esperam o frame em andamento e o stream espera as fotos.

O historico de gestos recomeca no primeiro frame de cada conexao e depois de
mais de 5 s sem frames (`GESTURE_IDLE_RESET_SECONDS`).

Eventos gravados:

- `ALUNO` e `NAO_ALUNO`: presenca de rosto, um evento por episodio, como os
  alertas. O aluno grava quando aparece e so grava de novo depois de sumir de
  um frame; o desconhecido continua no episodio enquanto aparece perto de onde
  estava. `NAO_ALUNO` guarda o recorte do rosto.
- `ALERTA_GESTO`: um evento por episodio. Enquanto o mesmo alerta continua no
  mesmo track em frames seguidos, nao grava de novo; se ele some de um frame e
  volta, grava outra vez. Guarda o recorte da pessoa, os alertas ativos
  (`alertas`), os que acabaram de comecar (`alertas_novos`) e, em `evidencia`,
  quantas observacoes seguidas e quanto tempo sustentaram cada alerta.

## Rotas da API

### `GET /`

Retorna status do servidor e nome do banco configurado:

```json
{
  "status": "online",
  "banco": "recon-db"
}
```

### `GET /stream`

Pagina do cliente de stream, que abre a camera e envia os frames pelo
WebSocket.

### `GET /cadastros` e `GET /cadastro`

Pagina de cadastro de rostos.

### `GET /access-info`

URL base publica do servidor e URL da pagina de cadastro, usadas pelo QR code.

### `GET /qrcode/cadastro.png`

QR code em PNG apontando para a pagina de cadastro. O parametro opcional `url`
troca o endereco codificado.

### `GET /config/client`

Parametros que o cliente usa no stream: FPS, qualidade JPEG, resolucao e
quantos frames podem ficar pendentes.

### `POST /cadastro`

Cadastra um aluno no MongoDB com embedding facial.

Formato: `multipart/form-data`

- `nome` obrigatorio
- `foto` obrigatoria, de 1 a 5 arquivos no mesmo campo, em `.jpg`, `.jpeg` ou
  `.png`. Com mais de uma, o cadastro guarda a media dos embeddings, que
  reconhece melhor que uma foto so; a pagina de cadastro sugere de 3 a 5.

Regras importantes:

- Cada imagem precisa conter exatamente 1 rosto; o erro diz qual foto falhou.
- Uma foto que nao parece ser da mesma pessoa das outras recusa o cadastro,
  para nao misturar dois rostos no mesmo embedding.
- Se a pipeline ainda nao estiver pronta, a rota retorna `503`.
- Ao cadastrar com sucesso, o embedding tambem e sincronizado imediatamente com a pipeline em memoria.
- Enquanto as fotos sao processadas, o stream fica parado: no Pi, cada foto
  custa uma deteccao e um embedding, alguns segundos.
- Recadastrar um nome cria outro cadastro, e o antigo continua valendo. A rota
  nao apaga nada: no Pi, `tools/cadastros.py` lista os cadastros, apaga os
  antigos de uma pessoa depois do recadastro (`--manter-ultimo`) ou tira a
  pessoa (`--apagar`), junto com as referencias aprendidas dela. Os eventos
  ficam e saem pelo prazo. Depois de apagar, reinicie a API.

Exemplo em PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/cadastro" `
  -F "nome=Joao Silva" `
  -F "foto=@C:\caminho\foto1.jpg" `
  -F "foto=@C:\caminho\foto2.jpg" `
  -F "foto=@C:\caminho\foto3.jpg"
```

Resposta de sucesso:

```json
{
  "mensagem": "Sucesso! Rosto de 'Joao Silva' cadastrado.",
  "status": "sucesso"
}
```

Com `FACE_LEARN_FROM_STREAM`, ligado no perfil do Pi e desligado no padrao, o
sistema tambem aprende com o stream. Um rosto reconhecido com folga, com
semelhanca de 0,60 ou mais e perto do cadastro, vira mais uma referencia da
pessoa, guardada na colecao
`rostos_aprendidos`. Sao ate 5 por pessoa, uma a cada 10 min, e o MongoDB apaga
as com mais de 30 dias. Com as vagas cheias, uma nova troca a referencia menos
parecida com o cadastro, se for mais parecida que ela, e assim uma referencia
errada e a primeira a sair. Uma quase igual a uma referencia guardada so
disputa a vaga dela, para as 5 nao virarem copias da mesma foto.
`tools/limpar_aprendidos.py` mostra e apaga as referencias.

### `GET /logs`

Lista os eventos mais recentes, do mais novo para o mais antigo. O `tipo` e
`ALUNO`, `NAO_ALUNO` ou `ALERTA_GESTO`. Cada evento guarda um recorte em JPEG
de quem apareceu, e o proprio MongoDB apaga os eventos com mais de
`EVENT_RETENTION_DAYS` dias (30 por padrao; 0 guarda tudo).

Query param:

- `limite` opcional, padrao `50`

Exemplo: `GET http://127.0.0.1:8000/logs?limite=20`

```json
[
  {
    "id": "67f0...",
    "nome": "Joao Silva",
    "tipo": "ALUNO",
    "data_hora": "28/03/2026 - 19:20:11",
    "imagem_url": null
  }
]
```

### `WS /stream`

Stream de reconhecimento em tempo real. O cliente envia frames em bytes JPEG e
recebe, para cada frame, uma resposta com o numero do frame na conexao
(`frame`, a partir de 1), inclusive nas respostas de erro:

```json
{
  "frame": 1,
  "rostos": [
    {"nome": "Joao Silva", "bbox": [100, 80, 220, 260], "confidence": 0.87}
  ],
  "pessoas": [
    {"bbox": [90, 60, 260, 430], "confidence": 0.81}
  ],
  "gestos": [
    {"track_id": 3, "bbox": [95, 70, 255, 420], "alerts": ["Rendicao"], "confidence": 0.81}
  ]
}
```

Com `ENABLE_PERFORMANCE_METRICS=1`, a resposta traz tambem `metrics`, com os
tempos de cada estagio, a RAM do processo e a temperatura; `DEBUG_PIPELINE=1`
inclui os tempos e um bloco `debug`. O significado de cada metrica esta em
[docs/BENCHMARK.md](docs/BENCHMARK.md).

O cliente web casa cada resposta com o frame enviado por esse numero. Se um
frame fica 30 s sem resposta, ele reconecta; ao pausar ou perder a conexao,
apaga os ultimos resultados, para nao deixar caixas antigas sobre o video.

## Teste rapido com o cliente web

Com a API no ar, abra `http://127.0.0.1:8000/stream` na propria maquina do
servidor. A pagina e servida pela API e se conecta ao WebSocket do mesmo host.

O navegador so libera camera em contexto seguro: `https://` ou `localhost`. De
outra maquina da rede, suba a API com certificado (no Pi, `tools/run_rpi.py`
com `--ssl-certfile` e `--ssl-keyfile`) e abra
`https://IP_DA_MAQUINA_DO_SERVIDOR:8000/stream`. Para um teste rapido sem
certificado no Chrome ou Edge, inicie o navegador com
`--unsafely-treat-insecure-origin-as-secure=http://IP_DA_MAQUINA_DO_SERVIDOR:8000`.

Se a camera nao estiver disponivel na maquina atual, o cliente ainda abre a
interface, mas nao envia frames validos para o servidor.

## Validacao local

Os testes nao exigem modelos, MongoDB nem o Raspberry:

```bash
python -m unittest discover -s tests -p 'test_*.py'
node --test tests/client_stream.test.cjs tests/client_cadastro.test.cjs
```

## Estado final e limitacoes conhecidas

O trabalho de desempenho foi encerrado em 27/09/2026 com o perfil do Pi
descrito acima, e as regras de gesto foram revistas em 29 e 30/09/2026. Em
01/10/2026 a pose passou a rodar em 416 px no perfil. O que ficou para depois
esta nos dois planos. Limitacoes conhecidas desta versao:

- **Regras de gesto no Pi.** Com o frame entre 4 e 6 s, cada regra exige de 2
  a 4 observacoes seguidas: uns 4 a 6 s para mao fechada e ameaca, 9 a 12 s
  para rendicao e mao oculta e 13 a 18 s para braco estendido. Os limiares de
  0,20 a 0,40 s so pesam com vazao alta. Conferido no Pi em 30/09/2026, numa
  rodada com gesto.
- **Nome herdado pelo rosto.** No perfil rpi3, um rosto no mesmo lugar herda
  por ate 15 s o nome ja reconhecido, sem reconhecer de novo, inclusive o de
  desconhecido. Nesse intervalo o nome aparece mesmo num frame em que o
  reconhecimento falharia, e a troca de nome entre duas pessoas que trocam de
  lugar nao foi testada. `FACE_REUSE_SECONDS=0` volta a reconhecer todo frame.
- **Reconhecimento perto do limite.** Em 02/10/2026, com duas pessoas na
  frente da webcam, o rosto cadastrado de frente teve semelhanca de 0,42 a 0,59
  com o cadastro, em volta do limite de 0,52, e foi reconhecido em 9, 21 e 25
  dos 30 frames das tres rodadas; nos outros, saiu como desconhecido. No PC, a
  causa principal foi o cadastro de uma foto so; desde 03/10/2026 ele aceita
  ate 5 fotos e guarda a media, mas so ajuda quem for recadastrado; no Pi
  ainda nao foi medido. O embedding no frame original, ligado no perfil do Pi
  desde 03/10/2026, subiu um pouco a semelhanca no Pi, mais nos rostos
  dificeis, sem custo de tempo. Aprender com o stream
  (`FACE_LEARN_FROM_STREAM=1`) levou, no PC, de 233 a 277 os frames
  reconhecidos em 360 com uma foto de cadastro. No Pi, com o video de carga e
  o filtro de copia, levou de 26 a 28 de 30 nas tres rodadas, sem custo de
  tempo, e passou a ligado no perfil do Pi em 03/10/2026. O video tem uma
  pessoa so, a do cadastro: o ganho com a camera de verdade, em outras
  condicoes, e com outras pessoas na cena nao foi medido.
- **Rostos do stream no banco.** Com o aprendizado ligado, padrao no perfil do
  Pi desde 03/10/2026, o banco guarda ate 5 embeddings por pessoa tirados do
  stream, sem a pessoa fazer nada, por ate 30 dias. Dado biometrico e dado
  pessoal sensivel na LGPD: confira a base legal e o prazo;
  `FACE_LEARN_FROM_STREAM=0` desliga. Uma referencia aprendida por engano sai
  quando chega outra mais parecida com o cadastro, quando vence ou quando e
  apagada com `tools/limpar_aprendidos.py`; enquanto as vagas da pessoa nao
  enchem, so as duas ultimas.
- **Pose em 416 px.** Na rodada com gesto em 416, a mao oculta de lado nao
  disparou e o punho levantado disparou um frame depois do que em 640. Com 6
  frames por situacao e a pessoa em outra posicao, nao deu para separar o
  efeito do tamanho do efeito da cena. `POSE_IMGSZ=0` volta aos 640.
- **Mao fechada.** Com a mao solta ao lado do corpo, o reconhecedor da mao nao
  separa o punho da mao relaxada. Por isso, desde 30/09/2026 o punho so conta,
  no alerta de mao fechada e no de ameaca, com o braco levantado a 45 graus ou
  mais da vertical: o punho com o braco solto nao alerta. Com o braco
  levantado, a mao aberta ou relaxada ainda e lida como fechada em parte dos
  frames, e na rodada no Pi o punho na ponta do braco estendido nao foi lido:
  a ameaca e pouco confiavel. Ver [docs/PLANO_GESTOS.md](docs/PLANO_GESTOS.md).
- **Braco apontado para a camera.** Aparece curto na imagem e nao conta como
  braco estendido; so o braco esticado para o lado dispara.
- **Mao oculta depende de como a pessoa esta virada.** De costas nao dispara, e
  de lado a mao que a camera nao ve conta como oculta.
- **Eventos por episodio.** Rostos e alertas gravam quando aparecem e nao se
  repetem enquanto continuam. Um frame em que o detector perde a pessoa encerra
  o episodio, e a volta grava um evento novo. Desde 03/10/2026 os eventos com
  mais de 30 dias sao apagados; quando o prazo e ligado, os mais antigos saem de
  uma vez, sem volta.
- **Memoria.** A API ocupa cerca de 710 MB numa placa de 906 MB. Em uso
  continuo o sistema manda uns 210 MB dela para o swap na primeira meia hora e
  depois fica parado: em 10,5 h seguidas, cerca de 500 MB em RAM, sem crescer e
  quase sem troca com o swap. Mais pessoas ou mais cameras podem esgotar a RAM.
- **Temperatura.** Sem dissipador, o Pi 3 B+ baixa o clock ao atingir o
  `temp_soft_limit`. No Pi do projeto o limite foi elevado para 70 C em
  27/09/2026; em 10,5 h seguidas com uma pessoa e o perfil atual, de 01 para
  02/10/2026, o Pi ficou entre 56 e 63,4 C, sem baixar o clock. Ver
  [docs/RASPBERRY_PI.md](docs/RASPBERRY_PI.md).
- **Rede do Pi por Wi-Fi.** Em 02/10/2026 o Wi-Fi se reconectou sozinho e o Pi
  ficou fora de alcance por uns 9 min, com a API rodando o tempo todo. Foi a
  unica reconexao em 24 h; a economia de energia do Wi-Fi esta ligada. Cabo de
  rede evita isso.
- **Fonte do Pi no limite.** Com a pose no `.pt`, a fonte atual segura o
  pipeline (`vcgencmd get_throttled` em `0x0` por 10,5 h). Com a pose em NCNN
  nas 4 threads do padrao, a tensao caiu durante o stream; com 2, padrao do
  perfil, ficou em `0x0` nas tres rodadas de 04/10/2026. O NCNN foi para uso
  sem teste longo, por decisao do responsavel: confira o `get_throttled` de
  vez em quando, e qualquer valor diferente de `0x0` pede a volta ao `.pt`.
- **Uma camera por processo.** O rastreador e o historico de gestos sao
  globais. Duas conexoes ao mesmo tempo dividem a thread de inferencia, um
  frame de cada vez, e misturam os rastros.

## Medicao e planos

- [docs/BENCHMARK.md](docs/BENCHMARK.md): protocolo de medicao do stream no Pi
- [docs/PLANO_OTIMIZACAO.md](docs/PLANO_OTIMIZACAO.md): plano e historico de desempenho, com o estado verificado
- [docs/PLANO_GESTOS.md](docs/PLANO_GESTOS.md): plano das regras de gesto e de quando um alerta dispara
