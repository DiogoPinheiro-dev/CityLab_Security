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
  runner self-hosted do proprio Pi: sincroniza os arquivos em
  `/home/citylab/CityLab_Security` e instala as dependencias em `citylab_venv`.
  O workflow nao reinicia a API.
- outras branches: features, correcoes e experimentos.

No Pi 3 B+, pare a API antes do push: com 1 GB de RAM o deploy disputa memoria
com ela, e o runner ja caiu num deploy feito com a API rodando.

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

`CITYLAB_PROFILE=rpi3` escolhe 1 thread para o ONNX Runtime, 3 para o PyTorch e
1 para o OpenCV, e um frame pendente por vez no cliente. Tambem liga a passada
unica de pessoas e pose e o gate de movimento, que pula a pose quando a cena
esta parada e desocupada. Um valor explicito no ambiente sempre prevalece sobre
o perfil.

Resultado medido no Pi com webcam, tres rodadas por cenario, mediana do tempo
por frame:

| Cenario | Linha de base (19/09/2026) | Perfil atual | Ganho |
|---|---|---|---|
| Cena vazia | 5,41 s | 1,68 s | -69% |
| Uma pessoa | 13,41 s | 5,95 s | -56% |
| Duas pessoas | 16,62 s | 6,17 s | -63% |

Nenhuma deteccao de rosto, pessoa ou gesto se perdeu em relacao a linha de
base, e a caixa de pessoa falsa que ela produzia sumiu. A cena vazia foi medida
antes do ajuste de threads do PyTorch, que so afeta frames com pose. Sob carga
continua o Pi atinge 60 C e o firmware baixa o clock para 1,2 GHz, o que custa
de 3% a 7%; um dissipador com ventoinha evita isso. Detalhes em
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
3. rosto e gesto rodam em paralelo; no perfil `rpi3` as pessoas saem da mesma
   passada do modelo de pose
4. os eventos vao para o MongoDB
5. o servidor responde com o payload consolidado

O historico de gestos recomeca no primeiro frame de cada conexao e depois de
mais de 5 s sem frames (`GESTURE_IDLE_RESET_SECONDS`).

Eventos gravados:

- `ALUNO` e `NAO_ALUNO`: presenca de rosto, no maximo um a cada 5 s por
  identidade. `NAO_ALUNO` guarda o recorte do rosto.
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
- `foto` obrigatoria, em `.jpg`, `.jpeg` ou `.png`

Regras importantes:

- A imagem precisa conter exatamente 1 rosto.
- Se a pipeline ainda nao estiver pronta, a rota retorna `503`.
- Ao cadastrar com sucesso, o embedding tambem e sincronizado imediatamente com a pipeline em memoria.

Exemplo em PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/cadastro" `
  -F "nome=Joao Silva" `
  -F "foto=@C:\caminho\foto.jpg"
```

Resposta de sucesso:

```json
{
  "mensagem": "Sucesso! Rosto de 'Joao Silva' cadastrado.",
  "status": "sucesso"
}
```

### `GET /logs`

Lista os eventos mais recentes, do mais novo para o mais antigo. O `tipo` e
`ALUNO`, `NAO_ALUNO` ou `ALERTA_GESTO`.

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
recebe, para cada frame:

```json
{
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
node --test tests/client_stream.test.cjs
```

## Estado final e limitacoes conhecidas

O trabalho de desempenho foi encerrado em 27/09/2026 com o perfil do Pi
descrito acima, e as regras de gesto foram revistas em 29/09/2026. O que ficou
para depois esta nos dois planos. Limitacoes conhecidas desta versao:

- **Regras de gesto no Pi.** Com o frame em cerca de 6 s, cada regra exige de 2
  a 4 observacoes seguidas: uns 6 s para mao fechada e ameaca, 12 s para
  rendicao e mao oculta e 18 s para braco estendido. Os limiares de 0,20 a
  0,40 s so pesam com vazao alta. As regras novas ainda nao rodaram no Pi.
- **Mao fechada.** Nas gravacoes de teste, o reconhecedor da mao confundiu mao
  aberta ou relaxada com punho em boa parte dos frames. Ver
  [docs/PLANO_GESTOS.md](docs/PLANO_GESTOS.md).
- **Mao oculta depende de como a pessoa esta virada.** De costas nao dispara, e
  de lado a mao que a camera nao ve conta como oculta.
- **Eventos de rosto.** O cooldown de 5 s e menor que o frame, entao cada rosto
  grava um evento por frame. Os alertas de gesto gravam um evento quando cada
  alerta comeca.
- **Memoria.** A API ocupa entre 660 e 745 MB numa placa de 906 MB. Mais
  pessoas ou mais cameras podem esgotar a RAM.
- **Temperatura.** Sem dissipador, o Pi 3 B+ baixa o clock ao atingir o
  `temp_soft_limit`. No Pi do projeto o limite foi elevado para 70 C em
  27/09/2026, sem medicao depois; ver [docs/RASPBERRY_PI.md](docs/RASPBERRY_PI.md).
- **Uma camera por processo.** O rastreador e o historico de gestos sao
  globais.

## Medicao e planos

- [docs/BENCHMARK.md](docs/BENCHMARK.md): protocolo de medicao do stream no Pi
- [docs/PLANO_OTIMIZACAO.md](docs/PLANO_OTIMIZACAO.md): plano e historico de desempenho, com o estado verificado
- [docs/PLANO_GESTOS.md](docs/PLANO_GESTOS.md): plano das regras de gesto e de quando um alerta dispara
