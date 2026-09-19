# CityLab Security - Servidor

API FastAPI para reconhecimento em tempo real com:

- reconhecimento facial
- deteccao de pessoas
- analise de gestos suspeitos
- deteccao de maos para refinar gestos

Hoje o servidor sobe usando diretamente a pipeline unificada em [App/recognition_pipeline.py](/e:/Codigos/CityLab_Security/App/recognition_pipeline.py), que orquestra:

- [App/FaceRecon/service.py](/e:/Codigos/CityLab_Security/App/FaceRecon/service.py)
- [App/GestureRecon/service.py](/e:/Codigos/CityLab_Security/App/GestureRecon/service.py)

## Estrutura atual

Arquivos principais do projeto:

- [Server/main.py](/e:/Codigos/CityLab_Security/Server/main.py): servidor FastAPI e websocket
- [Server/Db/database.py](/e:/Codigos/CityLab_Security/Server/Db/database.py): conexao com MongoDB
- [App/recognition_pipeline.py](/e:/Codigos/CityLab_Security/App/recognition_pipeline.py): pipeline unificada
- [App/FaceRecon/service.py](/e:/Codigos/CityLab_Security/App/FaceRecon/service.py): reconhecimento facial e deteccao de pessoas
- [App/GestureRecon/service.py](/e:/Codigos/CityLab_Security/App/GestureRecon/service.py): gestos, pose e maos
- [App/GestureRecon/detector.py](/e:/Codigos/CityLab_Security/App/GestureRecon/detector.py): regras de interpretacao de pose
- [App/camera_auto_config.py](/e:/Codigos/CityLab_Security/App/camera_auto_config.py): ajuste automatico de imagem
- [Client/teste_websocket.html](/e:/Codigos/CityLab_Security/Client/teste_websocket.html): cliente web simples para teste

## Estrategia de branches

- `main`: branch estavel do app
- `server`: branch principal de desenvolvimento
- outras branches: desenvolvimento de features, correcao e experimentos

O projeto usa um fluxo leve, sem bloqueio local para commits na `main`. O guia esta em [CONTRIBUTING.md](/e:/Codigos/CityLab_Security/CONTRIBUTING.md).

## Requisitos e instalacao

O perfil versionado nesta branch usa Raspberry Pi OS Bookworm 64-bit e Python 3.11.
Siga [docs/RASPBERRY_PI.md](docs/RASPBERRY_PI.md) para instalar OpenCV do sistema,
criar o ambiente e instalar `requirements-rpi-bookworm.txt`. As dependencias dos
modelos sao opcionais nesse arquivo e precisam ser instaladas e validadas para
executar o reconhecimento completo. MongoDB local ou Atlas e necessario.

O workflow de deploy usa o mesmo perfil e o ambiente `citylab_venv` com acesso aos
pacotes do sistema. Ele exige que OpenCV ja esteja instalado no host. Se o ambiente
existente usa outro Python, recrie-o com Python 3.11 antes do deploy.

Para medir o stream com webcam ou video fixo, consulte [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Variaveis de ambiente

Crie um arquivo `.env` na raiz do projeto.

Exemplo:

```env
MONGO_DETAILS=mongodb://localhost:27017
MONGO_DB_NAME=recon-db
MONGO_SERVER_SELECTION_TIMEOUT_MS=10000
```

## Como rodar o servidor

Com o ambiente virtual ativo:

```powershell
python -m uvicorn Server.main:app --reload --host 0.0.0.0 --port 8000

```

Endpoints uteis:

- Swagger UI: `http://0.0.0.0 :8000/docs`
- Home: `http://0.0.0.0 :8000/`

## Fluxo de processamento

Quando o servidor sobe:

1. valida a conexao com o MongoDB
2. instancia a pipeline unificada
3. carrega embeddings dos alunos do MongoDB para memoria
4. sincroniza esses embeddings com o `FaceRecognitionService`

Durante o stream websocket:

1. o cliente envia um frame JPEG
2. o servidor decodifica o frame
3. a pipeline roda face + pessoas + gestos
4. o servidor responde com o payload consolidado

## Rotas da API

### `GET /`

Retorna status do servidor e nome do banco configurado.

Resposta esperada:

```json
{
  "status": "online",
  "banco": "recon-db"
}
```

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
curl.exe -X POST "http://0.0.0.0:8000/cadastro" `
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

Lista os logs mais recentes de reconhecimento.

Query param:

- `limite` opcional, padrao `50`

Exemplo:

`GET http://0.0.0.0:8000/logs?limite=20`

Resposta de exemplo:

```json
[
  {
    "id": "67f0...",
    "nome": "Joao Silva",
    "tipo": "ALUNO",
    "data_hora": "28/03/2026 - 19:20:11",
    "imagem_url": "data:image/jpeg;base64,..."
  }
]
```

### `WS /stream`

Stream de reconhecimento em tempo real.

Entrada do cliente:

- enviar frames em bytes JPEG

Saida do servidor:

```json
{
  "rostos": [
    {
      "nome": "Joao Silva",
      "bbox": [100, 80, 220, 260],
      "confidence": 0.87
    }
  ],
  "pessoas": [
    {
      "bbox": [90, 60, 260, 430],
      "confidence": 0.81
    }
  ],
  "gestos": [
    {
      "track_id": 3,
      "bbox": [95, 70, 255, 420],
      "alerts": ["Rendicao"]
    }
  ],
}
```

## Teste rapido com o cliente web

Em outro terminal, sirva os arquivos do cliente:

```powershell
py -m http.server 5500 -d Client
```

Abra no navegador:

`http://0.0.0.0 :5500/teste_websocket.html`

Se a camera nao estiver disponivel na maquina atual, o cliente ainda pode abrir a interface, mas nao enviara frames validos para o servidor.
