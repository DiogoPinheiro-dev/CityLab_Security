# CityLab Security - Servidor

## Requisitos

- Python 3.13
- MongoDB local ou MongoDB Atlas
- Dependencias do projeto instaladas via `requirements.txt`

Arquivos de modelo esperados:

- `App/FaceRecon/yolov8n.pt`
- `App/FaceRecon/base_dados_alunos.pkl`
- `App/GestureRecon/yolov8n-pose.pt`

Observacao sobre InsightFace:

- O servidor usa modelos em `App/.insightface/models/buffalo_l`.
- Se esse modelo nao existir, o InsightFace tenta baixar automaticamente na primeira execucao.

## Configuracao do ambiente

Na raiz do projeto:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

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
py -m uvicorn Server.main:app --reload --host 127.0.0.1 --port 8000
```

Endpoints uteis:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Home: `http://127.0.0.1:8000/`

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

- `nome` (texto, obrigatorio)
- `foto` (arquivo `.jpg`, `.jpeg` ou `.png`, obrigatorio)

Regras importantes:

- A imagem precisa conter exatamente 1 rosto.
- Se a pipeline ainda nao estiver pronta, retorna `503`.

Exemplo (PowerShell):

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

Lista os logs mais recentes de reconhecimento.

Query param:

- `limite` (opcional, padrao: `50`)

Exemplo:

`GET http://127.0.0.1:8000/logs?limite=20`

Resposta (exemplo):

```json
[
  {
    "id": "67f0...",
    "nome": "Joao Silva",
    "tipo": "RECONHECIDO",
    "data_hora": "28/03/2026 - 19:20:11",
    "imagem_url": "data:image/jpeg;base64,..."
  }
]
```

### `WS /stream` (WebSocket)

Stream de reconhecimento em tempo real.

Entrada do cliente:

- Enviar frame em bytes (JPEG) a cada ciclo.

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
  ]
}
```

## Teste rapido com o cliente web

Em outro terminal, para servir os arquivos do cliente:

```powershell
py -m http.server 5500 -d Client
```

Abra:

`http://127.0.0.1:5500/teste_websocket.html`

Se a camera nao estiver disponivel no notebook atual, o cliente ainda conecta no servidor, mas nao envia frames.
