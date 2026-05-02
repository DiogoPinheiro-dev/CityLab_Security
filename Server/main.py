import asyncio
import base64
import io
import os
import socket
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLIENT_DIR = PROJECT_ROOT / "Client"

from App.recognition_pipeline import UnifiedRecognitionService, create_unified_service
from Server.Db.database import MONGO_DB_NAME, colecao_alunos, colecao_logs, validar_conexao_mongo


class LogResponse(BaseModel):
    id: str
    nome: str
    tipo: str
    data_hora: str
    imagem_url: Optional[str] = None


banco_rostos_memoria = {
    "nomes": [],
    "embeddings": [],
}

recognizer: Optional[UnifiedRecognitionService] = None


def _sync_memoria_para_recognizer() -> None:
    if recognizer is None:
        return

    recognizer.face_service.replace_known_faces(
        banco_rostos_memoria["nomes"],
        banco_rostos_memoria["embeddings"],
    )


def _get_local_network_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def _build_public_base_url(request: Request) -> str:
    configured_url = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if configured_url:
        return configured_url

    host = request.url.hostname or "127.0.0.1"
    if host in {"127.0.0.1", "localhost", "::1"}:
        host = _get_local_network_ip()

    default_port = 443 if request.url.scheme == "https" else 80
    port = request.url.port
    port_part = f":{port}" if port and port != default_port else ""

    return f"{request.url.scheme}://{host}{port_part}"


@asynccontextmanager
async def lifespan(_: FastAPI):
    global recognizer

    print("[INFO] Validando conexao com MongoDB...")
    await validar_conexao_mongo()
    print("[INFO] Conexao com MongoDB OK.")

    print("[INFO] API iniciada. Carregando pipeline unificado...")
    recognizer = create_unified_service()

    print("[INFO] Carregando alunos do MongoDB para memoria...")
    banco_rostos_memoria["nomes"].clear()
    banco_rostos_memoria["embeddings"].clear()

    cursor = colecao_alunos.find({})
    async for aluno in cursor:
        banco_rostos_memoria["nomes"].append(aluno["nome"])
        banco_rostos_memoria["embeddings"].append(np.array(aluno["embedding"]))

    _sync_memoria_para_recognizer()
    print(f"[INFO] {len(banco_rostos_memoria['nomes'])} alunos carregados.")

    yield
    print("[INFO] API desligada.")


app = FastAPI(title="API FaceRecon", lifespan=lifespan)

if CLIENT_DIR.exists():
    app.mount("/client", StaticFiles(directory=CLIENT_DIR), name="client")

@app.get("/")
async def home():
    return {"status": "online", "banco": MONGO_DB_NAME}

@app.get("/cadastros")
@app.get("/cadastro")
async def pagina_cadastros():
    cadastro_page = CLIENT_DIR / "cadastros.html"
    if not cadastro_page.exists():
        raise HTTPException(status_code=404, detail="Pagina de cadastros nao encontrada.")

    return FileResponse(cadastro_page)

@app.get("/access-info")
async def access_info(request: Request):
    public_base_url = _build_public_base_url(request)
    return {
        "base_url": public_base_url,
        "cadastro_url": f"{public_base_url}/cadastros",
    }

@app.get("/qrcode/cadastro.png")
async def qrcode_cadastro(request: Request, url: Optional[str] = None):
    try:
        import qrcode
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Dependencia qrcode nao instalada. Rode: pip install qrcode[pil]",
        ) from exc

    cadastro_url = url or f"{_build_public_base_url(request)}/cadastros"
    if not cadastro_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL invalida para QR Code.")

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=3,
    )
    qr.add_data(cadastro_url)
    qr.make(fit=True)

    image = qr.make_image(fill_color="#102330", back_color="#ffffff")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")

@app.post("/cadastro")
async def cadastrar_aluno(nome: str = Form(...), foto: UploadFile = File(...)):
    if not foto.filename or not foto.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Formato invalido. Use JPG ou PNG.")

    if recognizer is None:
        raise HTTPException(status_code=503, detail="Pipeline de reconhecimento nao esta pronta.")

    try:
        conteudo_arquivo = await foto.read()
        nparr = np.frombuffer(conteudo_arquivo, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Erro ao processar a imagem. Arquivo corrompido.")

        faces = recognizer.face_service.app_insight.get(img)

        if not faces:
            raise HTTPException(status_code=400, detail="Nenhum rosto encontrado na foto enviada.")
        if len(faces) > 1:
            raise HTTPException(
                status_code=400,
                detail="Multiplos rostos encontrados. Envie uma foto de apenas 1 pessoa.",
            )

        embedding_lista = faces[0].normed_embedding.tolist()

        novo_aluno = {
            "nome": nome,
            "embedding": embedding_lista,
            "cadastrado_em": datetime.now().strftime("%d/%m/%Y - %H:%M:%S"),
        }
        await colecao_alunos.insert_one(novo_aluno)

        banco_rostos_memoria["nomes"].append(nome)
        banco_rostos_memoria["embeddings"].append(np.array(embedding_lista))
        _sync_memoria_para_recognizer()

        return {
            "mensagem": f"Sucesso! Rosto de '{nome}' cadastrado.",
            "status": "sucesso",
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {exc}")

@app.get("/logs", response_model=List[LogResponse])
async def visualizar_logs(limite: int = 50):
    logs_db = []
    cursor = colecao_logs.find().sort("data_hora_raw", -1).limit(limite)

    async for documento in cursor:
        logs_db.append(
            {
                "id": str(documento["_id"]),
                "nome": documento["nome"],
                "tipo": documento["tipo"],
                "data_hora": documento["data_hora_formatada"],
                "imagem_url": documento.get("imagem_url") or documento.get("imagem_rosto"),
            }
        )

    return logs_db

@app.get("/stream")
async def pagina_stream():
    stream_page = CLIENT_DIR / "teste_websocket.html"
    if not stream_page.exists():
        raise HTTPException(status_code=404, detail="Pagina do stream nao encontrada.")

    return FileResponse(
        stream_page,
        headers={"Cache-Control": "no-store"},
    )

@app.websocket("/stream")
async def websocket_reconhecimento(websocket: WebSocket):
    await websocket.accept()
    print("[INFO] Cliente Web conectado ao stream de video.")

    recently_logged = {}
    log_cooldown_seconds = 5

    try:
        while True:
            if recognizer is None:
                await websocket.send_json({"erro": "Pipeline de reconhecimento nao inicializada."})
                await asyncio.sleep(0.2)
                continue

            bytes_frame = await websocket.receive_bytes()
            nparr = np.frombuffer(bytes_frame, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            results = recognizer.process_frame(frame)

            resultados_faces = []
            for face in results.get("faces", []):
                bbox = face.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                nome_detectado = face.get("name", "NAO ALUNO")
                resultado_face = {
                    "nome": nome_detectado,
                    "bbox": bbox,
                    "confidence": face.get("confidence"),
                }
                resultados_faces.append(resultado_face)

                tempo_atual = time.time()
                if nome_detectado in recently_logged and (tempo_atual - recently_logged[nome_detectado] <= log_cooldown_seconds):
                    continue

                recently_logged[nome_detectado] = tempo_atual

                x1, y1, x2, y2 = [int(valor) for valor in bbox]
                h_full, w_full = frame.shape[:2]
                crop_x1, crop_y1 = max(0, x1), max(0, y1)
                crop_x2, crop_y2 = min(w_full, x2), min(h_full, y2)
                rosto_recortado = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                imagem_base64 = ""
                if rosto_recortado.size > 0:
                    _, buffer = cv2.imencode(".jpg", rosto_recortado)
                    imagem_base64 = base64.b64encode(buffer).decode("utf-8")
                    imagem_base64 = f"data:image/jpeg;base64,{imagem_base64}"

                novo_log = {
                    "nome": nome_detectado,
                    "tipo": "RECONHECIDO" if nome_detectado != "NAO ALUNO" else "NAO_ALUNO",
                    "data_hora_formatada": datetime.now().strftime("%d/%m/%Y - %H:%M:%S"),
                    "data_hora_raw": datetime.now(),
                    "imagem_rosto": imagem_base64,
                }
                await colecao_logs.insert_one(novo_log)

            resultados_pessoas = []
            for person in results.get("persons", []):
                bbox = person.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                resultados_pessoas.append(
                    {
                        "bbox": bbox,
                        "confidence": person.get("confidence"),
                    }
                )

            resultados_gestos = []
            for gesture in results.get("gestures", []):
                bbox = gesture.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                resultados_gestos.append(
                    {
                        "track_id": int(gesture.get("track_id", -1)),
                        "bbox": [int(valor) for valor in bbox],
                        "alerts": [
                            str(alert)
                            for alert in gesture.get("alerts", [])
                        ],
                        "confidence": gesture.get("confidence"),
                    }
                )

            resposta = {
                "rostos": resultados_faces,
                "pessoas": resultados_pessoas,
                "gestos": resultados_gestos,
            }
            await websocket.send_json(resposta)

    except WebSocketDisconnect:
        print("[INFO] Cliente Web desconectado.")
    except Exception as exc:
        print(f"[ERRO WEBSOCKET] {exc}")
