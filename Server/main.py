import asyncio
import io
import os
import socket
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLIENT_DIR = PROJECT_ROOT / "Client"

from App.recognition_pipeline import UnifiedRecognitionService, create_unified_service
from App.settings import (
    DEBUG_PIPELINE,
    ENABLE_PERFORMANCE_METRICS,
    ENABLE_SYSTEM_MONITOR,
    JPEG_QUALITY,
    MAX_IN_FLIGHT_FRAMES,
    PIPELINE_MAX_WORKERS,
    PROCESS_SCALE,
    STREAM_FPS,
    STREAM_HEIGHT,
    STREAM_WIDTH,
)
from Server.Db.database import MONGO_DB_NAME, colecao_alunos, colecao_logs, validar_conexao_mongo
from Server.event_logger import EventLogger
from Server.system_monitor import SystemMonitor


class LogResponse(BaseModel):
    id: str
    nome: str
    tipo: str
    data_hora: str
    imagem_url: Optional[str] = None


class ClientStreamConfig(BaseModel):
    stream_fps: int
    jpeg_quality: float
    stream_width: int
    stream_height: int
    max_in_flight_frames: int


banco_rostos_memoria = {
    "nomes": [],
    "embeddings": [],
}

recognizer: Optional[UnifiedRecognitionService] = None
event_logger = EventLogger(colecao_logs)
system_monitor = SystemMonitor(enabled=ENABLE_SYSTEM_MONITOR)


def _sync_memoria_para_recognizer() -> None:
    current_recognizer = recognizer
    if current_recognizer is None or current_recognizer.face_service is None:
        return

    current_recognizer.face_service.replace_known_faces(
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
    recognizer = create_unified_service(
        max_workers=PIPELINE_MAX_WORKERS,
        process_scale=PROCESS_SCALE,
        enable_performance_metrics=ENABLE_PERFORMANCE_METRICS,
        debug_pipeline=DEBUG_PIPELINE,
    )

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

    if recognizer is not None:
        recognizer.close()
    print("[INFO] API desligada.")


app = FastAPI(title="API FaceRecon", lifespan=lifespan)

if CLIENT_DIR.exists():
    app.mount("/client", StaticFiles(directory=CLIENT_DIR), name="client")


@app.get("/")
async def home():
    return {"status": "online", "banco": MONGO_DB_NAME}


@app.get("/config/client", response_model=ClientStreamConfig)
async def client_stream_config():
    return ClientStreamConfig(
        stream_fps=STREAM_FPS,
        jpeg_quality=JPEG_QUALITY,
        stream_width=STREAM_WIDTH,
        stream_height=STREAM_HEIGHT,
        max_in_flight_frames=MAX_IN_FLIGHT_FRAMES,
    )


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
        from qrcode.constants import ERROR_CORRECT_M
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
        error_correction=ERROR_CORRECT_M,
        box_size=10,
        border=3,
    )
    qr.add_data(cadastro_url)
    qr.make(fit=True)

    image = qr.make_image(fill_color="#102330", back_color="#ffffff")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")


@app.post("/cadastro")
async def cadastrar_aluno(nome: str = Form(...), foto: UploadFile = File(...)):
    if not foto.filename or not foto.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Formato invalido. Use JPG ou PNG.")

    current_recognizer = recognizer
    if current_recognizer is None:
        raise HTTPException(status_code=503, detail="Pipeline de reconhecimento nao esta pronta.")
    if current_recognizer.face_service is None:
        raise HTTPException(status_code=503, detail="Servico facial nao esta disponivel neste ambiente.")

    try:
        conteudo_arquivo = await foto.read()
        nparr = np.frombuffer(conteudo_arquivo, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Erro ao processar a imagem. Arquivo corrompido.")

        faces = current_recognizer.face_service.app_insight.get(img)

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

    try:
        while True:
            current_recognizer = recognizer
            if current_recognizer is None:
                await websocket.send_json({"erro": "Pipeline de reconhecimento nao inicializada."})
                await asyncio.sleep(0.2)
                continue

            frame_started_at = time.perf_counter()
            bytes_frame = await websocket.receive_bytes()
            receive_ms = (time.perf_counter() - frame_started_at) * 1000.0

            decode_started_at = time.perf_counter()
            nparr = np.frombuffer(bytes_frame, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            decode_ms = (time.perf_counter() - decode_started_at) * 1000.0

            if frame is None:
                continue

            pipeline_started_at = time.perf_counter()
            results = current_recognizer.process_frame(frame)
            pipeline_ms = (time.perf_counter() - pipeline_started_at) * 1000.0

            faces = [face for face in results.get("faces", []) if not face.get("debug_only")]
            gestures = results.get("gestures", [])

            log_started_at = time.perf_counter()
            await event_logger.log_face_events(frame, faces)
            await event_logger.log_gesture_events(frame, gestures)
            logs_ms = (time.perf_counter() - log_started_at) * 1000.0

            resultados_faces = []
            for face in faces:
                bbox = face.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                resultados_faces.append(
                    {
                        "nome": face.get("name", "NAO ALUNO"),
                        "bbox": bbox,
                        "confidence": face.get("confidence"),
                    }
                )

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
            for gesture in gestures:
                bbox = gesture.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                resultados_gestos.append(
                    {
                        "track_id": int(gesture.get("track_id", -1)),
                        "bbox": [int(valor) for valor in bbox],
                        "alerts": [str(alert) for alert in gesture.get("alerts", [])],
                        "confidence": gesture.get("confidence"),
                    }
                )

            resposta = {
                "rostos": resultados_faces,
                "pessoas": resultados_pessoas,
                "gestos": resultados_gestos,
            }

            metrics = dict(results.get("metrics", {}))
            metrics["receive_ms"] = receive_ms
            metrics["decode_ms"] = decode_ms
            metrics["pipeline_ms"] = pipeline_ms
            metrics["logs_ms"] = logs_ms
            metrics["total_ms"] = (time.perf_counter() - frame_started_at) * 1000.0
            if metrics["total_ms"] > 0:
                metrics["effective_fps"] = 1000.0 / metrics["total_ms"]

            system_monitor.record_frame_metrics(metrics)
            system_monitor.maybe_log_snapshot()

            if DEBUG_PIPELINE or ENABLE_PERFORMANCE_METRICS:
                resposta["metrics"] = metrics
            if DEBUG_PIPELINE and "debug" in results:
                resposta["debug"] = results["debug"]

            await websocket.send_json(resposta)

    except WebSocketDisconnect:
        print("[INFO] Cliente Web desconectado.")
    except Exception as exc:
        print(f"[ERRO WEBSOCKET] {exc}")
