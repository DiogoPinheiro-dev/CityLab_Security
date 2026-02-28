import cv2
import numpy as np
import insightface
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional
from ultralytics import YOLO
import base64

# Atenção ao import: como o main.py está em 'Server' e o database em 'Server/Db', o import fica assim:
from Db.database import colecao_logs, colecao_alunos

# --- MODELOS PYDANTIC (Para padronizar as respostas da API) ---
class LogResponse(BaseModel):
    id: str
    nome: str
    tipo: str  # "RECONHECIDO" ou "NAO ALUNO"
    data_hora: str
    imagem_url: Optional[str] = None

# --- VARIÁVEIS GLOBAIS DE MEMÓRIA ---
# Vamos guardar os rostos aqui para não precisar consultar o banco a cada milissegundo
banco_rostos_memoria = {
    "nomes": [],
    "embeddings": []
}

# --- CARREGAMENTO DO MODELO ---
# Carregamos o InsightFace fora das rotas para que ele carregue apenas 1x quando a API ligar
print("[INFO] Carregando modelo InsightFace na CPU...")
app_insight = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
app_insight.prepare(ctx_id=0, det_size=(640, 640))
print("[INFO] InsightFace pronto.")

modelo_yolo = YOLO("../App/FaceRecon/yolov8n.pt") 

SIMILARITY_THRESHOLD = 0.52

# --- CICLO DE VIDA DA API ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] API Iniciada. Carregando alunos do MongoDB para a memória...")
    
    # Busca todos os alunos cadastrados
    cursor = colecao_alunos.find({})
    async for aluno in cursor:
        banco_rostos_memoria["nomes"].append(aluno["nome"])
        # Converte a lista do MongoDB de volta para um array do numpy
        banco_rostos_memoria["embeddings"].append(np.array(aluno["embedding"]))
        
    print(f"[INFO] {len(banco_rostos_memoria['nomes'])} alunos carregados com sucesso!")
    yield
    print("[INFO] API Desligada.")

app = FastAPI(title="API FaceRecon", lifespan=lifespan)

# --- ROTAS ---

@app.get("/")
async def home():
    """
    Rota Home: No futuro, essa rota (ou uma rota WebSocket /stream) 
    vai acionar o processamento de vídeo em tempo real.
    """
    return {"status": "online", "banco": "recon-db"}


@app.post("/cadastro")
async def cadastrar_aluno(nome: str = Form(...), foto: UploadFile = File(...)):
    """
    Recebe a foto via upload, extrai o embedding e salva no MongoDB.
    """
    if not foto.filename or not foto.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Formato inválido. Use JPG ou PNG.")

    try:
        # 1. Lê a imagem enviada na requisição em memória
        conteudo_arquivo = await foto.read()
        nparr = np.frombuffer(conteudo_arquivo, np.uint8)

        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Erro ao processar a imagem. Arquivo corrompido.")

        # 2. Passa a imagem no InsightFace
        faces = app_insight.get(img)

        # 3. Validações de face
        if not faces:
            raise HTTPException(status_code=400, detail="Nenhum rosto encontrado na foto enviada.")
        if len(faces) > 1:
            raise HTTPException(status_code=400, detail="Múltiplos rostos encontrados. Envie uma foto de apenas 1 pessoa.")

        # 4. Extrai a assinatura facial e converte para lista (MongoDB não aceita arrays do numpy nativamente)
        embedding_lista = faces[0].normed_embedding.tolist()

        # 5. Salva no MongoDB
        novo_aluno = {
            "nome": nome,
            "embedding": embedding_lista,
            "cadastrado_em": datetime.now().strftime("%d/%m/%Y - %H:%M:%S")
        }
        
        await colecao_alunos.insert_one(novo_aluno)

        banco_rostos_memoria["nomes"].append(nome)
        banco_rostos_memoria["embeddings"].append(np.array(embedding_lista))
        print(f"[DEBUG] Memória atualizada! Total de rostos agora: {len(banco_rostos_memoria['nomes'])}")

        return {
            "mensagem": f"Sucesso! Rosto de '{nome}' cadastrado.",
            "status": "sucesso"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {str(e)}")

@app.get("/logs", response_model=List[LogResponse])
async def visualizar_logs(limite: int = 50):
    """
    Rota de Logs: Retorna o histórico de reconhecimentos e alertas.
    """
    logs_db = []
    
    # Busca os logs no MongoDB ordenados do mais recente para o mais antigo
    cursor = colecao_logs.find().sort("data_hora_raw", -1).limit(limite)
    
    async for documento in cursor:
        logs_db.append({
            "id": str(documento["_id"]),
            "nome": documento["nome"],
            "tipo": documento["tipo"],
            "data_hora": documento["data_hora_formatada"], # Ex: 28/02/2026 - 14:30:15
            "imagem_url": documento.get("imagem_url", None)
        })

    return logs_db

# --- ROTA WEBSOCKET DE RECONHECIMENTO CONTÍNUO ---
@app.websocket("/stream")
async def websocket_reconhecimento(websocket: WebSocket):
    await websocket.accept()
    print("[INFO] Cliente Web conectado ao stream de vídeo.")
    
    # Dicionário para controlar o cooldown de logs (para não flodar o banco)
    recently_logged = {}
    LOG_COOLDOWN_SECONDS = 5 

    try:
        while True:
            # 1. Recebe o frame em bytes do HTML
            bytes_frame = await websocket.receive_bytes()
            
            # 2. Converte os bytes de volta para uma imagem OpenCV
            nparr = np.frombuffer(bytes_frame, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # 3. Processamento InsightFace (Reduzimos a escala para ficar mais rápido, igual no seu código original)
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            faces = app_insight.get(small_frame)

            # if len(faces) > 0:
            #     print(f"\n[DEBUG] --- Analisando frame: {len(faces)} rosto(s) detectado(s) na câmera ---")
            
            resultados_faces = []

            for face in faces:
                live_embedding = face.normed_embedding
                nome_detectado = "NAO ALUNO"
                
                # Se temos alunos na memória, fazemos a comparação
                if len(banco_rostos_memoria["embeddings"]) > 0:
                    # Calcula a similaridade (produto escalar) contra todos os rostos conhecidos
                    scores = np.dot(banco_rostos_memoria["embeddings"], live_embedding)
                    melhor_indice = np.argmax(scores)
                    melhor_score = scores[melhor_indice]

                    # total_banco = len(banco_rostos_memoria['nomes'])
                    # print(f"[DEBUG] Comparando rosto da câmera com {total_banco} pessoa(s) no banco:")
                    
                    # for i, score in enumerate(scores):
                    #     nome_banco = banco_rostos_memoria['nomes'][i]
                        # print(f"   -> Chance de ser '{nome_banco}': {score:.4f}")
                    
                    # print(f"   => VENCEDOR: '{banco_rostos_memoria['nomes'][melhor_indice]}' com score de {melhor_score:.4f} (Mínimo exigido: {SIMILARITY_THRESHOLD})")

                    if melhor_score > SIMILARITY_THRESHOLD:
                        nome_detectado = banco_rostos_memoria["nomes"][melhor_indice]
                        # print(f"   => STATUS: RECONHECIDO!")
                    
                    # else:
                    #     print(f"   => STATUS: DESCONHECIDO (Score abaixo de {SIMILARITY_THRESHOLD})")
                
                # Pega as coordenadas e ajusta a escala de volta para o tamanho original
                bbox = [int(coord / 0.5) for coord in face.bbox.astype(int)]

                x1, y1, x2, y2 = bbox
                
                # Adiciona o resultado para enviar ao HTML desenhar a caixa
                resultados_faces.append({"nome": nome_detectado, "bbox": bbox})
                
                # --- LÓGICA DE SALVAR LOGS E FOTOS NO MONGODB ---
                tempo_atual = time.time()
                # Verifica se passou o tempo de cooldown para não salvar 30 fotos por segundo da mesma pessoa
                if nome_detectado not in recently_logged or (tempo_atual - recently_logged[nome_detectado] > LOG_COOLDOWN_SECONDS):
                    recently_logged[nome_detectado] = tempo_atual
                    
                    # 1. Garante que as coordenadas não vazem dos limites da imagem (evita erro no OpenCV)
                    h_full, w_full = frame.shape[:2]
                    crop_x1, crop_y1 = max(0, x1), max(0, y1)
                    crop_x2, crop_y2 = min(w_full, x2), min(h_full, y2)
                    
                    # 2. Recorta apenas o rosto do frame original
                    rosto_recortado = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    # 3. Converte o recorte para Base64 (Texto)
                    imagem_base64 = ""
                    if rosto_recortado.size > 0:
                        _, buffer = cv2.imencode('.jpg', rosto_recortado)
                        imagem_base64 = base64.b64encode(buffer).decode('utf-8')
                        imagem_base64 = f"data:image/jpeg;base64,{imagem_base64}" # Formato pronto para web
                    
                    # 4. Monta o documento com Nome, Data, Hora e a Foto
                    novo_log = {
                        "nome": nome_detectado,
                        "tipo": "RECONHECIDO" if nome_detectado != "NAO ALUNO" else "NAO_ALUNO",
                        "data_hora_formatada": datetime.now().strftime("%d/%m/%Y - %H:%M:%S"),
                        "data_hora_raw": datetime.now(),
                        "imagem_rosto": imagem_base64 # A foto salva direto no banco!
                    }
                    
                    # 5. Salva no banco de dados
                    await colecao_logs.insert_one(novo_log)
                    print(f"[LOG DB] Salvo no banco: {nome_detectado} com a foto do rosto.")

            # 4. Processamento YOLO (Detectar Pessoas)
            resultados_pessoas = []
            results_yolo = modelo_yolo(small_frame, classes=[0], verbose=False)
            for r in results_yolo:
                for box in r.boxes:
                    bbox_person = [int(coord / 0.5) for coord in box.xyxy[0].numpy().astype(int)]
                    resultados_pessoas.append({"bbox": bbox_person})

            # 5. Devolve as caixas (Bounding Boxes) e os Nomes para o HTML desenhar
            resposta = {
                "rostos": resultados_faces,
                "pessoas": resultados_pessoas
            }
            await websocket.send_json(resposta)

    except WebSocketDisconnect:
        print("[INFO] Cliente Web desconectado.")
    except Exception as e:
        print(f"[ERRO WEBSOCKET] {e}")