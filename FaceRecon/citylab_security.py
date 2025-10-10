import cv2
import face_recognition
import os
import numpy as np
from ultralytics import YOLO
import threading # Importamos a biblioteca de threading
import time

# ================================
# Diretórios e Modelos (sem alteração)
# ================================
DIR_ALUNOS = "alunos"
known_face_encodings = []
known_face_names = []
print("[INFO] Carregando rostos conhecidos...")
for filename in os.listdir(DIR_ALUNOS):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(DIR_ALUNOS, filename)
        img = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(img, num_jitters=10)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])
print(f"[INFO] {len(known_face_encodings)} rostos carregados.")

model = YOLO("yolov8n.pt")

# ===============================================
# VARIÁVEIS COMPARTILHADAS ENTRE THREADS
# ===============================================
# Estas variáveis serão acessadas tanto pela thread principal quanto pela de processamento
latest_frame = None
last_known_faces = []
last_known_persons = []
# Um 'Lock' é essencial para evitar que as duas threads acessem as mesmas variáveis ao mesmo tempo
processing_lock = threading.Lock()
# Flag para sinalizar quando a thread de processamento deve parar
is_running = True

# ===============================================
# FUNÇÃO DA THREAD DE PROCESSAMENTO
# ===============================================
def process_frames():
    global latest_frame, last_known_faces, last_known_persons, is_running
    
    NUM_JITTERS_REALTIME = 5

    while is_running:
        # Pega uma cópia do último frame para processar
        with processing_lock:
            frame_to_process = latest_frame.copy() if latest_frame is not None else None

        if frame_to_process is None:
            time.sleep(0.01) # Espera um pouco se não houver frame
            continue

        # --- INÍCIO DO PROCESSAMENTO PESADO ---
        rgb_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
        
        # 1. DETECÇÃO DE ROSTOS
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        # 2. ENCODING
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=NUM_JITTERS_REALTIME)

        current_faces = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "NAO ALUNO"
            color = (0, 0, 255)
            if True in matches:
                idx = matches.index(True)
                name = known_face_names[idx]
                color = (0, 255, 0)
            box = (left, top, right, bottom)
            current_faces.append((box, name, color))

        # 3. DETECÇÃO COM YOLO
        results = model(frame_to_process, classes=[0], verbose=False)
        current_persons = []
        for r in results:
            for box in r.boxes:
                current_persons.append(box.xyxy[0].numpy().astype(int))
        # --- FIM DO PROCESSAMENTO PESADO ---

        # Atualiza as listas de resultados de forma segura (usando o lock)
        with processing_lock:
            last_known_faces = current_faces
            last_known_persons = current_persons

# ================================
# INICIALIZAÇÃO DA CÂMERA E DA THREAD
# ================================
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
print("🔒 CityLab Security rodando... Pressione 'q' para sair.")

# Inicia a thread de processamento em background
processing_thread = threading.Thread(target=process_frames, daemon=True)
processing_thread.start()

# ================================
# LOOP PRINCIPAL (THREAD PRINCIPAL)
# ================================
while True:
    ret, frame = cap.read()
    if not ret:
        is_running = False
        break

    # Atualiza o frame mais recente para a thread de processamento usar
    with processing_lock:
        latest_frame = frame.copy()
        
        # Pega uma cópia dos últimos resultados para desenhar, liberando o lock rapidamente
        faces_to_draw = list(last_known_faces)
        persons_to_draw = list(last_known_persons)

    # --- BLOCO DE DESENHO (SUPER RÁPIDO) ---
    for box, name, color in faces_to_draw:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    for box in persons_to_draw:
        x1, y1, x2, y2 = box
        is_suspect = False
        if is_suspect:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 3)
            cv2.putText(frame, "SUSPEITO", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 3)

    cv2.imshow("CityLab Security", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        is_running = False # Sinaliza para a thread parar
        break

# Encerramento limpo
processing_thread.join() # Espera a thread de processamento terminar
cap.release()
cv2.destroyAllWindows()