import cv2
import face_recognition
import os
import numpy as np
from ultralytics import YOLO

# ================================
# Diretórios e Modelos
# ================================
DIR_ALUNOS = "alunos"

known_face_encodings = []
known_face_names = []

print("[INFO] Carregando rostos conhecidos...")
for filename in os.listdir(DIR_ALUNOS):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(DIR_ALUNOS, filename)
        img = face_recognition.load_image_file(path)
        # Ao carregar, usamos mais jitters para criar encodings mais robustos
        encodings = face_recognition.face_encodings(img, num_jitters=10) 
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

print(f"[INFO] {len(known_face_encodings)} rostos carregados.")

# REMOVIDO: Não usaremos mais o HaarCascade, pois é menos robusto a rotações.
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

model = YOLO("yolov8n.pt")

# ================================
# Inicializar câmera
# ================================
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

print("🔒 CityLab Security rodando... Pressione 'q' para sair.")

# ===============================================
# VARIÁVEIS DE OTIMIZAÇÃO E CONFIGURAÇÃO
# ===============================================
PROCESS_EVERY_N_FRAMES = 10
frame_count = 0

# NOVO PARÂMETRO: Aumenta a robustez do reconhecimento a diferentes ângulos.
# Valores maiores são mais precisos, mas mais lentos. Comece com 5.
NUM_JITTERS_REALTIME = 5

last_known_faces = []
last_known_persons = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        last_known_faces = []
        last_known_persons = []

        # Converte o frame para RGB (não precisamos mais de tons de cinza)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # =========================================================================
        # --- RECONHECIMENTO FACIAL REFINADO ---
        # =========================================================================

        # 1. DETECÇÃO DE ROSTOS: Usando o modelo HOG, mais robusto que o HaarCascade.
        # Ele já retorna as coordenadas no formato (top, right, bottom, left).
        face_locations = face_recognition.face_locations(rgb_frame, model="hog") # 'hog' é mais rápido, 'cnn' é mais preciso

        # 2. ENCODING: Gerando os encodings para os rostos encontrados com num_jitters.
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=NUM_JITTERS_REALTIME)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "NAO ALUNO"
            color = (0, 0, 255)

            if True in matches:
                idx = matches.index(True)
                name = known_face_names[idx]
                color = (0, 255, 0)
            
            # Salva o resultado (bounding box, nome, cor) na lista
            # Atenção: Convertendo as coordenadas de volta para (x1, y1, x2, y2)
            box = (left, top, right, bottom)
            last_known_faces.append((box, name, color))

        # --- DETECÇÃO DE PESSOAS COM YOLO (sem alteração) ---
        results = model(frame, classes=[0], verbose=False)
        for r in results:
            for box in r.boxes:
                last_known_persons.append(box.xyxy[0].numpy().astype(int))

    # ===============================================================
    # BLOCO DE DESENHO (sem alteração)
    # ===============================================================
    for box, name, color in last_known_faces:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    for box in last_known_persons:
        x1, y1, x2, y2 = box
        is_suspect = False
        if is_suspect:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 3)
            cv2.putText(frame, "SUSPEITO", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 3)

    cv2.imshow("CityLab Security", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()