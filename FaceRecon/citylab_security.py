import cv2
import face_recognition
import os
import numpy as np
from ultralytics import YOLO

# ================================
# Configurações de diretórios
# ================================
DIR_ALUNOS = "alunos"
DIR_GESTOS = "gestos_suspeitos"

# ================================
# Carregar rostos de alunos
# ================================
known_face_encodings = []
known_face_names = []

for filename in os.listdir(DIR_ALUNOS):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(DIR_ALUNOS, filename)
        img = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(img)[0]
        known_face_encodings.append(encoding)
        name = os.path.splitext(filename)[0]
        known_face_names.append(name)

print(f"[INFO] {len(known_face_encodings)} alunos cadastrados carregados.")

# ================================
# Carregar modelo YOLO para gestos
# (treinado com dataset de gestos suspeitos)
# ================================
model = YOLO("yolov8n.pt")  # depois substituir pelo modelo customizado

# ================================
# Inicializar câmera
# ================================
cap = cv2.VideoCapture(0)  # Logitech C925e
cap.set(3, 1920)
cap.set(4, 1080)

print("🔒 CityLab Security rodando... Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =============================
    # Reconhecimento Facial
    # =============================
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "NÃO ALUNO"
        color = (0, 0, 255)  # vermelho

        if True in matches:
            first_match_index = matches.index(True)
            name = "ALUNO"
            color = (0, 255, 0)  # verde

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

    # =============================
    # Detecção de Gestos Suspeitos
    # =============================
    results = model(frame, stream=True, classes=[0])  # 0 = pessoa
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
            conf = float(box.conf[0])

            # Aqui entraria a lógica de "classificação de gesto suspeito"
            # (precisa de treinamento customizado com dataset)
            is_suspect = False  # <-- substituir por verificação real

            if is_suspect:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 3)
                cv2.putText(frame, "SUSPEITO", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 3)

    # =============================
    # Mostrar vídeo
    # =============================
    cv2.imshow("CityLab Security", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
