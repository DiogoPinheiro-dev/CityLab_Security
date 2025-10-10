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
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(DIR_ALUNOS, filename)
        img = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(img)

        if encodings:
            encoding = encodings[0]
            known_face_encodings.append(encoding)
            # Nome da pessoa = nome do arquivo sem extensão
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)

print(f"[INFO] {len(known_face_encodings)} rostos de alunos carregados.")

# ================================
# Carregar modelo YOLO (gestos suspeitos)
# ================================
model = YOLO("yolov8n.pt")  # modelo base ou customizado depois

# ================================
# Inicializar câmera
# ================================
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(3, 1280)  # largura
cap.set(4, 720)   # altura

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
        # Tolerância aumentada para melhorar reconhecimento com variações
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "NAO ALUNO"
        color = (0, 0, 255)  # vermelho

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]  # mostra o nome real
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

            # Placeholder: lógica real virá com modelo treinado
            is_suspect = False  

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
