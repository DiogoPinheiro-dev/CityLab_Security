import cv2
from ultralytics import YOLO
from detector import GestureAnalyzer

def main():
    # Inicializa o modelo de Pose. 
    # O 'yolov8n-pose.pt' é a versão "nano" (mais rápida, ideal para tempo real).
    # Baixará automaticamente se não existir.
    model = YOLO("yolov8n-pose.pt")
    
    # Inicializa o analisador de gestos
    analyzer = GestureAnalyzer(fps=30)
    
    # Tenta usar a câmera externa primeiro (índice 1, ou maior),
    # Se falhar ou não existir, usa a webcam nativa (índice 0).
    video_source = 1
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened() or not cap.read()[0]:
        print("Câmera externa não encontrada. Iniciando webcam nativa...")
        cap.release()
        video_source = 0
        cap = cv2.VideoCapture(video_source)
        
    if not cap.isOpened():
        print(f"Erro ao abrir qualquer fonte de vídeo.")
        return

    print("Iniciando detecção... Pressione 'q' para sair.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou erro na leitura.")
            break
            
        # Roda o YOLO Tracking no frame.
        # persist=True avisa o modelo que os frames pertencem ao mesmo vídeo.
        # tracker="bytetrack.yaml" usa o ByteTrack, excelente para multidões.
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        current_tracks = []

        if results and len(results) > 0:
            result = results[0]
            
            # Se encontrou pessoas e possui tracking IDs
            if result.boxes is not None and result.boxes.id is not None and result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                keypoints_batch = result.keypoints.data.cpu().numpy() # [N_pessoas, 17, 3]

                for box, track_id, keypoints in zip(boxes, track_ids, keypoints_batch):
                    current_tracks.append(track_id)
                    
                    # Analisa o comportamento da pessoa com base nas suas coordenadas dos membros e bounding box
                    alerts = analyzer.analyze(track_id, keypoints, box)
                    
                    # --- Desenho Visual ---
                    x1, y1, x2, y2 = map(int, box)
                    
                    if alerts:
                        color = (0, 0, 255) # BGR: Vermelho se tiver alerta
                    else:
                        color = (0, 255, 0) # Verde se estiver normal
                        
                    # Desenha a Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Desenha o ID da pessoa
                    label = f"ID: {track_id}"
                    cv2.putText(frame, label, (x1, max(0, y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Desenha os alertas um embaixo do outro
                    if alerts:
                        for i, alert in enumerate(alerts):
                            text_y = max(0, y1 - 10) + (i * 20)
                            cv2.putText(frame, alert, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
        # Limpa da memória IDs antigos que não aparecem mais na tela
        analyzer.clean_old_tracks(current_tracks)
        
        # Mostra o frame
        cv2.imshow("Suspicious Gesture Recognition (CityLab)", frame)
        
        # Saída com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
