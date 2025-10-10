from ultralytics import YOLO

# Carregar um modelo pré-treinado (yolov8n.pt é um bom começo por ser leve)
model = YOLO('yolov8n.pt')

# Treinar o modelo usando o seu dataset
if __name__ == '__main__':
    results = model.train(data='GestureRecon/gestos.yaml', epochs=50, imgsz=640)