import os
from deepface import DeepFace

# Configurações - devem ser as mesmas do seu script principal
DIR_ALUNOS = "alunos"
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "opencv" # Vamos testar primeiro com o detector padrão

print(f"--- Iniciando Validação das Imagens na Pasta '{DIR_ALUNOS}' ---")
print(f"Usando o modelo: {MODEL_NAME} e o detector: {DETECTOR_BACKEND}\n")

sucesso = 0
falha = 0

if not os.path.isdir(DIR_ALUNOS):
    print(f"[ERRO] A pasta '{DIR_ALUNOS}' não foi encontrada.")
else:
    for filename in os.listdir(DIR_ALUNOS):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(DIR_ALUNOS, filename)
            try:
                # Tentamos representar a imagem com a regra estrita
                embedding_obj = DeepFace.represent(
                    img_path=path, 
                    model_name=MODEL_NAME, 
                    enforce_detection=True, 
                    detector_backend=DETECTOR_BACKEND
                )
                print(f"[SUCESSO] ✅ - Rosto encontrado e processado em: {filename}")
                sucesso += 1
            except ValueError as e:
                # Se der erro, significa que o DeepFace não encontrou um rosto válido
                print(f"[FALHA]   ❌ - Nenhum rosto detectado em: {filename} -> Erro: {e}")
                falha += 1

print("\n--- Validação Concluída ---")
print(f"Total de imagens com sucesso: {sucesso}")
print(f"Total de imagens com falha: {falha}")