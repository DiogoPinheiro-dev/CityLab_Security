import os
import pickle
from pathlib import Path

import cv2
import insightface
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    app_dir = SCRIPT_DIR.parent
    dir_alunos = app_dir / "alunos"
    arquivo_base_dados = SCRIPT_DIR / "base_dados_alunos.pkl"

    print("--- INICIANDO PROCESSO DE CADASTRO DE ROSTOS (com InsightFace) ---")

    app = insightface.app.FaceAnalysis(
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("[INFO] Modelo InsightFace carregado.")

    known_face_embeddings = []
    known_face_names = []

    if dir_alunos.is_dir():
        for filename in os.listdir(dir_alunos):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                path = dir_alunos / filename
                try:
                    img = cv2.imread(str(path))
                    faces = app.get(img)

                    if faces and len(faces) == 1:
                        known_face_embeddings.append(faces[0].normed_embedding)
                        known_face_names.append(path.stem)
                        print(f"[SUCESSO] - Rosto de '{path.stem}' cadastrado.")
                    elif not faces:
                        print(f"[FALHA] - Nenhum rosto encontrado em '{filename}'.")
                    else:
                        print(f"[FALHA] - Multiplos rostos encontrados em '{filename}'. Apenas um e permitido.")
                except Exception as exc:
                    print(f"[ERRO] - Erro ao processar '{filename}': {exc}")

    if known_face_embeddings:
        data = {"embeddings": np.array(known_face_embeddings), "names": known_face_names}
        with open(arquivo_base_dados, "wb") as file:
            pickle.dump(data, file)
        print(f"\n[SUCESSO] Base de dados salva em '{arquivo_base_dados}' com {len(known_face_names)} rostos.")
    else:
        print("\n[ERRO] Nenhum rosto pode ser cadastrado. A base de dados nao foi criada.")

    print("--- PROCESSO DE CADASTRO CONCLUIDO ---")
