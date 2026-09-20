"""Compara reconhecimento facial antigo/otimizado nas mesmas imagens, sem salvar biometria."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def compare(service, context, np):
    original_modules = service.app_insight.models
    original_prefilter = service.prefilter
    original_match = service._match_face
    vectors = []

    def capture(embedding):
        vectors.append(embedding.copy())
        return original_match(embedding)

    try:
        service._match_face = capture
        service.prefilter = False
        baseline = service.recognize_faces(context)
        baseline_vectors = list(vectors)
        vectors.clear()
        service.app_insight.models = {name: model for name, model in original_modules.items()
                                      if name in {"detection", "recognition"}}
        service.prefilter = True
        optimized = service.recognize_faces(context)
        if len(baseline) != len(optimized):
            raise AssertionError("Quantidade de rostos mudou")
        if len(baseline_vectors) != len(baseline) or len(vectors) != len(optimized):
            raise AssertionError("Quantidade de embeddings mudou")
        for before, after, before_vec, after_vec in zip(baseline, optimized, baseline_vectors, vectors):
            if before["bbox"] != after["bbox"] or before["name"] != after["name"]:
                raise AssertionError("Caixa ou identidade mudou")
            if not np.allclose(before_vec, after_vec, rtol=1e-5, atol=1e-6):
                raise AssertionError("Embedding mudou alem da tolerancia")
            if not np.isclose(before["confidence"], after["confidence"], rtol=1e-5, atol=1e-6):
                raise AssertionError("Similaridade mudou alem da tolerancia")
        return len(baseline)
    finally:
        service._match_face = original_match
        service.prefilter = original_prefilter
        service.app_insight.models = original_modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    import cv2
    import numpy as np
    from App.FaceRecon.service import FaceRecognitionService
    from App.frame_context import build_frame_context
    from App.settings import PROCESS_SCALE, EXPERIMENTAL_GRAYSCALE

    # Uma unica copia dos modelos em RAM. Nao mede economia de memoria.
    service = FaceRecognitionService(lazy_person_model=True, minimal_modules=False,
                                     prefilter=False, onnx_threads=0)
    total = 0
    for index, path in enumerate(args.images, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Imagem {index} invalida")
        context = build_frame_context(frame, PROCESS_SCALE, EXPERIMENTAL_GRAYSCALE)
        count = compare(service, context, np)
        total += count
        print(f"Imagem {index}: equivalencia OK; {count} rostos aceitos")
    if not total:
        raise SystemExit("Inconclusivo: nenhuma imagem teve rosto aceito; inclua rostos validos.")
    print("Verificacao concluida sem gravar imagens, nomes ou embeddings.")


if __name__ == "__main__":
    main()
