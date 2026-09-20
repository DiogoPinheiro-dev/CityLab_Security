"""Exporta o peso de pose existente para comparacao opcional no Raspberry."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if not args.model.is_file() or args.model.suffix != ".pt":
        parser.error("Informe um arquivo .pt de pose existente")
    if args.imgsz <= 0 or args.imgsz % 32:
        parser.error("imgsz deve ser positivo e multiplo de 32")
    destination = args.model.with_name(args.model.stem + "_ncnn_model")
    if destination.exists():
        parser.error("Diretorio NCNN ja existe; use uma copia do peso em outro diretorio")

    from ultralytics import YOLO

    model = YOLO(str(args.model.resolve()), task="pose")
    result = model.export(format="ncnn", imgsz=args.imgsz, half=False, device="cpu")
    print(f"Exportado para: {result}")
    print("Valide caixas, keypoints, alertas e desempenho no Pi antes de ativar POSE_MODEL_PATH.")


if __name__ == "__main__":
    main()
